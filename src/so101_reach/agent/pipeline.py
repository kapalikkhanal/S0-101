"""Pipeline mode: code does the predictable parts, the agent only grasps and places.

For a task, an ordered list of pick-and-place steps (from --object/--place directly, or one
planner call for free-text tasks such as stacking / unstacking). For each step:

  1. code   park the arm out of the head camera's view, measure the object and the target
  2. code   open the gripper to the object's width + 2 cm, rise, RL travel to 5 cm above the object
  3. agent  GRASP phase (short prompt, fine-motion tools) - ends with done; code checks the
            holding signal and retries from 1 (up to `grasp_retries`) if nothing is held
  4. code   lift above everything nearby, RL travel to 5 cm above the target
  5. agent  PLACE phase - decides drop / lower inside / stack from the object and the target, releases
  6. code   rise straight up
and finally RL home. Works for any object the perception can measure: containers, surfaces and
other objects (stacking) are all just "targets" with a measured top height.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from so101_reach.agent.perception import Detection, annotate

GRASP_INSTRUCTIONS = """You control a 5-joint robot arm with a parallel-jaw gripper (units: cm, degrees; x forward,
y left, h = height above the table). Positions refer to the GRIP POINT: midway between the jaw tips,
1.5 cm in from the tips (red TIP circle in the images). The arm holds still between your tool calls.

YOUR JOB NOW: GRASP the object named in the task. The arm is already 5 cm above it with the gripper
open just wider than the object. Do not lift or carry it afterwards - the system does that.
- If the images show the jaws are not centred over the object, correct with small move_by steps.
- Turn jaw_yaw (set_orientation) only if the jaws are not across the object's narrow side.
- Lower straight down (move_by dz<0 or move_to) until the grip point is at about the middle of the
  object's height (for an object standing on another object: about 1/3 of its height below its top,
  so the jaws do not touch the object underneath). If a move is refused, read the reason (it gives
  the lowest allowed height) and use that.
- close_gripper: gentle for soft or fragile objects, firm for hard ones. The result says whether
  something was caught. If nothing was caught: open again, rise 2 cm, re-centre, retry.
- The phase ends by itself as soon as the object is held. Call done(success=false) only if you cannot grasp it.
Be quick: one action per turn, no unnecessary checks."""

PLACE_INSTRUCTIONS = """You control a 5-joint robot arm with a parallel-jaw gripper (units: cm, degrees; x forward,
y left, h = height above the table). Positions refer to the GRIP POINT between the jaw tips.

YOUR JOB NOW: PLACE the object you are holding on / into the target named in the task. The arm is
already above the target's centre; the task text gives the held object's bottom height, the
target's top (rim) height and pre-computed heights. Decide from what the object is and what the
target is:
- Sturdy object into an open container: release from ABOVE - lower (if needed) until the held
  object's bottom is about 2 cm above the rim, then open_gripper. Never go below the rim: the jaws
  would hit the rim and push the container.
- Fragile object (egg, glass, phone) into a container: lower inside until its bottom is ~0.5 cm
  above the container floor, keeping the jaws clear of the walls, then open gently.
- Onto another object (stacking): centre exactly over it (small move_by if the images show an
  offset), lower until the held bottom is ~0.4 cm above that object's top, then open_gripper just
  wider than the held object.
- Opening the gripper releases the object and ends the phase by itself. Do not move away or go home -
  the system does that.
Be quick: one action per turn."""

FINE_TOOLS = ["move_by", "move_to", "set_orientation", "open_gripper", "close_gripper"]
PLACE_TOOLS = ["move_by", "move_to", "set_orientation", "open_gripper"]


def _cm(v: float) -> float:
  return round(100 * v, 1)


class Pipeline:
  def __init__(self, agent, arm, per, grasp_retries: int = 2, grasp_turns: int = 10, place_turns: int = 6,
               hover_cm: float = 5.0, clear_view: bool = True, log=print):
    self.agent, self.arm, self.per = agent, arm, per
    self.grasp_retries, self.grasp_turns, self.place_turns = grasp_retries, grasp_turns, place_turns
    self.hover = hover_cm / 100
    self.use_clear_view = clear_view
    self.say = log
    self.tz = per.table_z

  # --- perception helpers ------------------------------------------------------------------
  def _measure(self, names: list[str]) -> dict[str, Detection]:
    if self.use_clear_view:
      self.arm.clear_view()
    self.per.views()
    out = {}
    for n in names:
      dets = self.per.find(n)
      if dets:
        out[n] = dets[0]
    if self.use_clear_view:
      self.arm.home()  # leave the parking pose on its checked joint path (RL travel starts from home)
    return out

  def _objects_text(self, dets: dict[str, Detection]) -> str:
    for i, d in enumerate(dets.values()):
      d.id = i
    return "\n".join(d.describe(self.tz) for d in dets.values()) or "(nothing detected)"

  def _head_b64(self, width: int = 640) -> str | None:
    views = self.per.views()
    if "head" not in views:
      return None
    img = annotate(views["head"], [], self.arm.geom(self.arm.q6()[0])["gp"], self.tz, "head camera", axes=True)
    h = int(img.shape[0] * width / img.shape[1])
    return base64.b64encode(cv2.imencode(".jpg", cv2.resize(img, (width, h)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()

  # --- motion helpers ----------------------------------------------------------------------
  def _rise_to(self, h_m: float) -> str:
    st = self.arm.state()
    h_now = self.tz + st["grip_point_cm"][2] / 100
    if h_now >= h_m - 0.005:
      return "ok"
    # move_by is capped at 10 cm; large rises still use checked straight segments.
    while h_now < h_m - 0.005:
      r = self.arm.move_by(0, 0, min(9.0, 100 * (h_m - h_now)))
      if not r.startswith("ok"):
        return r
      new_h = self.tz + self.arm.state()["grip_point_cm"][2] / 100
      if new_h <= h_now + 0.001:
        return "stopped: rise made no progress"
      h_now = new_h
    return "ok"

  def _go_above(self, d: Detection, h_m: float, jaw_yaw: float | None) -> str:
    x, y = _cm(d.center[0]), _cm(d.center[1])
    r = self.arm.move_to(x, y, _cm(h_m - self.tz), 0.0, jaw_yaw)
    if r.startswith("refused") or r.startswith("stopped") or r.startswith("travel"):
      r2 = self.arm.move_to(x, y, _cm(h_m - self.tz), None, jaw_yaw)  # keep the reachable tilt
      return r2 if not r2.startswith("ok") else "ok"
    return r

  def _jaw_for(self, obj: Detection, width: float) -> float:
    """Jaw closing direction square to the object's faces: across the narrow side of an elongated
    object; for a square / round one, whichever face direction needs the smaller wrist turn
    (closing diagonally on a square block catches only its corners and it slips out)."""
    if max(obj.size[:2]) > 1.3 * width:
      return obj.yaw_deg + 90
    now = self.arm.state()["jaw_yaw_deg"]
    cands = [obj.yaw_deg, obj.yaw_deg + 90]
    return min(cands, key=lambda c: abs((c - now + 90) % 180 - 90))

  # --- one pick-and-place step -------------------------------------------------------------
  def step(self, k: int, pick: str, place_on: str, all_names: list[str]) -> dict:
    res = {"step": k, "pick": pick, "place_on": place_on, "grasp_attempts": 0}
    held_below = None
    obj = None
    for attempt in range(self.grasp_retries + 1):
      res["grasp_attempts"] = attempt + 1
      dets = self._measure(all_names)
      if pick not in dets or place_on not in dets:
        res["result"] = f"not detected: {', '.join(n for n in (pick, place_on) if n not in dets)}"
        return res
      obj = dets[pick]
      clearance = max(d.top_z for d in dets.values()) + 0.03
      width = min(obj.size[:2])
      self.arm.gripper(min(80.0, self.arm.percent_for_gap(width + 0.02)), "medium")
      r = self._rise_to(max(clearance, self.arm.geom(self.arm.q6()[0])["gp"][2]))
      if not r.startswith("ok"):
        return {**res, "result": r}
      jaw = self._jaw_for(obj, width)
      r = self._go_above(obj, obj.top_z + self.hover, jaw)
      self.say(f"  [pipeline] step {k} attempt {attempt + 1}: above {pick} -> {r}")
      if not r.startswith("ok"):
        return {**res, "result": r}
      self.agent.phase = f"s{k}_grasp{attempt + 1}_"
      task = (f"GRASP the {pick}. {obj.describe(self.tz)}. The grip point is now about {100 * self.hover:.0f} cm above its top; "
              f"the gripper is open about {100 * (width + 0.02):.1f} cm.")
      out = self.agent.run(task, [pick], GRASP_INSTRUCTIONS, FINE_TOOLS, self.grasp_turns,
                           stop_when=lambda tool, res: tool == "close_gripper" and self.arm.state()["holding_signal"])
      st = self.arm.state()
      self.say(f"  [pipeline] grasp phase: agent says {out.get('success')} ({out.get('summary', '')[:80]}); holding={st['holding_signal']}")
      if st["holding_signal"]:
        gp = self.arm.geom(self.arm.q6()[0])["gp"]
        held_below = float(gp[2] - obj.support_z)
        self.arm.held = {"name": pick, "below_gp": held_below, "size": obj.size}
        break
      self.arm.gripper(min(80.0, self.arm.percent_for_gap(width + 0.02)), "medium")
      self._rise_to(obj.top_z + self.hover)
    if held_below is None:
      res["result"] = "grasp failed"
      return res
    # Carry: above everything nearby, then RL travel to 5 cm (held bottom) above the target.
    tgt = dets[place_on]
    others = [d.top_z for n, d in dets.items() if n != pick]
    carry = max(others + [tgt.top_z]) + held_below + 0.04
    r = self._rise_to(carry)
    if not r.startswith("ok"):
      return {**res, "result": r}
    h_above = max(carry, tgt.top_z + held_below + self.hover)
    r = self._go_above(tgt, h_above, None)
    self.say(f"  [pipeline] carried {pick} above {place_on} -> {r}")
    if not r.startswith("ok"):
      return {**res, "result": r}
    gp_h = self.arm.state()["grip_point_cm"][2] / 100
    bottom = gp_h - held_below
    container = bool(tgt.extra.get("container")) or max(tgt.size[:2]) > 1.8 * max(obj.size[:2])
    hint_drop = tgt.top_z - self.tz + 0.02 + held_below
    hint_stack = tgt.top_z - self.tz + 0.004 + held_below
    self.agent.phase = f"s{k}_place_"
    task = (f"PLACE the {pick} you are holding on/into the {place_on}.\n"
            f"Held: {pick}, size {' x '.join(f'{100 * v:.1f}' for v in obj.size)} cm; its bottom is now at h={100 * bottom:.1f} cm "
            f"({100 * held_below:.1f} cm below the grip point; grip point at h={100 * gp_h:.1f}).\n"
            f"Target: {tgt.describe(self.tz)}" + (" (an open container)" if container else "") + ".\n"
            f"Pre-computed grip-point heights: held bottom 2 cm above the target's top/rim -> h={100 * hint_drop:.1f} "
            f"(from here: move_by dz={100 * (hint_drop - gp_h):+.1f}); held bottom 0.4 cm above the target's top (stacking) -> "
            f"h={100 * hint_stack:.1f} (from here: move_by dz={100 * (hint_stack - gp_h):+.1f}).\n"
            f"Opening the gripper releases the object and ends this phase.")
    out = self.agent.run(task, [pick, place_on], PLACE_INSTRUCTIONS, PLACE_TOOLS, self.place_turns,
                         stop_when=lambda tool, res: tool == "open_gripper" and not self.arm.state()["holding_signal"])
    st = self.arm.state()
    if st["holding_signal"] or st["gripper_open_percent"] < 5:
      return {**res, "result": "place failed: object remains held"}
    self.arm.held = None
    self._rise_to(self.tz + self.arm.state()["grip_point_cm"][2] / 100 + 0.04)
    # Release alone does not prove placement. Refresh geometry after settling.
    self.arm.wait(0.5)
    self.per.views()
    measured = [d for d in self.per.find(pick) if not d.extra.get("note")]
    verified = any(np.linalg.norm(d.center[:2] - tgt.center[:2]) < max(tgt.size[:2]) / 2
                   and d.support_z <= tgt.top_z + 0.015 for d in measured)
    res["result"] = "placed" if out.get("success") and verified else "placement unverified"
    return res

  # --- whole task --------------------------------------------------------------------------
  def run(self, task: str, names: list[str], steps: list[dict] | None = None) -> dict:
    t0 = time.time()
    if steps is None:  # free-text task: one planner call
      dets = self._measure(names)
      steps = self.agent.plan(task, self._objects_text(dets), self._head_b64())
      self.say(f"  [pipeline] plan: {steps}")
    if not steps:
      return {"success": False, "steps": [], "error": "planner returned no steps", "wall_s": round(time.time() - t0, 1)}
    results = []
    for k, s in enumerate(steps, 1):
      if s.get("pick") not in names or s.get("place_on") not in names:
        results.append({"step": k, **s, "result": "unknown object name in the plan"})
        break
      r = self.step(k, s["pick"], s["place_on"], names)
      results.append(r)
      self.say(f"  [pipeline] step {k}: {r}")
      if r["result"] != "placed":
        break
    home = self.arm.home() if results and all(r["result"] == "placed" for r in results) else "skipped after failure"
    ok = len(results) == len(steps) and all(r["result"] == "placed" for r in results)
    return {"success": ok, "steps": results, "home": home, "wall_s": round(time.time() - t0, 1)}
