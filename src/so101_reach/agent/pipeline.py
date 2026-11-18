"""Pipeline mode: code does the predictable parts; the LLM is the fallback, not the default.

For a task, an ordered list of pick-and-place steps (from --object/--place directly, or one
planner call for free-text tasks such as stacking / unstacking). For each step:

  1. code   park the arm out of the head camera's view, measure the object and the target
  2. code   open the gripper to the object's width + 2 cm, rise, RL travel to 5 cm above the object
  3. local  GRASP, zero tokens: cached parameters from an earlier LLM success if the skill cache
            has a matching scene, else the behavior-cloned model's prediction (bc_grasp.py, if a
            trained model is given), else the scripted default (local_skills.py). Judged by the
            holding signal. Only if that fails does the LLM run the GRASP phase (with the local
            failure as context); a verified LLM grasp is distilled back into the cache.
  4. code   lift above everything nearby, RL travel to 5 cm above the target
  5. local  PLACE, zero tokens: cached or scripted release (drop above rim / stack), verified by
            re-perception; the LLM PLACE phase only on failure, successes distilled to the cache
  6. code   rise straight up
and finally RL home. `handoff="llm-only"` restores the old behaviour (LLM for every grasp/place)
for A/B token comparisons. Works for any object the perception can measure: containers, surfaces
and other objects (stacking) are all just "targets" with a measured top height.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from so101_reach.agent.local_skills import distill_grasp, distill_place, is_soft, jaw_for, scripted_grasp, scripted_place
from so101_reach.agent.perception import Detection, annotate
from so101_reach.agent.skill_cache import SkillCache, make_key

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
               hover_cm: float = 5.0, clear_view: bool = True, log=print,
               cache: SkillCache | None = None, handoff: str = "local-first", bc=None):
    """handoff: "local-first" (cached/scripted fine motion first, LLM only on failure) or
    "llm-only" (the LLM runs every grasp/place phase, the old behaviour).
    bc: optional BcGrasp model; its prediction is the grasp rung between a cache hit and the
    scripted default.
    If `per` has a `confident(names)` method (TrackedPerception), the arm is parked out of the
    head camera's view only when the tracker is not confident about the objects."""
    self.agent, self.arm, self.per = agent, arm, per
    self.grasp_retries, self.grasp_turns, self.place_turns = grasp_retries, grasp_turns, place_turns
    self.hover = hover_cm / 100
    self.use_clear_view = clear_view
    self.say = log
    self.tz = per.table_z
    self.cache, self.handoff, self.bc = cache, handoff, bc
    self.stats = {"llm_phases": 0, "local_phases": 0, "cache_hits": 0, "bc_phases": 0, "tokens_saved": 0,
                  "parks": 0, "parks_skipped": 0}

  # --- perception helpers ------------------------------------------------------------------
  def _measure(self, names: list[str]) -> dict[str, Detection]:
    park = self.use_clear_view
    if park and hasattr(self.per, "confident") and self.per.confident(names):
      park = False  # the Kalman tracker already knows where everything is: skip the ~5-8 s park
      self.stats["parks_skipped"] += 1
      self.say(f"  [pipeline] tracker confident about {names}: not parking the arm")
    if park:
      self.stats["parks"] += 1
      self.arm.clear_view()
    self.per.views()
    out = {}
    for n in names:
      dets = self.per.find(n)
      if dets:
        out[n] = dets[0]
    if park:
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
    return jaw_for(self.arm, obj, width)

  # --- one pick-and-place step -------------------------------------------------------------
  def step(self, k: int, pick: str, place_on: str, all_names: list[str]) -> dict:
    res = {"step": k, "pick": pick, "place_on": place_on, "grasp_attempts": 0}
    held_below = None
    obj = None
    grasp_note = ""
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
      gkey = make_key("grasp", pick, obj.size)
      if self.handoff == "local-first" and attempt == 0:
        # Zero-token rung: cached parameters from an earlier LLM success, else the BC model's
        # prediction, else the scripted default.
        entry = self.cache.lookup(gkey) if self.cache else None
        params = entry["params"] if entry else (self.bc.predict(obj, self.tz) if self.bc is not None else None)
        self.agent.phase = f"s{k}_local{attempt + 1}_"
        held, grasp_note = scripted_grasp(self.arm, obj, self.tz, params)
        if params is not None and not entry:
          grasp_note = grasp_note.replace("cached params", "BC prediction", 1)
        self.say(f"  [pipeline] grasp phase (local, 0 tokens): {grasp_note}")
        if held and params is not None and not entry:
          self.stats["bc_phases"] += 1
        if entry:
          self.cache.hit(entry, held)
          if held:
            self.stats["cache_hits"] += 1
            self.stats["tokens_saved"] += entry["stats"].get("tokens_phase", 0)
        if held:
          self.stats["local_phases"] += 1
      else:
        self.agent.phase = f"s{k}_grasp{attempt + 1}_"
        task = (f"GRASP the {pick}. {obj.describe(self.tz)}. The grip point is now about {100 * self.hover:.0f} cm above its top; "
                f"the gripper is open about {100 * (width + 0.02):.1f} cm."
                + (f" A scripted grasp attempt just failed ({grasp_note}): try something different." if grasp_note else ""))
        out = self.agent.run(task, [pick], GRASP_INSTRUCTIONS, FINE_TOOLS, self.grasp_turns,
                             stop_when=lambda tool, res: tool == "close_gripper" and self.arm.state()["holding_signal"])
        self.stats["llm_phases"] += 1
        held = self.arm.state()["holding_signal"]
        self.say(f"  [pipeline] grasp phase: agent says {out.get('success')} ({out.get('summary', '')[:80]}); holding={held}")
        if held and self.cache is not None:
          # The RL/local side learns from the LLM: distill the pose that worked into the cache.
          e = self.cache.record(gkey, distill_grasp(self.arm, obj, self.agent.last_close_force),
                                out.get("input_tokens", 0) + out.get("output_tokens", 0))
          self.say(f"  [pipeline] cached grasp params for '{pick}': {e['params']}")
      if held:
        gp = self.arm.geom(self.arm.q6()[0])["gp"]
        held_below = float(gp[2] - obj.support_z)
        self.arm.held = {"name": pick, "below_gp": held_below, "size": obj.size}
        if hasattr(self.per, "forget"):
          self.per.forget(pick)  # it travels with the gripper now: a stale track would fake the placement check
        break
      if hasattr(self.per, "invalidate"):
        self.per.invalidate(pick)  # the failed grasp may have nudged it: re-measure before retrying
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
    pkey = make_key("place", pick, obj.size, target_name=place_on, target_size=tgt.size, container=container)
    released_by, place_note, out = None, "", {}
    entry = None
    if self.handoff == "local-first":
      # Zero-token rung: cached release pose from an earlier LLM success, else the scripted rules.
      entry = self.cache.lookup(pkey) if self.cache else None
      self.agent.phase = f"s{k}_place_local_"
      ok, place_note = scripted_place(self.arm, tgt, obj.size, held_below, self.tz, hint_drop, hint_stack,
                                      container, is_soft(pick), entry["params"] if entry else None)
      self.say(f"  [pipeline] place phase (local, 0 tokens): {place_note}")
      if ok:
        released_by = "cache" if entry else "scripted"
        self.stats["local_phases"] += 1
      elif entry:
        self.cache.hit(entry, False)
        entry = None
    if released_by is None:
      self.agent.phase = f"s{k}_place_"
      task = (f"PLACE the {pick} you are holding on/into the {place_on}.\n"
              f"Held: {pick}, size {' x '.join(f'{100 * v:.1f}' for v in obj.size)} cm; its bottom is now at h={100 * bottom:.1f} cm "
              f"({100 * held_below:.1f} cm below the grip point; grip point at h={100 * gp_h:.1f}).\n"
              f"Target: {tgt.describe(self.tz)}" + (" (an open container)" if container else "") + ".\n"
              f"Pre-computed grip-point heights: held bottom 2 cm above the target's top/rim -> h={100 * hint_drop:.1f} "
              f"(from here: move_by dz={100 * (hint_drop - gp_h):+.1f}); held bottom 0.4 cm above the target's top (stacking) -> "
              f"h={100 * hint_stack:.1f} (from here: move_by dz={100 * (hint_stack - gp_h):+.1f}).\n"
              + (f"A scripted place attempt just failed ({place_note}): try something different.\n" if place_note else "")
              + "Opening the gripper releases the object and ends this phase.")
      out = self.agent.run(task, [pick, place_on], PLACE_INSTRUCTIONS, PLACE_TOOLS, self.place_turns,
                           stop_when=lambda tool, res: tool == "open_gripper" and not self.arm.state()["holding_signal"])
      self.stats["llm_phases"] += 1
      released_by = "llm"
    st = self.arm.state()
    if st["holding_signal"] or st["gripper_open_percent"] < 5:
      return {**res, "result": "place failed: object remains held"}
    release_gp = self.arm.geom(self.arm.q6()[0])["gp"]  # the pose that released; distilled on success
    self.arm.held = None
    if hasattr(self.per, "invalidate"):
      self.per.invalidate(place_on)  # the released object may have shifted the target
    self._rise_to(self.tz + self.arm.state()["grip_point_cm"][2] / 100 + 0.04)
    # Release alone does not prove placement. Refresh geometry after settling.
    self.arm.wait(0.5)
    self.per.views()
    def placed_on(dets, t):
      return any(np.linalg.norm(d.center[:2] - t.center[:2]) < max(t.size[:2]) / 2
                 and d.support_z <= t.top_z + 0.015 for d in dets if not d.extra.get("note"))

    verified = placed_on(self.per.find(pick), tgt)
    if not verified:
      # The arm hovers right above the drop and may hide the object, and `tgt` is the pre-place
      # (noisy) estimate. Re-measure both from a clear view before calling the placement a failure.
      fresh = self._measure([pick, place_on])
      if pick in fresh:
        verified = placed_on([fresh[pick]], fresh.get(place_on, tgt))
      self.say(f"  [pipeline] placement re-checked from a clear view -> {'verified' if verified else 'not verified'}")
    if released_by == "cache" and entry is not None:
      self.cache.hit(entry, verified)
      if verified:
        self.stats["cache_hits"] += 1
        self.stats["tokens_saved"] += entry["stats"].get("tokens_phase", 0)
    elif released_by == "llm" and verified and out.get("success") and self.cache is not None:
      e = self.cache.record(pkey, distill_place(release_gp, tgt, held_below, self.agent.last_open_percent),
                            out.get("input_tokens", 0) + out.get("output_tokens", 0))
      self.say(f"  [pipeline] cached place params for '{pick}' on '{place_on}': {e['params']}")
    res["result"] = "placed" if verified and (released_by != "llm" or out.get("success")) else "placement unverified"
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
    u = dict(self.agent.total_usage)
    self.say(f"  [pipeline] handoff {self.handoff}: {self.stats['llm_phases']} LLM phase(s), "
             f"{self.stats['local_phases']} local ({self.stats['cache_hits']} cache hit(s), {self.stats['bc_phases']} BC); "
             f"camera parks {self.stats['parks']} (skipped {self.stats['parks_skipped']} by the tracker); "
             f"tokens in {u['in']:,} / out {u['out']:,}; ~{self.stats['tokens_saved']:,} tokens saved by the cache"
             + (f"; {self.cache.summary()} in {self.cache.path}" if self.cache else ""))
    return {"success": ok, "steps": results, "home": home, "wall_s": round(time.time() - t0, 1),
            "handoff": self.handoff, "phases": dict(self.stats), "tokens": u,
            "tokens_saved_estimate": self.stats["tokens_saved"]}
