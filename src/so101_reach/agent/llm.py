"""The LLM agent loop: OpenAI Responses API, or any OpenAI-compatible chat-completions server
(a local vLLM / Ollama model, e.g. Cosmos 3 or Qwen3-VL) - with function tools and images.

Each turn the model gets the arm state, the measured objects and the annotated head + wrist
images, and answers with tool calls; every call runs to completion (the arm holds between
calls) and its result goes back with a fresh observation. Older images are dropped from the
context (text kept). Everything is logged to logs/agent/<time>/ (images, calls, results,
timings, token use).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from so101_reach.agent.perception import Detection, annotate
from so101_reach.sim import so101

LOG_DIR = so101.PROJECT_ROOT / "logs" / "agent"

INSTRUCTIONS = """You control a real 5-joint SO-101 robot arm with a parallel-jaw gripper, through tools.
Every tool call runs to completion; the arm holds still between calls, so take your time and check
the result of each step before the next.

Frame and units: centimetres and degrees. x points forward from the robot base, y to the robot's
left, h is the height above the table. The head camera sees the whole table (cyan arrows mark +x
and +y); the wrist camera is mounted on the gripper and looks along it.

Positions refer to the GRIP POINT: midway between the two jaw tips, 1.5 cm in from the tips -
where an object sits when the jaws close on it. It is drawn as a red circle labelled TIP.
- move_to(x, y, h, tilt, jaw_yaw): moves the grip point there. tilt = gripper angle from straight
  down (+ = tips leaning away from the robot); jaw_yaw = direction in the table plane along which
  the jaws close (0 = +x, 90 = +y). Pass null to keep the present value. Moves longer than 6 cm
  rise straight up to ~9 cm, travel across with a learned reaching policy, then come straight in
  to the target; short moves are straight lines (so near objects, move in short steps).
- move_by(dx, dy, dz): straight fine move, orientation kept (<= 10 cm).
- set_orientation(tilt, jaw_yaw): turn the gripper in place.
- open_gripper(percent): open to a percentage (100 = fully open); the state reports the jaw tip gap
  in cm. Opening releases whatever is held.
- close_gripper(force): close to GRASP. The jaws stop by themselves on contact and then hold with
  the chosen force (gentle for soft or fragile things, firm for hard ones). The result says whether
  something was caught between the jaws.
- find(name): detect objects by name (open-vocabulary detector + depth): centre, top height, size,
  long-side yaw, what it rests on. Objects you ask for are re-measured every turn.
- clear_view(): park the arm out of the head camera's view. Use it when an object you need is
  'not detected' or only remembered (the arm hides it) - the next observation re-measures it.
- home(): back to the start pose. done(success, summary): finish.

How grasping works with this gripper: open just wider than the object (jaw tip gap ~1.5-2.5 cm more
than its width across the jaws - the state reports the gap; opening much wider makes the open jaw
reach lower and can disturb neighbours), put the grip point above the object, turn jaw_yaw so the jaws
close across the object's narrow side (for an elongated object: long-side yaw + 90), lower the grip
point to about the object's middle or slightly below its top, close, then lift and check the
HOLDING SIGNAL in the state (jaws stopped short by the object) and the wrist image. Opening to a
smaller percentage does NOT grasp - always use close_gripper. A wide-open
jaw reaches lower than the grip point, so the arm refuses moves that would hit the table: open less
or grip higher. Objects can rest on other things (a block, a box) - use their measured top height.
Placing - for ANY object, decide how to release it from what it is and where it goes. You know
what you are holding (name, size, where its bottom is - see the state) and the target's measured
size and height. Reason about: is the object fragile (eggs, glass, ceramics, electronics), soft or
deformable, round (rolls), or sturdy? Is the target an open container (does the object fit through
the opening with margin? how high is the rim?), a flat surface, or the top of another object?
- Drop from above when the object is sturdy and will not roll out or break, and the opening is
  comfortably larger than it (>= 1.5 cm margin): centre it over the opening and release when its
  bottom is 1-3 cm above the rim. Do not move the gripper inside - the jaws would hit the walls and
  push the container.
- Lower it inside / onto the surface when it is fragile, deformable, could roll or bounce out, or
  the opening is tight: bring its bottom to ~0.3-0.5 cm above the floor or surface, keep the jaws
  clear of walls, then release.
- Placing on top of another object (stacking): only on a flat, stable top; centre the held object
  over it (use its measured centre and the wrist image), lower until the held bottom is ~0.3-0.5 cm
  above that top, release, rise straight up 3+ cm before moving sideways. Round or tall narrow
  objects are unstable on top - avoid stacking them unless asked.
- Taking things apart (unstacking, removing an object resting on another): always take the
  topmost object first; grip it in its upper part (grip point about 1/3 of its height below its
  top) so the jaws do not touch what is underneath; lift straight up before moving sideways.
After placing, check the result in the images and the measurements before finishing.
If a move is refused or stopped, read the reason, back off and try differently.
Keep fine moves near objects small (1-3 cm) and look at the wrist image after each."""


# Local models: reason explicitly before acting (they skip steps otherwise, e.g. closing a gripper
# that was never opened).
THINK_FIRST = """

Before every tool call, think briefly (a few sentences, not more): (1) what the images and the state
show now, (2) what has to be true before the next step (e.g. the gripper must be open wider than
the object BEFORE moving down to it; you must be HOLDING something before carrying it), (3) the one
next action and its exact numbers. Then call exactly ONE tool. Never repeat a failed action unchanged.
Read every tool result, including any WARNING in it."""


PLAN_INSTRUCTIONS = """Plan the task as an ORDERED list of pick-and-place steps for a robot arm with one
gripper. Each step picks ONE object and puts it on or into ONE target (a container, a surface, or
another object). Rules: only an object with nothing on top of it can be picked (take stacks apart
from the top down); when building a stack, place from the bottom up (the bottom object stays where
it is and is never picked); use the exact object names from the list. Call submit_plan."""


def load_api_key() -> str:
  if os.environ.get("OPENAI_API_KEY"):
    return os.environ["OPENAI_API_KEY"]
  env = so101.PROJECT_ROOT / ".env.local"
  if env.exists():
    for line in env.read_text().splitlines():
      m = re.match(r"\s*(?:export\s+)?(?:OPENAI_KEY|OPENAI_API_KEY)\s*=\s*['\"]?([^'\"\s]+)", line)
      if m:
        return m.group(1)
  raise RuntimeError("no OpenAI key: set OPENAI_API_KEY or OPENAI_KEY=... in .env.local")


def _tool(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
  return {"type": "function", "name": name, "description": desc, "strict": True,
          "parameters": {"type": "object", "properties": props, "required": required if required is not None else list(props),
                         "additionalProperties": False}}


_NUM = {"type": "number"}
_NUM_OR_NULL = {"type": ["number", "null"]}
TOOLS = [
  _tool("find", "Detect objects by name; returns their measured 3-D positions and sizes.", {"name": {"type": "string"}}),
  _tool("move_to", "Move the grip point to (x, y, h) cm with optional orientation (null keeps it).",
        {"x": _NUM, "y": _NUM, "h": _NUM, "tilt": _NUM_OR_NULL, "jaw_yaw": _NUM_OR_NULL}),
  _tool("move_by", "Fine straight move of the grip point by (dx, dy, dz) cm, orientation kept.", {"dx": _NUM, "dy": _NUM, "dz": _NUM}),
  _tool("set_orientation", "Turn the gripper in place (null keeps a value).", {"tilt": _NUM_OR_NULL, "jaw_yaw": _NUM_OR_NULL}),
  _tool("open_gripper", "Open the gripper to a percentage (releases anything held).", {"percent": _NUM}),
  _tool("close_gripper", "Close the gripper to grasp; stops on contact and holds with the given force.",
        {"force": {"type": "string", "enum": ["gentle", "medium", "firm"]}}),
  _tool("clear_view", "Park the arm out of the head camera's view, to see and measure objects it hides.", {}),
  _tool("home", "Move back to the start pose.", {}),
  _tool("done", "Finish the task.", {"success": {"type": "boolean"}, "summary": {"type": "string"}}),
]


class _TextCall:
  """A tool call recovered from text, shaped like the SDK's tool-call objects."""

  def __init__(self, cid: str, name: str, arguments: str):
    self.id = cid
    self.function = type("F", (), {"name": name, "arguments": arguments})()


def _calls_from_text(text: str, turn: int, allowed_names: set[str] | None = None) -> list:
  """Tool calls written in the reply text (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
  or a bare JSON object of that shape) - small local models sometimes do this."""
  out = []
  blobs = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S) or re.findall(r"(\{[^{}]*\"name\"[^{}]*\{.*?\}\s*\})", text, re.S)
  for i, b in enumerate(blobs):
    try:
      obj = json.loads(b)
    except json.JSONDecodeError:
      continue
    if isinstance(obj, dict) and obj.get("name") in (allowed_names if allowed_names is not None else {t["name"] for t in TOOLS}):
      out.append(_TextCall(f"text_{turn}_{i}", obj["name"], json.dumps(obj.get("arguments", {}))))
  return out


def _jpeg_b64(img: np.ndarray, width: int, quality: int = 70) -> str:
  h = int(img.shape[0] * width / img.shape[1])
  ok, buf = cv2.imencode(".jpg", cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, quality])
  return base64.b64encode(buf).decode()


class Agent:
  def __init__(self, arm, perception, model: str = "gpt-5.5", effort: str = "low", max_turns: int = 40,
               log_dir: Path | None = None, verbose: bool = True, base_url: str | None = None,
               head_width: int = 640, wrist_width: int = 512, detail: str = "high", max_input_tokens: int = 300_000):
    """base_url: an OpenAI-compatible chat-completions server (e.g. http://localhost:8000/v1 for vLLM,
    http://localhost:11434/v1 for Ollama); None = the OpenAI Responses API."""
    from openai import OpenAI

    self.base_url = base_url
    # Cost: images are most of the input tokens; old observations are trimmed (see _drop_old_images)
    # and a run stops once `max_input_tokens` have been sent.
    self.head_width, self.wrist_width, self.detail = head_width, wrist_width, detail
    self.max_input_tokens = max_input_tokens
    self.client = OpenAI(api_key=load_api_key()) if base_url is None else OpenAI(base_url=base_url, api_key="local")
    self.arm, self.per = arm, perception
    self.model, self.effort, self.max_turns = model, effort, max_turns
    self.log_dir = log_dir or LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{model.replace('/', '_')}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.verbose = verbose
    self._last_out = ""
    self.phase = ""  # log-file prefix of the current phase (pipeline mode)
    self.tracked: list[str] = []
    self.dets: list[Detection] = []
    self.events: list[dict] = []

  def _say(self, msg: str) -> None:
    if self.verbose:
      print(msg, flush=True)

  def _log(self, **ev) -> None:
    ev["t"] = round(time.time(), 2)
    self.events.append(ev)
    with open(self.log_dir / "events.jsonl", "a") as f:
      f.write(json.dumps(ev, default=str) + "\n")

  def _detect(self) -> str:
    self.dets = []
    lines = []
    held = self.arm.held["name"] if self.arm.held else None
    for name in self.tracked:
      for d in self.per.find(name):
        if held and d.name == held and d.extra.get("note"):
          continue  # A remembered position of the object now in the gripper.
        d.id = len(self.dets)
        self.dets.append(d)
        lines.append(d.describe(self.per.table_z))
      if not any(d.name == name or name in d.name for d in self.dets):
        lines.append(f"{name!r}: not detected")
    return "\n".join(lines) or "(no objects requested yet - use find)"

  def _observation(self, turn: int) -> list[dict]:
    views = self.per.views()  # fresh frames first: the objects are measured on these
    det_text = self._detect()
    q6, _ = self.arm.q6()
    gp = self.arm.geom(q6)["gp"]
    text = f"OBSERVATION (turn {turn}):\n{self.arm.state_text()}\nObjects:\n{det_text}"
    self._log(kind="observation", turn=turn, text=text)
    content = [{"type": "input_text", "text": text}]
    for cam, width in (("head", self.head_width), ("wrist", self.wrist_width)):
      if cam not in views:
        continue
      img = annotate(views[cam], self.dets, gp, self.per.table_z, f"{cam} camera", axes=cam == "head")
      cv2.imwrite(str(self.log_dir / f"{self.phase}turn{turn:02d}_{cam}.jpg"), img)
      content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{_jpeg_b64(img, width)}", "detail": self.detail})
    return [{"role": "user", "content": content}]

  def _near_object(self, gp: np.ndarray, radius: float = 0.04):
    """The measured object whose footprint is under / around the grip point, if any."""
    near = [d for d in self.dets if np.linalg.norm(d.center[:2] - gp[:2]) < radius + max(d.size[:2]) / 2
            and not d.extra.get("container")]
    return min(near, key=lambda d: np.linalg.norm(d.center[:2] - gp[:2])) if near else None

  def _warnings(self, name: str, args: dict, before: dict) -> str:
    """Hints about likely mistakes (state the model may have lost track of). Advisory only."""
    a = self.arm
    st = a.state()
    gp = np.array(st["grip_point_cm"]) / 100
    gap = st["jaw_tip_gap_cm"] / 100
    w = []
    if name in ("move_to", "move_by") and not a.held:
      went_down = st["grip_point_cm"][2] < before["grip_point_cm"][2] - 1.0
      d = self._near_object(np.array([gp[0], gp[1], 0.0]))
      if went_down and d is not None and gap < min(d.size[:2]) + 0.01:
        w.append(f"WARNING: the jaws are only {100 * gap:.1f} cm apart but the {d.name} below is {100 * min(d.size[:2]):.1f} cm "
                 f"wide - to grasp it, rise, open the gripper wider than the object, then come down again")
    if name == "close_gripper" and "NOTHING" in self._last_out:
      w.append("WARNING: nothing was grasped - open the gripper, check the wrist image, re-centre the grip point over the object "
               "and try again before carrying anything")
    if name in ("move_to",) and before.get("holding_signal") and not st["holding_signal"]:
      w.append("WARNING: the holding signal was lost during this move - the object may have dropped")
    return ("\n" + "\n".join(w)) if w else ""

  def _call_tool(self, name: str, args: dict) -> str:
    before = self.arm.state()
    self._last_out = self._run_tool(name, args)
    return self._last_out + self._warnings(name, args, before)

  def _run_tool(self, name: str, args: dict) -> str:
    a = self.arm
    if name == "find":
      self.per.views()  # fresh frames (the arm may have moved since the last observation)
      if args["name"] not in self.tracked:
        self.tracked.append(args["name"])
      return self._detect()
    if name == "move_to":
      return a.move_to(args["x"], args["y"], args["h"], args.get("tilt"), args.get("jaw_yaw"))
    if name == "move_by":
      return a.move_by(args["dx"], args["dy"], args["dz"])
    if name == "set_orientation":
      return a.set_orientation(args.get("tilt"), args.get("jaw_yaw"))
    if name == "open_gripper":
      return a.gripper(args["percent"], "medium")
    if name == "close_gripper":
      out = a.gripper(0, args["force"])
      if out.startswith("closed on something"):
        self._note_held()
      return out
    if name == "home":
      return a.home()
    if name == "clear_view":
      return a.clear_view()
    return f"unknown tool {name}"

  def _note_held(self) -> None:
    """Which object was just grasped (the measured one nearest the grip point) and how far its
    bottom hangs below the grip point."""
    gp = self.arm.geom(self.arm.q6()[0])["gp"]
    cands = [d for d in self.dets if np.linalg.norm(d.center[:2] - gp[:2]) < 0.04 and d.support_z - 0.02 < gp[2] < d.top_z + 0.02]
    if not cands:
      self.arm.held = None
      return
    d = min(cands, key=lambda d: np.linalg.norm(d.center[:2] - gp[:2]))
    self.arm.held = {"name": d.name, "below_gp": float(gp[2] - d.support_z), "size": d.size}
    self._log(kind="held", name=d.name, below_gp=round(float(gp[2] - d.support_z), 4))

  @staticmethod
  def _drop_old_images(inp: list, keep: int = 1) -> None:
    """Older observations keep only the arm-state line (no images, no object list): the newest
    observation always has everything."""
    obs = [i for i, it in enumerate(inp) if isinstance(it, dict) and it.get("role") == "user"]
    for i in obs[:-keep]:
      content = []
      for c in inp[i]["content"]:
        if c["type"] == "input_image":
          continue
        if c["type"] == "input_text" and c["text"].startswith("OBSERVATION") and "\nObjects:" in c["text"]:
          c = {"type": "input_text", "text": c["text"].split("\nObjects:")[0] + "\n(older observation, objects and images omitted)"}
        content.append(c)
      inp[i]["content"] = content

  @staticmethod
  def _tools(names: list[str] | None) -> list[dict]:
    return TOOLS if names is None else [t for t in TOOLS if t["name"] in set(names) | {"done"}]

  def run(self, task: str, track: list[str], instructions: str | None = None, tool_names: list[str] | None = None,
          max_turns: int | None = None, stop_when=None) -> dict:
    """Run the agent until it calls done (or the turn / token budget ends). `instructions`,
    `tool_names` and `max_turns` narrow it down to one phase (pipeline mode); `stop_when(tool, result)`
    ends the phase successfully as soon as its goal is reached (e.g. the object is held)."""
    if self.base_url is not None:
      return self._run_chat(task, track, instructions, tool_names, max_turns, stop_when)
    instructions = instructions or INSTRUCTIONS
    tools = self._tools(tool_names)
    self.tracked = list(track)
    self._log(kind="task", task=task, model=self.model, effort=self.effort, phase=self.phase)
    inp = [{"role": "user", "content": [{"type": "input_text", "text": f"TASK: {task}"}]}] + self._observation(0)
    result = {"success": False, "summary": "turn budget used up", "turns": 0}
    t_start = time.time()
    usage = {"in": 0, "out": 0}
    for turn in range(1, (max_turns or self.max_turns) + 1):
      if usage["in"] >= self.max_input_tokens:
        result = {"success": False, "summary": f"stopped: input-token budget ({self.max_input_tokens}) used up", "turns": turn - 1}
        self._say(f"  [agent] {result['summary']} - the arm holds where it is")
        break
      self._drop_old_images(inp)
      t0 = time.time()
      r = self.client.responses.create(model=self.model, instructions=instructions, input=inp, tools=tools,
                                       reasoning={"effort": self.effort})
      dt = time.time() - t0
      usage["in"] += r.usage.input_tokens
      usage["out"] += r.usage.output_tokens
      cached = getattr(getattr(r.usage, "input_tokens_details", None), "cached_tokens", 0) or 0
      usage["cached"] = usage.get("cached", 0) + cached
      inp += r.output
      calls = [o for o in r.output if o.type == "function_call"]
      text = " ".join(c.text for o in r.output if o.type == "message" for c in o.content if getattr(c, "text", None))
      if text:
        self._say(f"  [agent] {text[:300]}")
      self._log(kind="llm", turn=turn, seconds=round(dt, 2), input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                text=text, calls=[(c.name, c.arguments) for c in calls])
      if not calls:
        inp.append({"role": "user", "content": [{"type": "input_text", "text": "Use the tools (call done when finished)."}]})
        continue
      finished = False
      for c in calls:
        args = json.loads(c.arguments or "{}")
        if c.name == "done":
          result = {"success": bool(args.get("success")), "summary": args.get("summary", ""), "turns": turn}
          out, finished = "ok", True
        else:
          t1 = time.time()
          try:
            out = self._call_tool(c.name, args)
          except Exception as e:  # A tool error must not crash the run: the arm holds; tell the agent.
            out = f"error: {e}"
          self._say(f"  turn {turn:2d} ({dt:4.1f}s think) {c.name}({', '.join(f'{k}={v}' for k, v in args.items())}) -> {out}")
          self._log(kind="tool", turn=turn, name=c.name, args=args, result=out, seconds=round(time.time() - t1, 2))
          if stop_when is not None and stop_when(c.name, out):
            result, finished = {"success": True, "summary": f"phase goal reached after {c.name}", "turns": turn}, True
        inp.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
      if finished:
        break
      inp += self._observation(turn)
    return self._finish(result, t_start, usage)

  def _finish(self, result: dict, t_start: float, usage: dict) -> dict:
    result.update(wall_s=round(time.time() - t_start, 1), input_tokens=usage["in"], cached_input_tokens=usage.get("cached", 0),
                  output_tokens=usage["out"], log=str(self.log_dir))
    self._say(f"  [cost] {result.get('turns', 0)} turns, input {usage['in']:,} tokens ({usage.get('cached', 0):,} cached), "
              f"output {usage['out']:,} tokens, {result['wall_s']} s")
    self._log(kind="result", **result)
    (self.log_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

  # --- Planner (pipeline mode) -------------------------------------------------------------------
  def plan(self, task: str, objects_text: str, image_b64: str | None = None) -> list[dict]:
    """One model call: the task + measured objects -> ordered pick-and-place steps
    [{"pick": name, "place_on": name}]."""
    tool = _tool("submit_plan", "Submit the ordered pick-and-place steps.",
                 {"steps": {"type": "array", "items": {"type": "object", "properties": {
                   "pick": {"type": "string", "description": "exact name of the object to pick (from the list)"},
                   "place_on": {"type": "string", "description": "exact name of the object/container to put it on or into"}},
                   "required": ["pick", "place_on"], "additionalProperties": False}}})
    text = (f"TASK: {task}\n\nMeasured objects (exact names):\n{objects_text}\n\n" + PLAN_INSTRUCTIONS)
    t0 = time.time()
    if self.base_url is None:
      content = [{"type": "input_text", "text": text}]
      if image_b64:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}", "detail": self.detail})
      r = self.client.responses.create(model=self.model, input=[{"role": "user", "content": content}], tools=[tool],
                                       tool_choice={"type": "function", "name": "submit_plan"}, reasoning={"effort": self.effort})
      calls = [o for o in r.output if o.type == "function_call"]
      args = json.loads(calls[0].arguments) if calls else {"steps": []}
    else:
      content = [{"type": "text", "text": text}]
      if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
      ctool = {"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}}
      extra = {"think": False} if self.effort == "none" else {"reasoning_effort": self.effort}
      r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": content}], tools=[ctool],
                                              tool_choice="auto", temperature=0.2, max_tokens=4096, extra_body=extra)
      m = r.choices[0].message
      calls = list(m.tool_calls or []) or _calls_from_text(m.content or "", 0, {"submit_plan"})
      args = json.loads(calls[0].function.arguments) if calls else {"steps": []}
    self._log(kind="plan", task=task, steps=args.get("steps", []), seconds=round(time.time() - t0, 2))
    return args.get("steps", [])

  # --- Chat-completions backend (local OpenAI-compatible servers) ------------------------------
  @staticmethod
  def _chat_content(content: list[dict]) -> list[dict]:
    out = []
    for c in content:
      if c["type"] == "input_text":
        out.append({"type": "text", "text": c["text"]})
      elif c["type"] == "input_image":
        out.append({"type": "image_url", "image_url": {"url": c["image_url"]}})
    return out

  def _run_chat(self, task: str, track: list[str], instructions: str | None = None, tool_names: list[str] | None = None,
                max_turns: int | None = None, stop_when=None) -> dict:
    self.tracked = list(track)
    self._log(kind="task", task=task, model=self.model, backend=self.base_url, phase=self.phase)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
             for t in self._tools(tool_names)]
    first = self._observation(0)[0]["content"]
    msgs: list[dict] = [{"role": "system", "content": (instructions or INSTRUCTIONS) + THINK_FIRST},
                        {"role": "user", "content": [{"type": "text", "text": f"TASK: {task}"}] + self._chat_content(first)}]
    result = {"success": False, "summary": "turn budget used up", "turns": 0}
    t_start, usage = time.time(), {"in": 0, "out": 0}
    for turn in range(1, (max_turns or self.max_turns) + 1):
      if usage["in"] >= self.max_input_tokens:
        result = {"success": False, "summary": "input-token budget used up", "turns": turn - 1}
        break
      users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
      for i in users[:-1]:  # keep only the newest images
        if isinstance(msgs[i]["content"], list) and any(c["type"] == "image_url" for c in msgs[i]["content"]):
          msgs[i]["content"] = [c for c in msgs[i]["content"] if c["type"] != "image_url"] + [{"type": "text", "text": "(images of this turn omitted)"}]
      t0 = time.time()
      # Thinking level for local servers (Ollama: false / low / medium / xhigh via reasoning_effort).
      extra = {"think": False} if self.effort == "none" else {"reasoning_effort": self.effort}
      r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=tools, tool_choice="auto", temperature=0.2,
                                              max_tokens=4096, extra_body=extra or None)
      dt = time.time() - t0
      if r.usage:
        usage["in"] += r.usage.prompt_tokens
        usage["out"] += r.usage.completion_tokens
      m = r.choices[0].message
      calls = list(m.tool_calls or [])
      text = (m.content or "").strip()
      if not calls and text:
        calls = _calls_from_text(text, turn, {t["function"]["name"] for t in tools})
      if text:
        self._say(f"  [agent] {text[:300]}")
      self._log(kind="llm", turn=turn, seconds=round(dt, 2), input_tokens=r.usage.prompt_tokens if r.usage else None,
                output_tokens=r.usage.completion_tokens if r.usage else None, text=text,
                calls=[(c.function.name, c.function.arguments) for c in calls])
      msgs.append({"role": "assistant", "content": text or None,
                   "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                  for c in calls] or None})
      if not calls:
        msgs[-1].pop("tool_calls")
        msgs.append({"role": "user", "content": "Use the tools (call done when finished)."})
        continue
      finished = False
      for k, c in enumerate(calls):
        if k > 0:  # One action per turn: the result of each must be seen before the next.
          msgs.append({"role": "tool", "tool_call_id": c.id, "content": "skipped: only one action per turn - call it again next turn"})
          continue
        try:
          args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
          args = None
        if args is None:
          out = "error: the arguments were not valid JSON - call the tool again"
        elif c.function.name == "done":
          result = {"success": bool(args.get("success")), "summary": args.get("summary", ""), "turns": turn}
          out, finished = "ok", True
        else:
          t1 = time.time()
          try:
            out = self._call_tool(c.function.name, args)
          except Exception as e:  # The arm holds; tell the agent.
            out = f"error: {e}"
          self._say(f"  turn {turn:2d} ({dt:4.1f}s think) {c.function.name}({', '.join(f'{k}={v}' for k, v in args.items())}) -> {out}")
          self._log(kind="tool", turn=turn, name=c.function.name, args=args, result=out, seconds=round(time.time() - t1, 2))
          if stop_when is not None and stop_when(c.function.name, out):
            result, finished = {"success": True, "summary": f"phase goal reached after {c.function.name}", "turns": turn}, True
        msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
      if finished:
        break
      obs = self._observation(turn)[0]["content"]
      msgs.append({"role": "user", "content": self._chat_content(obs)})
    return self._finish(result, t_start, usage)
