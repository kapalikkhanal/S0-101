"""Shared harness for the report experiments (scripts/report/t*.py).

Methods compared on the same seeded pick-and-place scenes (one random object -> the white box):

  rl_scripted  RL travel + scripted grasp/place; failed grasps retried with perturbed scripted
               parameters (no learning, no LLM)
  rl_bc        RL travel + behavior-cloned grasp (data/il/bc_grasp.json), perturbed retries
  vla          RL travel + the vision-language agent runs EVERY grasp and place phase
               (pipeline handoff "llm-only": the VLA decides all fine motion from images + state)
  vla_free     the vision-language agent decides every motion itself, travel included (agent
               "free" mode: no code-side pipeline at all)
  hybrid       RL travel + BC grasp first; the VLA takes over only when the local attempt fails

Every method gets the same retry budget (3 grasp attempts). Success is judged on GROUND TRUTH:
the object's centre lies inside the box footprint and below its rim at the end of the episode.
Results are appended (one JSON line per episode) to results/report/<test>.jsonl, resumable:
episodes already in the file are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results" / "report"
RESULTS.mkdir(parents=True, exist_ok=True)

from so101_reach.agent.arm import AgentArm  # noqa: E402
from so101_reach.agent.bc_grasp import DEFAULT_MODEL as BC_MODEL  # noqa: E402
from so101_reach.agent.bc_grasp import BcGrasp  # noqa: E402
from so101_reach.agent.local_skills import heuristic_grasp_params, perturb_params, scripted_grasp  # noqa: E402
from so101_reach.agent.perception import SimPerception  # noqa: E402
from so101_reach.agent.sim_reality import make_world, wrap_perception  # noqa: E402
from so101_reach.agent.sim_world import OpenBox, StaticBox, WorldCfg  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.mouse_real import SafeTargets  # noqa: E402
from so101_reach.real.robot import SimRealPlant  # noqa: E402
from so101_reach.settings import AGENT_POLICY_DIR as POLICY  # noqa: E402
from so101_reach.sim import so101  # noqa: E402
from so101_reach.sim.grasp_scene import ObjectCfg  # noqa: E402

TZ = -0.0132
COLORS = {"red": (0.85, 0.1, 0.1, 1), "green": (0.1, 0.7, 0.2, 1), "blue": (0.1, 0.2, 0.9, 1),
          "yellow": (0.9, 0.8, 0.1, 1), "orange": (1.0, 0.45, 0.02, 1), "purple": (0.5, 0.1, 0.6, 1)}
METHODS = ("rl_scripted", "rl_bc", "vla", "vla_free", "hybrid")
LLM_METHODS = {"vla", "vla_free", "hybrid"}
DEFAULT_MODEL = os.environ.get("SO101_REPORT_MODEL", "qwen2.5vl-tools:7b")
DEFAULT_BASE_URL = os.environ.get("SO101_REPORT_BASE_URL", "http://127.0.0.1:11434/v1")
BOX = "white box"


# --- scenes ---------------------------------------------------------------------------------
def pick_place_scene(seed: int, p_stack: float = 0.25) -> tuple[WorldCfg, str, dict]:
  """One random object (box / bar / cylinder / ball, size +-30 %, random yaw, mass, colour,
  sometimes on a platform) in front of the arm, and the open white box to its right."""
  rng = np.random.default_rng(seed)
  shape = str(rng.choice(["box", "bar", "cylinder", "sphere"], p=[0.4, 0.2, 0.2, 0.2]))
  s = 0.034 * (1 + float(rng.uniform(-0.3, 0.3)))
  color = str(rng.choice(list(COLORS)))
  if shape == "box":
    size = (s / 2, s / 2 * float(rng.uniform(0.9, 1.1)), s / 2 * float(rng.uniform(0.7, 1.3)))
    mj, word, half_z = "box", "cube", size[2]
  elif shape == "bar":
    size = (s / 2 * float(rng.uniform(1.5, 2.2)), s / 2 * float(rng.uniform(0.4, 0.65)), s / 2 * float(rng.uniform(0.5, 0.9)))
    mj, word, half_z = "box", "bar", size[2]
  elif shape == "cylinder":
    size = (s / 2 * float(rng.uniform(0.5, 0.8)), s / 2 * float(rng.uniform(0.7, 1.2)))
    mj, word, half_z = "cylinder", "cylinder", size[1]
  else:
    size = (s / 2 * float(rng.uniform(0.55, 0.8)),)
    mj, word, half_z = "sphere", "ball", size[0]
  label = f"{color} {word}"
  xy = (float(rng.uniform(0.18, 0.25)), float(rng.uniform(-0.12, 0.03)))
  yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
  quat = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
  mass = float(rng.uniform(0.02, 0.06))
  platforms, support = [], TZ
  stacked = bool(rng.uniform() < p_stack) and shape != "sphere"
  if stacked:
    ph = float(rng.uniform(0.012, 0.025))
    platforms.append(StaticBox("platform", (*xy, TZ + ph), (0.035, 0.035, ph), (0.55, 0.35, 0.2, 1)))
    support = TZ + 2 * ph
  obj = ObjectCfg(mj, size, (*xy, support + half_z + 0.002), quat, mass, (1.0, 0.005, 0.0001), COLORS[color])
  bx = (0.33 + float(rng.uniform(-0.01, 0.01)), -0.19 + float(rng.uniform(-0.015, 0.015)))
  box = OpenBox(BOX, bx, (0.12, 0.16), 0.045)
  meta = {"shape": shape, "stacked": stacked, "mass": round(mass, 3), "obj_xy": [round(v, 3) for v in xy],
          "width_mm": round(2000 * (min(size[0], size[1]) if mj == "box" else size[0]), 1)}
  return WorldCfg(TZ, {label: obj}, platforms, [box]), label, meta


def in_box(world, label: str) -> bool:
  g = world.object_truth(label)
  s = next(s for s in world.surfaces() if s["label"] == BOX)
  c, box = g["center"], s["corners"]
  return bool(box[:, 0].min() < c[0] < box[:, 0].max() and box[:, 1].min() < c[1] < box[:, 1].max() and c[2] < s["top"])


# --- agents -----------------------------------------------------------------------------------
class ScriptedRetryAgent:
  """Stands in for the LLM in the non-LLM baselines: a "grasp" phase becomes one perturbed
  scripted grasp around the base parameters (BC prediction or heuristic); a "place" phase
  reports failure (the scripted place already ran)."""

  def __init__(self, arm, per, bc=None, seed: int = 0):
    self.arm, self.per, self.bc = arm, per, bc
    self.rng = np.random.default_rng([seed, 99])
    self.phase = ""
    self.total_usage = {"in": 0, "out": 0, "cached": 0}
    self.last_close_force, self.last_open_percent = "firm", 60.0
    self.llm_calls, self.llm_seconds = 0, 0.0

  def run(self, task, track, instructions=None, tool_names=None, max_turns=None, stop_when=None):
    if "close_gripper" not in (tool_names or []):  # place phase
      return {"success": False, "summary": "no fallback for place", "input_tokens": 0, "output_tokens": 0}
    dets = self.per.find(track[0])
    if not dets:
      return {"success": False, "summary": "object not detected", "input_tokens": 0, "output_tokens": 0}
    obj = dets[0]
    base = self.bc.predict(obj, self.per.table_z) if self.bc is not None else heuristic_grasp_params(self.arm, obj, self.per.table_z)
    ok, note = scripted_grasp(self.arm, obj, self.per.table_z, perturb_params(base, self.rng))
    return {"success": ok, "summary": note, "input_tokens": 0, "output_tokens": 0}

  def plan(self, *a, **k):
    return []


def make_llm_agent(arm, per, log_dir: Path, model: str, base_url: str | None, effort: str = "none", max_turns: int = 30):
  """The project's Agent, instrumented: counts LLM calls and their wall time (the arm holds
  still while the model thinks, so that time adds to the real execution time)."""
  from so101_reach.agent.llm import Agent

  class TimedAgent(Agent):
    def _log(self, **ev):
      if ev.get("kind") in ("llm", "plan"):
        self.llm_calls += 1
        self.llm_seconds += float(ev.get("seconds") or 0.0)
        self.llm_trace.append({"phase": self.phase, "s": ev.get("seconds"), "in": ev.get("input_tokens"),
                               "out": ev.get("output_tokens")})
      super()._log(**ev)

  log_dir.mkdir(parents=True, exist_ok=True)
  a = TimedAgent.__new__(TimedAgent)
  a.llm_calls, a.llm_seconds, a.llm_trace = 0, 0.0, []
  Agent.__init__(a, arm, per, model, effort, max_turns, log_dir=log_dir, verbose=False, base_url=base_url)
  return a


# --- one episode ------------------------------------------------------------------------------
_CTRL: dict = {}


def controller(speed: float = 0.5) -> ReachController:
  if speed not in _CTRL:
    _CTRL[speed] = ReachController(ControllerCfg(speed_factor=speed), policy_dir=POLICY)
  return _CTRL[speed]


def failure_category(res: dict, gt_ok: bool) -> str:
  if gt_ok:
    return "success"
  if "error" in res:
    return "crash"
  steps = res.get("steps") or []
  r = (steps[-1].get("result", "") if steps else res.get("summary", "")).lower()
  if not steps and res.get("summary") is not None:  # free mode
    s = r
    if "budget" in s or "turn" in s:
      return "timeout"
    return "agent gave up" if res.get("success") is False else "wrong placement"
  if r.startswith("not detected"):
    return "perception miss"
  if r == "grasp failed":
    return "grasp failed"
  if "remains held" in r:
    return "release failed"
  if r in ("placed", "placement unverified"):
    return "dropped / missed box"
  if any(k in r for k in ("refused", "stopped", "travel", "unreachable", "no progress")):
    return "motion refused"
  return "other"


def run_episode(method: str, seed: int, level: float, *, model: str = DEFAULT_MODEL, base_url: str | None = DEFAULT_BASE_URL,
                bc_path: Path = BC_MODEL, scene_fn=pick_place_scene, log_root: Path | None = None, speed: float = 0.5) -> dict:
  from so101_reach.agent.pipeline import Pipeline

  cfg, label, meta = scene_fn(seed)
  world = make_world(cfg, seed, level)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  arm = AgentArm(plant, controller(speed), SafeTargets(), TZ)
  llm = method in LLM_METHODS
  base = SimPerception(world)
  if not llm:  # no images needed: skip rendering (much faster)
    base.views = lambda: {}
  per = wrap_perception(base, seed, level, arm_points=arm.arm_points, render=llm)
  bc = BcGrasp.load(bc_path) if method in ("rl_bc", "hybrid") else None
  if method in ("rl_bc", "hybrid") and bc is None:
    raise SystemExit(f"{method} needs a BC model at {bc_path} (run scripts/report/t07_bc_data_efficiency.py first)")
  log_dir = (log_root or RESULTS / "agent_logs") / f"{method}_L{level:g}_s{seed}"
  agent = make_llm_agent(arm, per, log_dir, model, base_url) if llm else ScriptedRetryAgent(arm, per, bc, seed)
  task = f"Pick up the {label} and put it into the {BOX}, then go home."
  t_wall, sim_t0 = time.time(), float(world.data.time)
  try:
    if method == "vla_free":
      res = agent.run(task, [label, BOX])
    else:
      handoff = "llm-only" if method == "vla" else "local-first"
      pipe = Pipeline(agent, arm, per, clear_view=level > 0, cache=None, handoff=handoff, bc=bc, log=lambda *_: None)
      res = pipe.run(task, [label, BOX], [{"pick": label, "place_on": BOX}])
  except Exception as e:  # a crash is a failed episode, not a failed experiment
    res = {"success": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}
  for _ in range(25):  # let the released object settle before judging
    plant.wait()
  gt = in_box(world, label)
  truth = world.object_truth(label)
  sim_s = float(world.data.time) - sim_t0
  llm_s = float(getattr(agent, "llm_seconds", 0.0))
  steps = res.get("steps") or []
  phases = res.get("phases", {})
  out = {
    "method": method, "level": level, "seed": seed, "label": label, **meta,
    "success": gt, "claimed_success": bool(res.get("success")), "category": failure_category(res, gt),
    "grasped": bool(steps and steps[-1].get("result") != "grasp failed" and not str(steps[-1].get("result", "")).startswith("not detected"))
    if steps else None,
    "grasp_attempts": steps[-1].get("grasp_attempts") if steps else None,
    "result": steps[-1].get("result") if steps else res.get("summary", res.get("error", "")),
    "sim_s": round(sim_s, 2), "llm_s": round(llm_s, 2), "exec_s": round(sim_s + llm_s, 2),
    "compute_wall_s": round(time.time() - t_wall, 1),
    "llm_calls": int(getattr(agent, "llm_calls", 0)),
    "tokens_in": int(agent.total_usage.get("in", 0)), "tokens_out": int(agent.total_usage.get("out", 0)),
    "llm_phases": phases.get("llm_phases"), "local_phases": phases.get("local_phases"), "bc_phases": phases.get("bc_phases"),
    "final_obj_xyz": [round(float(v), 4) for v in truth["center"]],
    "model": model if llm else None, "error": res.get("error"),
  }
  if llm:
    out["llm_trace"] = agent.llm_trace
  world.close_renderers()
  return out


# --- resumable result files -----------------------------------------------------------------
def done_keys(path: Path, keys=("method", "level", "seed")) -> set:
  if not path.exists():
    return set()
  out = set()
  for line in path.read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      out.add(tuple(r.get(k) for k in keys))
  return out


def append(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "a") as f:
    f.write(json.dumps(row, default=float) + "\n")


def load(path: Path) -> list[dict]:
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run_grid(path: Path, jobs: list[tuple[str, int, float]], workers: int = 1, **kw) -> None:
  """Run (method, seed, level) jobs not yet in `path`. Non-LLM jobs run in parallel processes;
  LLM jobs run one at a time (one local GPU model)."""
  todo = [j for j in jobs if (j[0], j[2], j[1]) not in done_keys(path)]
  print(f"[grid] {path.name}: {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
  fast = [j for j in todo if j[0] not in LLM_METHODS]
  slow = [j for j in todo if j[0] in LLM_METHODS]
  t0 = time.time()
  n = 0

  def report(r):
    nonlocal n
    n += 1
    append(path, r)
    print(f"[grid] {n}/{len(todo)} {r['method']:12s} L{r['level']:<4g} seed {r['seed']} {r['label']:16s} "
          f"{'OK ' if r['success'] else 'FAIL'} {r['category']:22s} exec {r['exec_s']:6.1f}s "
          f"llm {r['llm_calls']:3d} calls  ({time.time() - t0:.0f}s)", flush=True)

  if fast:
    if workers > 1:
      from concurrent.futures import ProcessPoolExecutor, as_completed

      with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run_episode, m, s, lv, **kw) for m, s, lv in fast]
        for f in as_completed(futs):
          report(f.result())
    else:
      for m, s, lv in fast:
        report(run_episode(m, s, lv, **kw))
  for m, s, lv in slow:
    report(run_episode(m, s, lv, **kw))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
  if n == 0:
    return float("nan"), float("nan"), float("nan")
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return p, max(0.0, c - h), min(1.0, c + h)


_ = replace  # re-exported for test scripts
