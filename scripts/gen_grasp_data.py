"""Generate imitation-learning grasp data in randomized simulation scenes.

Each episode builds a fresh scene (random shape / size / pose / mass, sometimes on a
platform), approaches with RL travel, and tries to grasp with a ladder of parameter
sources:

  1. base policy: the BC model (--bc-model, for DAgger iterations) or the scripted
     heuristic (`heuristic_grasp_params`)
  2. perturbed retries around the base (self-supervised exploration)
  3. optionally an LLM expert (--expert local|openai) on the remaining failures -
     classic DAgger: the expert is only queried where the learner fails

Success = holding signal AND the object stays held through a 5 cm lift. Only then are the
actually-achieved parameters recorded (`distill_grasp`), so a weak expert can reduce the
amount of data but never its quality. One JSONL row per attempt (failures included - they
feed failure analysis and a future success predictor) appended to data/il/grasp_episodes.jsonl.

Examples:
  python scripts/gen_grasp_data.py --episodes 50 --seed 0                 # no LLM, no tokens
  python scripts/gen_grasp_data.py --episodes 200 --seed 100 --expert local
  python scripts/gen_grasp_data.py --episodes 30 --seed 900 --expert openai --max-expert 10

Train afterwards with scripts/train_bc_grasp.py, evaluate with scripts/eval_grasp_policy.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from so101_reach.agent.arm import AgentArm  # noqa: E402
from so101_reach.agent.bc_grasp import BcGrasp, features_for  # noqa: E402
from so101_reach.agent.local_skills import distill_grasp, heuristic_grasp_params, jaw_for, perturb_params, scripted_grasp  # noqa: E402
from so101_reach.agent.perception import SimPerception  # noqa: E402
from so101_reach.agent.sim_world import SimWorld, StaticBox, WorldCfg  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.mouse_real import SafeTargets  # noqa: E402
from so101_reach.real.robot import SimRealPlant  # noqa: E402
from so101_reach.sim import so101  # noqa: E402
from so101_reach.sim.grasp_scene import ObjectCfg  # noqa: E402
from so101_reach.settings import AGENT_POLICY_DIR as POLICY  # noqa: E402

OUT_DEFAULT = so101.PROJECT_ROOT / "data" / "il" / "grasp_episodes.jsonl"
TZ = -0.0132
COLORS = {"red": (0.85, 0.1, 0.1, 1), "green": (0.1, 0.7, 0.2, 1), "blue": (0.1, 0.2, 0.9, 1),
          "white": (0.95, 0.95, 0.95, 1), "yellow": (0.9, 0.8, 0.1, 1), "orange": (1.0, 0.45, 0.02, 1)}
HOVER = 0.05


def random_scene(seed: int, p_stack: float = 0.3) -> tuple[WorldCfg, str, dict]:
  """A seeded one-object grasp scene: shape box/bar/cylinder/sphere, size +-30%, random pose
  and mass, p_stack of standing on a platform (the 'stacked' case)."""
  rng = np.random.default_rng(seed)
  shape = str(rng.choice(["box", "bar", "cylinder", "sphere"], p=[0.4, 0.2, 0.2, 0.2]))
  s = 0.034 * (1 + float(rng.uniform(-0.3, 0.3)))
  color = str(rng.choice(list(COLORS)))
  if shape == "box":
    size = (s / 2, s / 2 * float(rng.uniform(0.9, 1.1)), s / 2 * float(rng.uniform(0.7, 1.3)))
    mj_shape, word, half_z = "box", "cube", size[2]
  elif shape == "bar":
    size = (s / 2 * float(rng.uniform(1.5, 2.2)), s / 2 * float(rng.uniform(0.4, 0.65)), s / 2 * float(rng.uniform(0.5, 0.9)))
    mj_shape, word, half_z = "box", "bar", size[2]
  elif shape == "cylinder":
    size = (s / 2 * float(rng.uniform(0.5, 0.8)), s / 2 * float(rng.uniform(0.7, 1.2)))
    mj_shape, word, half_z = "cylinder", "cylinder", size[1]
  else:
    size = (s / 2 * float(rng.uniform(0.55, 0.8)),)
    mj_shape, word, half_z = "sphere", "ball", size[0]
  label = f"{color} {word}"
  xy = (float(rng.uniform(0.19, 0.30)), float(rng.uniform(-0.17, 0.02)))
  yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
  quat = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
  mass = float(rng.uniform(0.02, 0.06))
  platforms, support = [], TZ
  stacked = bool(rng.uniform() < p_stack) and shape != "sphere"  # a stacked ball just rolls off
  if stacked:
    ph = float(rng.uniform(0.015, 0.03))
    platforms.append(StaticBox("platform", (*xy, TZ + ph), (0.04, 0.04, ph), (0.55, 0.35, 0.2, 1)))
    support = TZ + 2 * ph
  obj = ObjectCfg(mj_shape, size, (*xy, support + half_z + 0.002), quat, mass, (1.0, 0.005, 0.0001), COLORS[color])
  return WorldCfg(TZ, {label: obj}, platforms), label, {"shape": shape, "stacked": stacked, "mass": round(mass, 3)}


def _cm(v: float) -> float:
  return round(100 * v, 1)


def _rise_to(arm, h_m: float) -> bool:
  h = TZ + arm.state()["grip_point_cm"][2] / 100
  while h < h_m - 0.005:
    if not arm.move_by(0, 0, min(9.0, 100 * (h_m - h))).startswith("ok"):
      return False
    h2 = TZ + arm.state()["grip_point_cm"][2] / 100
    if h2 <= h + 0.001:
      return False
    h = h2
  return True


def _approach(arm, per, label):
  """Fresh measurement, open jaws, rise clear, RL travel to hover above the object.
  Returns the measured Detection or None."""
  dets = per.find(label)
  if not dets:
    return None
  obj = dets[0]
  width = float(min(obj.size[:2]))
  if width > 0.065:
    return None  # wider than the jaws can open
  arm.gripper(min(80.0, arm.percent_for_gap(width + 0.02)), "medium")
  if not _rise_to(arm, max(obj.top_z + 0.03, TZ + arm.state()["grip_point_cm"][2] / 100)):
    return None
  jaw = jaw_for(arm, obj, width)
  h_cm = _cm(obj.top_z + HOVER - TZ)
  r = arm.move_to(_cm(obj.center[0]), _cm(obj.center[1]), h_cm, 0.0, jaw)
  if not r.startswith("ok"):
    r = arm.move_to(_cm(obj.center[0]), _cm(obj.center[1]), h_cm, None, jaw)
  return obj if r.startswith("ok") else None


def _lift_verified(arm) -> bool:
  return (arm.state()["holding_signal"] and arm.move_by(0, 0, 5.0).startswith("ok")
          and arm.state()["holding_signal"])


def _expert_grasp(agent, arm, obj, note: str) -> dict:
  from so101_reach.agent.pipeline import FINE_TOOLS, GRASP_INSTRUCTIONS
  width = float(min(obj.size[:2]))
  task = (f"GRASP the {obj.name}. {obj.describe(TZ)}. The grip point is now about {100 * HOVER:.0f} cm above its top; "
          f"the gripper is open about {100 * (width + 0.02):.1f} cm."
          + (f" A scripted grasp attempt just failed ({note}): try something different." if note else ""))
  return agent.run(task, [obj.name], GRASP_INSTRUCTIONS, FINE_TOOLS, 10,
                   stop_when=lambda tool, res: tool == "close_gripper" and arm.state()["holding_signal"])


def run_episode(ep: int, seed: int, ctrl, bc, args, make_agent) -> list[dict]:
  cfg, label, meta = random_scene(seed)
  world = SimWorld(cfg)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  arm = AgentArm(plant, ctrl, SafeTargets(), TZ)
  per = SimPerception(world)
  rng = np.random.default_rng(seed + 7_000_000)
  rows: list[dict] = []

  def row(source: str, obj, params, success: bool, note: str) -> dict:
    return {"episode": ep, "seed": seed, "label": label, **meta, "t": round(time.time(), 1),
            "features": [round(f, 3) for f in features_for(obj, TZ)], "source": source,
            "params": params, "success": success, "note": note[:160]}

  try:
    obj0 = _approach(arm, per, label)
    if obj0 is None:
      return [{"episode": ep, "seed": seed, "label": label, **meta, "source": "abort",
               "success": False, "note": "not detected / unreachable / too wide"}]
    base = bc.predict(obj0, TZ) if bc else heuristic_grasp_params(arm, obj0, TZ)
    plans = [("bc" if bc else "heuristic", base)]
    plans += [(f"perturb{i + 1}", perturb_params(base, rng)) for i in range(args.perturb_retries)]
    for source, params in plans:
      obj = obj0 if not rows else _approach(arm, per, label)  # re-measure: the object may have moved
      if obj is None:
        rows.append(row(source, obj0, params, False, "object lost after a failed attempt"))
        return rows
      ok, note = scripted_grasp(arm, obj, TZ, params)
      # Label = the COMMANDED params (what replay executes). The achieved pose differs by the arm's
      # tracking error; replaying it as a command would add that error a second time. It is kept
      # as "achieved" for analysis, distilled now while closed (_lift_verified raises it 5 cm).
      achieved = distill_grasp(arm, obj, params["force"]) if ok else None
      ok = ok and _lift_verified(arm)
      rows.append({**row(source, obj, params, ok, note), "achieved": achieved if ok else None})
      if ok:
        return rows
      arm.gripper(min(80.0, arm.percent_for_gap(float(min(obj.size[:2])) + 0.02)), "medium")
      _rise_to(arm, obj.top_z + HOVER)
    if make_agent is not None:
      obj = _approach(arm, per, label)
      if obj is None:
        rows.append(row("expert", obj0, {}, False, "object lost before the expert attempt"))
        return rows
      agent = make_agent(arm, per)
      out = _expert_grasp(agent, arm, obj, rows[-1]["note"])
      p = distill_grasp(arm, obj, agent.last_close_force)  # before the 5 cm verification lift
      ok = _lift_verified(arm)
      p = p if ok else {}
      rows.append({**row("expert", obj, p, ok, out.get("summary", "")),
                   "tokens": out.get("input_tokens", 0) + out.get("output_tokens", 0)})
  finally:
    world.close_renderers()
  return rows


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--episodes", type=int, default=50)
  p.add_argument("--seed", type=int, default=0, help="first seed; episode k uses seed+k (keep ranges disjoint from eval)")
  p.add_argument("--out", type=Path, default=OUT_DEFAULT)
  p.add_argument("--perturb-retries", type=int, default=2)
  p.add_argument("--expert", default="none", choices=["none", "local", "openai"])
  p.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="expert=local: OpenAI-compatible server")
  p.add_argument("--model", default=None, help="expert model (default: qwen3-vl:8b-instruct local, gpt-5.5 openai)")
  p.add_argument("--effort", default=None, choices=["none", "low", "medium", "high"],
                 help="expert reasoning effort (default: none for local, low for openai)")
  p.add_argument("--max-expert", type=int, default=None, help="stop querying the expert after this many calls")
  p.add_argument("--bc-model", type=Path, default=None, help="use this BC model as the base policy (DAgger iteration)")
  p.add_argument("--speed", type=float, default=0.5)
  args = p.parse_args()

  bc = BcGrasp.load(args.bc_model) if args.bc_model else None
  if args.bc_model and bc is None:
    raise SystemExit(f"cannot load BC model {args.bc_model}")
  make_agent = None
  expert_calls = [0]
  if args.expert != "none":
    from so101_reach.agent.llm import Agent
    model = args.model or ("qwen3-vl:8b-instruct" if args.expert == "local" else "gpt-5.5")
    base_url = args.base_url if args.expert == "local" else None
    effort = args.effort or ("none" if args.expert == "local" else "low")

    def make_agent(arm, per, _m=model, _b=base_url, _e=effort):
      expert_calls[0] += 1
      return Agent(arm, per, _m, _e, 10, base_url=_b, verbose=False)

  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=POLICY)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  n_ok = 0
  by_source: dict[str, list[int]] = {}
  t0 = time.time()
  with open(args.out, "a") as f:
    for ep in range(args.episodes):
      mk = make_agent if make_agent is None or args.max_expert is None or expert_calls[0] < args.max_expert else None
      rows = run_episode(ep, args.seed + ep, ctrl, bc, args, mk)
      for r in rows:
        f.write(json.dumps(r) + "\n")
      f.flush()
      ok = any(r["success"] for r in rows)
      n_ok += ok
      src = rows[-1]["source"]
      by_source.setdefault(src, [0, 0])
      by_source[src][0] += ok
      by_source[src][1] += 1
      print(f"[gen] ep {ep + 1}/{args.episodes} seed {args.seed + ep} {rows[-1].get('label', '?'):16s} "
            f"{'OK  ' if ok else 'FAIL'} via {src:10s} ({len(rows)} attempt(s)); "
            f"running success {n_ok}/{ep + 1}", flush=True)
  print(f"[gen] done: {n_ok}/{args.episodes} episodes succeeded in {time.time() - t0:.0f} s -> {args.out}")
  for src, (ok, tot) in sorted(by_source.items()):
    print(f"[gen]   decided by {src:10s}: {ok}/{tot}")
  if make_agent is not None:
    print(f"[gen]   expert calls: {expert_calls[0]}")


if __name__ == "__main__":
  main()
