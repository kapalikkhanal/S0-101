"""An LLM agent picks (and places) objects with an RL reach policy doing the travelling.

Two modes:
  --mode pipeline (default): code measures, travels (RL) to 5 cm above the object and above the
      target, retries failed grasps and goes home; the agent only does the GRASP and the PLACE
      (src/so101_reach/agent/pipeline.py). Free-text tasks (stacking, unstacking, several objects)
      are turned into ordered pick-and-place steps by one planner call.
  --mode free: the agent decides every motion itself (more flexible, slower, less predictable).

  python scripts/agent_pick.py --sim table  --object "red cube" --place "white box"   # simulated arm
  python scripts/agent_pick.py --sim on_box --object "red cube" --place "white box"
  python scripts/agent_pick.py --real --object "black cube" --place "white container" # REAL ARM
  python scripts/agent_pick.py --task "stack the red cube on the white cube" --track "red cube" "white cube"

Options: --model (default gpt-5.5; e.g. gpt-6-luna, gpt-5.4-mini), --effort low|medium|high,
--max-turns, --speed (RL travel speed 0.05-1).
Local grasp rung: a trained BC model (data/il/bc_grasp.json, scripts/train_bc_grasp.py) is used
between a skill-cache hit and the scripted default; --no-bc disables it.
Perception: --tracker kf wraps perception in per-object Kalman filters (src/so101_reach/agent/tracker.py);
the arm is then parked out of the head camera's view only when the tracker is not confident.
In sim, --sim-noise adds camera-like noise / outliers / arm occlusion and --sim-clear-view parks
the arm like on the real robot (to measure the parks the tracker saves). The key is read from OPENAI_API_KEY or
.env.local (OPENAI_KEY=...). Logs (images, calls, timing, tokens): logs/agent/<time>/.
On the real arm the arm holds its pose at the end (scripts/real_release.py lets it go limp).
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from so101_reach.agent.arm import AgentArm  # noqa: E402
from so101_reach.agent.llm import Agent  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.mouse_real import SafeTargets  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

from so101_reach.settings import AGENT_POLICY_DIR as POLICY


def run_task(args, agent, arm, per, task: str, track: list[str], clear_view: bool) -> dict:
  if args.mode == "free":
    return agent.run(task, track)
  from so101_reach.agent.pipeline import Pipeline
  from so101_reach.agent.skill_cache import SkillCache

  cache = None if args.no_cache else (SkillCache(args.cache_dir) if args.cache_dir else SkillCache())
  steps = [{"pick": args.object, "place_on": args.place}] if (args.object and args.place and not args.task) else None
  return Pipeline(agent, arm, per, clear_view=clear_view, cache=cache, handoff=args.handoff,
                  bc=load_bc(args)).run(task, track, steps)


def load_bc(args):
  if args.no_bc or args.handoff != "local-first":
    return None
  from so101_reach.agent.bc_grasp import BcGrasp

  bc = BcGrasp.load(args.bc_model)
  print(f"[agent] BC grasp model: {args.bc_model if bc is not None else 'none (' + str(args.bc_model) + ' missing)'}")
  return bc


def wrap_perception(args, per, track: list[str], arm_points=None):
  if args.tracker == "kf":
    from so101_reach.agent.tracker import TrackedPerception

    print("[agent] perception: Kalman object tracker")
    return TrackedPerception(per, track, arm_points=arm_points)
  return per


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--object", default=None)
  p.add_argument("--place", default=None)
  p.add_argument("--task", default=None, help="free-text task instead of --object/--place")
  p.add_argument("--mode", default="pipeline", choices=["pipeline", "free"])
  p.add_argument("--track", nargs="*", default=None, help="object names measured every turn (default: object, place)")
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--sim", choices=["table", "on_box", "stack", "tower"], default="table", help="simulated scene (default: table)")
  mode.add_argument("--real", dest="sim", action="store_const", const=None, help="explicitly operate the real arm")
  p.add_argument("--policy-dir", type=Path, default=POLICY, help="exported ONNX policy directory")
  p.add_argument("--model", default=os.environ.get("SO101_OPENAI_MODEL", "gpt-5.5"))
  p.add_argument("--effort", default="low", choices=["none", "low", "medium", "high"])
  p.add_argument("--base-url", default=None,
                 help="local OpenAI-compatible server instead of the OpenAI API, e.g. http://localhost:8000/v1 (vLLM: Cosmos 3) "
                      "or http://localhost:11434/v1 (Ollama); --model is then the served model name")
  p.add_argument("--max-turns", type=int, default=30)
  p.add_argument("--max-input-tokens", type=int, default=300_000, help="stop the run (arm holds) after this many input tokens")
  p.add_argument("--head-width", type=int, default=640, help="head image width sent to the model [px]")
  p.add_argument("--wrist-width", type=int, default=512, help="wrist image width sent to the model [px]")
  p.add_argument("--detail", default="high", choices=["low", "high", "auto"], help="image detail (low = cheapest)")
  p.add_argument("--speed", type=float, default=None, help="RL travel speed 0.05-1.0 (default: 0.3 real, 0.5 sim)")
  p.add_argument("--handoff", default="local-first", choices=["local-first", "llm-only"],
                 help="pipeline mode: local-first = cached/scripted grasp+place, LLM only on failure (default); "
                      "llm-only = the LLM runs every grasp/place phase (A/B token baseline)")
  p.add_argument("--no-cache", action="store_true", help="pipeline mode: do not read or write the skill cache")
  p.add_argument("--cache-dir", type=Path, default=None,
                 help="skill-cache directory (default: data/agent_cache)")
  p.add_argument("--bc-model", type=Path, default=None, help="BC grasp model (default: data/il/bc_grasp.json if present)")
  p.add_argument("--no-bc", action="store_true", help="do not use the BC grasp model")
  p.add_argument("--tracker", default="none", choices=["none", "kf"], help="kf: Kalman object tracking (fewer camera parks)")
  p.add_argument("--sim-noise", action="store_true", help="sim: camera-like measurement noise, outliers and arm occlusion")
  p.add_argument("--noise-seed", type=int, default=0)
  p.add_argument("--sim-clear-view", action="store_true", help="sim: park the arm out of view before measuring, like the real arm")
  p.add_argument("--yes", action="store_true", help="real arm: start without asking")
  args = p.parse_args()
  if args.bc_model is None:
    from so101_reach.agent.bc_grasp import DEFAULT_MODEL

    args.bc_model = DEFAULT_MODEL
  if args.task is None and args.object is None:
    p.error("give --object (and optionally --place) or --task")
  task = args.task or (f"Pick up the {args.object}" + (f" and put it into the {args.place}" if args.place else " and lift it about 5 cm")
                       + ", then go home.")
  track = args.track if args.track is not None else [n for n in (args.object, args.place) if n]
  if args.base_url is None:
    from so101_reach.agent.llm import load_api_key
    load_api_key()  # Check credentials before any hardware engagement.
  if not args.sim:
    from so101_reach.settings import require_hardware_config
    require_hardware_config(cameras=True, servo_fit=True)
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed or (0.5 if args.sim else 0.3)), policy_dir=args.policy_dir)

  if args.sim:
    from so101_reach.agent.perception import SimPerception
    from so101_reach.agent.sim_world import SimWorld, scene
    from so101_reach.real.robot import SimRealPlant

    world = SimWorld(scene(args.sim))
    world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
    plant = SimRealPlant(world, world.jmap, world.fcal)
    plant.engage()
    for _ in range(25):
      plant.wait()
    arm = AgentArm(plant, ctrl, SafeTargets(), world.cfg.table_z)
    per = SimPerception(world)
    if args.sim_noise:
      from so101_reach.agent.sim_noise import NoisyPerception

      per = NoisyPerception(per, arm_points=arm.arm_points, seed=args.noise_seed)
    per = wrap_perception(args, per, track, arm_points=arm.arm_points)
    agent = Agent(arm, per, args.model, args.effort, args.max_turns, base_url=args.base_url,
                  head_width=args.head_width, wrist_width=args.wrist_width, detail=args.detail, max_input_tokens=args.max_input_tokens)
    print(f"[agent] SIM {args.sim}: {task}  (model {args.model}, effort {args.effort}); log {agent.log_dir}")
    res = run_task(args, agent, arm, agent.per, task, track, clear_view=args.sim_clear_view)
    print(f"[agent] result: {res}")
    if args.object and args.place:
      truth = [s for s in world.surfaces() if args.place in s["label"] or s["label"] in args.place]
      obj = [lab for lab in world.labels if all(w in lab for w in args.object.split()) or args.object.split()[-1] in lab]
      if truth and obj:
        c, box = world.object_truth(obj[0])["center"], truth[0]["corners"]
        inside = box[:, 0].min() < c[0] < box[:, 0].max() and box[:, 1].min() < c[1] < box[:, 1].max() and c[2] < truth[0]["top"]
        print(f"[agent] ground truth: {obj[0]} {'IS' if inside else 'is NOT'} inside the {truth[0]['label']} (centre {np.round(c, 3)})")
    for lab in world.labels:
      g = world.object_truth(lab)
      print(f"[agent] ground truth: {lab} centre ({100 * g['center'][0]:.1f}, {100 * g['center'][1]:.1f}) cm, "
            f"bottom h={100 * (g['bottom'] - world.cfg.table_z):.1f} cm")
    world.close_renderers()
    return

  from so101_reach.agent.perception import RealPerception
  from so101_reach.real.cameras import open_camera
  from so101_reach.real.feetech_bus import FeetechBus, load_calibration
  from so101_reach.real.handeye import WRIST_EXTRINSICS_PATH, HeadCameraCalibration, WristCameraCalibration, WristCameraPose
  from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap
  from so101_reach.real.orange import OrangeDetector
  from so101_reach.real.robot import RealPlant

  jmap, fcal = JointMap.load(), load_calibration()
  hcal = HeadCameraCalibration.load()
  holder: dict = {}

  def joints() -> dict[str, float]:
    if "arm" not in holder:
      return {**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS}
    q6, _ = holder["arm"].q6()
    return dict(zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), q6, strict=True))

  with contextlib.ExitStack() as stack:
    head = stack.enter_context(open_camera("head"))
    head.warmup(1.5)
    table_z = OrangeDetector(head, hcal).estimate_table_z(head.read())
    if table_z is None:
      raise SystemExit("cannot measure the table height from the head camera")
    wrist, wpose = None, None
    if WRIST_EXTRINSICS_PATH.exists():
      wrist = stack.enter_context(open_camera("wrist"))
      wrist.warmup(1.0)
      wpose = WristCameraPose(WristCameraCalibration.load(), joints)
    per = RealPerception(head, wrist, hcal, wpose, table_z, arm_points=lambda: holder["arm"].arm_points() if "arm" in holder else None)
    per = wrap_perception(args, per, track)
    bus = stack.enter_context(FeetechBus(FOLLOWER_PORT, calib=fcal))
    plant = RealPlant(bus, jmap, fcal, torque_limit=400)
    if not args.yes:
      input(f"[agent] REAL ARM: {task}\n        model {args.model}; table at {100 * table_z:+.1f} cm. Enter = start (Ctrl+C = abort) ")
    plant.engage()
    try:
      arm = AgentArm(plant, ctrl, SafeTargets(), table_z)
      holder["arm"] = arm
      agent = Agent(arm, per, args.model, args.effort, args.max_turns, base_url=args.base_url,
                    head_width=args.head_width, wrist_width=args.wrist_width, detail=args.detail, max_input_tokens=args.max_input_tokens)
      print(f"[agent] log {agent.log_dir}")
      res = run_task(args, agent, arm, per, task, track, clear_view=True)
      print(f"[agent] result: {res}")
    finally:
      plant.hold()
      print("[agent] the arm holds its pose.")


if __name__ == "__main__":
  main()
