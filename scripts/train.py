"""Train the SO-101 reaching policy with PPO (MjLab + RSL-RL), then evaluate and export.

This wraps MjLab's own training entry point (``mjlab.scripts.train.launch_training``), so
the equivalent raw command is:
  train So101-Reach --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.seed 42

Examples:
  python scripts/train.py --seed 42 --num-envs 4096 --iterations 6000
  python scripts/train.py --resume-run 2026-10-26_20-14-11_ppo_s42 --resume-checkpoint model_600.pt --iterations 5400
  python scripts/train.py --seed 7 --no-eval

Logs/checkpoints: logs/rsl_rl/so101_reach/<timestamp>_<run-name>/ (TensorBoard events,
model_<it>.pt every 100 iterations with ONNX export, params/{env,agent}.yaml).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import mjlab.tasks  # noqa: E402,F401  (populates the task registry incl. So101-Reach)
from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402

from so101_reach.checkpoints.policy_io import LOG_ROOT, latest_checkpoint  # noqa: E402

TASK = "So101-Reach"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", choices=["So101-Reach", "So101-Reach-Real"], default=TASK)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=6000)
  p.add_argument("--run-name", default=None)
  p.add_argument("--resume-run", default=None, help="run directory name (regex) under logs/rsl_rl/so101_reach")
  p.add_argument("--resume-checkpoint", default="model_.*.pt")
  p.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
  p.add_argument("--no-eval", action="store_true", help="skip automatic evaluation/export after training")
  args = p.parse_args()

  cfg = TrainConfig.from_task(args.task)
  cfg.env.scene.num_envs = args.num_envs
  agent = dataclasses.replace(
    cfg.agent,
    seed=args.seed,
    max_iterations=args.iterations,
    run_name=args.run_name or f"ppo_s{args.seed}" + ("_resume" if args.resume_run else ""),
    logger=args.logger,
    resume=args.resume_run is not None,
    load_run=args.resume_run or ".*",
    load_checkpoint=args.resume_checkpoint,
  )
  cfg = dataclasses.replace(cfg, agent=agent)
  before = set(LOG_ROOT.glob("*")) if LOG_ROOT.exists() else set()
  launch_training(args.task, cfg)

  if args.no_eval:
    return
  new_runs = sorted(set(LOG_ROOT.glob("*")) - before)
  ckpt = latest_checkpoint(new_runs[-1]) if new_runs else latest_checkpoint()
  tag = f"auto_{ckpt.parent.name}"
  py = sys.executable
  subprocess.run([py, str(ROOT / "scripts/evaluate.py"), "--controllers", "native", "ik", "ik_joint", "--checkpoint", str(ckpt), "--tag", tag, "--actuator-model", "identified" if args.task.endswith("-Real") else "menagerie"], check=True)
  subprocess.run([py, str(ROOT / "scripts/export_policy.py"), "--checkpoint", str(ckpt)], check=True)


if __name__ == "__main__":
  main()
