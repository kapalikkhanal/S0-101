"""Benchmark reaching controllers on the frozen evaluation suite.

Controllers:
  native   trained MjLab PPO policy (--checkpoint, default: latest under logs/)
  ik       damped-least-squares resolved-rate IK through the same action interface
  ik_joint global multi-restart IK + synchronized trapezoidal joint interpolation
  nvidia   nvidia/so101-reach ONNX policy via the adapter env

Examples:
  python scripts/evaluate.py --controllers nvidia ik
  python scripts/evaluate.py --controllers native ik nvidia --checkpoint logs/rsl_rl/so101_reach/<run>/model_6000.pt
  python scripts/evaluate.py --controllers nvidia --nvidia-rebase every_10ms --nvidia-actuator isaac_pd --tag nv_isaac
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from so101_reach.checkpoints import policy_io  # noqa: E402
from so101_reach.evaluation.benchmark import (  # noqa: E402
  INIT_FAMILIES,
  EvalSuite,
  Perturbation,
  build_report,
  episode_metrics,
  make_suite,
  run_suite,
)
from so101_reach.evaluation.ik_controller import (  # noqa: E402
  DlsIkCfg,
  DlsIkController,
  JointInterpolationController,
  solve_ik_batch,
)
from so101_reach.sim import so101  # noqa: E402
from so101_reach.tasks.eval_env_cfg import make_eval_env_cfg  # noqa: E402

RESULTS = so101.PROJECT_ROOT / "results"
SUITE_PATH = so101.PROJECT_ROOT / "data" / "eval" / "suite_v2.npz"

ROWS = [
  ("mean_err_mm", "Mean settled error [mm]", "{:.2f}"),
  ("median_err_mm", "Median error [mm]", "{:.2f}"),
  ("p95_err_mm", "95th pct error [mm]", "{:.2f}"),
  ("success@50mm", "Success @50 mm", "{:.1%}"),
  ("success@20mm", "Success @20 mm", "{:.1%}"),
  ("success@10mm", "Success @10 mm", "{:.1%}"),
  ("success@5mm", "Success @5 mm", "{:.1%}"),
  ("mean_reach_time@10mm_s", "Reach time @10 mm [s]", "{:.2f}"),
  ("mean_settle_time@10mm_s", "Settle time @10 mm [s]", "{:.2f}"),
  ("mean_settle_time@5mm_s", "Settle time @5 mm [s]", "{:.2f}"),
  ("collision_rate", "Collision rate", "{:.1%}"),
  ("soft_limit_excursion_rate", "Soft-limit excursions", "{:.1%}"),
  ("hard_limit_violation_rate", "Hard-limit violations", "{:.1%}"),
  ("mean_overshoot", "Overshoot past 10 mm [mm]", "{:.2f}"),
  ("mean_rms_joint_acc", "RMS joint accel [rad/s^2]", "{:.1f}"),
  ("mean_rms_tcp_jerk", "RMS TCP jerk [m/s^3]", "{:.1f}"),
  ("mean_rms_action_rate", "RMS action rate", "{:.3f}"),
  ("mean_path_ratio", "Path length / straight line", "{:.2f}"),
  ("mean_peak_tcp_speed", "Peak TCP speed [m/s]", "{:.2f}"),
]


def markdown_table(reports: dict[str, dict], section: str = "overall") -> str:
  names = list(reports)
  lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
  for key, label, fmt in ROWS:
    vals = []
    for n in names:
      v = reports[n][section].get(key, float("nan"))
      vals.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
    lines.append(f"| {label} | " + " | ".join(vals) + " |")
  return "\n".join(lines)


def family_table(reports: dict[str, dict], key: str = "success@10mm", fmt: str = "{:.1%}") -> str:
  names = list(reports)
  lines = ["| Initial configuration | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
  for fam in INIT_FAMILIES:
    lines.append(f"| {fam} | " + " | ".join(fmt.format(reports[n]["by_init"][fam][key]) for n in names) + " |")
  return "\n".join(lines)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--controllers", nargs="+", default=["native", "ik", "nvidia"], choices=["native", "ik", "ik_joint", "nvidia"])
  p.add_argument("--checkpoint", type=Path, default=None)
  p.add_argument("--num-targets", type=int, default=500, help="targets per initial-configuration family")
  p.add_argument("--horizon", type=float, default=5.0)
  p.add_argument("--nvidia-rebase", choices=["policy_step", "every_10ms"], default="policy_step")
  p.add_argument("--nvidia-actuator", choices=["menagerie", "isaac_pd"], default="menagerie")
  p.add_argument("--ik-gain", type=float, default=DlsIkCfg.gain)
  p.add_argument("--ik-vmax", type=float, default=DlsIkCfg.v_max)
  p.add_argument("--delay-steps", type=int, default=0, help="robustness: fixed action delay [20 ms steps]")
  p.add_argument("--kp-scale", type=float, default=1.0, help="robustness: servo stiffness scale")
  p.add_argument("--kv-scale", type=float, default=1.0, help="robustness: servo damping scale")
  p.add_argument("--actuator-model", choices=["menagerie", "identified"], default="menagerie",
                 help="servo model of the evaluation physics (identified = fitted on the real arm)")
  p.add_argument("--tag", default="")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--rebuild-suite", action="store_true")
  args = p.parse_args()

  if args.rebuild_suite or not SUITE_PATH.exists():
    make_suite(args.num_targets).save(SUITE_PATH)
  suite = EvalSuite.load(SUITE_PATH)
  print(f"[eval] suite: {len(suite)} episodes ({len(suite) // len(INIT_FAMILIES)} targets x {INIT_FAMILIES}), "
        f"{int(suite.in_nvidia_box.sum())} in NVIDIA goal box")

  perturb = Perturbation(args.delay_steps, args.kp_scale, args.kv_scale)
  reports: dict[str, dict] = {}
  for ctrl in args.controllers:
    name = ctrl
    if ctrl == "nvidia":
      env = ManagerBasedRlEnv(
        make_eval_env_cfg("nvidia", len(suite), args.nvidia_rebase, args.nvidia_actuator), device=args.device
      )
      policy = policy_io.load_nvidia_obs_policy(args.device)
      name = f"nvidia[{args.nvidia_rebase},{args.nvidia_actuator}]"
      lat = policy_io.measure_latency(policy.module, 20, "cpu") | policy_io.measure_latency(policy.module, 20, args.device)
      from so101_reach.checkpoints.nvidia import DEFAULT_DIR, ONNX_FILE

      lat |= policy_io.measure_onnx_latency(DEFAULT_DIR / ONNX_FILE)
    else:
      env = ManagerBasedRlEnv(make_eval_env_cfg("native", len(suite), actuator_model=args.actuator_model), device=args.device)
      if ctrl == "ik":
        policy = DlsIkController(env, DlsIkCfg(gain=args.ik_gain, v_max=args.ik_vmax))
        name = "dls_ik"
        lat = {}
      elif ctrl == "ik_joint":
        import time

        t0 = time.perf_counter()
        q_goal = solve_ik_batch(suite.targets, suite.init_q)
        solve_ms = (time.perf_counter() - t0) / len(suite) * 1e3
        policy = JointInterpolationController(env, torch.tensor(q_goal))
        name = "ik_joint_interp"
        lat = {"ik_solve_per_target_ms_parallel": solve_ms}
      else:
        ckpt = args.checkpoint or policy_io.latest_checkpoint()
        policy = policy_io.load_native_policy(env, ckpt)
        name = f"native_rl[{ckpt.parent.name}/{ckpt.stem}]"
        obs_dim = env.observation_manager.group_obs_dim["actor"][0]
        lat = policy_io.measure_latency(policy.module, obs_dim, "cpu") | policy_io.measure_latency(policy.module, obs_dim, args.device)
        onnx = ckpt.parent / f"{ckpt.parent.name}.onnx"
        if onnx.exists():
          lat |= policy_io.measure_onnx_latency(onnx)
    traces = run_suite(env, policy, suite, args.horizon, perturb=perturb)
    m = episode_metrics(traces)
    rep = build_report(name, suite, m)
    rep.latency = lat
    rep.notes = {"perturbation": perturb.label}
    out = RESULTS / (args.tag or "latest") / name.replace("/", "_").replace("[", "_").replace("]", "").replace(",", "_")
    rep.save(out, m)
    reports[name] = json.loads((out / "summary.json").read_text())
    o = rep.overall
    print(f"[eval] {name}: mean {o['mean_err_mm']:.2f} mm, median {o['median_err_mm']:.2f} mm, p95 {o['p95_err_mm']:.2f} mm, "
          f"success@10 {o['success@10mm']:.1%}, @5 {o['success@5mm']:.1%}, collisions {o['collision_rate']:.1%}  -> {out}")
    del env, policy
    torch.cuda.empty_cache()

  md = [f"# Reaching benchmark ({len(suite)} episodes, horizon {args.horizon:.1f} s, {perturb.label})\n", "## All targets\n", markdown_table(reports)]
  md += ["\n## Targets inside the NVIDIA training goal box\n", markdown_table(reports, "nvidia_goal_box")]
  md += ["\n## Success @10 mm by initial configuration\n", family_table(reports)]
  md += ["\n## Mean settled error [mm] by initial configuration\n", family_table(reports, "mean_err_mm", "{:.2f}")]
  md += ["\n## Inference latency (batch 1)\n"] + [f"- **{n}**: " + ", ".join(f"{k}={v:.1f}" for k, v in r["latency"].items()) for n, r in reports.items() if r["latency"]]
  out_md = RESULTS / (args.tag or "latest") / "benchmark.md"
  out_md.write_text("\n".join(md) + "\n")
  print("\n".join(md))
  print(f"\n[eval] wrote {out_md}")


if __name__ == "__main__":
  main()
