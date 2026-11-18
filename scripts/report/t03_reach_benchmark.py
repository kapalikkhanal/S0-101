"""Test 3 - reaching: RL (PPO) vs classical IK controllers, nominal and under perturbations.

Runs scripts/evaluate.py (batched MjLab/GPU benchmark on the frozen suite: targets x 4 initial-
configuration families) for:
  nominal     native RL (sim policy), native RL (identified-servo fine-tuned policy), DLS IK,
              global IK + joint interpolation
  delay       action delay 1..3 steps (20 ms each)       - communication latency
  kp / kv     servo stiffness / damping scaled 0.6..1.4   - servo model mismatch
  identified  physics with the servo model fitted on the real arm (sim-to-real proxy)
  friction    identified physics, joint friction (stiction) x0.5..3 - the dominant unknown on
              cheap servos (arXiv 2503.01255); sim policy, real-arm policy and DLS IK
  deadband    identified physics, servo dead band 1..8 ticks (0.09..0.7 deg)
  realistic   identified physics + friction x1.5 + dead band 2 ticks + 1-step delay (combined)
  noservodr   the identified-servo policy trained WITHOUT servo randomization
              (So101-Reach-Real-NoServoDR, --noservodr-ckpt) on identified/friction/realistic:
              completes the identification-vs-randomization ablation

Per-episode metrics (episodes.npz) and summaries land in results/<tag>/<controller>/; feeds the
error-CDF, workspace error map, smoothness/safety and reach-robustness graphs.

  python scripts/report/t03_reach_benchmark.py
  python scripts/report/t03_reach_benchmark.py --num-targets 100 --only nominal
  python scripts/report/t03_reach_benchmark.py --only friction deadband realistic
  python scripts/report/t03_reach_benchmark.py --only noservodr --noservodr-ckpt logs/rsl_rl/so101_reach/<run>/model_9997.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_CKPT = ROOT / "checkpoints" / "so101_reach_policy" / "model_5999.pt"
REAL_CKPT = ROOT / "checkpoints" / "so101_reach_policy_real" / "model_9997.pt"
FRICTION_SCALES = (0.5, 1.5, 2.0, 3.0)  # 1.0 = the "identified" run
DEADBAND_TICKS = (1, 2, 4, 8)  # 0 = the "identified" run
REALISTIC = ["--friction-scale", "1.5", "--deadband-ticks", "2", "--delay-steps", "1"]


def run(tag: str, controllers: list[str], extra: list[str], ckpt: Path, n: int) -> None:
  out = ROOT / "results" / f"report_reach_{tag}" / "benchmark.md"
  if out.exists():
    print(f"[t03] {tag}: already done ({out})")
    return
  cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--controllers", *controllers, "--checkpoint", str(ckpt),
         "--num-targets", str(n), "--tag", f"report_reach_{tag}", *extra]
  print("[t03]", " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--num-targets", type=int, default=250, help="targets per initial-configuration family")
  p.add_argument("--only", nargs="*", default=None,
                 help="subset of: nominal real_policy delay kp kv identified friction deadband realistic noservodr")
  p.add_argument("--noservodr-ckpt", type=Path, default=None, help="checkpoint trained on So101-Reach-Real-NoServoDR")
  args = p.parse_args()
  want = lambda k: args.only is None or k in args.only  # noqa: E731
  n = args.num_targets
  # The suite is built once (fixed seed) and reused by every run.
  if want("nominal"):
    run("nominal", ["native", "ik", "ik_joint"], [], SIM_CKPT, n)
  if want("real_policy"):
    run("real_policy", ["native"], [], REAL_CKPT, n)
  if want("identified"):
    run("identified", ["native", "ik"], ["--actuator-model", "identified"], SIM_CKPT, n)
    run("identified_realpolicy", ["native"], ["--actuator-model", "identified"], REAL_CKPT, n)
  if want("delay"):
    for d in (1, 2, 3):
      run(f"delay{d}", ["native", "ik"], ["--delay-steps", str(d)], SIM_CKPT, n)
  if want("kp"):
    for s in (0.6, 0.8, 1.2, 1.4):
      run(f"kp{s:g}", ["native", "ik"], ["--kp-scale", str(s)], SIM_CKPT, n)
  if want("kv"):
    for s in (0.6, 1.4):
      run(f"kv{s:g}", ["native", "ik"], ["--kv-scale", str(s)], SIM_CKPT, n)
  ident = ["--actuator-model", "identified"]
  if want("friction"):
    for s in FRICTION_SCALES:
      run(f"fric{s:g}", ["native", "ik"], [*ident, "--friction-scale", str(s)], SIM_CKPT, n)
      run(f"fric{s:g}_realpolicy", ["native"], [*ident, "--friction-scale", str(s)], REAL_CKPT, n)
  if want("deadband"):
    for d in DEADBAND_TICKS:
      run(f"db{d:g}", ["native", "ik"], [*ident, "--deadband-ticks", str(d)], SIM_CKPT, n)
      run(f"db{d:g}_realpolicy", ["native"], [*ident, "--deadband-ticks", str(d)], REAL_CKPT, n)
  if want("realistic"):
    run("realistic", ["native", "ik"], [*ident, *REALISTIC], SIM_CKPT, n)
    run("realistic_realpolicy", ["native"], [*ident, *REALISTIC], REAL_CKPT, n)
  if want("noservodr"):
    if args.noservodr_ckpt is None:
      p.error("--only noservodr needs --noservodr-ckpt (train with: python scripts/train.py --task So101-Reach-Real-NoServoDR ...)")
    ck = args.noservodr_ckpt.resolve()
    run("identified_noservodr", ["native"], ident, ck, n)
    run("realistic_noservodr", ["native"], [*ident, *REALISTIC], ck, n)
    for s in FRICTION_SCALES:
      run(f"fric{s:g}_noservodr", ["native"], [*ident, "--friction-scale", str(s)], ck, n)


if __name__ == "__main__":
  main()
