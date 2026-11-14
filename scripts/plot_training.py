"""Summarise / plot training curves from the TensorBoard event files of a run.

  python scripts/plot_training.py                       # latest run
  python scripts/plot_training.py --run logs/rsl_rl/so101_reach/<run> --every 250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from so101_reach.checkpoints.policy_io import LOG_ROOT  # noqa: E402

TAGS = {
  "reward": "Train/mean_reward",
  "stage": "Curriculum/stage/stage",
  "success": "Curriculum/stage/success_rate",
  "tol_mm": "Curriculum/stage/success_tol_mm",
  "err_m": "Metrics/ee_target/position_error",
  "std": "Policy/mean_std",
  "self_coll": "Episode_Termination/self_collision",
  "ground": "Episode_Termination/ground_contact",
}


def load(run: Path) -> dict[str, tuple[list[int], list[float]]]:
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  avail = set(ea.Tags()["scalars"])
  out = {}
  for k, tag in TAGS.items():
    cands = [tag] + [t for t in avail if t.endswith(tag.split("/")[-1]) and k in ("reward", "std")]
    tag = next((t for t in cands if t in avail), None)
    if tag:
      ev = ea.Scalars(tag)
      out[k] = ([e.step for e in ev], [e.value for e in ev])
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=Path, default=None)
  p.add_argument("--every", type=int, default=200)
  args = p.parse_args()
  run = args.run or sorted(p for p in LOG_ROOT.iterdir() if p.is_dir())[-1]
  d = load(run)
  keys = [k for k in TAGS if k in d]
  print(f"run: {run}")
  print("iter  " + "  ".join(f"{k:>9s}" for k in keys))
  steps = d[keys[0]][0]
  for i, s in enumerate(steps):
    if s % args.every == 0 or i == len(steps) - 1:
      print(f"{s:5d} " + "  ".join(f"{d[k][1][min(i, len(d[k][1]) - 1)]:9.4f}" for k in keys))
  fig, axes = plt.subplots(2, 3, figsize=(15, 7))
  for ax, k in zip(axes.flat, ("reward", "success", "err_m", "stage", "std", "self_coll"), strict=False):
    if k in d:
      ax.plot(*d[k])
      ax.set_title(TAGS[k])
      ax.set_xlabel("iteration")
      ax.grid(alpha=0.3)
  fig.tight_layout()
  out = run / "training_curves.png"
  fig.savefig(out, dpi=110)
  print(f"saved {out}")


if __name__ == "__main__":
  main()
