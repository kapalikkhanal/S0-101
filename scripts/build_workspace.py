"""Generate the train / eval reachable-workspace banks (data/workspace/*.npz).

Usage:
  python scripts/build_workspace.py                # both splits with default sizes
  python scripts/build_workspace.py --split eval --num-samples 1000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so101_reach.sim.workspace import WorkspaceSpec, bank_path, build_bank  # noqa: E402

DEFAULTS = {"train": WorkspaceSpec(num_samples=3_000_000, seed=0), "eval": WorkspaceSpec(num_samples=1_000_000, seed=1)}


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--split", choices=["train", "eval", "both"], default="both")
  p.add_argument("--num-samples", type=int, default=None)
  p.add_argument("--workers", type=int, default=None)
  args = p.parse_args()
  splits = ["train", "eval"] if args.split == "both" else [args.split]
  for split in splits:
    spec = DEFAULTS[split]
    if args.num_samples:
      spec = WorkspaceSpec(**{**spec.__dict__, "num_samples": args.num_samples})
    t0 = time.time()
    bank = build_bank(spec, args.workers)
    bank.save(bank_path(split))
    t = bank.tcp_pos
    r = np.linalg.norm(t[:, :2], axis=1)
    print(
      f"[{split}] {len(bank)} points from {spec.num_samples} samples in {time.time() - t0:.1f}s -> {bank_path(split)}\n"
      f"  x [{t[:, 0].min():+.3f}, {t[:, 0].max():+.3f}]  y [{t[:, 1].min():+.3f}, {t[:, 1].max():+.3f}]  "
      f"z [{t[:, 2].min():+.3f}, {t[:, 2].max():+.3f}]  radial [{r.min():.3f}, {r.max():.3f}] m\n"
      f"  occupied volume ~ {len(np.unique(np.floor(t / spec.voxel), axis=0)) * spec.voxel**3 * 1e3:.1f} L"
    )


if __name__ == "__main__":
  main()
