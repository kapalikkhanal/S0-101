"""Test 2 - robustness: success vs reality-gap noise level.

Level 0 = exact simulator, 1 = our estimate of the real arm + camera, 2 = pessimistic. One knob
scales servo noise / calibration bias, object physics randomization and camera noise together.
Same seeds as test 1, so levels 0 and 1 are shared with it (results/report/manip.jsonl).

  python scripts/report/t02_noise_sweep.py
  python scripts/report/t02_noise_sweep.py --levels 0 1 2 3 --n-fast 50 --n-llm 0
"""

from __future__ import annotations

import argparse

from common import DEFAULT_BASE_URL, DEFAULT_MODEL, LLM_METHODS, RESULTS, run_grid

from t01_manip_methods import SEED0


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--methods", nargs="+", default=["rl_scripted", "rl_bc", "vla", "hybrid"])
  p.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
  p.add_argument("--n-fast", type=int, default=100)
  p.add_argument("--n-llm", type=int, default=15, help="LLM episodes per extra level (0 = skip LLM methods)")
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--base-url", default=DEFAULT_BASE_URL)
  args = p.parse_args()
  jobs = []
  for i in range(max(args.n_fast, args.n_llm)):
    for m in args.methods:
      n = args.n_llm if m in LLM_METHODS else args.n_fast
      if i < n:
        jobs += [(m, SEED0 + i, lv) for lv in args.levels]
  run_grid(RESULTS / "manip.jsonl", jobs, workers=args.workers, model=args.model, base_url=args.base_url)


if __name__ == "__main__":
  main()
