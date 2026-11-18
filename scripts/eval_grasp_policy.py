"""Evaluate grasp-parameter policies on held-out randomized simulation scenes.

Every seed is one scene from `gen_grasp_data.random_scene` (use a seed range disjoint from the
training data). Each condition rebuilds the same scene from scratch, so all conditions see
identical objects and poses:

  heuristic        scripted default parameters, one attempt
  bc               the BC model (data/il/bc_grasp.json), one attempt
  heuristic+retry  heuristic, then --retries perturbed retries
  bc+retry         BC, then --retries perturbed retries
  (+expert)        optional: the LLM grasp phase on what bc+retry still failed (--expert)

Success = holding signal AND still held after a 5 cm lift (same check as data generation).
Prints per-condition success rates by shape / stacked bucket, mean attempts, and every
failure (seed, object, last note) - not only aggregates. Writes a JSON report to results/.

Examples:
  python scripts/eval_grasp_policy.py --episodes 40 --seed 100000
  python scripts/eval_grasp_policy.py --episodes 20 --seed 100000 --conditions heuristic bc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script with the scene helpers

import gen_grasp_data as G  # noqa: E402  (sets MUJOCO_GL before mujoco is imported)

from so101_reach.agent.bc_grasp import DEFAULT_MODEL, BcGrasp  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.settings import AGENT_POLICY_DIR as POLICY  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

CONDITIONS = ("heuristic", "bc", "heuristic+retry", "bc+retry")


def run_condition(cond: str, seed: int, ctrl, bc, retries: int, make_agent=None) -> dict:
  use_bc = cond.startswith("bc")
  k = retries if cond.endswith("+retry") else 0
  rows = G.run_episode(0, seed, ctrl, bc if use_bc else None, SimpleNamespace(perturb_retries=k), make_agent)
  ok = any(r["success"] for r in rows)
  first = rows[0]
  return {"seed": seed, "label": first.get("label", "?"), "shape": first.get("shape", "?"),
          "stacked": bool(first.get("stacked", False)), "success": ok,
          "first_try": bool(first.get("success")), "attempts": sum(r["source"] != "abort" for r in rows),
          "aborted": first.get("source") == "abort", "decided_by": rows[-1]["source"],
          "note": rows[-1].get("note", ""), "tokens": sum(r.get("tokens", 0) for r in rows)}


def summarize(results: list[dict]) -> dict:
  def rate(rs):
    return {"ok": sum(r["success"] for r in rs), "n": len(rs),
            "rate": round(sum(r["success"] for r in rs) / max(1, len(rs)), 3)}
  valid = [r for r in results if not r["aborted"]]
  out = {"all": rate(valid), "aborted": sum(r["aborted"] for r in results),
         "first_try_rate": round(sum(r["first_try"] for r in valid) / max(1, len(valid)), 3),
         "mean_attempts": round(sum(r["attempts"] for r in valid) / max(1, len(valid)), 2),
         "tokens": sum(r["tokens"] for r in results), "by_shape": {}, "by_stacked": {}}
  for shape in sorted({r["shape"] for r in valid}):
    out["by_shape"][shape] = rate([r for r in valid if r["shape"] == shape])
  for st in (False, True):
    rs = [r for r in valid if r["stacked"] == st]
    if rs:
      out["by_stacked"]["stacked" if st else "table"] = rate(rs)
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--episodes", type=int, default=40)
  p.add_argument("--seed", type=int, default=100_000, help="first held-out seed (keep disjoint from training seeds)")
  p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
  p.add_argument("--retries", type=int, default=2, help="perturbed retries for the +retry conditions")
  p.add_argument("--bc-model", type=Path, default=DEFAULT_MODEL)
  p.add_argument("--expert", default="none", choices=["none", "local", "openai"],
                 help="add a bc+retry+expert column: the LLM grasps what bc+retry failed")
  p.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
  p.add_argument("--model", default=None)
  p.add_argument("--max-expert", type=int, default=10)
  p.add_argument("--speed", type=float, default=0.5)
  p.add_argument("--out", type=Path, default=None, help="JSON report (default: results/grasp_eval_<time>.json)")
  args = p.parse_args()

  bc = BcGrasp.load(args.bc_model)
  conds = [c for c in args.conditions if bc is not None or not c.startswith("bc")]
  if len(conds) < len(args.conditions):
    print(f"[eval] no BC model at {args.bc_model}: skipping the bc conditions (run scripts/train_bc_grasp.py)")
  make_agent, expert_calls = None, [0]
  if args.expert != "none":
    from so101_reach.agent.llm import Agent
    model = args.model or ("qwen3-vl:8b-instruct" if args.expert == "local" else "gpt-5.5")
    base_url, effort = (args.base_url, "none") if args.expert == "local" else (None, "low")

    def make_agent(arm, per):
      expert_calls[0] += 1
      return Agent(arm, per, model, effort, 10, base_url=base_url, verbose=False)

  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=POLICY)
  results: dict[str, list[dict]] = {c: [] for c in conds}
  expert_col = "bc+retry+expert" if bc is not None else "heuristic+retry+expert"
  if make_agent is not None:
    results[expert_col] = []
  t0 = time.time()
  for i in range(args.episodes):
    seed = args.seed + i
    line = []
    for c in conds:
      r = run_condition(c, seed, ctrl, bc, args.retries)
      results[c].append(r)
      line.append(f"{c}={'OK' if r['success'] else ('--' if r['aborted'] else 'FAIL')}")
    if make_agent is not None:
      mk = make_agent if expert_calls[0] < args.max_expert else None
      r = run_condition("bc+retry" if bc is not None else "heuristic+retry", seed, ctrl, bc, args.retries, mk)
      results[expert_col].append(r)
      line.append(f"expert={'OK' if r['success'] else 'FAIL'}")
    label = next(iter(results.values()))[-1]["label"]
    print(f"[eval] {i + 1}/{args.episodes} seed {seed} {label:16s} " + " ".join(line), flush=True)

  summary = {c: summarize(rs) for c, rs in results.items()}
  print(f"\n[eval] {args.episodes} held-out scenes (seeds {args.seed}-{args.seed + args.episodes - 1}), "
        f"{time.time() - t0:.0f} s")
  shapes = sorted({s for v in summary.values() for s in v["by_shape"]})
  head = f"{'condition':24s} {'success':>10s} {'1st try':>8s} {'attempts':>8s} " + " ".join(f"{s:>9s}" for s in shapes) \
      + f" {'table':>9s} {'stacked':>9s}"
  print(head)
  print("-" * len(head))
  for c, s in summary.items():
    cell = lambda d: f"{d['ok']}/{d['n']}" if d else "-"  # noqa: E731
    print(f"{c:24s} {cell(s['all']):>10s} {s['first_try_rate']:>8.2f} {s['mean_attempts']:>8.2f} "
          + " ".join(f"{cell(s['by_shape'].get(sh)):>9s}" for sh in shapes)
          + f" {cell(s['by_stacked'].get('table')):>9s} {cell(s['by_stacked'].get('stacked')):>9s}")
  print("\n[eval] failures per condition:")
  for c, rs in results.items():
    fails = [r for r in rs if not r["success"]]
    print(f"  {c}: {len(fails)}")
    for r in fails:
      print(f"    seed {r['seed']} {r['label']:16s} {'ABORT ' if r['aborted'] else ''}{r['note'][:110]}")

  out = args.out or so101.PROJECT_ROOT / "results" / f"grasp_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "summary": summary,
                             "bc_meta": bc.meta if bc is not None else None, "episodes": results}, indent=1))
  print(f"\n[eval] report -> {out}")


if __name__ == "__main__":
  main()
