"""Make every report figure (PNG + PDF) and a summary table from results/report/.

Each figure is skipped (with a note) when its test has not produced data yet, so this can be
run at any time. Output: results/report/figures/, results/report/summary.md

  python scripts/report/plot_all.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "report"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode), categorical slots in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
METHOD_ORDER = ["rl_scripted", "rl_bc", "hybrid", "vla", "vla_free"]
METHOD_NAME = {"rl_scripted": "RL + scripted", "rl_bc": "RL + BC", "hybrid": "Hybrid\n(BC → VLA)",
               "vla": "VLA\n(+ RL travel)", "vla_free": "VLA only\n(free)"}
METHOD_COLOR = {m: SLOTS[i] for i, m in enumerate(METHOD_ORDER)}
LEVEL_NAME = {0.0: "exact sim", 0.5: "low noise", 1.0: "realistic noise", 1.5: "high noise", 2.0: "pessimistic"}

plt.rcParams.update({
  "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
  "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
  "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
  "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
  "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
made, skipped = [], []


def load(name: str) -> list[dict]:
  p = R / name
  return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def wilson(k: int, n: int, z: float = 1.96):
  if n == 0:
    return np.nan, np.nan, np.nan
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return p, max(0, c - h), min(1, c + h)


def save(fig, name: str, note: str = "") -> None:
  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)
  made.append((name, note))


def methods_in(rows):
  return [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]


def pct_axis(ax, lim=(0, 1.05)):
  ax.set_ylim(*lim)
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


# --- 1. headline success: method x {exact, realistic} ---------------------------------------
def fig_success(man):
  levels = [lv for lv in (0.0, 1.0) if any(r["level"] == lv for r in man)]
  ms = methods_in(man)
  if not ms:
    return skipped.append("01_success_by_method (no manip.jsonl)")
  fig, ax = plt.subplots(figsize=(7.5, 4))
  w = 0.38
  for j, lv in enumerate(levels):
    for i, m in enumerate(ms):
      rs = [r for r in man if r["method"] == m and r["level"] == lv]
      p, lo, hi = wilson(sum(r["success"] for r in rs), len(rs))
      x = i + (j - (len(levels) - 1) / 2) * w
      ax.bar(x, p, w - 0.03, color=METHOD_COLOR[m], alpha=1.0 if lv else 0.45, hatch=None if lv else "///",
             edgecolor=SURF, linewidth=0, label=None)
      ax.errorbar(x, p, [[p - lo], [hi - p]], color=INK2, capsize=3, lw=1)
      ax.text(x, hi + 0.02, f"{p:.0%}\nn={len(rs)}", ha="center", va="bottom", fontsize=7.5, color=INK2)
  ax.set_xticks(range(len(ms)), [METHOD_NAME[m] for m in ms])
  pct_axis(ax, (0, 1.25))
  ax.set_ylabel("task success (ground truth)")
  from matplotlib.patches import Patch
  ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.45, hatch="///", label=LEVEL_NAME[0.0]),
                     Patch(facecolor=MUTED, label=LEVEL_NAME[1.0])][: len(levels)], loc="upper right", ncols=2)
  ax.set_title("Pick-and-place success: VLA vs RL-based methods (95% Wilson CI)")
  save(fig, "01_success_by_method", "Bars: success rate per method; hatched = exact sim, solid = realistic noise.")


# --- 2. success by condition heatmap ---------------------------------------------------------
def fig_condition_heatmap(man, level=1.0):
  rows = [r for r in man if r["level"] == level] or [r for r in man if r["level"] == 0.0]
  ms = methods_in(rows)
  if not ms:
    return skipped.append("02_success_by_condition")
  conds = [("box", "cube"), ("bar", "bar"), ("cylinder", "cylinder"), ("sphere", "ball")]
  cols = [c[1] for c in conds] + ["on table", "on platform"]
  M = np.full((len(ms), len(cols)), np.nan)
  N = np.zeros_like(M, dtype=int)
  for i, m in enumerate(ms):
    rm = [r for r in rows if r["method"] == m]
    groups = [[r for r in rm if r["shape"] == s] for s, _ in conds] + [[r for r in rm if not r["stacked"]],
                                                                        [r for r in rm if r["stacked"]]]
    for j, g in enumerate(groups):
      if g:
        M[i, j] = np.mean([r["success"] for r in g])
        N[i, j] = len(g)
  fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ms) + 1.6))
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
  im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
  ax.grid(False)
  for i in range(len(ms)):
    for j in range(len(cols)):
      if not np.isnan(M[i, j]):
        ax.text(j, i, f"{M[i, j]:.0%}\n({N[i, j]})", ha="center", va="center", fontsize=8,
                color="white" if M[i, j] > 0.55 else INK)
  ax.set_xticks(range(len(cols)), cols)
  ax.set_yticks(range(len(ms)), [METHOD_NAME[m].replace("\n", " ") for m in ms])
  ax.axvline(len(conds) - 0.5, color=SURF, lw=4)
  cb = fig.colorbar(im, ax=ax, fraction=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
  cb.outline.set_visible(False)
  lvl = rows[0]["level"]
  ax.set_title(f"Success by object shape and support ({LEVEL_NAME.get(lvl, lvl)}; episodes in brackets)")
  save(fig, "02_success_by_condition", "Where each method fails: shape and stacked/table breakdown.")


# --- 3. execution time ---------------------------------------------------------------------
def fig_time(man, level=1.0):
  rows = [r for r in man if r["level"] == level and r["success"]] or [r for r in man if r["success"]]
  ms = methods_in(rows)
  if not ms:
    return skipped.append("03_execution_time")
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.2, 1]})
  data = [[r["exec_s"] for r in rows if r["method"] == m] for m in ms]
  bp = a1.boxplot(data, widths=0.5, patch_artist=True, showfliers=True, medianprops={"color": INK, "lw": 1.5},
                  flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": "none"})
  for b, m in zip(bp["boxes"], ms):
    b.set(facecolor=METHOD_COLOR[m], alpha=0.85, edgecolor=SURF)
  a1.set_xticks(range(1, len(ms) + 1), [METHOD_NAME[m] for m in ms], fontsize=8)
  a1.set_yscale("log")
  a1.set_ylabel("time to complete task [s] (log)")
  a1.set_title("Execution time, successful episodes")
  sim = [np.mean([r["sim_s"] for r in rows if r["method"] == m]) for m in ms]
  llm = [np.mean([r["llm_s"] for r in rows if r["method"] == m]) for m in ms]
  y = np.arange(len(ms))
  a2.barh(y, sim, color=SLOTS[0], height=0.55, label="arm motion (sim time)")
  a2.barh(y, llm, left=sim, color=SLOTS[1], height=0.55, label="VLA thinking (arm waits)")
  a2.set_yticks(y, [METHOD_NAME[m].replace("\n", " ") for m in ms], fontsize=8)
  a2.invert_yaxis()
  a2.set_xlabel("mean seconds per episode")
  a2.set_title("Where the time goes")
  a2.legend(loc="lower right", fontsize=8)
  a2.grid(axis="y", visible=False)
  save(fig, "03_execution_time", "Left: distribution; right: motion vs model-thinking split.")


# --- 4. latency ----------------------------------------------------------------------------
def fig_latency():
  p = R / "latency.json"
  if not p.exists():
    return skipped.append("04_decision_latency (no latency.json)")
  d = {k: np.array(v) for k, v in json.loads(p.read_text()).items() if not k.endswith("_tokens") and v}
  man = load("manip.jsonl")
  calls = [1000 * float(t["s"]) for r in man for t in r.get("llm_trace", []) if t.get("s")]
  if calls:
    d["vla (in episodes)"] = np.array(calls)
  names = {"ppo_onnx": "PPO policy (ONNX, CPU)", "bc_mlp": "BC grasp MLP", "ik_solve": "IK pose solve"}
  keys = sorted(d, key=lambda k: np.median(d[k]))
  fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(keys) + 1.5))
  for i, k in enumerate(keys):
    v = d[k]
    med, lo, hi = np.median(v), np.percentile(v, 5), np.percentile(v, 95)
    c = SLOTS[0] if not k.startswith("vla") else SLOTS[3]
    ax.barh(i, med, color=c, height=0.55)
    ax.errorbar(med, i, xerr=[[med - lo], [hi - med]], color=INK2, capsize=3, lw=1)
    hz = 1000 / med
    ax.text(hi * 1.15, i, f"{med:.3g} ms  (≈{hz:.3g} Hz)" if hz >= 1 else f"{med / 1000:.3g} s  ({1 / hz:.2g} s/decision)",
            va="center", fontsize=8, color=INK2)
  ax.set_yticks(range(len(keys)), [names.get(k, k.replace("vla_", "VLA turn: ")) for k in keys], fontsize=8)
  ax.set_xscale("log")
  ax.set_xlabel("decision latency [ms], log (bar = median, whisker = 5-95 %)")
  ax.axvline(20, color=MUTED, lw=1, ls="--")
  ax.text(20, len(keys) - 0.4, " 50 Hz control tick", color=MUTED, fontsize=8)
  ax.set_xlim(right=max(np.percentile(v, 95) for v in d.values()) * 30)
  ax.grid(axis="y", visible=False)
  ax.set_title("Decision latency: RL/learned policies vs the VLA")
  save(fig, "04_decision_latency", "Median per-decision latency; VLA rows are local Qwen2.5-VL on an RTX 3060 laptop.")


# --- 5. language generalization -----------------------------------------------------------
def fig_language():
  rows = load("language.jsonl")
  if not rows:
    return skipped.append("05_language_generalization")
  fams = [f for f in ("direct", "paraphrase", "synonym", "spatial", "multi", "stack", "negation") if any(r["family"] == f for r in rows)]
  pls = sorted({r["planner"] for r in rows}, key=lambda p: (p != "keyword", p))
  fig, ax = plt.subplots(figsize=(8.5, 4))
  w = 0.8 / len(pls)
  colors = {pl: (MUTED if pl == "keyword" else SLOTS[3 + i]) for i, pl in enumerate([p for p in pls if p != "keyword"])}
  colors["keyword"] = SLOTS[0]
  for j, pl in enumerate(pls):
    for i, f in enumerate(fams):
      rs = [r for r in rows if r["planner"] == pl and r["family"] == f]
      p, lo, hi = wilson(sum(r["correct"] for r in rs), len(rs))
      x = i + (j - (len(pls) - 1) / 2) * w
      ax.bar(x, p, w - 0.03, color=colors[pl], label=(pl.replace("vla_", "VLA planner ") if pl != "keyword" else "rule-based parser") if i == 0 else None)
      ax.errorbar(x, p, [[p - lo], [hi - p]], color=INK2, capsize=2, lw=0.8)
  ax.set_xticks(range(len(fams)), fams)
  pct_axis(ax, (0, 1.12))
  ax.set_ylabel("correct plan")
  ax.legend(loc="upper center", ncols=len(pls), bbox_to_anchor=(0.5, -0.12))
  ax.set_title("Instruction generalization: plan accuracy by instruction type")
  save(fig, "05_language_generalization", "VLA planner vs keyword parser on 7 instruction families.")


# --- 6. noise sweep ---------------------------------------------------------------------------
def fig_noise(man):
  ms = methods_in(man)
  if not ms or len({r["level"] for r in man}) < 2:
    return skipped.append("06_robustness_noise")
  fig, ax = plt.subplots(figsize=(7, 4))
  for m in ms:
    lvls = sorted({r["level"] for r in man if r["method"] == m})
    st = [wilson(sum(r["success"] for r in man if r["method"] == m and r["level"] == lv),
                 sum(1 for r in man if r["method"] == m and r["level"] == lv)) for lv in lvls]
    p = np.array([s[0] for s in st])
    ax.fill_between(lvls, [s[1] for s in st], [s[2] for s in st], color=METHOD_COLOR[m], alpha=0.12, lw=0)
    ax.plot(lvls, p, "-o", color=METHOD_COLOR[m], ms=6, mec=SURF, mew=1.5, label=METHOD_NAME[m].replace("\n", " "))
    ax.text(lvls[-1] + 0.05, p[-1], METHOD_NAME[m].split("\n")[0], color=INK2, fontsize=8, va="center")
  ax.set_xlabel("reality-gap noise level (0 = exact sim, 1 = realistic, 2 = pessimistic)")
  ax.set_ylabel("task success")
  pct_axis(ax)
  ax.set_xlim(-0.1, max(r["level"] for r in man) + 0.6)
  ax.legend(loc="lower left", fontsize=8)
  ax.set_title("Robustness to sensing / actuation / physics noise (95% CI band)")
  save(fig, "06_robustness_noise", "Servo noise+bias, object physics, camera noise+outliers+occlusion scaled together.")


# --- 7. PPO curves ----------------------------------------------------------------------------
def fig_ppo():
  p = R / "ppo_curves.csv"
  if not p.exists():
    return skipped.append("07_ppo_training (no ppo_curves.csv)")
  rows = list(csv.DictReader(open(p)))
  panels = [("reward", "mean reward"), ("success", "curriculum success rate"), ("err_m", "TCP position error [mm]"),
            ("stage", "curriculum stage")]
  fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
  seeds = sorted({r["seed"] for r in rows})
  for ax, (k, lab) in zip(axes, panels):
    for i, s in enumerate(seeds):
      rs = [r for r in rows if r["seed"] == s and r.get(k)]
      x = np.array([float(r["env_steps"]) for r in rs]) / 1e6
      y = np.array([float(r[k]) for r in rs]) * (1000 if k == "err_m" else 1)
      if len(y) > 25:
        y = np.convolve(y, np.ones(25) / 25, mode="same")
      ax.plot(x, y, color=SLOTS[i], lw=1.6, label=f"seed {s}")
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("environment steps [M]")
  if len(seeds) > 1:
    axes[0].legend(fontsize=8)
  last = [r for r in rows if r.get("wall_s")]
  if last:
    fig.suptitle(f"PPO reach training ({float(last[-1]['wall_s']) / 3600:.1f} GPU-h for "
                 f"{float(last[-1]['env_steps']) / 1e6:.0f} M steps on an RTX 3060 laptop)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
  save(fig, "07_ppo_training", "Training curves (25-iteration moving average).")


# --- 8-10, 13. reach benchmark ------------------------------------------------------------------
def reach_runs():
  out = {}
  for d in sorted((ROOT / "results").glob("report_reach_*")):
    tag = d.name.replace("report_reach_", "")
    for c in d.iterdir():
      if (c / "summary.json").exists():
        name = "RL (PPO)" if c.name.startswith("native") else {"dls_ik": "DLS IK", "ik_joint_interp": "IK + joint interp."}.get(c.name, c.name)
        ep = np.load(c / "episodes.npz") if (c / "episodes.npz").exists() else None
        out[(tag, name)] = (json.loads((c / "summary.json").read_text()), ep)
  return out


CTRL_COLOR = {"RL (PPO)": SLOTS[0], "DLS IK": SLOTS[1], "IK + joint interp.": SLOTS[2], "RL (PPO, real-arm fine-tune)": SLOTS[6]}


def fig_reach_cdf(runs):
  sel = [(n, v) for (t, n), v in runs.items() if t == "nominal" and v[1] is not None]
  rp = runs.get(("real_policy", "RL (PPO)"))
  if rp and rp[1] is not None:
    sel.append(("RL (PPO, real-arm fine-tune)", rp))
  if not sel:
    return skipped.append("08_reach_error_cdf")
  fig, ax = plt.subplots(figsize=(7, 4))
  for n, (s, ep) in sel:
    e = np.sort(ep["settled_err"])
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=CTRL_COLOR.get(n, MUTED), label=f"{n} (median {np.median(e):.1f} mm)")
  for t in (5, 10, 20):
    ax.axvline(t, color=MUTED, lw=0.8, ls=":")
    ax.text(t, 0.02, f" {t} mm", color=MUTED, fontsize=8)
  ax.set_xscale("log")
  ax.set_xlabel("settled TCP error [mm] (log)")
  ax.set_ylabel("fraction of episodes ≤ error")
  pct_axis(ax)
  ax.legend(loc="upper left", fontsize=8)
  ax.set_title("Reaching accuracy: cumulative error distribution")
  save(fig, "08_reach_error_cdf", "All suite episodes (targets × 4 start configurations).")


def fig_workspace(runs):
  suite = ROOT / "data" / "eval" / "suite_v2.npz"
  sel = [(n, v) for (t, n), v in runs.items() if t == "nominal" and v[1] is not None and n in ("RL (PPO)", "DLS IK")]
  if not sel or not suite.exists():
    return skipped.append("09_workspace_error_map")
  tg = np.load(suite)["targets"]
  fig, axes = plt.subplots(1, len(sel), figsize=(5 * len(sel), 4.2), squeeze=False)
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
  for ax, (n, (s, ep)) in zip(axes[0], sel):
    e = ep["settled_err"][: len(tg)]
    k = len(e) // 4
    em = e.reshape(4, k).mean(0) if len(e) % 4 == 0 else e  # average over the 4 start configurations
    t = tg[:k] if len(e) % 4 == 0 else tg
    sc = ax.scatter(t[:, 1] * 100, t[:, 0] * 100, c=np.clip(em, 0, 20), cmap=cmap, s=14, vmin=0, vmax=20, edgecolors="none")
    ax.set_xlabel("y [cm] (+ = robot's left)")
    ax.set_ylabel("x [cm] (forward)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{n}: mean error per target", fontsize=10)
  cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.03)
  cb.set_label("settled error [mm] (clipped at 20)")
  cb.outline.set_visible(False)
  save(fig, "09_workspace_error_map", "Top view of the reachable workspace, colour = error averaged over start poses.")


def fig_smooth(runs):
  sel = [(n, v[0]) for (t, n), v in runs.items() if t == "nominal"]
  if not sel:
    return skipped.append("10_smoothness_safety")
  keys = [("mean_rms_tcp_jerk", "RMS TCP jerk [m/s³]"), ("mean_rms_joint_acc", "RMS joint accel. [rad/s²]"),
          ("mean_overshoot", "overshoot past 10 mm [mm]"), ("mean_path_ratio", "path length / straight line"),
          ("collision_rate", "collision rate"), ("mean_settle_time@10mm_s", "settle time @10 mm [s]")]
  fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
  for ax, (k, lab) in zip(axes.flat, keys):
    vals = [s["overall"].get(k, np.nan) for _, s in sel]
    ax.bar(range(len(sel)), vals, color=[CTRL_COLOR.get(n, MUTED) for n, _ in sel], width=0.6)
    for i, v in enumerate(vals):
      ax.text(i, v, f"{v:.1%}" if "rate" in k else f"{v:.3g}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(sel)), [n for n, _ in sel], fontsize=8)
    ax.set_title(lab, fontsize=10)
    ax.grid(axis="x", visible=False)
  fig.suptitle("Motion quality and safety (lower is better)", x=0.01, ha="left", fontweight="bold")
  save(fig, "10_smoothness_safety", "Nominal reach benchmark, all episodes.")


def fig_reach_robust(runs):
  def series(prefix, vals, nominal=None):
    out = defaultdict(list)
    for v in vals:
      tag = f"{prefix}{v:g}" if not (prefix == "delay" and v == 0) and not (prefix != "delay" and v == 1.0) else "nominal"
      for n in ("RL (PPO)", "DLS IK"):
        if (tag, n) in runs:
          out[n].append((v, runs[(tag, n)][0]["overall"]["success@10mm"]))
    return out
  panels = [("delay", [0, 1, 2, 3], "action delay [20 ms steps]"), ("kp", [0.6, 0.8, 1.0, 1.2, 1.4], "servo stiffness scale kp"),
            ("kv", [0.6, 1.0, 1.4], "servo damping scale kv")]
  data = [(series(p, v), lab) for p, v, lab in panels]
  if not any(d for d, _ in data):
    return skipped.append("13_reach_robustness")
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
  for ax, (d, lab) in zip(axes, data):
    for n, pts in d.items():
      if len(pts) > 1:
        x, y = zip(*sorted(pts))
        ax.plot(x, y, "-o", color=CTRL_COLOR[n], ms=6, mec=SURF, mew=1.5, label=n)
    ax.set_xlabel(lab)
  ident = [(n, runs[(t, n)][0]["overall"]["success@10mm"]) for t in ("identified",) for n in ("RL (PPO)", "DLS IK") if (t, n) in runs]
  axes[0].set_ylabel("success @10 mm")
  pct_axis(axes[0])
  axes[0].legend(fontsize=8, loc="lower left")
  if ident:
    axes[2].text(0.02, 0.05, "identified real-servo physics:\n" + "\n".join(f"{n}: {v:.0%}" for n, v in ident),
                 transform=axes[2].transAxes, fontsize=8, color=INK2)
  fig.suptitle("Reach robustness to actuation mismatch: RL vs IK", x=0.01, ha="left", fontweight="bold")
  save(fig, "13_reach_robustness", "Same suite, perturbed physics.")


def fig_reach_servo(runs):
  """Sim-to-real: identified physics with friction / dead band mismatch, per policy variant."""
  variants = [("RL, sim policy", "", "RL (PPO)", SLOTS[0]), ("RL, identified + servo DR", "_realpolicy", "RL (PPO)", SLOTS[6]),
              ("RL, identified, no servo DR", "_noservodr", "RL (PPO)", SLOTS[3]), ("DLS IK", "", "DLS IK", SLOTS[1])]
  base = {"": "identified", "_realpolicy": "identified_realpolicy", "_noservodr": "identified_noservodr"}

  def series(prefix, vals, nominal_val, suffix, ctrl, key):
    pts = []
    for v in vals:
      tag = base[suffix] if v == nominal_val else f"{prefix}{v:g}{suffix}"
      if (tag, ctrl) in runs:
        pts.append((v, runs[(tag, ctrl)][0]["overall"][key]))
    return pts

  panels = [("fric", [0.5, 1.0, 1.5, 2.0, 3.0], 1.0, "joint friction scale (x identified)"),
            ("db", [0, 1, 2, 4, 8], 0, "servo dead band [ticks, 0.088 deg]")]
  if not any(series(pf, v, nv, sfx, c, "success@10mm") for pf, v, nv, _ in panels for _, sfx, c, _ in variants):
    return skipped.append("15_reach_servo_mismatch")
  fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
  for ax, (pf, vals, nv, lab) in zip(axes[:2], panels):
    for name, sfx, ctrl, col in variants:
      pts = series(pf, vals, nv, sfx, ctrl, "mean_err_mm")
      if len(pts) > 1:
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=col, ms=6, mec=SURF, mew=1.5, label=name)
    ax.set_xlabel(lab)
  axes[0].set_ylabel("mean settled error [mm]")
  axes[0].legend(fontsize=8, loc="upper left")
  ax = axes[2]
  rows = []
  for name, sfx, c, col in variants:
    a = runs.get((base[sfx], c))
    b = runs.get(("realistic" + sfx, c))
    if a or b:
      rows.append((name, a[0]["overall"]["success@10mm"] if a else np.nan, b[0]["overall"]["success@10mm"] if b else np.nan, col))
  if rows:
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[1] for r in rows], 0.38, color=[r[3] for r in rows], alpha=0.45, label="identified")
    ax.barh(y - 0.2, [r[2] for r in rows], 0.38, color=[r[3] for r in rows], label="realistic (fric x1.5, db 2, delay 1)")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("success @10 mm")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
  fig.suptitle("Reach under servo mismatch on identified physics: identification vs randomization", x=0.01, ha="left",
               fontweight="bold")
  save(fig, "15_reach_servo_mismatch", "Friction / dead band swept around the fitted servo model (scripts/report/t03).")


# --- 11. data efficiency ------------------------------------------------------------------------
def fig_bc():
  rows = load("bc_efficiency.jsonl")
  if not rows:
    return skipped.append("11_bc_data_efficiency")
  fig, ax = plt.subplots(figsize=(7, 4))
  for j, lv in enumerate(sorted({r["level"] for r in rows})):
    rs = [r for r in rows if r["level"] == lv and not r["aborted"]]
    ns = sorted({r["n_demos"] for r in rs if r["n_demos"] > 0})
    st = [wilson(sum(r["success"] for r in rs if r["n_demos"] == n), sum(1 for r in rs if r["n_demos"] == n)) for n in ns]
    c = SLOTS[j]
    ax.fill_between(ns, [s[1] for s in st], [s[2] for s in st], color=c, alpha=0.12, lw=0)
    ax.plot(ns, [s[0] for s in st], "-o", color=c, ms=6, mec=SURF, mew=1.5, label=f"BC, {LEVEL_NAME.get(lv, lv)}")
    h = [r["success"] for r in rs if r["n_demos"] == 0]
    if h:
      ax.axhline(np.mean(h), color=c, ls="--", lw=1.2, label=f"scripted heuristic, {LEVEL_NAME.get(lv, lv)}")
  ax.set_xscale("log")
  ax.set_xlabel("verified demonstrations used for training (log)")
  ax.set_ylabel("first-try grasp success (held-out scenes)")
  pct_axis(ax)
  ax.legend(fontsize=8, loc="lower right")
  st = R / "bc_data_stats.json"
  if st.exists():
    s = json.loads(st.read_text())
    ax.text(0.01, 0.97, f"{s['successes']} verified grasps from {s['episodes']} self-supervised sim episodes, 0 LLM tokens",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
  ax.set_title("Data efficiency of the learned grasp policy")
  save(fig, "11_bc_data_efficiency", "BC trained on growing subsets; dashed = scripted baseline.")


# --- 12. failure modes --------------------------------------------------------------------------
def fig_failures(man, level=1.0):
  rows = [r for r in man if r["level"] == level] or man
  ms = methods_in(rows)
  if not ms:
    return skipped.append("12_failure_modes")
  cats = [c for c, _ in Counter(r["category"] for r in rows if r["category"] != "success").most_common()]
  if len(cats) > 6:
    cats = cats[:5] + ["other"]
  fig, ax = plt.subplots(figsize=(8, 0.55 * len(ms) + 1.8))
  left = np.zeros(len(ms))
  palette = [SLOTS[i] for i in (0, 1, 2, 3, 6, 7)]
  for c, col in zip(["success"] + cats, ["#c9c8c3"] + palette):
    v = []
    for m in ms:
      rm = [r for r in rows if r["method"] == m]
      cat = lambda r: r["category"] if r["category"] in cats or r["category"] == "success" else "other"  # noqa: E731
      v.append(sum(cat(r) == c for r in rm) / max(1, len(rm)))
    ax.barh(range(len(ms)), v, left=left, color=col, height=0.6, edgecolor=SURF, linewidth=2, label=c)
    for i, x in enumerate(v):
      if x >= 0.07:
        ax.text(left[i] + x / 2, i, f"{x:.0%}", ha="center", va="center", fontsize=7.5, color=INK)
    left += v
  ax.set_yticks(range(len(ms)), [METHOD_NAME[m].replace("\n", " ") for m in ms], fontsize=8)
  ax.invert_yaxis()
  ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
  ax.set_xlim(0, 1)
  ax.grid(axis="y", visible=False)
  ax.legend(ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
  lvl = rows[0]["level"]
  ax.set_title(f"Outcome breakdown per method ({LEVEL_NAME.get(lvl, lvl)})")
  save(fig, "12_failure_modes", "Share of episodes per outcome category.")


# --- 14. reach sim2real (agent stack) ---------------------------------------------------------
def fig_reach_s2r():
  rows = load("reach_sim2real.jsonl")
  if not rows:
    return skipped.append("14_reach_sim2real")
  ctrls = [c for c in ("rl", "ik") if any(r["controller"] == c for r in rows)]
  lvls = sorted({r["level"] for r in rows})
  fig, ax = plt.subplots(figsize=(7, 4))
  w = 0.8 / len(ctrls)
  name = {"rl": "RL travel (PPO)", "ik": "IK straight-line travel"}
  for j, c in enumerate(ctrls):
    data = [[r["err_true_mm"] for r in rows if r["controller"] == c and r["level"] == lv] for lv in lvls]
    pos = [i + (j - (len(ctrls) - 1) / 2) * w for i in range(len(lvls))]
    bp = ax.boxplot(data, positions=pos, widths=w * 0.8, patch_artist=True, medianprops={"color": INK},
                    flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": "none"})
    for b in bp["boxes"]:
      b.set(facecolor=SLOTS[j], edgecolor=SURF)
    ax.plot([], [], "s", color=SLOTS[j], label=name[c])
  ax.set_xticks(range(len(lvls)), [LEVEL_NAME.get(lv, lv) for lv in lvls])
  ax.set_yscale("log")
  ax.set_ylabel("true grip-point error [mm] (log)")
  ax.legend(fontsize=8)
  ax.set_title("Reach error on the real-arm model: sim vs noisy servos")
  save(fig, "14_reach_sim2real", "Ground-truth error; encoder noise and calibration bias are invisible to the controller.")


# --- 15. VLA cost / handoff ---------------------------------------------------------------------
def fig_cost(man):
  rows = [r for r in man if r["method"] in ("vla", "vla_free", "hybrid")]
  if not rows:
    return skipped.append("15_vla_usage")
  ms = methods_in(rows)
  lvls = sorted({r["level"] for r in rows})
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
  w = 0.8 / len(lvls)
  for j, lv in enumerate(lvls):
    for i, m in enumerate(ms):
      rs = [r for r in rows if r["method"] == m and r["level"] == lv]
      x = i + (j - (len(lvls) - 1) / 2) * w
      for ax, k in ((a1, "llm_calls"), (a2, None)):
        v = np.mean([r[k] for r in rs]) if k else np.mean([r["tokens_in"] + r["tokens_out"] for r in rs]) / 1000
        ax.bar(x, v, w - 0.04, color=METHOD_COLOR[m], alpha=0.45 if lv == 0 else 1.0, hatch="///" if lv == 0 else None)
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
  for ax, lab in ((a1, "VLA calls per episode"), (a2, "tokens per episode [k]")):
    ax.set_xticks(range(len(ms)), [METHOD_NAME[m] for m in ms], fontsize=8)
    ax.set_title(lab, fontsize=10)
    ax.grid(axis="x", visible=False)
  fig.suptitle("VLA usage: hybrid hand-off only calls the model when the local policy fails (hatched = exact sim)",
               x=0.01, ha="left", fontweight="bold", fontsize=10)
  save(fig, "15_vla_usage", "Model calls and tokens per episode.")


# --- 16. retry benefit -------------------------------------------------------------------------
def fig_retry(man, level=1.0):
  rows = [r for r in man if r["level"] == level and r.get("grasp_attempts")]
  ms = [m for m in methods_in(rows) if m != "vla_free"]
  if not ms:
    return skipped.append("16_grasp_attempts")
  fig, ax = plt.subplots(figsize=(6.5, 3.8))
  for m in ms:
    rm = [r for r in rows if r["method"] == m]
    ys = [np.mean([bool(r["grasped"]) and r["grasp_attempts"] <= k for r in rm]) for k in (1, 2, 3)]
    ax.plot([1, 2, 3], ys, "-o", color=METHOD_COLOR[m], ms=7, mec=SURF, mew=1.5, label=METHOD_NAME[m].replace("\n", " "))
  ax.set_xticks([1, 2, 3])
  ax.set_xlabel("grasp attempts allowed")
  ax.set_ylabel("object grasped")
  pct_axis(ax)
  ax.legend(fontsize=8, loc="lower right")
  ax.set_title("Benefit of retries (cumulative grasp success)")
  save(fig, "16_grasp_attempts", "Grasp success within k attempts.")


# --- 19. perception tracker under camera noise (scripts/eval_tracker.py) ----------------------
TRACK_COND = [("noisy", "raw noisy detections"), ("noisy+hold3", "3-frame hold (current)"),
              ("noisy+kf", "Kalman tracker (new)")]


def fig_tracker(path=ROOT / "results" / "tracker" / "tracker30.json"):
  if not path.exists():
    return skipped.append("19_tracker")
  d = json.loads(path.read_text())
  fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.25, 1]})
  # (a) CDF of the xy position error over every measurement that returned an estimate
  for i, (c, name) in enumerate(TRACK_COND):
    errs = np.sort([r["err_xy_mm"] for e in d["episodes"].get(c, []) for r in e["recs"]
                    if r.get("available") and r.get("err_xy_mm") is not None])
    if not len(errs):
      continue
    y = np.arange(1, len(errs) + 1) / len(errs)
    ax.plot(errs, y, color=SLOTS[i], label=f"{name}  (median {np.median(errs):.1f} mm)")
  ax.axvline(8, color=MUTED, ls=":", lw=1.2)
  ax.text(8.5, 0.42, "8 mm: tracker\n'confident'", fontsize=8, color=INK2)
  ax.set_xlim(0, 30)
  ax.set_xlabel("object position error, xy [mm]")
  ax.set_ylabel("share of measurements with error ≤ x")
  pct_axis(ax)
  ax.legend(fontsize=8, loc="lower right")
  ax.set_title("Position error under camera noise")
  # (b) availability of an estimate: overall vs while the arm hides the object
  s = d["summary"]
  groups = [("available", "all measurements"), ("available_when_occluded", "arm blocks the camera")]
  x = np.arange(len(groups))
  bw = 0.26
  for i, (c, name) in enumerate(TRACK_COND):
    vals = [s[c][k] for k, _ in groups]
    bars = bx.bar(x + (i - 1) * (bw + 0.02), vals, bw, color=SLOTS[i], label=name)
    for b, v in zip(bars, vals):
      bx.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
  bx.set_xticks(x, [g for _, g in groups])
  bx.grid(axis="x", visible=False)
  pct_axis(bx, (0, 1.12))
  bx.set_ylabel("object position estimate available")
  bx.set_title("Can the robot still locate the object?")
  n = s["noisy+kf"]["episodes"]
  fig.text(0.01, -0.02, f"{n} held-out sim scenes; camera noise σ = {1000 * d['noise']['sigma_xy']:.0f} mm, "
           f"{d['noise']['p_outlier']:.0%} outliers, {d['noise']['p_miss']:.0%} misses, arm occlusion.",
           fontsize=8, color=INK2)
  save(fig, "19_tracker", "Kalman tracker vs raw / 3-frame-hold perception: error CDF and availability under occlusion.")


# --- 20. learned (BC) vs scripted grasp on held-out scenes (scripts/eval_grasp_policy.py) -------
def fig_bc_heldout(path=ROOT / "results" / "bc_grasp" / "heuristic_vs_bc_40.json"):
  if not path.exists():
    return skipped.append("20_bc_vs_scripted")
  d = json.loads(path.read_text())
  conds = [("heuristic", "scripted grasp", METHOD_COLOR["rl_scripted"]), ("bc", "learned grasp (BC)", METHOD_COLOR["rl_bc"])]
  cats = [("all", None, "all objects"), ("by_shape", "box", "cubes / boxes"), ("by_shape", "bar", "bars"),
          ("by_shape", "sphere", "balls"), ("by_shape", "cylinder", "cylinders"), ("by_stacked", "stacked", "stacked")]
  fig, ax = plt.subplots(figsize=(9, 4))
  x = np.arange(len(cats))
  bw = 0.36
  for i, (c, name, col) in enumerate(conds):
    s = d["summary"][c]
    ks = [(s[a] if b is None else s[a][b]) for a, b, _ in cats]
    st = [wilson(k["ok"], k["n"]) for k in ks]
    pos = x + (i - 0.5) * (bw + 0.03)
    ax.bar(pos, [v[0] for v in st], bw, color=col, label=name)
    ax.errorbar(pos, [v[0] for v in st], yerr=[[v[0] - v[1] for v in st], [v[2] - v[0] for v in st]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for p_, k in zip(pos, ks):
      ax.text(p_, 0.02, f"{k['ok']}/{k['n']}", ha="center", fontsize=7.5, color=SURF, fontweight="bold")
  ax.set_xticks(x, [c for _, _, c in cats])
  ax.grid(axis="x", visible=False)
  pct_axis(ax, (0, 1.12))
  ax.set_ylabel("first-try grasp success")
  ax.legend(fontsize=8, loc="upper left", ncol=2)
  m = d.get("bc_meta", {})
  fig.text(0.01, -0.02, f"BC trained on {m.get('n_train', '?')} self-supervised sim grasps (0 LLM tokens); "
           f"{len(d['episodes']['bc'])} held-out scenes; error bars = 95% Wilson CI; numbers = successes / trials.",
           fontsize=8, color=INK2)
  ax.set_title("Learned vs scripted grasp on unseen scenes")
  save(fig, "20_bc_vs_scripted", "First-try success by object type, BC vs scripted heuristic (held-out seeds).")


def summary(man):
  lines = ["# Report results summary\n", "## Pick-and-place (ground-truth success)\n",
           "| method | level | n | success | 95% CI | median exec s (succ.) | VLA calls/ep | tokens/ep |", "|---|---|---|---|---|---|---|---|"]
  for m in methods_in(man):
    for lv in sorted({r["level"] for r in man if r["method"] == m}):
      rs = [r for r in man if r["method"] == m and r["level"] == lv]
      p, lo, hi = wilson(sum(r["success"] for r in rs), len(rs))
      ex = [r["exec_s"] for r in rs if r["success"]]
      lines.append(f"| {METHOD_NAME[m].replace(chr(10), ' ')} | {lv:g} | {len(rs)} | {p:.1%} | {lo:.0%}-{hi:.0%} | "
                   f"{np.median(ex) if ex else float('nan'):.1f} | {np.mean([r['llm_calls'] for r in rs]):.1f} | "
                   f"{np.mean([r['tokens_in'] + r['tokens_out'] for r in rs]):.0f} |")
  lines += ["\n## Figures\n"] + [f"- `{n}` - {d}" for n, d in made]
  if skipped:
    lines += ["\n## Not yet available\n"] + [f"- {s}" for s in skipped]
  (R / "summary.md").write_text("\n".join(lines) + "\n")


def main():
  man = load("manip.jsonl")
  fig_success(man)
  fig_condition_heatmap(man)
  fig_time(man)
  fig_latency()
  fig_language()
  fig_noise(man)
  fig_ppo()
  runs = reach_runs()
  fig_reach_cdf(runs)
  fig_workspace(runs)
  fig_smooth(runs)
  fig_bc()
  fig_failures(man)
  fig_reach_robust(runs)
  fig_reach_servo(runs)
  fig_reach_s2r()
  fig_cost(man)
  fig_retry(man)
  fig_tracker()
  fig_bc_heldout()
  summary(man)
  print(f"[plot] {len(made)} figures -> {FIG}")
  for s in skipped:
    print(f"[plot] skipped: {s}")


if __name__ == "__main__":
  main()
