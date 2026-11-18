"""Train the behavior-cloned grasp model from verified grasp episodes.

Reads the JSONL written by scripts/gen_grasp_data.py, keeps only successful rows (the
parameters there were distilled from a grasp that held through a 5 cm lift), and fits a
small MLP: features (8) -> [dx_mm, dy_mm, h_rel_top_mm, sin(2*jaw), cos(2*jaw), 3 force logits].

Pure numpy (manual backprop + Adam): the model has ~1.5k weights and trains in seconds on a
CPU, so there is no torch dependency at training or inference time. The regression-target
standardization is folded into the last layer before saving, so the runtime forward pass in
src/so101_reach/agent/bc_grasp.py reads the JSON as-is.

The train/validation split is by scene seed (seed % val_mod == 0 -> validation), so two
attempts on the same scene never land on both sides.

Examples:
  python scripts/train_bc_grasp.py                                   # data/il/grasp_episodes.jsonl
  python scripts/train_bc_grasp.py --data a.jsonl b.jsonl --epochs 3000 --hidden 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_reach.agent.bc_grasp import DEFAULT_MODEL, FEATURE_NAMES, FORCE_CLASSES, BcGrasp  # noqa: E402
from so101_reach.agent.local_skills import fold_jaw_rel, square_footprint  # noqa: E402

DATA_DEFAULT = ROOT / "data" / "il" / "grasp_episodes.jsonl"
N_REG = 5  # dx, dy, h, sin2a, cos2a
N_OUT = N_REG + len(FORCE_CLASSES)


def load_rows(paths: list[Path]) -> list[dict]:
  rows = []
  for p in paths:
    with open(p) as f:
      for line in f:
        line = line.strip()
        if line:
          rows.append(json.loads(line))
  return rows


def to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Successful rows -> (X features, Y regression targets, force class ids, scene seeds)."""
  X, Y, F, S = [], [], [], []
  for r in rows:
    p = r.get("params") or {}
    if not r.get("success") or "features" not in r or "dxy_obj_mm" not in p:
      continue
    if len(r["features"]) != len(FEATURE_NAMES) or p.get("force") not in FORCE_CLASSES:
      continue  # written by an older feature/param schema
    # Square / round footprint: jaw 0 and 90 deg grip equally well, and which one the heuristic
    # used depends on the wrist's previous angle (not a feature). Fold to one representative so
    # the label is single-valued; replay picks 0 or 90 by the smaller wrist turn.
    l_mm, w_mm = r["features"][0], r["features"][1]
    a = np.radians(2.0 * fold_jaw_rel(float(p["jaw_rel_deg"]), l_mm, w_mm))
    X.append(r["features"])
    Y.append([p["dxy_obj_mm"][0], p["dxy_obj_mm"][1], p["h_rel_top_mm"], np.sin(a), np.cos(a)])
    F.append(FORCE_CLASSES.index(p["force"]))
    S.append(int(r.get("seed", len(S))))
  return (np.asarray(X, float).reshape(-1, len(FEATURE_NAMES)), np.asarray(Y, float).reshape(-1, N_REG),
          np.asarray(F, int), np.asarray(S, int))


def init_layers(sizes: list[int], rng) -> list[list[np.ndarray]]:
  """He-initialised (W, b) pairs, W shaped (in, out) to match `x @ W + b` in BcGrasp.forward."""
  return [[rng.normal(0.0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)] for a, b in zip(sizes[:-1], sizes[1:])]


def forward(layers, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
  acts = [x]
  for i, (W, b) in enumerate(layers):
    x = x @ W + b
    if i < len(layers) - 1:
      x = np.maximum(x, 0.0)
    acts.append(x)
  return x, acts


def softmax(z: np.ndarray) -> np.ndarray:
  z = z - z.max(axis=1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=1, keepdims=True)


def loss_and_grads(layers, x, y_reg, y_cls, w_cls: float, l2: float):
  """MSE on the (standardized) regression outputs + w_cls * cross-entropy on the force logits."""
  out, acts = forward(layers, x)
  n = len(x)
  d_reg = out[:, :N_REG] - y_reg
  prob = softmax(out[:, N_REG:])
  ce = -np.log(prob[np.arange(n), y_cls] + 1e-12).mean()
  loss = float((d_reg ** 2).mean() + w_cls * ce + l2 * sum((W ** 2).sum() for W, _ in layers))
  g = np.zeros_like(out)
  g[:, :N_REG] = 2.0 * d_reg / (n * N_REG)
  onehot = np.eye(N_OUT - N_REG)[y_cls]
  g[:, N_REG:] = w_cls * (prob - onehot) / n
  grads = [None] * len(layers)
  for i in range(len(layers) - 1, -1, -1):
    W, _ = layers[i]
    grads[i] = [acts[i].T @ g + 2.0 * l2 * W, g.sum(axis=0)]
    if i > 0:
      g = (g @ W.T) * (acts[i] > 0)  # ReLU derivative of the previous layer's output
  return loss, grads


def train(X, Y, F, *, hidden: int = 32, epochs: int = 2000, lr: float = 3e-3, w_cls: float = 0.5,
          l2: float = 1e-4, batch: int = 64, seed: int = 0, log_every: int = 0):
  """Fit the MLP; returns (layers in raw units, normalization dict)."""
  rng = np.random.default_rng(seed)
  x_mean, x_std = X.mean(axis=0), X.std(axis=0)
  x_std = np.where(x_std < 1e-6, 1.0, x_std)  # constant features (e.g. 'soft' never set) stay finite
  y_mean, y_std = Y.mean(axis=0), Y.std(axis=0)
  y_std = np.where(y_std < 1e-6, 1.0, y_std)
  y_mean[3:5], y_std[3:5] = 0.0, 1.0  # sin/cos are already unit scale; keep them unshifted
  xn, yn = (X - x_mean) / x_std, (Y - y_mean) / y_std
  layers = init_layers([X.shape[1], hidden, hidden, N_OUT], rng)
  m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
  v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
  b1, b2, step = 0.9, 0.999, 0
  for ep in range(epochs):
    order = rng.permutation(len(xn))
    for s in range(0, len(order), batch):
      idx = order[s:s + batch]
      loss, grads = loss_and_grads(layers, xn[idx], yn[idx], F[idx], w_cls, l2)
      step += 1
      for li in range(len(layers)):
        for k in range(2):
          m[li][k] = b1 * m[li][k] + (1 - b1) * grads[li][k]
          v[li][k] = b2 * v[li][k] + (1 - b2) * grads[li][k] ** 2
          mh, vh = m[li][k] / (1 - b1 ** step), v[li][k] / (1 - b2 ** step)
          layers[li][k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    if log_every and (ep + 1) % log_every == 0:
      full, _ = loss_and_grads(layers, xn, yn, F, w_cls, l2)
      print(f"[train] epoch {ep + 1}/{epochs} loss {full:.4f}")
  # Fold the target standardization into the last layer: out_raw = out_n * y_std + y_mean.
  W, b = layers[-1]
  scale = np.concatenate([y_std, np.ones(N_OUT - N_REG)])
  shift = np.concatenate([y_mean, np.zeros(N_OUT - N_REG)])
  layers[-1] = [W * scale, b * scale + shift]
  return layers, {"mean": x_mean.tolist(), "std": x_std.tolist()}


def to_spec(layers, norm: dict, meta: dict) -> dict:
  return {"norm": norm, "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers], "meta": meta}


def jaw_err_deg(pred, true, X) -> np.ndarray:
  """|pred - true| with the object's symmetry: period 90 for square / round footprints, else 180."""
  period = np.array([90.0 if square_footprint(x[0], x[1]) else 180.0 for x in X])
  return np.abs((pred - true + period / 2) % period - period / 2)


def metrics(model: BcGrasp, X, Y, F) -> dict:
  """Validation errors in physical units (the same decoding BcGrasp.predict uses)."""
  if len(X) == 0:
    return {}
  out = np.array([model.forward(x) for x in X])
  jaw_pred = np.degrees(np.arctan2(out[:, 3], out[:, 4])) / 2
  jaw_true = np.degrees(np.arctan2(Y[:, 3], Y[:, 4])) / 2
  jaw_err = jaw_err_deg(jaw_pred, jaw_true, X)
  return {"n": int(len(X)),
          "dxy_mae_mm": round(float(np.linalg.norm(out[:, :2] - Y[:, :2], axis=1).mean()), 2),
          "h_mae_mm": round(float(np.abs(out[:, 2] - Y[:, 2]).mean()), 2),
          "jaw_mae_deg": round(float(jaw_err.mean()), 2),
          "force_acc": round(float((out[:, N_REG:].argmax(axis=1) == F).mean()), 3)}


def mean_baseline(Ytr, Ftr, X, Y, F) -> dict:
  """Errors of a constant predictor (training mean / majority force): the bar BC must beat."""
  if len(Y) == 0:
    return {}
  mu = Ytr.mean(axis=0)
  jaw_mu = np.degrees(np.arctan2(mu[3], mu[4])) / 2
  jaw_true = np.degrees(np.arctan2(Y[:, 3], Y[:, 4])) / 2
  return {"dxy_mae_mm": round(float(np.linalg.norm(Y[:, :2] - mu[:2], axis=1).mean()), 2),
          "h_mae_mm": round(float(np.abs(Y[:, 2] - mu[2]).mean()), 2),
          "jaw_mae_deg": round(float(jaw_err_deg(jaw_mu, jaw_true, X).mean()), 2),
          "force_acc": round(float((F == np.bincount(Ftr, minlength=N_OUT - N_REG).argmax()).mean()), 3)}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--data", type=Path, nargs="+", default=[DATA_DEFAULT])
  p.add_argument("--out", type=Path, default=DEFAULT_MODEL)
  p.add_argument("--hidden", type=int, default=32)
  p.add_argument("--epochs", type=int, default=2000)
  p.add_argument("--lr", type=float, default=3e-3)
  p.add_argument("--w-cls", type=float, default=0.5, help="weight of the force cross-entropy term")
  p.add_argument("--l2", type=float, default=1e-4)
  p.add_argument("--val-mod", type=int, default=5, help="scene seeds with seed %% val_mod == 0 go to validation")
  p.add_argument("--min-rows", type=int, default=20, help="refuse to train on fewer successful rows")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  rows = load_rows(args.data)
  X, Y, F, S = to_arrays(rows)
  print(f"[train] {len(rows)} rows read, {len(X)} successful grasps usable")
  if len(X) < args.min_rows:
    raise SystemExit(f"[train] only {len(X)} successful rows (< --min-rows {args.min_rows}); generate more data first")
  val = S % args.val_mod == 0
  if val.all() or not val.any():
    val = np.arange(len(X)) % args.val_mod == 0  # degenerate seed set: fall back to a row split
  t0 = time.time()
  layers, norm = train(X[~val], Y[~val], F[~val], hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                       w_cls=args.w_cls, l2=args.l2, seed=args.seed, log_every=max(1, args.epochs // 5))
  sources: dict[str, int] = {}
  for r in rows:
    if r.get("success"):
      sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1
  model = BcGrasp(to_spec(layers, norm, {}))
  m_tr, m_val = metrics(model, X[~val], Y[~val], F[~val]), metrics(model, X[val], Y[val], F[val])
  base = mean_baseline(Y[~val], F[~val], X[val], Y[val], F[val])
  meta = {"features": list(FEATURE_NAMES), "force_classes": list(FORCE_CLASSES), "hidden": args.hidden,
          "epochs": args.epochs, "n_train": int((~val).sum()), "n_val": int(val.sum()), "success_sources": sources,
          "train_metrics": m_tr, "val_metrics": m_val, "val_mean_baseline": base,
          "data": [str(d) for d in args.data], "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(to_spec(layers, norm, meta)))
  print(f"[train] done in {time.time() - t0:.1f} s; successes by source: {sources}")
  print(f"[train] train      : {m_tr}")
  print(f"[train] validation : {m_val}")
  print(f"[train] mean-model : {base}  (constant predictor on the same validation rows)")
  print(f"[train] saved -> {args.out}")


if __name__ == "__main__":
  main()
