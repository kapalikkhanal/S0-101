"""Export a trained checkpoint to TorchScript + ONNX + a JSON interface contract.

  python scripts/export_policy.py --checkpoint logs/rsl_rl/so101_reach/<run>/model_6000.pt
  -> checkpoints/so101_reach_policy/{policy.pt, policy.onnx, policy_contract.json}

The exported modules take the 25-D actor observation (see policy_contract.json) and return
the 4-D action (mean of the Gaussian policy; clip to [-1, 1] before applying).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from so101_reach.checkpoints import policy_io  # noqa: E402
from so101_reach.sim import so101  # noqa: E402
from so101_reach.tasks.eval_env_cfg import make_eval_env_cfg  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, default=None)
  p.add_argument("--out", type=Path, default=so101.PROJECT_ROOT / "checkpoints" / "so101_reach_policy")
  args = p.parse_args()
  ckpt = args.checkpoint or policy_io.latest_checkpoint()
  env = ManagerBasedRlEnv(make_eval_env_cfg("native", 1), device="cpu")
  actor = policy_io.load_native_actor(env, ckpt, device="cpu")
  obs_dim = env.observation_manager.group_obs_dim["actor"][0]
  paths = policy_io.export_actor(actor, args.out, obs_dim, policy_io.native_contract(ckpt))
  # Verify exports reproduce the source module.
  import onnxruntime as ort

  x = torch.randn(64, obs_dim)
  ref = actor(x).detach().numpy()
  ts = torch.jit.load(str(paths["torchscript"]))(x).detach().numpy()
  sess = ort.InferenceSession(str(paths["onnx"]))
  ox = np.concatenate([sess.run(None, {"obs": xi[None].numpy()})[0] for xi in x])
  np.testing.assert_allclose(ts, ref, rtol=1e-4, atol=5e-5)
  np.testing.assert_allclose(ox, ref, rtol=1e-4, atol=5e-5)
  print(f"[export] {ckpt} -> {args.out}")
  print(f"[export] max |torchscript - torch| = {np.abs(ts - ref).max():.2e}, max |onnx - torch| = {np.abs(ox - ref).max():.2e}")
  print(policy_io.measure_latency(actor, obs_dim, "cpu") | policy_io.measure_onnx_latency(paths["onnx"]))


if __name__ == "__main__":
  main()
