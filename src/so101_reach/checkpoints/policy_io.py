"""Checkpoint utilities: load native RSL-RL checkpoints, wrap external policies, export.

Native checkpoints (``logs/rsl_rl/so101_reach/<run>/model_<it>.pt``) contain the full
training state (actor, critic, optimiser, iteration, curriculum stage) and are resumable
with ``--agent.resume True``. For inference the actor is converted to a self-contained
module (observation normaliser + MLP + mean action) that takes the concatenated 25-D actor
observation; it can be exported as TorchScript (``policy.pt``) or ONNX (``<run>.onnx``,
written by the runner at every save).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from so101_reach.sim import so101

LOG_ROOT: Path = so101.PROJECT_ROOT / "logs" / "rsl_rl" / "so101_reach"


class ObsDictPolicy:
  """Adapts a module taking the actor tensor to MjLab's obs-dict interface."""

  def __init__(self, module: torch.nn.Module, group: str = "actor"):
    self.module = module.eval()
    self.group = group

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    return self.module(obs[self.group])


def latest_checkpoint(log_root: Path = LOG_ROOT) -> Path:
  """Highest-iteration checkpoint of the newest run under `log_root` (or inside a run dir)."""
  found = list(log_root.glob("model_*.pt")) or list(log_root.glob("*/model_*.pt"))
  ckpts = sorted(found, key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
  if not ckpts and log_root == LOG_ROOT:
    # A fresh clone can use the final simulation policy without a local training run.
    bundled = so101.PROJECT_ROOT / "checkpoints" / "so101_reach_policy" / "model_5999.pt"
    if bundled.is_file():
      return bundled
  if not ckpts:
    raise FileNotFoundError(f"No checkpoints under {log_root}")
  return ckpts[-1]


def load_native_actor(env, checkpoint: Path, device: str | None = None) -> torch.nn.Module:
  """Load an RSL-RL checkpoint into a standalone actor module (deterministic)."""
  from mjlab.rl import RslRlVecEnvWrapper

  from so101_reach.tasks.rl_cfg import so101_reach_ppo_runner_cfg
  from so101_reach.tasks.runner import So101ReachRunner

  device = device or str(env.device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = So101ReachRunner(wrapped, asdict(so101_reach_ppo_runner_cfg()), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.alg.get_policy().as_jit().to(device).eval()


def load_native_policy(env, checkpoint: Path, device: str | None = None) -> ObsDictPolicy:
  return ObsDictPolicy(load_native_actor(env, checkpoint, device))


def load_nvidia_obs_policy(device: str) -> ObsDictPolicy:
  from so101_reach.checkpoints.nvidia import load_nvidia_policy

  actor, _ = load_nvidia_policy(device=device)
  return ObsDictPolicy(actor)


def export_actor(actor: torch.nn.Module, out_dir: Path, obs_dim: int, meta: dict) -> dict[str, Path]:
  """Export TorchScript + ONNX (batch 1) and a JSON interface contract."""
  out_dir.mkdir(parents=True, exist_ok=True)
  cpu = actor.to("cpu").eval()
  scripted = torch.jit.script(cpu)
  ts_path = out_dir / "policy.pt"
  scripted.save(str(ts_path))
  onnx_path = out_dir / "policy.onnx"
  torch.onnx.export(
    cpu, (torch.zeros(1, obs_dim),), str(onnx_path), input_names=["obs"], output_names=["actions"],
    opset_version=18, dynamo=False,
  )
  meta_path = out_dir / "policy_contract.json"
  meta_path.write_text(json.dumps(meta, indent=2))
  return {"torchscript": ts_path, "onnx": onnx_path, "contract": meta_path}


def native_contract(checkpoint: Path) -> dict:
  from so101_reach.tasks.mdp.actions import IncrementalJointPositionActionCfg
  from so101_reach.tasks.reach_env_cfg import STEP_DT

  a = IncrementalJointPositionActionCfg(entity_name="robot")
  return {
    "checkpoint": checkpoint.name,
    "control_hz": 1.0 / STEP_DT,
    "frame": "robot base frame (base body origin, +x forward, +z up), metres",
    "tcp": {"body": "gripper", "offset_m": list(so101.NVIDIA_FINGERTIP_OFFSET), "note": "URDF gripper_frame_link"},
    "observation": [
      ["arm_joint_pos_rad", 4], ["arm_joint_vel_rad_s", 4], ["tcp_pos_m", 3], ["target_pos_m", 3],
      ["target_minus_tcp_m", 3], ["last_clipped_action", 4], ["joint_target_minus_joint_pos_rad", 4],
    ],
    "joint_order": list(so101.ARM_JOINTS),
    "action": {
      "dim": 4,
      "semantics": "joint_target += clip(a,-1,1) * scale; clamp to soft limits; clamp to q +- max_lead",
      "scale_rad": a.scale,
      "max_lead_rad": a.max_lead,
      "soft_limit_factor": so101.SOFT_JOINT_LIMIT_FACTOR,
      "fixed_joint_targets": a.fixed_targets,
    },
  }


def measure_latency(module: torch.nn.Module, obs_dim: int, device: str, n: int = 2000) -> dict[str, float]:
  """Single-observation (batch 1) inference latency statistics in microseconds."""
  module = module.to(device).eval()
  x = torch.zeros(1, obs_dim, device=device)
  with torch.no_grad():
    for _ in range(100):
      module(x)
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    ts = []
    for _ in range(n):
      t0 = time.perf_counter()
      module(x)
      if device.startswith("cuda"):
        torch.cuda.synchronize()
      ts.append(time.perf_counter() - t0)
  ts_us = np.array(ts) * 1e6
  return {f"{device}_mean_us": float(ts_us.mean()), f"{device}_p99_us": float(np.percentile(ts_us, 99))}


def measure_onnx_latency(onnx_path: Path, n: int = 2000) -> dict[str, float]:
  import onnxruntime as ort

  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
  inp = sess.get_inputs()[0]
  x = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
  for _ in range(100):
    sess.run(None, {inp.name: x})
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    sess.run(None, {inp.name: x})
    ts.append(time.perf_counter() - t0)
  ts_us = np.array(ts) * 1e6
  return {"onnxruntime_cpu_1thread_mean_us": float(ts_us.mean()), "onnxruntime_cpu_1thread_p99_us": float(np.percentile(ts_us, 99))}
