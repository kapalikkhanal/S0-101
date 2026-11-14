"""Loader + interface contract for ``nvidia/so101-reach`` (policy_fingertip_delta.onnx).

The checkpoint is an inference-only Isaac Lab actor exported to ONNX (opset 18):
  obs[1, 20] = joint_pos_rel(6) | fingertip_pos_root(3) | goal(7: xyz + wxyz quat) | last_action(4)
  actions[1, 4] (unbounded) -> clip [-1, 1]; target = q_measured + 0.20 * a,
  clamped to the descriptor joint limits, then to q_measured +- 0.10 rad, at 50 Hz.
The empirical observation normaliser is embedded in the graph.

``NvidiaContract`` is parsed from the published Isaac Lab I/O descriptor
(policy_fingertip_delta.yaml) rather than hard-coded, and ``load_nvidia_policy`` returns a
batched torch re-implementation of the ONNX graph (Sub/Div normaliser + 3x ELU Gemm + Gemm)
whose outputs are verified against onnxruntime before use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from so101_reach.sim import so101

REPO_ID = "nvidia/so101-reach"
ONNX_FILE = "policy_fingertip_delta.onnx"
YAML_FILE = "policy_fingertip_delta.yaml"
ONNX_SHA256 = "25e08afb6bc6cd62e669996303e8cd1f7a0a2aaa98e6b808b544a7d2b51e308f"
DEFAULT_DIR: Path = so101.PROJECT_ROOT / "checkpoints" / "external" / "nvidia_so101_reach"

# Values from the model card that the descriptor does not carry.
MAX_STEP_RAD = 0.10
CONTROL_HZ = 50.0


@dataclass(frozen=True)
class NvidiaContract:
  obs_joint_names: tuple[str, ...]
  obs_joint_offsets: tuple[float, ...]
  action_joint_names: tuple[str, ...]
  action_scale: float
  joint_limits: dict[str, tuple[float, float]]
  default_joint_pos: dict[str, float]
  stiffness: dict[str, float]
  damping: dict[str, float]
  physics_dt: float
  decimation: int
  obs_dim: int = 20
  max_step: float = MAX_STEP_RAD


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def download(target_dir: Path = DEFAULT_DIR) -> Path:
  from huggingface_hub import hf_hub_download

  target_dir.mkdir(parents=True, exist_ok=True)
  for f in (ONNX_FILE, YAML_FILE, "README.md"):
    hf_hub_download(REPO_ID, f, local_dir=target_dir)
  return target_dir


def load_contract(ckpt_dir: Path = DEFAULT_DIR) -> NvidiaContract:
  d = yaml.safe_load((ckpt_dir / YAML_FILE).read_text())
  obs = {t["name"]: t for t in d["observations"]["policy"]}
  names = tuple(n for n in (t["name"] for t in d["observations"]["policy"]))
  if names != ("joint_pos_rel", "ee_position", "generated_commands", "last_action"):
    raise ValueError(f"Unexpected observation layout {names}")
  dims = sum(int(np.prod(t["shape"])) for t in d["observations"]["policy"])
  if dims != 20:
    raise ValueError(f"Expected 20-D observation, descriptor gives {dims}")
  act = d["actions"][0]
  if act["full_path"].split(".")[-1] != "DelayedDeltaJointPositionAction":
    raise ValueError(f"Unexpected action type {act['full_path']}")
  art = d["articulations"]["robot"]
  jn = art["joint_names"]
  return NvidiaContract(
    obs_joint_names=tuple(obs["joint_pos_rel"]["joint_names"]),
    obs_joint_offsets=tuple(float(x) for x in obs["joint_pos_rel"]["joint_pos_offsets"]),
    action_joint_names=tuple(act["joint_names"]),
    action_scale=float(act["scale"]),
    joint_limits={n: (float(lo), float(hi)) for n, (lo, hi) in zip(jn, art["default_joint_pos_limits"], strict=True)},
    default_joint_pos={n: float(v) for n, v in zip(jn, art["default_joint_pos"], strict=True)},
    stiffness={n: float(v) for n, v in zip(jn, art["default_joint_stiffness"], strict=True)},
    damping={n: float(v) for n, v in zip(jn, art["default_joint_damping"], strict=True)},
    physics_dt=float(d["scene"]["physics_dt"]),
    decimation=int(d["scene"]["decimation"]),
  )


class NvidiaActor(torch.nn.Module):
  """Torch twin of the ONNX graph: (obs - mean) / std -> MLP[256,128,64] ELU -> 4."""

  def __init__(self, weights: dict[str, np.ndarray]):
    super().__init__()
    self.register_buffer("mean", torch.tensor(weights["normalizer._mean"]))
    self.register_buffer("std", torch.tensor(weights["normalizer._std"]))
    layers = []
    for i in (0, 2, 4, 6):
      lin = torch.nn.Linear(*weights[f"actor.{i}.weight"].shape[::-1])
      lin.weight.data = torch.tensor(weights[f"actor.{i}.weight"])
      lin.bias.data = torch.tensor(weights[f"actor.{i}.bias"])
      layers += [lin, torch.nn.ELU()] if i < 6 else [lin]
    self.mlp = torch.nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp((obs - self.mean) / self.std)


def _onnx_weights(onnx_path: Path) -> dict[str, np.ndarray]:
  import onnx
  from onnx import numpy_helper

  m = onnx.load(str(onnx_path))
  init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
  ops = [n.op_type for n in m.graph.node]
  if ops != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
    raise ValueError(f"Unexpected ONNX graph {ops}")
  div = next(n for n in m.graph.node if n.op_type == "Div")
  init["normalizer._std"] = init[div.input[1]]
  return init


def load_nvidia_policy(
  ckpt_dir: Path = DEFAULT_DIR, device: str = "cpu", verify: bool = True
) -> tuple[NvidiaActor, NvidiaContract]:
  onnx_path = ckpt_dir / ONNX_FILE
  if not onnx_path.exists():
    raise FileNotFoundError(f"{onnx_path} missing; run scripts/download_checkpoints.py")
  digest = sha256(onnx_path)
  if digest != ONNX_SHA256:
    raise ValueError(f"SHA-256 mismatch for {onnx_path}: {digest}")
  contract = load_contract(ckpt_dir)
  actor = NvidiaActor(_onnx_weights(onnx_path)).eval()
  if verify:
    max_err = verify_against_onnxruntime(actor, onnx_path)
    if max_err > 1e-4:
      raise RuntimeError(f"Torch twin deviates from onnxruntime by {max_err}")
  return actor.to(device), contract


def verify_against_onnxruntime(actor: NvidiaActor, onnx_path: Path, n: int = 256) -> float:
  import onnxruntime as ort

  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  rng = np.random.default_rng(0)
  obs = rng.normal(0, 0.5, size=(n, 20)).astype(np.float32)
  ref = np.concatenate([sess.run(None, {name: o[None]})[0] for o in obs])
  with torch.no_grad():
    out = actor(torch.tensor(obs)).numpy()
  return float(np.abs(out - ref).max())
