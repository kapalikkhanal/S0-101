"""Deployment-side reach controller (no MjLab, no GPU): exported ONNX policy + the exact
observation/action contract of the training environment, driving any `Plant`.

Plants:
  * SimPlant   - CPU MuJoCo with the same MJCF/servos; sensing can be made real-like
                 (positions quantised to encoder ticks, velocity by finite differences).
  * RealPlant  - the SO-101 follower over the Feetech bus (see real/robot.py).

Per 20 ms step:  read q -> estimate qd -> FK (TCP) -> 25-D obs -> ONNX -> clip
                 -> joint_target += 0.04 * a (soft limits inward-only, |target - q| <= 0.15)
                 -> plant.command(joint_target).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from so101_reach.sim import so101
from so101_reach.sim.workspace import soft_joint_limits

from so101_reach.settings import POLICY_DIR
TICK_RAD = 2 * np.pi / 4096


class Plant(Protocol):
  dt: float

  def read_q(self) -> np.ndarray: ...  # (4,) arm joint angles [rad]

  def command(self, joint_target: np.ndarray) -> None: ...  # (4,) [rad]

  def wait(self) -> None: ...  # advance / sleep to the next 20 ms tick


class Kinematics:
  """CPU MuJoCo forward kinematics of the TCP (same model as training)."""

  def __init__(self):
    self.model = so101.make_cpu_model()
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in so101.ARM_JOINTS])
    self.fadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in so101.FIXED_JOINTS])
    self.site = self.model.site(so101.TCP_SITE).id
    self.data.qpos[self.fadr] = [so101.FIXED_JOINT_TARGETS[n] for n in so101.FIXED_JOINTS]

  def set_fixed(self, fixed: dict[str, float]) -> None:
    """Use measured wrist-roll / gripper angles (the policy assumes them fixed)."""
    for i, n in enumerate(so101.FIXED_JOINTS):
      if n in fixed:
        self.data.qpos[self.fadr[i]] = fixed[n]

  def tcp(self, q: np.ndarray) -> np.ndarray:
    self.data.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.model, self.data)
    return self.data.site_xpos[self.site].copy()


@dataclass
class ControllerCfg:
  scale: float = 0.04
  max_lead: float = 0.15
  speed_factor: float = 1.0
  """Multiplies the per-step increment limit (e.g. 0.25 for first hardware tests)."""
  vel_filter: float = 0.5
  """EMA factor for finite-difference joint velocity (1 = no filtering)."""


class ReachController:
  def __init__(self, cfg: ControllerCfg | None = None, policy_dir: Path = POLICY_DIR):
    import onnxruntime as ort

    self.cfg = cfg or ControllerCfg()
    contract = json.loads((policy_dir / "policy_contract.json").read_text())
    assert contract["action"]["scale_rad"] == self.cfg.scale, "policy contract / controller scale mismatch"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    self.sess = ort.InferenceSession(str(policy_dir / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    self.input_name = self.sess.get_inputs()[0].name
    self.kin = Kinematics()
    lim = soft_joint_limits(self.kin.model)
    self.lo, self.hi = lim[:, 0], lim[:, 1]
    self.reset(None)

  def reset(self, q: np.ndarray | None) -> None:
    self.joint_target = None if q is None else q.copy()
    self.prev_q = None if q is None else q.copy()
    self.qd = np.zeros(4)
    self.last_action = np.zeros(4)

  def observation(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    tcp = self.kin.tcp(q)
    obs = np.concatenate([q, self.qd, tcp, target, target - tcp, self.last_action, self.joint_target - q])
    return obs.astype(np.float32)[None]

  def step(self, q: np.ndarray, target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One control step. Returns (new joint target [rad], TCP position)."""
    if self.joint_target is None:
      self.reset(q)
    raw_qd = (q - self.prev_q) / dt
    a = self.cfg.vel_filter
    self.qd = a * raw_qd + (1 - a) * self.qd
    self.prev_q = q.copy()
    obs = self.observation(q, target)
    act = np.clip(np.nan_to_num(self.sess.run(None, {self.input_name: obs})[0][0]), -1.0, 1.0)
    self.last_action = act
    prev = self.joint_target
    tgt = np.clip(prev + act * self.cfg.scale * self.cfg.speed_factor, np.minimum(self.lo, prev), np.maximum(self.hi, prev))
    self.joint_target = np.clip(tgt, q - self.cfg.max_lead, q + self.cfg.max_lead)
    return self.joint_target.copy(), obs[0, 8:11].copy()


class SimPlant:
  """CPU MuJoCo plant (5 ms physics, 4 substeps per 20 ms control step)."""

  dt = 0.02

  def __init__(self, q0: np.ndarray, quantize: bool = True, actuator_model: str = "menagerie"):
    self.model = so101.make_cpu_model(actuator_model)
    self.data = mujoco.MjData(self.model)
    self.quantize = quantize
    m = self.model
    self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in so101.ARM_JOINTS])
    self.aid = np.array([m.actuator(n).id for n in so101.ARM_JOINTS])
    for n in so101.FIXED_JOINTS:
      self.data.qpos[m.jnt_qposadr[m.joint(n).id]] = so101.FIXED_JOINT_TARGETS[n]
      self.data.ctrl[m.actuator(n).id] = so101.FIXED_JOINT_TARGETS[n]
    self.data.qpos[self.qadr] = q0
    self.data.ctrl[self.aid] = q0
    mujoco.mj_forward(m, self.data)
    self.substeps = round(self.dt / m.opt.timestep)

  def read_q(self) -> np.ndarray:
    q = self.data.qpos[self.qadr].copy()
    return np.round(q / TICK_RAD) * TICK_RAD if self.quantize else q

  def command(self, joint_target: np.ndarray) -> None:
    self.data.ctrl[self.aid] = joint_target

  def wait(self) -> None:
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

  def true_q(self) -> np.ndarray:
    return self.data.qpos[self.qadr].copy()


def rollout(controller: ReachController, plant, target: np.ndarray, steps: int) -> dict[str, np.ndarray]:
  """Run the controller on a plant for `steps` control steps and record the trajectory."""
  controller.reset(plant.read_q())
  rec = {"q": [], "joint_target": [], "tcp": [], "err": []}
  for _ in range(steps):
    q = plant.read_q()
    jt, tcp = controller.step(q, target, plant.dt)
    plant.command(jt)
    plant.wait()
    rec["q"].append(q)
    rec["joint_target"].append(jt)
    rec["tcp"].append(tcp)
    rec["err"].append(np.linalg.norm(target - tcp))
  return {k: np.array(v) for k, v in rec.items()}
