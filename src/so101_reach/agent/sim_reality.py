"""Reality-gap noise for the agent simulation: one knob (`level`) that scales every source.

`level` = 0 is the exact simulator; 1 is our estimate of the real SO-101 + head-camera setup;
2 is a pessimistic setup. All sources are seeded per episode, so every method sees the same
noise realisation for the same (seed, level):

  servo     encoder read noise (Gaussian, ticks) on every joint and a per-episode calibration
            bias per arm joint: the servo reports q + b, so the controller commands the arm
            to a pose that is off by -b (what a slightly wrong calibration does on the real arm)
  actuator  per-episode, per-joint log-uniform scaling of the identified servo model (kp, kv,
            Coulomb friction / stiction) and a position deadband: inside it the servo applies
            no position torque, so friction can hold the joint short of the goal. Unknown
            friction is the sim-to-real factor that breaks transfer most often (static-friction
            DR study, arXiv 2503.01255); the fit itself has residual error and drifts with
            temperature / load, so level 1 widens it moderately and level 2 reaches the
            training randomization range (x[0.5, 2] friction)
  physics   object mass and friction scaled randomly (unknown real objects)
  camera    NoisyPerception (centre / size / yaw noise, DINO outliers, misses, arm occlusion)
            plus a per-episode systematic offset of every measurement (head-camera extrinsic
            calibration error)

Use `make_world(cfg, seed, level)` instead of `SimWorld(cfg)` and `wrap_perception(per, ...)`
around `SimPerception`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from so101_reach.agent.sim_noise import NoiseCfg, NoisyPerception
from so101_reach.agent.sim_world import SimWorld, WorldCfg

ARM_BIAS_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Joints whose servo model was identified on the real arm (configs/sim/servo_model_fit.json).
SERVO_FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def log_scale(rng: np.random.Generator, rel: float, level: float) -> float:
  """Multiplier in [1 / (1 + rel * level), 1 + rel * level], log-uniform (1 at level 0)."""
  w = np.log1p(max(0.0, rel * level))
  return float(np.exp(rng.uniform(-w, w))) if w > 0 else 1.0


@dataclass
class RealityCfg:
  """Values at level 1 (realistic). Everything scales linearly with the level."""
  enc_sigma_ticks: float = 1.5  # STS3215: 4096 ticks/rev -> 1.5 ticks ~ 0.13 deg
  calib_bias_deg: float = 0.8  # per-joint calibration offset (1 sigma)
  mass_rel: float = 0.25  # mass * U(1 - x, 1 + x)
  friction_rel: float = 0.30  # sliding friction * U(1 - x, 1 + x / 2): real objects are mostly slipperier
  cam_bias_xy: float = 0.004  # head-camera extrinsic error [m] (1 sigma)
  cam_bias_z: float = 0.002
  servo_kp_rel: float = 0.15  # kp * logU(1 / (1 + x), 1 + x) per joint
  servo_kv_rel: float = 0.25
  servo_friction_rel: float = 0.5  # joint frictionloss (stiction): x[0.67, 1.5] at level 1, x[0.5, 2] at 2
  deadband_ticks: float = 1.0  # STS3215 CW/CCW dead band register, ticks (0.088 deg each)
  perception: NoiseCfg = None  # type: ignore[assignment]

  def __post_init__(self):
    if self.perception is None:
      self.perception = NoiseCfg()


def scaled_noise(base: NoiseCfg, level: float) -> NoiseCfg:
  return NoiseCfg(sigma_xy=base.sigma_xy * level, sigma_z=base.sigma_z * level, sigma_size=base.sigma_size * level,
                  sigma_yaw_deg=base.sigma_yaw_deg * level, p_outlier=min(0.5, base.p_outlier * level),
                  outlier_m=base.outlier_m, p_miss=min(0.5, base.p_miss * level), occl_radius=base.occl_radius)


class NoisySimWorld(SimWorld):
  """SimWorld whose servo bus reports noisy / biased encoder ticks."""

  def __init__(self, cfg: WorldCfg, level: float, rc: RealityCfg, rng: np.random.Generator):
    super().__init__(cfg)
    self.level, self.rc, self.rng = level, rc, rng
    per_rad = {n: 1.0 / max(abs(self.jmap.to_rad(n, 2049) - self.jmap.to_rad(n, 2048)), 1e-9) for n in ARM_BIAS_JOINTS}
    self.bias_ticks = {n: 0.0 for n in self.qadr}
    for n in ARM_BIAS_JOINTS:
      self.bias_ticks[n] = float(rng.normal(0, np.radians(rc.calib_bias_deg) * level)) * per_rad[n]
    # Actuator mismatch: perturb the identified servo model in place (this world's own model).
    m = self.model
    self.servo_scale: dict[str, dict[str, float]] = {}
    for n in SERVO_FIT_JOINTS:
      a, dof = self.aid[n], m.jnt_dofadr[m.joint(n).id]
      sk, sv, sf = (log_scale(rng, r, level) for r in (rc.servo_kp_rel, rc.servo_kv_rel, rc.servo_friction_rel))
      kp = m.actuator_gainprm[a, 0] * sk
      m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = kp, -kp
      m.actuator_biasprm[a, 2] *= sv
      m.dof_frictionloss[dof] *= sf
      self.servo_scale[n] = {"kp": sk, "kv": sv, "friction": sf}
    # Dead band [rad] per arm joint, and the commanded joint angle the servo is driving to.
    self.deadband = {n: rc.deadband_ticks * level / per_rad[n] for n in SERVO_FIT_JOINTS}
    self.cmd_rad: dict[str, float] = {}

  def set_arm(self, q: dict[str, float]) -> None:
    super().set_arm(q)
    self.cmd_rad.update({n: float(v) for n, v in q.items() if n in self.deadband})

  def step_physics(self) -> None:
    """Servo dead band: within it the position loop applies no torque (target = position),
    so stiction can hold the joint where it stopped."""
    d = self.data
    for _ in range(self.substeps):
      for n, c in self.cmd_rad.items():
        q = d.qpos[self.qadr[n]]
        d.ctrl[self.aid[n]] = q if abs(c - q) <= self.deadband[n] else c
      mujoco.mj_step(self.model, d)

  def _noisy_ticks(self, names) -> dict[str, int]:
    s = self.rc.enc_sigma_ticks * self.level
    return {n: int(round(self.jmap.to_ticks(n, self.data.qpos[self.qadr[n]]) + self.bias_ticks[n]
                         + (self.rng.normal(0, s) if s > 0 else 0.0))) for n in names}

  def sync_read(self, reg: str, names=None):
    from so101_reach.real.feetech_bus import JOINT_ORDER

    names = JOINT_ORDER if names is None else names
    if reg == "Present_Position":
      return self._noisy_ticks(names)
    if reg == "Goal_Position":
      present = self._noisy_ticks(names)
      return {n: self.goal_ticks.get(n, present[n]) for n in names}
    return super().sync_read(reg, names)

  def sync_write(self, reg: str, values: dict[str, int]) -> None:
    if reg == "Goal_Position":
      for n, t in values.items():
        self.goal_ticks[n] = int(t)
        # The servo drives its own (biased) encoder to the goal: the true joint ends at goal - bias.
        self.data.ctrl[self.aid[n]] = self.jmap.to_rad(n, t - self.bias_ticks[n])
        if n in self.deadband:
          self.cmd_rad[n] = float(self.data.ctrl[self.aid[n]])
      return
    super().sync_write(reg, values)


def randomize_physics(cfg: WorldCfg, level: float, rc: RealityCfg, rng: np.random.Generator) -> WorldCfg:
  if level <= 0:
    return cfg
  objs = {}
  for k, o in cfg.objects.items():
    m = o.mass * float(rng.uniform(1 - rc.mass_rel * level, 1 + rc.mass_rel * level))
    f = o.friction[0] * float(rng.uniform(max(0.2, 1 - rc.friction_rel * level), 1 + rc.friction_rel * level / 2))
    objs[k] = replace(o, mass=max(0.005, m), friction=(f, *o.friction[1:]))
  return replace(cfg, objects=objs)


def make_world(cfg: WorldCfg, seed: int, level: float, rc: RealityCfg | None = None) -> SimWorld:
  """Exact SimWorld at level 0, else a NoisySimWorld on physics-randomized objects."""
  rc = rc or RealityCfg()
  if level <= 0:
    return SimWorld(cfg)
  rng = np.random.default_rng([seed, 4242])
  return NoisySimWorld(randomize_physics(cfg, level, rc, rng), level, rc, rng)


class BiasedNoisyPerception(NoisyPerception):
  """NoisyPerception plus one systematic offset for the whole episode (extrinsic error)."""

  def __init__(self, per, arm_points=None, cfg: NoiseCfg | None = None, seed: int = 0, render: bool = True,
               bias: np.ndarray | None = None):
    super().__init__(per, arm_points, cfg, seed, render)
    self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)

  def _noisy(self, d):
    nd = super()._noisy(d)
    c = np.asarray(nd.center, float) + self.bias
    return replace(nd, center=c, top_z=nd.top_z + float(self.bias[2]), support_z=nd.support_z + float(self.bias[2]))


def wrap_perception(per, seed: int, level: float, arm_points=None, rc: RealityCfg | None = None, render: bool = True):
  """SimPerception at level 0, else noisy + biased perception."""
  rc = rc or RealityCfg()
  if level <= 0:
    return per
  rng = np.random.default_rng([seed, 777])
  bias = rng.normal(0, [rc.cam_bias_xy * level, rc.cam_bias_xy * level, rc.cam_bias_z * level])
  return BiasedNoisyPerception(per, arm_points=arm_points, cfg=scaled_noise(rc.perception, level), seed=seed,
                               render=render, bias=bias)
