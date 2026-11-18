"""Checks for the actuator part of the reality-gap model (CPU MuJoCo, no GPU):

  PYTHONPATH=src python3 -m pytest tests/test_sim_reality.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101_reach.agent.sim_reality import SERVO_FIT_JOINTS, NoisySimWorld, RealityCfg, log_scale, make_world  # noqa: E402
from so101_reach.agent.sim_world import SimWorld, WorldCfg  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

TZ = -0.0132


def servo_params(w: SimWorld) -> dict[str, tuple[float, float, float]]:
  m = w.model
  return {n: (float(m.actuator_gainprm[w.aid[n], 0]), float(-m.actuator_biasprm[w.aid[n], 2]),
              float(m.dof_frictionloss[m.jnt_dofadr[m.joint(n).id]])) for n in SERVO_FIT_JOINTS}


def test_log_scale_bounds_and_identity():
  rng = np.random.default_rng(0)
  assert log_scale(rng, 0.5, 0.0) == 1.0
  v = np.array([log_scale(rng, 0.5, 2.0) for _ in range(2000)])
  assert v.min() >= 0.5 - 1e-9 and v.max() <= 2.0 + 1e-9
  assert abs(np.median(np.log(v))) < 0.05  # log-uniform: symmetric in log space


def test_level0_is_exact_and_levels_perturb_servos_deterministically():
  base = servo_params(make_world(WorldCfg(TZ), seed=3, level=0))
  w1 = make_world(WorldCfg(TZ), seed=3, level=1)
  assert isinstance(w1, NoisySimWorld)
  p1 = servo_params(w1)
  assert p1 == servo_params(make_world(WorldCfg(TZ), seed=3, level=1))  # same seed -> same realisation
  assert p1 != servo_params(make_world(WorldCfg(TZ), seed=4, level=1))
  rc = RealityCfg()
  for n in SERVO_FIT_JOINTS:
    s = w1.servo_scale[n]
    for i, (k, rel) in enumerate((("kp", rc.servo_kp_rel), ("kv", rc.servo_kv_rel), ("friction", rc.servo_friction_rel))):
      assert 1 / (1 + rel) - 1e-9 <= s[k] <= 1 + rel + 1e-9
      assert p1[n][i] == pytest.approx(base[n][i] * s[k])
    # Position actuator stays consistent: bias = -kp.
    assert w1.model.actuator_biasprm[w1.aid[n], 1] == pytest.approx(-p1[n][0])


def test_deadband_holds_position_target_at_measured_angle():
  w = make_world(WorldCfg(TZ), seed=1, level=2)
  q0 = {**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS}
  w.set_arm(q0)
  n = "elbow_flex"
  db = w.deadband[n]
  assert db == pytest.approx(2 * RealityCfg().deadband_ticks * 2 * np.pi / 4096, rel=0.05)
  # Command inside the band: the servo must not drive the joint (ctrl follows the measured angle).
  w.cmd_rad[n] = float(w.data.qpos[w.qadr[n]]) + 0.5 * db
  w.step_physics()
  assert w.data.ctrl[w.aid[n]] == pytest.approx(w.data.qpos[w.qadr[n]], abs=1e-3)
  # Command outside the band: the servo drives toward it.
  goal = float(w.data.qpos[w.qadr[n]]) + 0.2
  w.cmd_rad[n] = goal
  w.step_physics()
  assert w.data.ctrl[w.aid[n]] == pytest.approx(goal)
