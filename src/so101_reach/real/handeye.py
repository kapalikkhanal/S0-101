"""Eye-to-hand calibration of the fixed head camera from a tag on the gripper.

For each robot pose i the camera measures the 3-D tag centre p_i (colour pixel + aligned
depth -> camera frame) and forward kinematics gives the gripper body pose (R_i, t_i) in the
robot base frame. With unknown camera pose (R_bc, t_bc) and tag offset o in the gripper
frame:

    R_bc p_i + t_bc = R_i o + t_i

solved by robust nonlinear least squares (initialised by Kabsch with o = 0). Using the
depth-measured centre makes the result independent of the printed tag size.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from so101_reach.sim import so101

EXTRINSICS_PATH: Path = so101.PROJECT_ROOT / "configs" / "head_camera_extrinsics.json"


def gripper_pose(model: mujoco.MjModel, data: mujoco.MjData, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
  """Gripper body rotation (3x3) and position (3,) in the base frame for joint angles q."""
  for n, v in q.items():
    data.qpos[model.jnt_qposadr[model.joint(n).id]] = v
  mujoco.mj_kinematics(model, data)
  b = model.body("gripper").id
  return data.xmat[b].reshape(3, 3).copy(), data.xpos[b].copy()


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """R, t minimising |R a + t - b|."""
  ca, cb = a.mean(0), b.mean(0)
  u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
  d = np.sign(np.linalg.det(vt.T @ u.T))
  r = vt.T @ np.diag([1, 1, d]) @ u.T
  return r, cb - r @ ca


@dataclass
class HeadCameraCalibration:
  R_base_cam: list[list[float]]
  t_base_cam: list[float]
  tag_offset_gripper: list[float]
  rms_mm: float
  loo_rms_mm: float
  num_poses: int

  @property
  def T_base_cam(self) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array(self.R_base_cam)
    T[:3, 3] = self.t_base_cam
    return T

  def cam_to_base(self, p_cam: np.ndarray) -> np.ndarray:
    return np.array(self.R_base_cam) @ p_cam + np.array(self.t_base_cam)

  def pose(self) -> tuple[np.ndarray, np.ndarray]:
    return np.array(self.R_base_cam), np.array(self.t_base_cam)

  def save(self, path: Path = EXTRINSICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(self), indent=2) + "\n")

  @staticmethod
  def load(path: Path = EXTRINSICS_PATH) -> HeadCameraCalibration:
    if not path.exists():
      raise FileNotFoundError(f"{path} missing - run scripts/calibrate_head_camera.py")
    return HeadCameraCalibration(**json.loads(path.read_text()))


def _residuals(x: np.ndarray, p: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
  rbc = Rotation.from_rotvec(x[:3]).as_matrix()
  pred_base = p @ rbc.T + x[3:6]
  fk = np.einsum("nij,j->ni", R, x[6:9]) + t
  return (pred_base - fk).ravel()


def _fit(p: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
  r0, t0 = kabsch(p, t)
  x0 = np.concatenate([Rotation.from_matrix(r0).as_rotvec(), t0, np.zeros(3)])
  return least_squares(_residuals, x0, args=(p, R, t), loss="soft_l1", f_scale=0.005).x


def solve(p_cam: np.ndarray, R_bg: np.ndarray, t_bg: np.ndarray) -> tuple[HeadCameraCalibration, np.ndarray]:
  """Returns the calibration and per-pose residual norms [m]."""
  x = _fit(p_cam, R_bg, t_bg)
  res = np.linalg.norm(_residuals(x, p_cam, R_bg, t_bg).reshape(-1, 3), axis=1)
  loo = []
  for i in range(len(p_cam)):
    keep = np.arange(len(p_cam)) != i
    xi = _fit(p_cam[keep], R_bg[keep], t_bg[keep])
    loo.append(np.linalg.norm(_residuals(xi, p_cam[i : i + 1], R_bg[i : i + 1], t_bg[i : i + 1])))
  cal = HeadCameraCalibration(
    R_base_cam=Rotation.from_rotvec(x[:3]).as_matrix().tolist(),
    t_base_cam=x[3:6].tolist(),
    tag_offset_gripper=x[6:9].tolist(),
    rms_mm=float(np.sqrt(np.mean(res**2)) * 1e3),
    loo_rms_mm=float(np.sqrt(np.mean(np.square(loo))) * 1e3),
    num_poses=len(p_cam),
  )
  return cal, res


# Eye-in-hand: the wrist camera (D405) on the gripper.

WRIST_EXTRINSICS_PATH: Path = so101.PROJECT_ROOT / "configs" / "wrist_camera_extrinsics.json"


@dataclass
class WristCameraCalibration:
  """Pose of the wrist camera in the gripper body frame (+ depth offset to the orange centre)."""

  R_gripper_cam: list[list[float]]
  t_gripper_cam: list[float]
  depth_offset: float
  """Added to the measured surface depth along the viewing ray to reach the fruit centre
  (radius + the sensor's skin bias), estimated jointly with the pose."""
  rms_mm: float
  loo_rms_mm: float
  num_poses: int

  def save(self, path: Path = WRIST_EXTRINSICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(self), indent=2) + "\n")

  @staticmethod
  def load(path: Path = WRIST_EXTRINSICS_PATH) -> WristCameraCalibration:
    if not path.exists():
      raise FileNotFoundError(f"{path} missing - run scripts/calibrate_wrist_camera.py")
    return WristCameraCalibration(**json.loads(path.read_text()))


class WristCameraPose:
  """Live wrist-camera pose in the base frame: FK(measured joints) x camera mount."""

  def __init__(self, calib: WristCameraCalibration, joints_fn):
    self.R_gc = np.array(calib.R_gripper_cam)
    self.t_gc = np.array(calib.t_gripper_cam)
    self.joints_fn = joints_fn  # () -> dict joint name -> angle [rad]
    self.model = so101.make_cpu_model()
    self.data = mujoco.MjData(self.model)

  def pose(self) -> tuple[np.ndarray, np.ndarray]:
    R_bg, t_bg = gripper_pose(self.model, self.data, self.joints_fn())
    return R_bg @ self.R_gc, R_bg @ self.t_gc + t_bg


def _wrist_residuals(x, rays, depths, g):
  R = Rotation.from_rotvec(x[:3]).as_matrix()
  p_cam = rays * (depths + x[6])[:, None]
  return (p_cam @ R.T + x[3:6] - g).ravel()


def _wrist_fit(rays, depths, g, offset0: float = 0.03):
  r0, t0 = kabsch(rays * (depths + offset0)[:, None], g)
  x0 = np.concatenate([Rotation.from_matrix(r0).as_rotvec(), t0, [offset0]])
  return least_squares(_wrist_residuals, x0, args=(rays, depths, g), loss="soft_l1", f_scale=0.005).x


def solve_wrist(rays: np.ndarray, depths: np.ndarray, R_bg: np.ndarray, t_bg: np.ndarray, p_orange: np.ndarray):
  """rays: (N,3) unit viewing rays to the orange in the wrist-camera frame; depths: (N,) surface
  distance along each ray [m]; R_bg, t_bg: gripper poses; p_orange: (3,) or (N,3) orange centre
  in the base frame (from the head camera). Returns (calibration, per-pose residual norms)."""
  p = np.broadcast_to(p_orange, t_bg.shape)
  g = np.einsum("nji,nj->ni", R_bg, p - t_bg)  # orange centre in each gripper frame
  x = _wrist_fit(rays, depths, g)
  res = np.linalg.norm(_wrist_residuals(x, rays, depths, g).reshape(-1, 3), axis=1)
  loo = []
  for i in range(len(rays)):
    k = np.arange(len(rays)) != i
    xi = _wrist_fit(rays[k], depths[k], g[k])
    loo.append(np.linalg.norm(_wrist_residuals(xi, rays[i : i + 1], depths[i : i + 1], g[i : i + 1])))
  cal = WristCameraCalibration(
    R_gripper_cam=Rotation.from_rotvec(x[:3]).as_matrix().tolist(),
    t_gripper_cam=x[3:6].tolist(),
    depth_offset=float(x[6]),
    rms_mm=float(np.sqrt(np.mean(res**2)) * 1e3),
    loo_rms_mm=float(np.sqrt(np.mean(np.square(loo))) * 1e3),
    num_poses=len(rays),
  )
  return cal, res
