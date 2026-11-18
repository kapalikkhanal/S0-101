"""Simulated world for the agent: SO-101 (identified servos, real torque limits) on the table,
free objects, static platforms / open boxes, and the head + wrist cameras placed from their
real calibrations. Exposes the Feetech bus interface used by RealPlant, so the agent's tools
run unchanged on it (physics advances in `step_physics`, one 20 ms control period).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from so101_reach.real.feetech_bus import JOINT_ORDER, load_calibration
from so101_reach.real.handeye import HeadCameraCalibration, WristCameraCalibration
from so101_reach.real.joint_map import JointMap
from so101_reach.sim.grasp_scene import STALL_TORQUE, ObjectCfg, SceneCfg, build_spec
from so101_reach.sim import so101

HEAD_FOVY, HEAD_SIZE = 42.0, (720, 1280)  # D435i colour
WRIST_FOVY, WRIST_SIZE = 58.0, (480, 848)  # D405


@dataclass
class StaticBox:
  name: str
  center: tuple[float, float, float]  # base frame [m]
  half: tuple[float, float, float]
  rgba: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)


@dataclass
class OpenBox:
  name: str
  center_xy: tuple[float, float]
  outer: tuple[float, float]  # outer size x, y [m]
  height: float
  wall: float = 0.004
  rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0)


@dataclass
class WorldCfg:
  table_z: float = -0.0132
  objects: dict[str, ObjectCfg] = field(default_factory=dict)  # label -> object
  platforms: list[StaticBox] = field(default_factory=list)
  boxes: list[OpenBox] = field(default_factory=list)


def scene(name: str) -> WorldCfg:
  """Test scenes. 'table': a cube on the table; 'on_box': the cube on a 5 cm block; both
  with an open white box to place into."""
  tz = -0.0132
  cube = ObjectCfg("box", (0.02, 0.02, 0.02), (0.25, -0.08, tz + 0.02), mass=0.05, friction=(1.0, 0.005, 0.0001),
                   rgba=(0.85, 0.1, 0.1, 1.0))
  box = OpenBox("white box", (0.33, -0.19), (0.12, 0.16), 0.045)
  cfg = WorldCfg(tz, {"red cube": cube}, [], [box])
  if name in ("stack", "tower"):
    s = 0.017
    mk = lambda pos, rgba: ObjectCfg("box", (s, s, s), pos, mass=0.03, friction=(1.0, 0.005, 0.0001), rgba=rgba)
    colors = {"white cube": (0.95, 0.95, 0.95, 1.0), "black cube": (0.08, 0.08, 0.08, 1.0), "red cube": (0.85, 0.1, 0.1, 1.0)}
    if name == "stack":
      spots = {"white cube": (0.24, -0.10), "black cube": (0.20, -0.17), "red cube": (0.30, -0.04)}
      objs = {k: mk((*spots[k], tz + s), colors[k]) for k in colors}
    else:  # white bottom, black middle, red top
      objs = {k: mk((0.25, -0.10, tz + s + 2 * s * i), colors[k]) for i, k in enumerate(("white cube", "black cube", "red cube"))}
    return WorldCfg(tz, objs, [], [box])
  if name == "on_box":
    cfg.platforms.append(StaticBox("block", (0.25, -0.06, tz + 0.025), (0.04, 0.04, 0.025), (0.55, 0.35, 0.2, 1.0)))
    cube.pos = (0.25, -0.06, tz + 0.05 + 0.02)
  elif name != "table":
    raise ValueError(f"unknown scene {name!r} (table, on_box, stack, tower)")
  return cfg


def _cv_to_mj_quat(R_cv: np.ndarray) -> np.ndarray:
  """MuJoCo cameras look along -z with +y up; OpenCV cameras along +z with +y down."""
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, (R_cv @ np.diag([1.0, -1.0, -1.0])).flatten())
  return q


class SimWorld:
  def __init__(self, cfg: WorldCfg, substeps: int | None = None):
    self.cfg = cfg
    self.jmap = JointMap.load(so101.SIM_CONFIG_DIR / "real_joint_map.json")
    self.fcal = load_calibration(so101.SIM_CONFIG_DIR / "sim_motor_calibration.json")
    objs = list(cfg.objects.values())
    spec = build_spec(SceneCfg(table_z=cfg.table_z, objects=objs, torque_limits={n: 1000 for n in JOINT_ORDER}))
    wb = spec.worldbody
    for p in cfg.platforms:
      wb.add_geom(name=f"static_{p.name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=p.center, size=p.half, rgba=p.rgba)
    for b in cfg.boxes:
      (cx, cy), (sx, sy), h, w = b.center_xy, b.outer, b.height, b.wall
      z0 = cfg.table_z
      parts = [((cx, cy, z0 + w / 2), (sx / 2, sy / 2, w / 2))]
      parts += [((cx + s * (sx / 2 - w / 2), cy, z0 + h / 2), (w / 2, sy / 2, h / 2)) for s in (-1, 1)]
      parts += [((cx, cy + s * (sy / 2 - w / 2), z0 + h / 2), (sx / 2, w / 2, h / 2)) for s in (-1, 1)]
      for k, (pos, half) in enumerate(parts):
        wb.add_geom(name=f"box_{b.name}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=b.rgba)
    hc = HeadCameraCalibration.load(so101.SIM_CONFIG_DIR / "head_camera_extrinsics.json")
    wb.add_camera(name="head", pos=hc.t_base_cam, quat=_cv_to_mj_quat(np.array(hc.R_base_cam)), fovy=HEAD_FOVY)
    wc = WristCameraCalibration.load(so101.SIM_CONFIG_DIR / "wrist_camera_extrinsics.json")
    spec.body("gripper").add_camera(name="wrist", pos=wc.t_gripper_cam, quat=_cv_to_mj_quat(np.array(wc.R_gripper_cam)),
                                    fovy=WRIST_FOVY)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in JOINT_ORDER}
    self.aid = {n: m.actuator(n).id for n in JOINT_ORDER}
    self.labels = list(cfg.objects)
    self.obj_body = {label: m.body(f"object{i}").id for i, label in enumerate(self.labels)}
    self.substeps = substeps or int(round(0.02 / m.opt.timestep))
    self.torque = dict.fromkeys(JOINT_ORDER, 1)
    self.goal_ticks: dict[str, int] = {}
    self.limits: dict[str, int] = {}
    self._renderers: dict[str, mujoco.Renderer] = {}

  def close_renderers(self) -> None:
    for r in self._renderers.values():
      r.close()
    self._renderers.clear()

  # --- Feetech bus look-alike -------------------------------------------------------------
  def connect(self) -> None:
    pass

  def close(self) -> None:
    pass

  def set_arm(self, q: dict[str, float]) -> None:
    for n, v in q.items():
      self.data.qpos[self.qadr[n]] = v
      self.data.ctrl[self.aid[n]] = v
    mujoco.mj_forward(self.model, self.data)

  def sync_read(self, reg: str, names=JOINT_ORDER) -> dict[str, int]:
    if reg == "Torque_Enable":
      return {n: self.torque[n] for n in names}
    if reg == "Present_Load":
      return {n: int(1000 * self.data.actuator_force[self.aid[n]] / STALL_TORQUE) for n in names}
    if reg == "Goal_Position":
      present = {n: self.jmap.to_ticks(n, self.data.qpos[self.qadr[n]]) for n in names}
      return {n: self.goal_ticks.get(n, present[n]) for n in names}
    assert reg == "Present_Position", reg
    return {n: self.jmap.to_ticks(n, self.data.qpos[self.qadr[n]]) for n in names}

  def sync_write(self, reg: str, values: dict[str, int]) -> None:
    if reg == "Torque_Enable":
      self.torque.update(values)
    elif reg == "Goal_Position":
      for n, t in values.items():
        self.goal_ticks[n] = int(t)
        self.data.ctrl[self.aid[n]] = self.jmap.to_rad(n, t)

  def write(self, name: str, reg: str, value: int) -> None:
    if reg == "Torque_Limit":
      # The arm servos' identified model was fitted at the real Torque_Limit (400): applying the
      # cap again would double it. The gripper's limit sets its grip force.
      self.limits[name] = int(value)
      if name == "gripper":
        self.model.actuator_forcerange[self.aid[name]] = np.array([-1.0, 1.0]) * STALL_TORQUE * value / 1000.0
    elif reg == "Goal_Position":
      self.sync_write("Goal_Position", {name: value})

  def read(self, name: str, reg: str) -> int:
    if reg == "Goal_Position":
      return self.goal_ticks.get(name, self.sync_read("Present_Position", (name,))[name])
    if reg == "Torque_Limit":
      return self.limits.get(name, 1000)
    return self.sync_read(reg, (name,))[name]

  def step_physics(self) -> None:
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

  # --- Cameras and ground truth -----------------------------------------------------------
  def render(self, cam: str) -> np.ndarray:
    h, w = HEAD_SIZE if cam == "head" else WRIST_SIZE
    if cam not in self._renderers:
      self._renderers[cam] = mujoco.Renderer(self.model, h, w)
    r = self._renderers[cam]
    r.update_scene(self.data, camera=cam)
    if cam == "wrist":
      self.hide_wrist_housing(r.scene)
    return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

  def hide_wrist_housing(self, scn: mujoco.MjvScene) -> None:
    """Drop the camera_mount geoms from a wrist-camera render. The calibrated D405 optical centre
    lies ~2.6 cm inside the mount mesh, so without this the wrist image is the inside of its own
    housing; the real lens looks out of the housing's front. Rendering only, physics unchanged."""
    mount = self.model.body("camera_mount").id
    for i in range(scn.ngeom):
      g = scn.geoms[i]
      if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and self.model.geom_bodyid[g.objid] == mount:
        g.rgba[3] = 0.0

  def camera_model(self, cam: str) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """(R_base_cam (OpenCV axes), t_base_cam, fx, fy, cx, cy) of a camera right now."""
    m, d = self.model, self.data
    cid = m.camera(cam).id
    R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    h, w = HEAD_SIZE if cam == "head" else WRIST_SIZE
    f = (h / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    return R, d.cam_xpos[cid].copy(), f, f, w / 2, h / 2

  def object_truth(self, label: str) -> dict:
    """Ground-truth centre, top height, size and yaw of an object (base frame)."""
    m, d = self.model, self.data
    b = self.obj_body[label]
    g = m.body_geomadr[b]
    R = d.xmat[b].reshape(3, 3)
    half = m.geom_size[g].copy()
    kind=m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
      half = np.array([half[0]] * 3)
      # Sphere bounds are invariant to the object's free-joint rotation.
      extent=half
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
      half = np.array([half[0], half[0], half[1]])
      axis=R[:,2]
      extent=half[0]*np.sqrt(np.maximum(0,1-axis**2))+half[2]*np.abs(axis)
    else:
      extent=None
    corners = np.array([d.xpos[b] + (R @ (half * (2 * np.array(s) - 1)) if extent is None else extent*(2*np.array(s)-1)) for s in np.ndindex(2, 2, 2)])
    ax = R[:, 0]
    return {"center": d.xpos[b].copy(), "top": float(corners[:, 2].max()), "bottom": float(corners[:, 2].min()),
            "size": (2 * half).tolist(), "yaw_deg": float(np.degrees(np.arctan2(ax[1], ax[0]))), "corners": corners,
            "linear_velocity":d.cvel[b,3:].copy()}

  def surfaces(self) -> list[dict]:
    """Static structures (platforms, boxes) for ground-truth 'find'."""
    out = []
    for p in self.cfg.platforms:
      c = np.array(p.center)
      out.append({"label": p.name, "center": c, "top": c[2] + p.half[2], "size": (2 * np.array(p.half)).tolist(), "yaw_deg": 0.0,
                  "corners": np.array([c + np.array(p.half) * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])})
    for b in self.cfg.boxes:
      c = np.array([*b.center_xy, self.cfg.table_z + b.height / 2])
      half = np.array([b.outer[0] / 2, b.outer[1] / 2, b.height / 2])
      out.append({"label": b.name, "center": c, "top": c[2] + half[2], "size": (2 * half).tolist(), "yaw_deg": 0.0,
                  "corners": np.array([c + half * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)]), "container": True})
    return out
