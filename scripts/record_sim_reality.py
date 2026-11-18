"""Record a slide video of the local pick-and-place pipeline in the noisy ("sim-to-reality") simulator.

Scene: the table with three cubes (white, black, red) and the open white box. The world is built with
`sim_reality.make_world` / `wrap_perception`, so at --level > 0 the servos report noisy, biased encoder
ticks, object mass / friction are randomized and the head-camera measurements carry noise, outliers,
misses, arm occlusion and an extrinsic offset. No LLM is used: grasps and places are local (cache /
BC / scripted); if a local phase fails the run just records the failure.

Each frame: a wide overview camera, plus an inset of the head camera with the latest measurement of
every object (green = ground truth, red = what the noisy perception reported, grey X = missed/occluded),
an inset of the wrist (gripper) camera, and a caption with the current pipeline phase.

  .venv/bin/python scripts/record_sim_reality.py                         # level 1 -> videos/sim_reality_l1.mp4
  .venv/bin/python scripts/record_sim_reality.py --level 2 --seed 3
  .venv/bin/python scripts/record_sim_reality.py --tracker               # + Kalman tracker -> ..._kf.mp4
  .venv/bin/python scripts/record_sim_reality.py --steps "red cube>white cube"   # stack instead of the default
                                                                           # (every cube into the white box)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from so101_reach.agent.arm import AgentArm  # noqa: E402
from so101_reach.agent.bc_grasp import DEFAULT_MODEL, BcGrasp  # noqa: E402
from so101_reach.agent.perception import SimPerception  # noqa: E402
from so101_reach.agent.pipeline import Pipeline  # noqa: E402
from so101_reach.agent.sim_reality import make_world, wrap_perception  # noqa: E402
from so101_reach.agent.sim_world import scene  # noqa: E402
from so101_reach.agent.tracker import TrackedPerception  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.mouse_real import SafeTargets  # noqa: E402
from so101_reach.real.robot import SimRealPlant  # noqa: E402
from so101_reach.settings import AGENT_POLICY_DIR as POLICY  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

W, H = 1280, 720
INSET_W, INSET_H = 448, 252
HEAD_ZOOM = 2.0  # the inset is a 2x crop of the head image around the workspace
WORK_CENTER = (0.26, -0.10, 0.0)  # base frame [m]
TABLE_RGBA = (0.56, 0.43, 0.32, 1.0)
LEG_RGBA = (0.35, 0.25, 0.17, 1.0)
ROOM_RGBA = (0.55, 0.56, 0.58, 1.0)


class NoLLM:
  """Stands in for the LLM agent: the pipeline only calls it when a local phase fails."""
  phase = ""
  total_usage = {"in": 0, "out": 0}
  last_close_force = "medium"
  last_open_percent = 80.0

  def run(self, *a, **k):
    return {"success": False, "summary": "no LLM in this recording"}

  def plan(self, *a, **k):
    return []


class Recorder:
  def __init__(self, world, per, out: Path | None, fps: int, every: int, level: float, title: str,
               view=(0.72, 230.0, -30.0)):
    self.world, self.per, self.every, self.level, self.title = world, per, every, level, title
    self.n = 0
    self.caption = "starting"
    m = world.model
    # Hide the checker floor in these renderers only (physics unchanged) and draw a wooden table.
    m.geom_group[m.geom("floor").id] = 4
    self.vopt = mujoco.MjvOption()
    self.vopt.geomgroup[4] = 0
    self.table_geoms = self._table(world.cfg.table_z)
    self.overview = mujoco.Renderer(m, H, W)
    self.hw, self.hh = int(INSET_W * HEAD_ZOOM), int(INSET_H * HEAD_ZOOM)
    self.head = mujoco.Renderer(m, self.hh, self.hw)
    self.wrist = mujoco.Renderer(m, INSET_H, INSET_W)  # D405 on the gripper: 848x480, same aspect as the inset
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.lookat[:] = (0.20, -0.10, 0.03)
    self.cam.distance, self.cam.azimuth, self.cam.elevation = view
    self.ffmpeg = None if out is None else subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    self._step = world.step_physics
    world.step_physics = self.step

  @staticmethod
  def _table(tz: float) -> list[tuple]:
    """(size, pos, rgba) boxes: table top (its surface is the physics table plane), legs, room floor."""
    cx, cy, hx, hy, th, legh = 0.18, -0.05, 0.42, 0.40, 0.025, 0.72
    boxes = [((hx, hy, th), (cx, cy, tz - th), TABLE_RGBA)]
    for sx in (-1, 1):
      for sy in (-1, 1):
        boxes.append(((0.025, 0.025, legh / 2), (cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), tz - 2 * th - legh / 2), LEG_RGBA))
    boxes.append(((3.0, 3.0, 0.01), (cx, cy, tz - 2 * th - legh - 0.01), ROOM_RGBA))
    return boxes

  def _render(self, r: mujoco.Renderer, camera) -> np.ndarray:
    r.update_scene(self.world.data, camera=camera, scene_option=self.vopt)
    scn = r.scene
    if camera == "wrist":
      self.world.hide_wrist_housing(scn)
    for size, pos, rgba in self.table_geoms:
      if scn.ngeom >= scn.maxgeom:
        break
      mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array(size), np.array(pos),
                          np.eye(3).flatten(), np.array(rgba, np.float32))
      scn.ngeom += 1
    return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

  def say(self, msg: str) -> None:
    print(msg, flush=True)
    s = msg.strip().removeprefix("[pipeline]").strip()
    if s:
      self.caption = s[:95]

  def step(self) -> None:
    self._step()
    self.n += 1
    if self.n % self.every == 0:
      self.write(self.frame())

  def _project_head(self, p) -> tuple[int, int] | None:
    R, t, f, _, cx, cy = self.world.camera_model("head")
    pc = R.T @ (np.asarray(p, float) - t)
    if pc[2] <= 0.01:
      return None
    s = self.hw / (2 * cx)  # camera_model is for the 1280x720 head image
    u, v = (f * pc[0] / pc[2] + cx) * s, (f * pc[1] / pc[2] + cy) * s
    return int(round(u - self.crop[0])), int(round(v - self.crop[1]))

  def frame(self) -> np.ndarray:
    d = self.world.data
    img = self._render(self.overview, self.cam)
    full = self._render(self.head, "head")
    self.crop = (0, 0)
    c = self._project_head(WORK_CENTER) or (self.hw // 2, self.hh // 2)
    x = int(np.clip(c[0] - INSET_W // 2, 0, self.hw - INSET_W))
    y = int(np.clip(c[1] - INSET_H // 2, 0, self.hh - INSET_H))
    self.crop = (x, y)
    inset = np.ascontiguousarray(full[y:y + INSET_H, x:x + INSET_W])
    latest = {}
    for e in getattr(self.per, "log", []):
      latest[e["label"]] = e
    for e in latest.values():
      pt = self._project_head(e["truth"])
      if pt:
        cv2.circle(inset, pt, 7, (60, 220, 60), 2, cv2.LINE_AA)
      if e["meas"] is not None:
        pm = self._project_head(e["meas"])
        if pm:
          col = (40, 40, 255) if e["kind"] != "outlier" else (0, 140, 255)
          cv2.drawMarker(inset, pm, col, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
          if pt:
            cv2.line(inset, pt, pm, col, 1, cv2.LINE_AA)
      elif pt:
        cv2.drawMarker(inset, pt, (160, 160, 160), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    if isinstance(self.per, TrackedPerception):  # the Kalman estimate of every tracked object
      for nt in self.per.names.values():
        for tr in nt.tracks:
          pk = self._project_head(tr.x[:3])
          if pk:
            cv2.drawMarker(inset, pk, (230, 160, 40), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
    legend = "o truth  + noisy meas.  <> Kalman" if isinstance(self.per, TrackedPerception) else \
        "head camera (zoom)   o truth   + noisy measurement"
    self._paste(img, inset, W - INSET_W - 16, ("head cam  " if "Kalman" in legend else "") + legend)
    self._paste(img, self._render(self.wrist, "wrist"), 16, "wrist camera (on the gripper)")
    band = img[:92].copy()
    cv2.rectangle(band, (0, 0), (W, 92), (20, 20, 20), -1)
    img[:92] = cv2.addWeighted(img[:92], 0.35, band, 0.65, 0)
    cv2.putText(img, self.title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"t = {d.time:5.1f} s   {self.caption}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (210, 210, 210), 1, cv2.LINE_AA)
    return img

  @staticmethod
  def _paste(img: np.ndarray, inset: np.ndarray, x0: int, label: str) -> None:
    """Framed inset with a caption bar, bottom-aligned at x0."""
    y0 = H - INSET_H - 16
    cv2.rectangle(img, (x0 - 3, y0 - 26), (x0 + INSET_W + 2, y0 + INSET_H + 2), (30, 30, 30), -1)
    img[y0:y0 + INSET_H, x0:x0 + INSET_W] = inset
    cv2.putText(img, label, (x0 + 4, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

  def write(self, img: np.ndarray) -> None:
    if self.ffmpeg is not None:
      self.ffmpeg.stdin.write(img.tobytes())

  def close(self, hold_s: float, fps: int) -> None:
    last = self.frame()
    for _ in range(int(hold_s * fps)):
      self.write(last)
    self.ffmpeg.stdin.close()
    self.ffmpeg.wait()
    self.overview.close()
    self.head.close()
    self.wrist.close()


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--level", type=float, default=1.0, help="reality-gap level: 0 exact sim, 1 realistic, 2 pessimistic")
  p.add_argument("--seed", type=int, default=0, help="noise realisation")
  p.add_argument("--scene", default="stack", choices=["stack", "tower", "table", "on_box"])
  p.add_argument("--steps", nargs="+", default=["red cube>white box", "black cube>white box", "white cube>white box"],
                 help='pick-and-place steps "object>target"')
  p.add_argument("--speed", type=float, default=0.5, help="RL travel speed 0.05-1")
  p.add_argument("--fps", type=int, default=30)
  p.add_argument("--playback", type=float, default=1.5, help="video speed vs. sim time (1 = real time)")
  p.add_argument("--tracker", action="store_true",
                 help="Kalman object tracker on top of the noisy perception: the arm parks only when it is unsure")
  p.add_argument("--no-clear-view", action="store_true", help="do not park the arm out of view before measuring")
  p.add_argument("--view", type=float, nargs=3, default=(0.66, 225.0, -30.0), metavar=("DIST", "AZIM", "ELEV"),
                 help="overview camera")
  p.add_argument("--still", type=Path, default=None, help="write the first frame as PNG and exit (to frame the shot)")
  p.add_argument("--out", type=Path, default=None)
  args = p.parse_args()
  if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found")

  steps = [dict(zip(("pick", "place_on"), (s.strip() for s in st.split(">")), strict=True)) for st in args.steps]
  names = list(dict.fromkeys(n for s in steps for n in (s["pick"], s["place_on"])))
  out = args.out or so101.PROJECT_ROOT / "videos" / f"sim_reality_l{args.level:g}_s{args.seed}{'_kf' if args.tracker else ''}.mp4"
  out.parent.mkdir(parents=True, exist_ok=True)

  world = make_world(scene(args.scene), args.seed, args.level)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=POLICY)
  arm = AgentArm(plant, ctrl, SafeTargets(), world.cfg.table_z)
  per = wrap_perception(SimPerception(world), args.seed, args.level, arm_points=arm.arm_points)
  if args.tracker:
    per = TrackedPerception(per, names, arm_points=arm.arm_points)

  sim_dt = world.substeps * world.model.opt.timestep  # 0.02 s per step_physics call
  every = max(1, int(round(args.playback / (sim_dt * args.fps))))
  title = ("Exact simulation (no noise)" if args.level <= 0 else
           f"Sim-to-real noise level {args.level:g}: servo noise + calib. bias, mass/friction, camera noise")
  if args.tracker:
    title = f"Sim-to-real noise level {args.level:g} + Kalman object tracker" if args.level > 0 else title + " + Kalman tracker"
  rec = Recorder(world, per, None if args.still else out, args.fps, every, args.level, title, tuple(args.view))
  if args.still:
    cv2.imwrite(str(args.still), rec.frame())
    return
  rec.caption = "  ".join(f"{s['pick']} -> {s['place_on']}" for s in steps)
  for _ in range(int(1.0 / sim_dt)):  # a short still intro
    world.step_physics()
  try:
    pipe = Pipeline(NoLLM(), arm, per, clear_view=not args.no_clear_view, log=rec.say, cache=None,
                    handoff="local-first", bc=BcGrasp.load(DEFAULT_MODEL))
    res = pipe.run("record", names, steps)
    rec.caption = "SUCCESS: all steps placed" if res["success"] else \
        f"stopped: {res['steps'][-1].get('result', '?') if res['steps'] else '?'}"
    print(f"[record] result: success={res['success']} steps={[s.get('result') for s in res['steps']]}")
  finally:
    rec.close(2.0, args.fps)
    world.close_renderers()
  print(f"[record] video -> {out}  ({rec.n * sim_dt:.1f} s sim, playback x{every * sim_dt * args.fps:.2f})")


if __name__ == "__main__":
  main()
