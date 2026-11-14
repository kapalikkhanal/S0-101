"""Wrist D405 (fixed on the gripper) for the last centimetres: centring and grasp evidence.

The D405's absolute pose in the base frame is not reliable (measured 2026-11-07: the table
appears 21 mm high and tilted ~15 deg with the stored calibration), and its depth is missing on
the black jaws and on the white cube's top face. The jaws themselves are sharp in colour,
and the fixed jaw is rigid relative to the camera. Everything here is therefore *relative*,
in the image: the object's centre against the jaw tips' midpoint, both measured in the same
frame. Only the camera's rough rotation is used to turn a pixel error into a horizontal
correction, and the correction is iterated, so moderate calibration error only slows
convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from so101_reach.agent.perception import _color_mask, color_words
from so101_reach.sim import so101

REFINED = so101.PROJECT_ROOT / "configs" / "wrist_camera_refined.json"
ORIGINAL = so101.PROJECT_ROOT / "configs" / "wrist_camera_extrinsics.json"


def fit_sphere_points(points,radius):
  """Metric centre of a visible fruit surface; reject sparse, flat or bad fits."""
  from scipy.optimize import least_squares
  p=np.asarray(points)
  if len(p)<80 or np.ptp(np.percentile(p[:,2],[5,95]))<.003:return None
  p=p[::max(1,len(p)//800)]
  init=np.median(p,axis=0)+[0,0,radius]
  fit=least_squares(lambda c:np.linalg.norm(p-c,axis=1)-radius,init,loss="soft_l1",f_scale=.002,max_nfev=80)
  rms=float(np.sqrt(np.mean((np.linalg.norm(p-fit.x,axis=1)-radius)**2)))
  if rms>.0025 or not np.all(np.isfinite(fit.x)):return None
  return fit.x,rms


def jaw_tip_pixels(bgr: np.ndarray,predicted=None) -> dict[str, tuple[float, float]]:
  """Leftmost points of the two large dark jaw silhouettes entering from the right edge:
  'upper' = moving jaw, 'lower' = fixed jaw in this mounting. A tip on the image border is
  clipped and is not returned."""
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  dark = cv2.morphologyEx((hsv[..., 2] < 55).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
  h, w = dark.shape
  blobs = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 3000 and st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] >= w - 5]
  tips = []
  for i in sorted(blobs, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]:
    ys, xs = np.where(lab == i)
    sel = xs <= xs.min() + 6
    tips.append((float(np.median(xs[sel])), float(np.median(ys[sel]))))
  tips.sort(key=lambda t: t[1])
  out = {}
  if len(tips) == 2:
    out = {"upper": tips[0], "lower": tips[1]}
  elif len(tips) == 1:
    out = {"lower" if tips[0][1] > h / 2 else "upper": tips[0]}
  # Dark objects can join a jaw silhouette. Its leftmost pixel then belongs
  # to the object, not the jaw. Reject that association rather than moving
  # the grasp centre to it. measure() already falls back to calibrated tips
  # for missing observations, and the grasp score discounts inferred tips.
  return {k: v for k, v in out.items() if 8 <= v[1] <= h - 8 and v[0] >= 8
          and (predicted is None or k not in predicted or np.linalg.norm(np.subtract(v,predicted[k]))<=40)}


def object_blob(bgr: np.ndarray, name: str, exclude_x: float | None = None) -> dict | None:
  """Largest blob of the colour named in `name` (left of the jaw tips if exclude_x is given)."""
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  masks = [m for m in (_color_mask(hsv, w) for w in name.lower().split()) if m is not None]
  if not masks:
    return None
  mask = np.any(masks, axis=0).astype(np.uint8)
  if exclude_x is not None:
    mask[:, int(exclude_x) + 150:] = 0  # far right: gripper body, cables and background
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
  if n < 2:
    return None
  i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
  if st[i, cv2.CC_STAT_AREA] < 400:
    return None
  x, y, bw, bh = st[i, :4]
  return {"centroid": (float(cen[i][0]), float(cen[i][1])), "area": int(st[i, cv2.CC_STAT_AREA]), "bbox": [int(x), int(y), int(x + bw), int(y + bh)]}


def detect_object(bgr: np.ndarray, name: str, near: np.ndarray, expected_px: float, model=None) -> dict | None:
  """Grounding DINO candidate for `name` whose size matches the expected pixel size (rejects the
  big white box and small glints; measured: a table reflection fooled a colour threshold),
  nearest to `near`; centre refined by the object's (relaxed) colour inside the box."""
  if model is None:
    from so101_reach.real.objects import DinoModel
    model = DinoModel.shared()
  cands = []
  query="fruit" if name.lower().split()[-1] in ("orange","oranges") else name
  for score, box in model.detect(bgr, query, 0.25, 0.25):
    w, h = box[2] - box[0], box[3] - box[1]
    size = float(np.sqrt(max(w * h, 1.0)))
    if not 0.5 * expected_px <= size <= 2.2 * expected_px:
      continue
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    cands.append((float(np.linalg.norm(c - near)), float(score), box, c))
  if not cands:
    return None
  _, score, box, c = min(cands, key=lambda t: t[0])
  x0, y0, x1, y1 = np.clip(np.rint(box).astype(int), 0, [bgr.shape[1], bgr.shape[0]] * 2)
  hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
  words = name.lower().split()
  if "white" in words:
    mask = (hsv[..., 1] <= 90) & (hsv[..., 2] >= 130)
  else:
    ms = [m for m in (_color_mask(hsv, w) for w in color_words(name)) if m is not None]
    mask = np.any(ms, axis=0) if ms else np.ones(hsv.shape[:2], bool)
  if mask.mean() >= 0.3:
    ys, xs = np.nonzero(mask)
    c = np.array([x0 + xs.mean(), y0 + ys.mean()])
  return {"centroid": (float(c[0]), float(c[1])), "area": int(mask.sum()), "bbox": [int(x0), int(y0), int(x1), int(y1)],
          "score": score, "candidates": len(cands)}


@dataclass
class ServoStep:
  error_px: tuple[float, float]
  correction_m: np.ndarray
  tips: dict
  blob: dict | None
  target_px: np.ndarray | None = None


class WristServo:
  def __init__(self, cam, kin):
    self.cam, self.kin = cam, kin
    k = cam.intrinsics
    self.f = np.array([k.fx, k.fy]); self.c = np.array([k.cx, k.cy])
    calib = json.loads((REFINED if REFINED.exists() else ORIGINAL).read_text())
    self.R_gc = np.array(calib["R_gripper_cam"]); self.t_gc = np.array(calib["t_gripper_cam"])
    m = kin.m
    self.gr = m.body("gripper").id
    self.tip_ids = {"lower": [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)],
                    "upper": [m.geom(f"moving_jaw_sph_tip{i}").id for i in (1, 2, 3)]}

  def fresh(self):
    """Newest frame: the stream is idle during long moves, so drain buffered (old) frames."""
    fr = None
    for _ in range(5):
      fr = self.cam.read()
    self.last_frame = fr.color  # shown in the live preview
    return fr

  def _tip_cam(self, q6, side) -> np.ndarray:
    self.kin.fk(q6)
    d = self.kin.d
    Rg = d.xmat[self.gr].reshape(3, 3)
    p = np.mean([d.geom_xpos[g] for g in self.tip_ids[side]], axis=0)
    return self.R_gc.T @ (Rg.T @ (p - d.xpos[self.gr]) - self.t_gc)

  def grip_pixel(self, q6, tips_obs: dict, bias_m=0.0) -> np.ndarray:
    """Image position of the grip point (15 mm inside the tip midpoint, where a grasped object's
    centre sits), from the refined calibration, shifted by the observed-minus-predicted tip
    residual so it stays anchored to the jaws actually seen."""
    from so101_reach.agent.arm import GRIP_IN_FROM_TIPS
    self.kin.fk(q6)
    d = self.kin.d
    Rg = d.xmat[self.gr].reshape(3, 3)
    tips = [np.mean([d.geom_xpos[g] for g in self.tip_ids[s]], axis=0) for s in ("upper", "lower")]
    a = d.site_xpos[self.kin.site] - d.xpos[self.gr]
    a /= np.linalg.norm(a)
    gp = (tips[0] + tips[1]) / 2 - (GRIP_IN_FROM_TIPS+bias_m) * a
    c = self.R_gc.T @ (Rg.T @ (gp - d.xpos[self.gr]) - self.t_gc)
    px = self.f * c[:2] / c[2] + self.c
    pred = self.predicted_tips(q6)
    res = [np.subtract(tips_obs[s], pred[s]) for s in ("upper", "lower") if s in tips_obs and tips_obs[s] != pred[s]]
    return px + (np.mean(res, axis=0) if res else 0.0)

  def fruit_depth(self,q6,frame,blob,width_m):
    """Fruit centre relative to jaws in wrist coordinates, independent of the base pose."""
    x0,y0,x1,y1=blob['bbox'];bw=x1-x0;bh=y1-y0
    yy,xx=np.mgrid[int(y0+.18*bh):int(y1-.18*bh):2,int(x0+.18*bw):int(x1-.18*bw):2]
    z=frame.depth[yy,xx];ok=(z>.05)&(z<.25)
    if np.count_nonzero(ok)<80:return None
    median=float(np.median(z[ok]));ok &= np.abs(z-median)<.6*width_m
    p=np.c_[(xx[ok]-self.c[0])/self.f[0]*z[ok],(yy[ok]-self.c[1])/self.f[1]*z[ok],z[ok]]
    fit=fit_sphere_points(p,width_m/2)
    if fit is None:return None
    center,rms=fit;tip=np.mean([self._tip_cam(q6,s) for s in ('upper','lower')],axis=0)
    _,a=self.kin.fk(q6);rg=self.kin.d.xmat[self.gr].reshape(3,3)
    axis=self.R_gc.T@rg.T@a
    return {'center_cam_m':center,'rms_mm':1000*rms,'insertion_mm':1000*float((tip-center)@axis),
            'center_px':self.f*center[:2]/center[2]+self.c}

  def predicted_tips(self, q6) -> dict[str, tuple[float, float]]:
    out = {}
    for side in ("upper", "lower"):
      c = self._tip_cam(q6, side)
      out[side] = tuple((self.f * c[:2] / c[2] + self.c).tolist())
    return out

  def measure(self, q6, name: str, frame=None, width_m: float | None = None) -> ServoStep | None:
    """Pixel error of the object's centre from the grasp centre (jaw-tip midpoint), and the
    horizontal base-frame correction it implies (at the jaw-tip depth)."""
    fr = frame if frame is not None else self.fresh()
    pred = self.predicted_tips(q6)
    tips = jaw_tip_pixels(fr.color,predicted=pred)
    for side in ("upper", "lower"):
      tips.setdefault(side, pred[side])  # out of view (wide opening): refined-calibration prediction
    # Target: the tip midpoint (grasped trials 17/23). The grip-point projection (grip_pixel) was
    # tried and moved good grasps by ~11 mm because the object's visible side faces bias its centre.
    mid = (np.array(tips["upper"]) + np.array(tips["lower"])) / 2
    z = float(np.mean([self._tip_cam(q6, s)[2] for s in ("upper", "lower")]))
    if width_m is not None:
      blob = detect_object(fr.color, name, mid, width_m * self.f[0] / max(z, 0.03))
    else:
      blob = object_blob(fr.color, name, exclude_x=max(tips["upper"][0], tips["lower"][0]))
    if blob is None:
      return None
    e = np.array(blob["centroid"]) - mid
    d_cam = np.array([e[0] * z / self.f[0], e[1] * z / self.f[1], 0.0])
    self.kin.fk(q6)
    Rg = self.kin.d.xmat[self.gr].reshape(3, 3)
    d_base = Rg @ (self.R_gc @ d_cam)
    corr = np.array([d_base[0], d_base[1], 0.0])
    return ServoStep((float(e[0]), float(e[1])), corr, {k: tuple(map(float, v)) for k, v in tips.items()}, blob, mid)

  def px_to_base(self, q6, v_px) -> np.ndarray:
    """Horizontal base-frame displacement for an image vector at the jaw-tip depth."""
    z = float(np.mean([self._tip_cam(q6, s)[2] for s in ("upper", "lower")]))
    d_cam = np.array([v_px[0] * z / self.f[0], v_px[1] * z / self.f[1], 0.0])
    self.kin.fk(q6)
    d = self.kin.d.xmat[self.gr].reshape(3, 3) @ (self.R_gc @ d_cam)
    return np.array([d[0], d[1]])

  def offset_pixels(self,q6,point_px,offset_world):
    """Project a requested grasp-point offset; the servo must not undo it."""
    z=float(np.mean([self._tip_cam(q6,s)[2] for s in ('upper','lower')]))
    self.kin.fk(q6);rg=self.kin.d.xmat[self.gr].reshape(3,3)
    c=np.r_[(np.asarray(point_px)-self.c)/self.f*z,z]
    shifted=c+(rg@self.R_gc).T@np.asarray(offset_world)
    if shifted[2]<=.03:return np.zeros(2)
    return self.f*shifted[:2]/shifted[2]+self.c-np.asarray(point_px)

  def held_blob(self, name: str, tips: dict, frame=None) -> dict | None:
    fr = frame if frame is not None else self.fresh()
    return object_blob(fr.color, name, exclude_x=max(tips["upper"][0], tips["lower"][0]))

  @staticmethod
  def annotate(bgr, step: ServoStep, path: Path):
    img = bgr.copy()
    for side, p in step.tips.items():
      cv2.circle(img, tuple(int(v) for v in p), 6, (255, 0, 255), 2)
    mid = step.target_px if step.target_px is not None else (np.array(step.tips["upper"]) + np.array(step.tips["lower"])) / 2
    cv2.drawMarker(img, tuple(int(v) for v in mid), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    if step.blob:
      cv2.rectangle(img, tuple(step.blob["bbox"][:2]), tuple(step.blob["bbox"][2:]), (0, 200, 255), 2)
      cv2.circle(img, tuple(int(v) for v in step.blob["centroid"]), 5, (0, 200, 255), -1)
      if 'grasp_point_px' in step.blob:
        cv2.drawMarker(img,tuple(int(v) for v in step.blob['grasp_point_px']),(255,255,0),cv2.MARKER_CROSS,16,2)
    cv2.imwrite(str(path), img)
