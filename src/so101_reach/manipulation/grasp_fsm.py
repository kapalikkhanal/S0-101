"""Bounded grasp state machine for the wrist-camera path (real SO-101).

One pass, no fixed up/side/down correction cycles:

  ALIGN    one wrist observation above the object; at most one lateral move; set the opening
  DESCEND  one checked straight descent. A stall under load in the contact zone ends it and
           no further downward command is issued.
  ASSESS   grasp score from measured wrist/arm geometry, plus independent constraints
  CLOSE    score >= threshold and constraints pass. No further centring is attempted.
  CORRECT  only when the jaws miss the capture region: one small move, then ASSESS again
  RETRY    after an empty close: one nearby retry (withdraw, re-observe, descend). The arm does
           not go home or to the observation pose unless the object is lost or moved substantially.

The grasp score is NOT a probability. Each component is a piecewise-linear ramp over a measured
quantity, with end points taken from the hardware notes cited below. They have not been calibrated
against labelled outcomes. An offline check of 20 logged real cube closes showed that the
closing-axis image span, the along-pad offset and a wrist-depth tip height do not separate caught
from missed closes, so the score is a gate on necessary conditions, not a predictor.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from so101_reach.agent.arm import GRIP_IN_FROM_TIPS
from so101_reach.manipulation.motion import MotionInterrupted, TableContact

SCORE_THRESHOLD = 0.60
# Coverage: physical clearance between the nearer jaw and the object across the jaws,
# (gap - width) / 2 - |object centre offset|, the offset measured on the object's own image scale.
# Below 0 a jaw tip lands on the object. The scale includes visible side faces (~10-15 % for the
# cube), which understates the offset slightly, hence 0 at -2 mm and 1 at +1 mm. All 21 logged
# real cube closes (caught or missed) lie at +3.4 to +9.7 mm.
COVER_FULL_MM, COVER_ZERO_MM = 1.0, -2.0
# Pad overlap across the closing axis: pads extend 10 mm either side of the grip point
# (fixed_jaw_box6/7). Visible side faces bias the measured centroid by up to ~11 mm.
PAD_HALF_MM = 10.0
PAD_FULL_FRAC, PAD_ZERO_FRAC = 0.60, 0.20
# Opening: margin of the jaw gap over the object width.
OPEN_MARGIN_FULL_MM, OPEN_MARGIN_ZERO_MM = 6.0, 1.0
# Engagement: modelled object top minus the lowest jaw tip at the measured joints. The verified
# real grasp of 2026-11-07 (cube_grasp_revision_real_1) closed with 5.3 mm. The hinge effect
# (34-48 mm openings held the 34 mm cube 4/4, 51-60 mm 0/8: the moving tip rises as it opens) is
# handled where the opening is commanded (MAX_OPEN_GAP_M), not in the score, because wider
# objects (fruit) need wider openings and the model's jaw heights are not accurate enough to
# gate on the higher jaw.
ENGAGE_FULL_MM, ENGAGE_ZERO_MM = 6.0, 1.0
# Measurement quality.
AGE_FULL_S, AGE_ZERO_S = 0.5, 2.0
DET_FULL, DET_LOW = 0.40, 0.25

ALIGN_TOL_PX = 25.0  # coarse alignment target; the final check is the capture region
ALIGN_MAX_M = 0.020  # one alignment move at most 2 cm
ALIGN_DISAGREE_M = 0.035  # wrist vs head larger than this: wrong object or object moved
CORRECT_MAX_M = 0.010  # the one capture-region correction at grasp height
PAD_TARGET_MM = 6.0  # along-pad correction aims to leave at most this offset (bias tolerance)
RETRY_MOVED_M = 0.025  # after an empty close, larger displacement means a full re-approach
MAX_OPEN_GAP_M = 0.049
DESCENT_TOL_M = 0.030  # P-servo friction stops the arm 6-17 mm short near the table
WITHDRAW_M = 0.004  # "withdraw slightly" before a lateral move when the jaws may touch


class ObjectLost(MotionInterrupted):
  """The wrist cannot find the object, or it moved substantially: the caller may re-approach."""


def ramp(x, zero, full):
  """0 at `zero`, 1 at `full`, linear between (either direction)."""
  if full == zero:
    return float(x >= full)
  return float(np.clip((x - zero) / (full - zero), 0.0, 1.0))


def pad_overlap_fraction(offset_mm, object_len_mm, pad_half_mm=PAD_HALF_MM):
  """Fraction of the usable pad length that lies on the object, for a pad centre `offset_mm`
  from the object centre along the pad direction."""
  half = object_len_mm / 2
  overlap = min(offset_mm + pad_half_mm, half) - max(offset_mm - pad_half_mm, -half)
  return float(max(0.0, overlap) / max(1e-6, min(2 * pad_half_mm, object_len_mm)))


def grasp_score(obs: dict) -> dict:
  """Score components in [0, 1] and their product. `obs` keys (mm, s):
  clearance_mm, along_mm, object_len_mm, object_width_mm, gap_mm, engagement_mm, age_s,
  detector_score, tips_observed (0-2), bbox_clipped."""
  coverage = ramp(obs["clearance_mm"], COVER_ZERO_MM, COVER_FULL_MM)
  pad = ramp(pad_overlap_fraction(abs(obs["along_mm"]), obs["object_len_mm"]), PAD_ZERO_FRAC, PAD_FULL_FRAC)
  opening = ramp(obs["gap_mm"] - obs["object_width_mm"], OPEN_MARGIN_ZERO_MM, OPEN_MARGIN_FULL_MM)
  engagement = ramp(obs["engagement_mm"], ENGAGE_ZERO_MM, ENGAGE_FULL_MM)
  det = obs.get("detector_score")
  quality = (ramp(obs["age_s"], AGE_ZERO_S, AGE_FULL_S)
             * (1.0 if det is None else 0.5 + 0.5 * ramp(det, DET_LOW, DET_FULL))
             * {2: 1.0, 1: 0.85}.get(int(obs.get("tips_observed", 2)), 0.7)
             * (0.8 if obs.get("bbox_clipped") else 1.0))
  comps = {"coverage": coverage, "pad_overlap": pad, "opening": opening, "engagement": engagement, "quality": quality}
  return {"score": float(np.prod(list(comps.values()))), "components": comps}


def decide(score: dict, constraints: dict, contact_stop: bool, corrections_left: int, threshold=SCORE_THRESHOLD):
  """Decision for one assessment: close | correct | redescend | fail, with a reason.

  Constraints are independent of the score: a high score never overrides them, and they never
  ask for more centring."""
  bad = [k for k, ok in constraints.items() if not ok]
  if bad:
    return "fail", "constraint not met: " + ", ".join(bad)
  if score["score"] >= threshold:
    return "close", f"score {score['score']:.2f} >= {threshold:.2f}"
  c = score["components"]
  weakest = min(c, key=c.get)
  if corrections_left <= 0:
    return "fail", f"score {score['score']:.2f} < {threshold:.2f} after the allowed correction (weakest: {weakest} {c[weakest]:.2f})"
  if weakest in ("coverage", "pad_overlap"):
    return "correct", f"jaws miss the capture region ({weakest} {c[weakest]:.2f})"
  if weakest == "engagement":
    if contact_stop:
      return "fail", (f"descent stopped by contact above the capture height (engagement {c['engagement']:.2f}); "
                      "no further downward command")
    return "redescend", f"jaws above the capture height (engagement {c['engagement']:.2f}); one more descent"
  return "fail", f"score {score['score']:.2f} limited by {weakest} {c[weakest]:.2f}, which a move cannot fix"


class GraspFSM:
  def __init__(self, ex, threshold=SCORE_THRESHOLD, local_retries=1, stats=None):
    self.ex, self.arm, self.motion, self.guard, self.wrist = ex, ex.arm, ex.motion, ex.motion.guard, ex.wrist
    self.threshold, self.local_retries = threshold, local_retries
    self.stats = stats if stats is not None else {
      "controller": "grasp_fsm", "threshold": threshold, "closes": 0, "corrections": 0, "align_moves": 0,
      "contact_stops": 0, "local_retries": 0, "re_approaches": 0, "assessments": [], "states": []}
    self.round = 0

  # --- helpers ----------------------------------------------------------------------------
  def state(self, name, **data):
    self.stats["states"].append(name)
    self.ex.set_phase(name, **data)

  def gp(self):
    return self.arm.geom(self.arm.q6()[0])["gp"].copy()

  def measure(self):
    """Fresh wrist frame and the object measurement in it (None if not visible)."""
    fr = self.wrist.fresh()
    q, _ = self.arm.q6()
    st = self.wrist.measure(q, self.step.pick, frame=fr, width_m=float(min(self.ob.size[:2])))
    return fr, q, st

  def _save(self, fr, st, name, lines):
    img = fr.color.copy()
    for i, t in enumerate(lines):
      cv2.putText(img, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
    self.wrist.annotate(img, st, self.ex.log_dir / name)

  def set_contact_region(self, xy):
    """The jaws may touch the object (and the table next to it) where the wrist found it."""
    from dataclasses import replace
    g = self.guard
    g.contact_override[self.step.pick] = replace(self.ob, center=np.r_[xy, self.ob.center[2]])
    if g.table_touch_center is not None:
      g.table_touch_center = np.array(xy, float)

  # --- states -----------------------------------------------------------------------------
  def run(self, step, ob, jaw, xy):
    """Returns the grip point after a close that made contact. Raises ObjectLost (the caller may
    re-approach) or MotionInterrupted(hard_stop=True) with a specific diagnosis."""
    self.step, self.ob, self.jaw = step, ob, jaw
    self.head_xy = np.array(xy, float)
    ex = self.ex
    ex._wrist_obj_size, ex._wrist_top_z, ex.wrist_width = ob.size, ob.top_z, None
    xy = self.align(np.array(xy, float))
    for attempt in range(self.local_retries + 1):
      corrections_left = 1
      target = self.plan_descent(xy)
      contact = self.descend(target)
      while True:
        a = self.assess(contact, corrections_left)
        if a["decision"] == "close":
          break
        if a["decision"] in ("correct", "redescend"):
          corrections_left -= 1
          xy, target, contact = self.correct(a, xy, target)
          continue
        self.withdraw()
        raise MotionInterrupted(f"grasp not attempted: {a['reason']}", hard_stop=True)
      if self.close():
        return self.gp()
      if attempt == self.local_retries:
        self.withdraw()
        raise MotionInterrupted(f"{self.stats['closes']} empty closes; stopped above the object", hard_stop=True)
      xy = self.local_retry(xy)
    raise AssertionError("unreachable")

  def align(self, xy):
    ex, g = self.ex, self.guard
    z = self.ob.top_z + 0.012 + GRIP_IN_FROM_TIPS  # jaw tips ~12 mm above the object's top
    ex._align_z = z
    self.state("align")
    g.move_peak_deviation = 0.0
    g.dev_samples = []
    self.motion.goto([*xy, z], yaw=self.jaw, tolerance=0.015)
    xy = self._observe_and_shift(xy, z, "wrist_align_0.jpg", from_head=True)
    self._set_opening()
    self.set_contact_region(xy)
    return xy

  def _observe_and_shift(self, xy, z, image, from_head):
    """One wrist observation above the object and at most one lateral move toward it."""
    ex = self.ex
    ex.wait(0.15)
    fr, q, st = self.measure()
    if st is None:
      raise ObjectLost("wrist camera does not see the object above it")
    shift = self.wrist.offset_pixels(q, np.array(st.blob["centroid"]), np.r_[ex.grasp_shift_xy, 0.0])
    err = np.array(st.error_px, float) + shift
    corr = self.wrist.px_to_base(q, err)
    self._save(fr, st, image, [f"align error {np.hypot(*err):.0f}px"])
    self._wrist_width(fr, q, st)
    moved = False
    if np.hypot(*err) > ALIGN_TOL_PX:
      c = corr * min(1.0, ALIGN_MAX_M / max(np.linalg.norm(corr), 1e-9))
      ref = self.head_xy if from_head else xy
      limit = ALIGN_DISAGREE_M if from_head else RETRY_MOVED_M
      if np.linalg.norm(xy + corr - ref) > limit:
        raise ObjectLost(f"object is {1000 * np.linalg.norm(xy + corr - ref):.0f} mm from where it was expected")
      xy = xy + c
      self.motion.goto([*xy, z], yaw=self.jaw, tolerance=0.015)
      self.stats["align_moves"] += 1
      moved = True
    ex.wrist_seen_at = ex.source.now()
    ex.event("wrist_align", error_px=err, correction_mm=1000 * corr, moved=moved, tips=st.tips, blob=st.blob)
    return xy

  def _wrist_width(self, fr, q, st):
    """Object width across the jaws from the spread of its own pixels (the detector box includes
    side faces). Used only to widen the opening when it reads clearly larger than the head."""
    ex = self.ex
    b, tips = st.blob["bbox"], st.tips
    axis = np.subtract(tips["lower"], tips["upper"])
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    crop = cv2.cvtColor(fr.color[b[1]:b[3], b[0]:b[2]], cv2.COLOR_BGR2HSV)
    obj = (crop[..., 1] <= 90) & (crop[..., 2] >= 130) if "white" in self.step.pick.lower().split() else np.ones(crop.shape[:2], bool)
    ys, xs = np.nonzero(obj)
    if len(xs) > 200:
      proj = xs * axis[0] + ys * axis[1]
      ext = float(np.percentile(proj, 95) - np.percentile(proj, 5)) / 0.9
    else:
      ext = abs(axis[0]) * (b[2] - b[0]) + abs(axis[1]) * (b[3] - b[1])
    zt = float(np.mean([self.wrist._tip_cam(q, s)[2] for s in ("upper", "lower")])) + 0.012
    ex.wrist_width = float(ext * zt / self.wrist.f[1])

  def _set_opening(self):
    ex = self.ex
    w_head = float(min(self.ob.size[:2]))
    w = w_head if ex.wrist_width is None or ex.wrist_width <= w_head + 0.006 else ex.wrist_width
    closing = self.arm.geom(self.arm.q6()[0])["axis"][:2]
    closing = closing / max(np.linalg.norm(closing), 1e-9)
    cross = abs(float(ex.grasp_shift_xy @ closing))
    need = w + 0.012 + 2 * cross
    # The cap keeps the hinged moving jaw low; never below a 4 mm margin over the object.
    gap = max(min(need, MAX_OPEN_GAP_M), w + 0.004)
    self.opening = gap
    self.arm.gripper(self.arm.percent_for_gap(gap), "medium")
    ex.event("wrist_aligned", wrist_width_mm=1000 * (ex.wrist_width or 0), head_width_mm=1000 * w_head,
             opening_gap_mm=1000 * gap, offset_from_head_mm=1000 * (self.gp()[:2] - self.head_xy))

  def plan_descent(self, xy):
    """Lowest collision-free grip point over xy at the present opening (the hinged jaw's lowest
    point depends on it), with the clearance margin taken from the measured tracking error."""
    from so101_reach.manipulation.skills import GRASP_PLANNING_MARGIN
    g, arm, ob = self.guard, self.arm, self.ob
    q, _ = arm.q6()
    low = ob.support_z + 0.55 * ob.size[2]
    dev = g.move_peak_deviation
    if len(g.dev_samples) >= 10:
      dev = float(np.percentile(g.dev_samples, 75))
    margin = max(GRASP_PLANNING_MARGIN, g.margin + dev)
    old, reason = g.margin, None
    try:
      g.margin = margin
      for z in np.arange(low, ob.top_z + 0.005, 0.002):
        p = np.array([*xy, z])
        qs, err = arm._pose(p, 0, self.jaw, q)
        reason = arm._check(qs) if err < 0.003 else f"IK error {1000 * err:.1f} mm"
        if reason is None:
          self.ex.event("grasp_height_planned", target_m=p, margin_mm=1000 * margin,
                        engagement_planned_mm=1000 * (ob.top_z - arm.geom(qs)["lowest"]))
          return p
    finally:
      g.margin = old
    raise MotionInterrupted(f"no collision-free grasp pose with this opening (last: {reason})", hard_stop=True)

  def descend(self, target):
    """One checked straight descent with the contact monitor armed only for this move."""
    g = self.guard
    self.state("descend", target_m=target)
    g.arm_descent(self.ob.top_z)
    contact = False
    try:
      self.motion.goto(target, yaw=self.jaw, tolerance=DESCENT_TOL_M)
    except TableContact as e:
      contact = True
      self.stats["contact_stops"] += 1
      self.ex.event("contact_stop", reason=str(e), **(g.descent or {}).get("stop", {}) or {})
    except MotionInterrupted as e:
      if not str(e).startswith("stopped: final grip-point error"):
        raise
      self.ex.event("descent_short", reason=str(e))
    finally:
      self.last_descent = g.disarm_descent()
    return contact

  def observe_capture(self, tag):
    """Grasp-height measurement: object extent between the jaw tips (closing axis), along-pad
    offset, opening, engagement and measurement quality."""
    ex, arm, ob = self.ex, self.arm, self.ob
    fr, q, st = self.measure()
    if st is None:
      return None, None, None
    up, lo = np.array(st.tips["upper"], float), np.array(st.tips["lower"], float)
    axis = (lo - up) / max(np.linalg.norm(lo - up), 1e-9)
    span = float(np.linalg.norm(lo - up))
    along = np.array([axis[1], -axis[0]])
    goal = self.wrist.grip_pixel(q, st.tips, bias_m=getattr(ex, "longitudinal_bias_m", 0.0)) \
      if hasattr(self.wrist, "grip_pixel") else (up + lo) / 2
    center = np.array(st.blob["centroid"], float)
    b = st.blob["bbox"]
    corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]], float)
    proj = (corners - up) @ axis
    clear_px = float(min(proj.min(), span - proj.max()))
    off_px = float((proj.min() + proj.max()) / 2 - span / 2)
    along_px = float((center - goal) @ along)
    to_mm = lambda v: 1000 * float(np.linalg.norm(self.wrist.px_to_base(q, v)))
    geom = arm.geom(q)
    pred = self.wrist.predicted_tips(q) if hasattr(self.wrist, "predicted_tips") else {}
    h, w = fr.color.shape[:2] if getattr(fr, "color", None) is not None else (480, 848)
    stamp = getattr(fr, "stamp", None)
    width_mm, gap_mm = 1000 * float(min(ob.size[:2])), 1000 * float(geom["gap"])
    # Object-centred geometry. The visible top face is ~20 mm nearer the camera than the jaw tips,
    # so it is magnified relative to the tip plane (5.7 vs ~3.9 px/mm in trial_1). Comparing its
    # box edges with the tip pixels overstated the overlap: -3.9 mm for a cube with ~4.5 mm of real
    # clearance (trial_2). Measure the offsets on the object's own pixel scale; compare the
    # across-jaw offset with the physical slack (gap - width) / 2.
    scale = max(float(proj.max() - proj.min()), 1.0) / max(width_mm, 1.0)  # px per mm on the object
    offset_mm, along_mm = off_px / scale, along_px / scale
    obs = {
      "clearance_mm": (gap_mm - width_mm) / 2 - abs(offset_mm),
      "closing_offset_mm": offset_mm,
      "along_mm": along_mm,
      "edge_clearance_mm": float(np.sign(clear_px)) * to_mm(abs(clear_px) * axis),  # former metric, for comparison
      "object_scale_px_per_mm": scale,
      "object_len_mm": 1000 * float(max(ob.size[:2])), "object_width_mm": width_mm,
      "gap_mm": gap_mm,
      "engagement_mm": 1000 * float(ob.top_z - geom["lowest"]),
      # Logged for calibration only: in the verified grasp of cube_grasp_revision_real_1 the model
      # put the moving jaw ~7 mm above the cube top, so it cannot gate the score.
      "engagement_higher_jaw_mm": 1000 * float(ob.top_z - geom.get("jaw_low_max", geom["lowest"])),
      "age_s": 0.0 if stamp is None else max(0.0, time.time() - stamp),
      "detector_score": st.blob.get("score"),
      "tips_observed": sum(1 for s in ("upper", "lower") if s in pred and tuple(st.tips[s]) != tuple(pred[s])) if pred else 2,
      "bbox_clipped": bool(b[0] <= 2 or b[1] <= 2 or b[2] >= w - 3 or b[3] >= h - 3),
      "closing_offset_px": off_px, "along_px": along_px, "clearance_px": clear_px, "jaw_span_px": span,
      "object_span_px": [float(proj.min()), float(proj.max())],
      "lowest_tip_above_table_mm": 1000 * float(geom["lowest"] - ex.source.table_z),
    }
    # Correction toward the capture region: centre across the jaws; along the pads only the
    # part beyond PAD_TARGET_MM (the measured centroid carries a side-face bias).
    # Directions from the camera model, magnitudes from the object-scale offsets.
    unit = lambda v: (lambda d: d / max(np.linalg.norm(d), 1e-12))(np.asarray(self.wrist.px_to_base(q, v), float))
    c = np.zeros(2)
    if abs(off_px) > 1e-6:
      c += unit(off_px * axis) * abs(offset_mm) / 1000
    if abs(along_px) > 1e-6:
      c += unit(along_px * along) * max(0.0, abs(along_mm) - PAD_TARGET_MM) / 1000
    c = c * min(1.0, CORRECT_MAX_M / max(np.linalg.norm(c), 1e-9))
    st.target_px = goal
    return obs, np.asarray(c, float), (fr, st)

  def constraints(self):
    """Independent of the score: no sustained downward effort, arm settled, no hand in view."""
    g, ex = self.guard, self.ex
    out = {}
    base = (self.last_descent or {}).get("base")
    push = g.downward_push() if hasattr(g, "downward_push") else None
    excess = None if push is None or base is None else push - base
    out["effort"] = excess is None or excess < g.descent_push_limit
    if not out["effort"]:
      # The servos are still commanded below the jaws (sag-correction aim held against friction or
      # a surface). Relieve without moving: hold the measured pose with the gravity balance measured
      # at rest before the descent. goal = present would let the arm sag ~12 mm-equivalent
      # (trial_5). Rise slightly only if that did not work.
      ex.event("effort_relief", excess_push_mm=1000 * excess)
      g.hold_balanced((self.last_descent or {}).get("base_offsets", {}))
      ex.wait(0.1)
      excess = g.downward_push() - base
      if excess >= g.descent_push_limit:
        gp = self.gp()
        try:
          self.motion.goto([*gp[:2], gp[2] + WITHDRAW_M], tilt=None, yaw=self.jaw, tolerance=0.015)
          excess = g.downward_push() - base
        except MotionInterrupted as e:
          ex.event("effort_relief_failed", reason=str(e))
      out["effort"] = excess < g.descent_push_limit
    for _ in range(5):  # up to 0.5 s for the arm to come to rest
      a = self.gp()
      ex.wait(0.1)
      out["settled"] = bool(np.linalg.norm(self.gp() - a) < 0.0015)
      if out["settled"]:
        break
    # A hand in view pauses the decision (arm held), as everywhere else; it does not end the grasp.
    limit = getattr(g, "auto_pause_timeout", None) or 60.0
    t0 = ex.source.now()
    while getattr(ex.source.snapshot(), "hands", None) and ex.source.now() - t0 < limit:
      g.pause_reason = "hand in view: grasp decision waits"
      ex.wait(0.2)
    g.pause_reason = None
    out["no_hand"] = not getattr(ex.source.snapshot(), "hands", None)
    self.effort_excess_mm = None if excess is None else 1000 * excess
    return out

  def assess(self, contact, corrections_left):
    ex = self.ex
    self.state("assess")
    self.round += 1
    ex.wait(0.15)
    obs, corr, view = self.observe_capture(self.round)
    if obs is None:
      ex.wait(0.2)  # one re-measure: the detector can miss a single frame
      obs, corr, view = self.observe_capture(self.round)
    if obs is None:
      self.withdraw()
      raise ObjectLost("wrist camera does not see the object at grasp height")
    score = grasp_score(obs)
    cons = self.constraints()
    decision, reason = decide(score, cons, contact, corrections_left, self.threshold)
    rec = {"round": self.round, "score": round(score["score"], 3),
           "components": {k: round(v, 3) for k, v in score["components"].items()},
           "observation": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in obs.items()},
           "constraints": cons, "effort_excess_mm": self.effort_excess_mm, "contact_stop": contact,
           "decision": decision, "reason": reason, "correction_mm": (1000 * corr).round(1).tolist()}
    self.stats["assessments"].append(rec)
    ex.event("grasp_assessment", **rec)
    fr, st = view
    self._save(fr, st, f"wrist_assess_{self.round}.jpg",
               [f"score {score['score']:.2f} -> {decision}",
                " ".join(f"{k[:4]}={v:.2f}" for k, v in score["components"].items())])
    if hasattr(fr, "depth"):
      np.savez_compressed(ex.log_dir / f"wrist_depth_assess_{self.round}.npz", depth=fr.depth, joints=self.arm.q6()[0],
                          tips=np.array([st.tips["upper"], st.tips["lower"]]), bbox=np.array(st.blob["bbox"]),
                          K=np.r_[self.wrist.f, self.wrist.c])
    return {**rec, "correction_m": corr}

  def correct(self, a, xy, target):
    """The one capture-region correction for this descent."""
    self.stats["corrections"] += 1
    self.state("correct", action=a["decision"], correction_mm=a["correction_mm"])
    if a["decision"] == "redescend":
      return xy, target, self.descend(target)
    gp = self.gp()
    c = np.asarray(a["correction_m"], float)
    obs = a["observation"]
    a_jaw_may_be_on_it = a["components"]["coverage"] < 0.5
    touching = a["contact_stop"] or obs["lowest_tip_above_table_mm"] < 3.0
    if a_jaw_may_be_on_it:
      # A jaw tip may be over the object: rise until it clears the top before moving sideways.
      lift = min(0.025, max(WITHDRAW_M, self.ob.top_z + 0.004 - self.arm.geom(self.arm.q6()[0])["lowest"]))
    else:
      lift = WITHDRAW_M if touching else 0.0
    if lift > 0:
      self.motion.goto([*gp[:2], gp[2] + lift], tilt=None, yaw=self.jaw, tolerance=0.015)
    new_xy = gp[:2] + c
    self.motion.goto([*new_xy, gp[2] + lift], yaw=self.jaw, tolerance=0.015)
    self.set_contact_region(new_xy)
    if lift == 0:
      # Slid sideways while straddling the object, at the same height: just reassess.
      return new_xy, np.array([*new_xy, target[2]]), False
    target = self.plan_descent(new_xy)
    return new_xy, target, self.descend(target)

  def close(self):
    ex = self.ex
    self.state("close")
    self.guard.watch(None)
    self.stats["closes"] += 1
    r = self.arm.gripper(0, self.step.force)
    held = bool(self.arm.state()["holding_signal"])
    ex.event("gripper", result=r, close_index=self.stats["closes"], contact=held)
    if not held:
      ex._wrist_blob(self.step, f"miss_{self.stats['closes']}")
    return held

  def withdraw(self):
    """Rise clear of the object's top (jaws keep their state); no downward motion."""
    try:
      gp = self.gp()
      z = max(gp[2] + 0.012, getattr(self.ex, "_align_z", gp[2] + 0.012))
      self.motion.goto([*gp[:2], z], tilt=None, yaw=self.jaw, tolerance=0.015)
    except MotionInterrupted as e:
      self.ex.event("withdraw_failed", reason=str(e))

  def local_retry(self, xy):
    """After an empty close: withdraw above the object, reopen, observe once, at most one lateral
    move. No observation pose, no head re-approach."""
    ex = self.ex
    self.stats["local_retries"] += 1
    self.state("local_retry", reason="empty close", attempt=self.stats["local_retries"])
    gp = self.gp()
    self.motion.goto([*gp[:2], ex._align_z], tilt=None, yaw=self.jaw, tolerance=0.015)
    self.arm.gripper(self.arm.percent_for_gap(self.opening), "medium")
    xy = self._observe_and_shift(gp[:2].copy(), ex._align_z, f"wrist_retry_{self.stats['local_retries']}.jpg", from_head=False)
    self.set_contact_region(xy)
    return xy
