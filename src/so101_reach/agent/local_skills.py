"""Scripted fine-motion primitives: the zero-token rung of the local-first handoff.

These run the nominal grasp/place strategies from GRASP_INSTRUCTIONS / PLACE_INSTRUCTIONS
in code, optionally steered by cached parameters distilled from an earlier successful LLM
phase (see `skill_cache.py`). They use the same `AgentArm` calls the LLM's tools map to,
so they behave identically in simulation and on the real arm. Success is judged by the
same hardware signals the pipeline already trusts (holding signal; re-perception for
place happens in the pipeline).

Each primitive returns ``(ok, note)``; on failure `note` describes what was tried so the
pipeline can hand it to the LLM as context.
"""

from __future__ import annotations

import re

import numpy as np

SOFT_WORDS = ("egg", "glass", "phone", "orange", "berry", "tomato", "banana", "cup")


def is_soft(name: str) -> bool:
  n = name.lower()
  return any(w in n for w in SOFT_WORDS)


def _cm(v: float) -> float:
  return round(100 * v, 1)


def _wrap_deg(a: float) -> float:
  """Jaw yaw difference folded to [-90, 90) (the jaws are symmetric under 180 deg)."""
  return (a + 90.0) % 180.0 - 90.0


def jaw_for(arm, obj, width: float) -> float:
  """Jaw closing direction square to the object's faces: across the narrow side of an elongated
  object; for a square / round one, whichever face direction needs the smaller wrist turn
  (closing diagonally on a square block catches only its corners and it slips out)."""
  if max(obj.size[:2]) > 1.3 * width:
    return obj.yaw_deg + 90
  now = arm.state()["jaw_yaw_deg"]
  cands = [obj.yaw_deg, obj.yaw_deg + 90]
  return min(cands, key=lambda c: abs((c - now + 90) % 180 - 90))


def heuristic_grasp_params(arm, obj, tz: float) -> dict:
  """The scripted default grasp written as an explicit parameter set (the same format
  `distill_grasp` produces and `scripted_grasp` replays): centre, mid-height (1/3 below the
  top when stacked), jaw square to the faces, force by softness. Used as the baseline that
  perturbation retries and the BC model are compared against."""
  stacked = obj.support_z - tz > 0.01
  h = obj.top_z - obj.size[2] / 3 if stacked else obj.support_z + 0.5 * obj.size[2]
  jaw = jaw_for(arm, obj, float(min(obj.size[:2])))
  return {"dxy_obj_mm": [0.0, 0.0], "h_rel_top_mm": round(1000 * (h - obj.top_z), 1),
          "jaw_rel_deg": round(_wrap_deg(jaw - obj.yaw_deg), 1),
          "force": "gentle" if is_soft(obj.name) else "firm"}


def perturb_params(params: dict, rng) -> dict:
  """A jittered copy of grasp params for retry exploration (self-supervised data)."""
  out = {k: (list(v) if isinstance(v, list) else v) for k, v in params.items()}
  out["dxy_obj_mm"] = [round(params["dxy_obj_mm"][0] + rng.uniform(-6, 6), 1),
                       round(params["dxy_obj_mm"][1] + rng.uniform(-6, 6), 1)]
  out["h_rel_top_mm"] = round(params["h_rel_top_mm"] + rng.uniform(-6, 6), 1)
  out["jaw_rel_deg"] = round(_wrap_deg(params["jaw_rel_deg"] + rng.uniform(-20, 20)), 1)
  if rng.uniform() < 0.25:
    out["force"] = {"gentle": "medium", "medium": "firm", "firm": "medium"}[params.get("force", "firm")]
  return out


def _move_gp(arm, x_m: float, y_m: float, h_m: float, tz: float, jaw_yaw: float | None) -> str:
  """move_to the grip point (base-frame metres -> the arm's cm convention), retrying once at
  the lowest allowed height if the floor/collision check names one."""
  r = arm.move_to(_cm(x_m), _cm(y_m), _cm(h_m - tz), None, jaw_yaw)
  if r.startswith("ok"):
    return r
  m = re.search(r"lowest allowed h[^0-9]*([\d.]+)", r)
  if m:
    r2 = arm.move_to(_cm(x_m), _cm(y_m), float(m.group(1)) + 0.1, None, jaw_yaw)
    if r2.startswith("ok"):
      return r2
  return r


def scripted_grasp(arm, obj, tz: float, params: dict | None = None) -> tuple[bool, str]:
  """Grasp `obj` from the hover pose the pipeline already reached (centred, jaws open).

  Default: lower straight to mid-height (1/3 below the top for a stacked object) and close
  (gentle for soft-named objects, else firm). With cached `params`, use the distilled
  offset / height / jaw / force instead. Returns (holding, note)."""
  if params:
    x, y = obj.center[0] + params["dxy_obj_mm"][0] / 1000, obj.center[1] + params["dxy_obj_mm"][1] / 1000
    h = obj.top_z + params["h_rel_top_mm"] / 1000
    jaw = obj.yaw_deg + params.get("jaw_rel_deg", 0.0)
    force = params.get("force", "firm")
  else:
    x, y = obj.center[0], obj.center[1]
    stacked = obj.support_z - tz > 0.01
    h = obj.top_z - obj.size[2] / 3 if stacked else obj.support_z + 0.5 * obj.size[2]
    jaw = None  # keep the jaw the pipeline already set with _jaw_for
    force = "gentle" if is_soft(obj.name) else "firm"
  src = "cached params" if params else "scripted default"
  r = _move_gp(arm, x, y, h, tz, jaw)
  if not r.startswith("ok"):
    return False, f"{src}: lowering to h={_cm(h - tz)} cm was refused ({r})"
  out = arm.gripper(0, force)
  if arm.state()["holding_signal"]:
    return True, f"{src}: held after closing at h={_cm(h - tz)} cm ({force})"
  return False, f"{src}: closed ({force}) at h={_cm(h - tz)} cm but {out}"


def distill_grasp(arm, obj, force: str) -> dict:
  """Object-relative parameters of the grasp pose the arm is in right now (called at the
  moment the holding signal is true, before the carry)."""
  gp = arm.geom(arm.q6()[0])["gp"]
  jaw = float(arm.state()["jaw_yaw_deg"])
  return {"dxy_obj_mm": [round(1000 * float(gp[0] - obj.center[0]), 1), round(1000 * float(gp[1] - obj.center[1]), 1)],
          "h_rel_top_mm": round(1000 * float(gp[2] - obj.top_z), 1),
          "jaw_rel_deg": round(_wrap_deg(jaw - obj.yaw_deg), 1),
          "force": force}


def scripted_place(arm, tgt, obj_size, held_below: float, tz: float, hint_drop: float, hint_stack: float,
                   container: bool, soft: bool, params: dict | None = None) -> tuple[bool, str]:
  """Release the held object over `tgt` from the pose above it the pipeline already reached.

  Default rules (same as the task text's precomputed hints): container -> lower until the held
  bottom is 2 cm above the rim and open; otherwise stack -> held bottom 0.4 cm above the top,
  open just wider than the object. A soft/fragile object into a container is deferred to the
  LLM (it must be lowered inside, between the walls). Returns (released, note)."""
  width = float(min(obj_size[:2]))
  if params:
    x, y = tgt.center[0] + params["dxy_tgt_mm"][0] / 1000, tgt.center[1] + params["dxy_tgt_mm"][1] / 1000
    h = tgt.top_z + params["release_h_rel_top_mm"] / 1000 - tz
    open_pct, src = float(params.get("open_percent", 60.0)), "cached params"
  elif container and soft:
    return False, "scripted default does not lower a fragile object inside a container"
  elif container:
    x, y, h = tgt.center[0], tgt.center[1], hint_drop
    open_pct, src = min(80.0, arm.percent_for_gap(width + 0.03)), "scripted default (drop above rim)"
  else:
    x, y, h = tgt.center[0], tgt.center[1], hint_stack
    open_pct, src = min(80.0, arm.percent_for_gap(width + 0.015)), "scripted default (stack)"
  r = _move_gp(arm, x, y, h + tz, tz, None)
  if not r.startswith("ok"):
    return False, f"{src}: lowering to h={_cm(h)} cm was refused ({r})"
  arm.gripper(open_pct, "medium")
  st = arm.state()
  if st["holding_signal"] or st["gripper_open_percent"] < 5:
    return False, f"{src}: opened at h={_cm(h)} cm but the object remains held"
  return True, f"{src}: released at h={_cm(h)} cm"


def distill_place(release_gp: np.ndarray, tgt, held_below: float, open_percent: float) -> dict:
  """Target-relative parameters of the release pose (grip point captured at release)."""
  rel_top = float(release_gp[2] - tgt.top_z)
  return {"dxy_tgt_mm": [round(1000 * float(release_gp[0] - tgt.center[0]), 1), round(1000 * float(release_gp[1] - tgt.center[1]), 1)],
          "release_h_rel_top_mm": round(1000 * rel_top, 1),
          "open_percent": round(float(open_percent), 1),
          "strategy": "stack" if rel_top < held_below + 0.012 else "drop"}
