"""Check the requested physical relation from current named measurements."""
import numpy as np
from so101_reach.manipulation.scene import footprint_contains


def placement_relation(payload,target,relation,table_z):
  effective='in' if target.extra.get('container') else 'on'
  if relation=='on' and not target.extra.get('container'):effective='on'
  # A cube seen inside a box may have a bad support-ring height. Its measured
  # footprint provides an independent height estimate; do not infer this for arbitrary shapes.
  height=float(payload.size[2])
  words=payload.name.lower().split()
  if 'cube' in words:height=float(np.median(payload.size[:2]))
  bottom=float(payload.top_z-height)
  radius=max(payload.size[:2])/2
  inside=footprint_contains(target,payload.center[:2],radius,margin=.003)
  if effective=='in':
    height_ok=table_z-.006<=bottom<target.top_z-.005
  else:
    height_ok=abs(bottom-target.top_z)<=.008
  return {'satisfied':bool(inside and height_ok),'effective_relation':effective,
          'inside':bool(inside),'height_ok':bool(height_ok),'estimated_bottom_m':bottom}
