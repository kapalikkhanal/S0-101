"""Destination geometry and release evidence, independent of detector labels.

An object's semantic identity and dimensions do not change when the arm occludes
it. Measure rim/interior geometry from a clear view, then verify that geometry
before release instead of reclassifying the destination under the payload.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import numpy as np

from so101_reach.agent.perception import Detection, _color_mask, color_words


def copy_detection(d: Detection) -> Detection:
  return replace(d, center=np.array(d.center).copy(), size=tuple(d.size),
                 bbox=None if d.bbox is None else np.array(d.bbox).copy(), extra=copy.deepcopy(d.extra))


def region_points(depth, view, reference, scale=1.12, step=2):
  half=np.array(reference.size,float)/2
  a=np.radians(reference.yaw_deg)
  rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  xy=np.array([[x,y] for x in (-1,1) for y in (-1,1)])*half[:2]*scale
  xy=xy@rot.T+reference.center[:2]
  px=[view.project([*p,z]) for p in xy for z in (reference.support_z,reference.top_z+.02)]
  px=[p for p in px if p is not None]
  if not px:return np.empty((0,3)),np.empty((0,2))
  px=np.asarray(px);h,w=depth.shape
  x0,y0=np.maximum(px.min(0),0);x1,y1=np.minimum(px.max(0)+1,[w,h])
  vs,us=np.mgrid[int(y0):int(y1):step,int(x0):int(x1):step]
  zz=depth[vs,us];ok=np.isfinite(zz)&(zz>0)
  pts=np.column_stack(((us[ok]-view.cx)/view.fx*zz[ok],(vs[ok]-view.cy)/view.fy*zz[ok],zz[ok]))@view.R.T+view.t
  local=(pts[:,:2]-reference.center[:2])@rot
  inside=np.all(np.abs(local)<=half[:2]*scale,axis=1)
  return pts[inside],local[inside]


def destination_geometry(depth,view,reference,table_z):
  """Rim and interior from world-space height samples, not image-box centre pixels.

  Points above the established object's top are occluders, not its new surface.
  Insufficient evidence returns None rather than changing a container to a solid.
  """
  pts,local=region_points(depth,view,reference,scale=1.0)
  if len(pts)<100:return None
  keep=(pts[:,2]>=table_z-.004)&(pts[:,2]<=reference.top_z+.012)
  pts,local=pts[keep],local[keep]
  norm=np.abs(local)/np.maximum(np.array(reference.size[:2])/2,.001)
  inner=pts[np.all(norm<.55,axis=1),2]
  ring=pts[(np.max(norm,axis=1)>.68),2]
  if len(inner)<30 or len(ring)<30:return None
  rim=float(np.percentile(ring,90))
  floor=float(np.percentile(inner,35))
  hollow=rim-floor>=.012
  # A visible upper plane is required for a closed support; missing interior is unknown.
  if not hollow and abs(rim-floor)>.008:return None
  return {"container":bool(hollow),"floor_z":floor,"rim_z":rim,
          "interior_points":int(len(inner)),"rim_points":int(len(ring)),"evidence":"rgbd_rim_and_interior"}


@dataclass
class PlacementTarget:
  detection: Detection
  relation: str
  requested_relation: str
  release_xy: np.ndarray | None = None

  @property
  def release_clearance(self):
    return .012 if self.relation=="in" else .004


def prepare_target(step,payload: Detection,target: Detection) -> PlacementTarget:
  """Resolve and validate the complete placement before picking anything up."""
  if any(not np.isfinite(v) or v<=0 for v in target.size):
    raise ValueError("destination has invalid geometry")
  container=bool(target.extra.get("container"))
  relation=step.relation
  # Ordinary 'put on the plate/in the bowl/on the container' means deposit,
  # not balance on an unsupported rim. Geometry must establish the opening first.
  if relation=="on" and container:relation="in"
  # No verified open interior (e.g. a shallow plate: rim < 12 mm above its floor): the
  # 'in' refusal was removed at the user's request (2026-11-07); place on its measured top.
  if relation=="in" and not container:relation="on"
  # The pre-grasp opening-fit check for 'in' (payload + 15 mm vs the smallest opening) was
  # removed at the user's request (2026-11-07). The execution monitor still checks the held
  # payload against observed rim points during the carry and release.
  if relation=="on" and max(payload.size[:2])>min(target.size[:2])+.003:
    raise ValueError("placement surface is too small for this object's footprint")
  return PlacementTarget(copy_detection(target),relation,step.relation)


def free_release_point(depth,view,target,payload,occupants=()):
  """Choose observed free interior space rather than drop onto existing contents."""
  if not target.extra.get("container"):return target.center[:2].copy()
  pts,local=region_points(depth,view,target,scale=1,step=2)
  floor=float(target.extra["floor_z"])
  half=np.array(target.size[:2])/2
  # Fit the observed interior floor: a single percentile height plus depth
  # noise/plane tilt otherwise labels much of a white floor as an obstacle.
  inner=np.all(np.abs(local)<.75*half,axis=1)&(pts[:,2]<target.top_z-.008)
  # Refusals for an unestablished/uncertain floor were removed at the user's request
  # (2026-11-07): fall back to the measured flat floor height and keep choosing by contents.
  plane=np.array([0.0,0.0,floor])
  if np.count_nonzero(inner)>=50:
    from scipy.optimize import least_squares
    design=np.c_[local[inner],np.ones(np.count_nonzero(inner))]
    fit=least_squares(lambda p:design@p-pts[inner,2],[0,0,floor],loss='soft_l1',f_scale=.003,max_nfev=50)
    if abs(fit.x[2]-floor)<=.015 and np.mean(np.abs(design@fit.x-pts[inner,2])<.006)>=.5:
      plane=fit.x
    else:
      target.extra['placement_note']="floor plane uncertain; measured flat floor used"
  else:
    target.extra['placement_note']="interior floor sparsely observed; measured flat floor used"
  target.extra['placement_floor_plane']=plane.tolist()
  height=pts[:,2]-(np.c_[local,np.ones(len(local))]@plane)
  rim=target.top_z
  contents=(pts[:,2]<rim-.004)|(pts[:,2]>rim+.008)
  occupied=local[(height>.012)&contents&np.all(np.abs(local)<half-.006,axis=1)]
  radius=float(np.linalg.norm(payload.size[:2])/2 if 'cube' in payload.name.lower().split() else max(payload.size[:2])/2)
  a=np.radians(target.yaw_deg);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  known=[((d.center[:2]-target.center[:2])@rot,
          float(np.linalg.norm(d.size[:2])/2 if 'cube' in d.name.lower().split() else max(d.size[:2])/2))
         for d in occupants if d.name!=payload.name and np.all(np.abs((d.center[:2]-target.center[:2])@rot)<half)]
  bounds=half-radius-.008
  # Tight fit (opening-fit check removed on request): release over the interior centre.
  if np.any(bounds<=0):return target.center[:2].copy()
  from scipy.spatial import cKDTree
  tree=cKDTree(occupied) if len(occupied)>=10 else None
  options=[]
  for x in np.linspace(-bounds[0],bounds[0],13):
    for y in np.linspace(-bounds[1],bounds[1],13):
      xy=np.array([x,y])
      obstacle=np.inf if tree is None else float(tree.query(xy)[0])-radius
      remembered=min([np.linalg.norm(xy-c)-radius-r for c,r in known],default=np.inf)
      wall=float(np.min(half-np.abs(xy)))-radius
      options.append((min(wall,obstacle,remembered),-float(np.linalg.norm(xy)),xy))
  # The point with the most room among the observed contents. The former refusal when no point
  # kept 6 mm from them ("no observed free space in the container") was removed at the user's
  # request (2026-11-07); a tighter spot is used and recorded instead.
  best=max(options,key=lambda p:p[:2])
  target.extra['placement_clearance_mm']=round(1000*best[0],1) if np.isfinite(best[0]) else None
  if best[0]<=.006:
    target.extra['placement_note']=f"tight: best spot has {1000*best[0]:.1f} mm to contents or wall (6 mm preferred)"
  point=best[2];a=np.radians(target.yaw_deg)
  return target.center[:2]+np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@point


def packed_release_point(target,payload,occupants):
  """Reserve room for the complete payload, including arbitrary cube yaw."""
  if not target.extra.get('container'):return target.center[:2].copy()
  def radius(d):
    return float(np.linalg.norm(d.size[:2])/2 if 'cube' in d.name.split() else max(d.size[:2])/2)
  r=radius(payload);half=np.array(target.size[:2])/2;bounds=half-r-.008
  if np.any(bounds<=0):return target.center[:2].copy()
  a=np.radians(target.yaw_deg);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  occupied=[((d.center[:2]-target.center[:2])@rot,radius(d)) for d in occupants
            if d.name!=payload.name and np.all(np.abs((d.center[:2]-target.center[:2])@rot)<half)]
  candidates=[]
  for x in np.linspace(-bounds[0],bounds[0],13):
    for y in np.linspace(-bounds[1],bounds[1],13):
      p=np.array([x,y]);wall=float(np.min(half-np.abs(p)))-r
      gap=min([np.linalg.norm(p-c)-r-other for c,other in occupied],default=np.inf)
      candidates.append((min(wall,gap),-float(np.linalg.norm(p)),p))
  best=max(candidates,key=lambda p:p[:2])
  target.extra['packing_clearance_mm']=1000*float(best[0])
  return target.center[:2]+rot@best[2]


def release_geometry(depth,view,target: Detection,payload: Detection,table_z,baseline_depth=None):
  """Segment the placed object above the destination's *pre-release* support.

  The container's walls are excluded by requiring height above its interior floor.
  For a solid support, only geometry above its old top can be the payload. This
  avoids measuring the merged destination + payload as a new oversized object.
  """
  pts,_=region_points(depth,view,target,scale=1.02,step=1)
  if len(pts)<30:return None
  floor=float(target.extra.get("floor_z",table_z+.004)) if target.extra.get("container") else target.top_z
  ceiling=(target.top_z if target.extra.get("container") else floor)+payload.size[2]+.020
  high=pts[(pts[:,2]>floor+max(.008,.35*payload.size[2]))&(pts[:,2]<ceiling)]
  if len(high)<30:return None
  # Pixel-connected components prevent a box rim and a same-colour cube from merging.
  camera=(high-view.t)@view.R
  uu=np.rint(view.fx*camera[:,0]/camera[:,2]+view.cx).astype(int)
  vv=np.rint(view.fy*camera[:,1]/camera[:,2]+view.cy).astype(int)
  import cv2
  image=np.zeros(depth.shape,np.uint8)
  inside=(uu>=0)&(uu<image.shape[1])&(vv>=0)&(vv<image.shape[0])
  if baseline_depth is not None:
    # Fixed head camera: pre-existing contents/rim are not the newly released item.
    changed=np.zeros(len(inside),bool)
    old=baseline_depth[vv[inside],uu[inside]];new=depth[vv[inside],uu[inside]]
    changed[inside]=(old>0)&(old-new>.008)
    inside &= changed
  # Depth alone can split a short piece of rim into a cube-sized component.
  # Appearance is additional evidence, never a substitute for metric geometry.
  if view.image is not None:
    hsv=cv2.cvtColor(view.image,cv2.COLOR_BGR2HSV)
    colors=[_color_mask(hsv,w) for w in color_words(payload.name)]
    colors=[c for c in colors if c is not None]
    if colors:
      matching=np.zeros(len(inside),bool)
      matching[inside]=np.any(colors,axis=0)[vv[inside],uu[inside]]
      inside &= matching
  image[vv[inside],uu[inside]]=1
  _,labels,stats,_=cv2.connectedComponentsWithStats(image,8)
  component=labels[vv[inside],uu[inside]]
  high=high[inside]
  options=[]
  for lab in range(1,len(stats)):
    points=high[component==lab]
    if len(points)<25:continue
    lo,hi=np.percentile(points[:,:2],[3,97],axis=0)
    size=hi-lo
    # Reject the full rim/perimeter and tiny glints; retain a size-compatible top patch.
    if max(size)>1.55*max(payload.size[:2])+.010 or min(size)<.25*min(payload.size[:2]):continue
    center=(lo+hi)/2
    # A rim fragment may fit the size test but cannot support a whole payload.
    a=np.radians(target.yaw_deg)
    local=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])@(center-target.center[:2])
    if np.any(np.abs(local)+max(payload.size[:2])/2>np.array(target.size[:2])/2-.003):continue
    top=float(np.percentile(points[:,2],85))
    support=floor
    if target.extra.get("container"):
      support=max(floor,top-payload.size[2])
      if top<floor+.5*payload.size[2] or support>target.top_z-.005:continue
    elif abs(top-(floor+payload.size[2]))>.015:continue
    options.append((len(points),center,top,size,support))
  if len(options)!=1:return None  # ambiguous contents require another view
  _,xy,top,size,support=options[0]
  return Detection(0,payload.name,0,np.array([*xy,(top+support)/2]),top,support,
                   (*size,max(.001,top-support)),payload.yaw_deg,"head",extra={"evidence":"segmented_release_geometry"})
