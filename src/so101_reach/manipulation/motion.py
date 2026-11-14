"""Collision-checked execution for the existing RL/IK arm adapter.

MuJoCo collision primitives cover the entire robot, including the wrist camera.
Sim uses the actual scene geometry; camera mode checks observed depth surfaces.
This is a conservative software monitor, not a certified hardware safety system.
"""
from __future__ import annotations

import time
import mujoco
import numpy as np

from so101_reach.agent.arm import AgentArm
from so101_reach.real.feetech_bus import JOINT_ORDER
from so101_reach.sim import so101
from so101_reach.manipulation.scene import SceneUnavailable, attached_boxes, fresh_object


class MotionInterrupted(RuntimeError):
  def __init__(self,message,hard_stop=False):
    super().__init__(message)
    self.hard_stop=hard_stop


class TableContact(MotionInterrupted):
  """Stop the descent on servo effort rise; inspect the pose before closing."""


class ReplanAfterPause(MotionInterrupted):
  """The old trajectory is stale after holding; rebuild it from measured joints."""


def point_distance(points, position, rotation, kind, size, radius):
  """Signed distance to collision primitives; meshes use enclosing spheres."""
  p=(points-position) @ rotation
  if kind == mujoco.mjtGeom.mjGEOM_BOX:
    v=np.abs(p)-size
    return np.linalg.norm(np.maximum(v,0),axis=1)+np.minimum(v.max(axis=1),0)
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return np.linalg.norm(p,axis=1)-size[0]
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    p=p.copy();p[:,2]-=np.clip(p[:,2],-size[1],size[1])
    return np.linalg.norm(p,axis=1)-size[0]
  if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
    v=np.column_stack((np.linalg.norm(p[:,:2],axis=1)-size[0],np.abs(p[:,2])-size[1]))
    return np.linalg.norm(np.maximum(v,0),axis=1)+np.minimum(v.max(axis=1),0)
  return np.linalg.norm(p,axis=1)-radius


class SceneGuard:
  def __init__(self, arm: AgentArm, source, margin=0.003, max_age=0.8, max_tracking=0.020):
    self.arm,self.source,self.margin,self.max_age=arm,source,margin,max_age
    self.max_tracking=max_tracking
    self.auto_pause_timeout=None  # None disables; 0 waits indefinitely
    self.last_hand_time=-1e9
    self.pause_reason=None
    self.pause_events=[]
    self._checking_paused=False
    self.max_deviation=0.0
    self.recent_deviation=0.0
    self.fresh_waits=[]
    self.move_peak_deviation=0.0  # peak measured deviation since the last reset (see skills)
    self.dev_samples=[]  # per-tick measured deviations since the last reset
    self.allow_occluded=0.0
    self.occluded_uses=0
    self._q_meas=None
    self._cloud_cache=None
    self.table_baseline=None
    self._table_margin=self.margin
    self.last_block=None  # (part, offending cloud points, pose) of the latest observed-obstacle refusal
    self.contact_override={}  # name -> Detection: wrist-corrected pose of the object the jaws may touch
    self.table_touch_center=None
    self.table_touch_radius=.05
    self.touch_loads={}
    self._touch_sample=-1e9
    # Explicit descent contact monitor (grasp state machine): armed only for one descent move.
    self.descent=None
    self.release_zone=None  # open container being filled: its rim/contents may touch the held object
    # 7 mm of commanded-but-unachieved downward grip-point motion. On the P-only STS3215,
    # Present_Load is about 4.5 units per tick of goal-present error (trial logs), so this matches
    # the previous 100-unit shoulder load limit, expressed at the jaws with a sign.
    self.descent_push_limit=0.007
    self.active=None
    self.anchor=None
    self.grasp_contact=None
    self.held_name=None
    self.held_below=0.0
    self.held_size=None
    self.held_transform=None
    self.held_baseline={}
    self.checks=0
    self.interruptions=[]
    self.actual_contacts=[]
    self.last_q=None
    self.world=getattr(source,"world",None)
    self.m=self.world.model if self.world is not None else so101.make_cpu_model()
    if self.world is None:
      self.m.geom_pos[self.m.geom("floor").id,2]=source.table_z
    self.d=mujoco.MjData(self.m)
    self.qadr=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_ORDER])
    moving={self.m.body(n).id for n in ("shoulder","upper_arm","lower_arm","wrist","gripper","moving_jaw_so101_v1","camera_mount")}
    jaws={self.m.body(n).id for n in ("gripper","moving_jaw_so101_v1")}
    robot={self.m.body(n).id for n in ("base","shoulder","upper_arm","lower_arm","wrist","gripper","moving_jaw_so101_v1","camera_mount")}
    self.robot_geoms=np.array([i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in moving and (self.m.geom_contype[i] or self.m.geom_conaffinity[i])],int)
    self.all_robot={i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in robot}
    self.jaw_geoms={i for i in self.robot_geoms if self.m.geom_bodyid[i] in jaws}
    self.env_geoms=np.array([i for i in range(self.m.ngeom) if i not in self.all_robot and (self.m.geom_contype[i] or self.m.geom_conaffinity[i])],int)
    self._snapshot=None
    # Camera mode: collision meshes are checked against their sampled surfaces, not enclosing spheres.
    self.mesh_samples={}
    for g in self.robot_geoms:
      if self.m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=self.m.geom_dataid[g]
        v=self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]]
        _,idx=np.unique(np.floor(v/0.003).astype(np.int64),axis=0,return_index=True)
        self.mesh_samples[int(g)]=np.asarray(v[idx],float)
    # Camera mode also protects measured attached hardware (wrist D405 + cable exit).
    self.attached=[(self.m.body(b).id,c,h,n) for b,c,h,n in attached_boxes()] if self.world is None else []

  def watch(self,name=None,anchor=None,allow_occluded=0.0):
    self.active=name
    self.anchor=None if anchor is None else np.array(anchor).copy()
    self.allow_occluded=allow_occluded if name else 0.0

  def allow_jaw_table_touch(self,ob,sample_loads=True):
    """sample_loads: legacy implicit load stop (one baseline per grasp). The grasp state machine
    passes False and arms the explicit per-descent monitor instead (arm_descent)."""
    self.table_touch_center=np.array(ob.center[:2]).copy()
    self.table_touch_radius=max(ob.size[:2])/2+.02
    self.touch_loads={}
    if sample_loads and type(self.arm.plant).__name__=='RealPlant':
      self.touch_loads={n:abs(self.arm.bus.read(n,'Present_Load')) for n in ('shoulder_lift','elbow_flex')}

  def set_release_zone(self,destination,pad=0.010,above=0.015):
    """Container footprint (+pad) up to its rim (+above) where the held object may touch."""
    if destination is None:
      self.release_zone=None;return
    self.release_zone={"center":np.array(destination.center[:2],float),"half":np.array(destination.size[:2],float)/2+pad,
                       "yaw":float(destination.yaw_deg),"top":float(destination.top_z)+above}

  @staticmethod
  def _in_release_zone(points,zone):
    a=np.radians(zone["yaw"]);rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    local=np.abs((np.asarray(points)[:,:2]-zone["center"])@rot.T)
    return np.all(local<=zone["half"],axis=1)&(np.asarray(points)[:,2]<=zone["top"])

  def downward_push(self,q=None,raw=None):
    """Grip-point displacement (m) the arm servos are commanded *below* the measured pose.

    The STS3215 position loop is proportional only, so its effort is proportional to
    goal - present; projected through the grip point's vertical Jacobian this is the downward
    effort expressed at the jaws, with a sign (upward lifting effort is negative)."""
    if raw is None or q is None:
      q,raw=self.arm.q6()
    goal=self.arm.bus.sync_read("Goal_Position")
    names=JOINT_ORDER[:4]
    self._last_goal_offsets={n:int(goal[n])-int(raw[n]) for n in names}
    dq=np.array([self.arm.jmap.to_rad(n,goal[n])-self.arm.jmap.to_rad(n,raw[n]) for n in names])
    z0=self.arm.geom(q)["gp"][2];jz=np.zeros(4)
    for j in range(4):
      qq=np.array(q,float).copy();qq[j]+=1e-3
      jz[j]=(self.arm.geom(qq)["gp"][2]-z0)/1e-3
    return float(-jz@dq)

  def arm_descent(self,top_z,zone=0.010,stall=0.0005,consecutive=2,period=0.06,breakaway=0.005):
    """Arm the contact monitor for one descent. The baseline is taken now, at rest at the
    pre-descent pose (same configuration and gravity load as the descent), and discarded by
    disarm_descent, so it cannot leak into other moves or later attempts.

    It stops the descent only when all hold, on `consecutive` samples: the lowest jaw tip is within
    `zone` of the object's top or lower (the jaws can touch the object or the table), the grip point
    has not descended by `stall` over the last 0.12 s, and the downward push exceeds the
    per-descent baseline by descent_push_limit; and the arm has already moved down by `breakaway`
    since arming. Static friction holds a resting arm until the commanded push reaches the friction
    band (8-12 mm at the jaws): on 2026-11-07 (trial_3) a 9.5 mm breakaway push stopped a descent
    3 mm after it began, 9 mm above the cube top. Nothing can be touched there: a descent starts
    with the tips 12 mm above the object."""
    q,raw=self.arm.q6()
    base=self.downward_push(q,raw)
    self.descent={"top_z":float(top_z),"base":base,"base_offsets":dict(self._last_goal_offsets),"zone":zone,"stall":stall,
                  "z0":float(self.arm.geom(q)["gp"][2]),"breakaway":breakaway,
                  "need":consecutive,"period":period,"hits":0,"hist":[],"last":-1e9,"max_excess":0.0,
                  "samples":0,"stop":None}

  def hold_balanced(self,offsets):
    """Hold the measured arm pose with the goal-present offsets measured at rest before the
    descent (the servos' own gravity balance there). Unlike hold() (goal = present), this neither
    lets the P-servos sag under gravity nor adds downward push. Wrist roll and gripper keep goals."""
    try:
      raw=self.arm.bus.sync_read("Present_Position")
      goal={n:self.arm.plant._clamp(n,int(raw[n])+int(offsets.get(n,0))) for n in JOINT_ORDER[:4]}
      self.arm.bus.sync_write("Goal_Position",goal)
    except (OSError,AttributeError,KeyError,TypeError):
      return self.hold()
    return None

  def disarm_descent(self):
    d,self.descent=self.descent,None
    return d

  def _descent_check(self,q,raw,now):
    d=self.descent
    if now-d["last"]<d["period"]:
      return
    d["last"]=now;d["samples"]+=1
    geom=self.arm.geom(q);z=float(geom["gp"][2])
    d["hist"]=[h for h in d["hist"] if now-h[0]<=0.5]+[(now,z)]
    excess=self.downward_push(q,raw)-d["base"]
    d["max_excess"]=max(d["max_excess"],excess)
    past=[h for h in d["hist"] if now-h[0]>=0.12]
    stalled=bool(past) and z-past[-1][1]>-d["stall"]
    in_zone=geom["lowest"]<=d["top_z"]+d["zone"]
    moved=d.get("z0",np.inf)-z>=d.get("breakaway",0.0)
    d["hits"]=d["hits"]+1 if (excess>=self.descent_push_limit and stalled and in_zone and moved) else 0
    if d["hits"]>=d["need"]:
      loads={}
      try:loads={n:int(self.arm.bus.read(n,"Present_Load")) for n in JOINT_ORDER[:4]}
      except Exception:pass
      d["stop"]={"excess_push_mm":round(1000*excess,1),"baseline_push_mm":round(1000*d["base"],1),
                 "lowest_tip_above_object_top_mm":round(1000*(geom["lowest"]-d["top_z"]),1),
                 "lowest_tip_above_table_mm":round(1000*(geom["lowest"]-self.source.table_z),1),
                 "grip_point_m":geom["gp"].tolist(),"loads":loads}
      self.hold_balanced(d["base_offsets"])
      raise TableContact("descent stopped: stalled under load near the object (contact or friction); "
                         "no further downward command")

  def _jaw_touch(self,g,pos):
    return (self.table_touch_center is not None and g in self.jaw_geoms and
            np.linalg.norm(pos[:2]-self.table_touch_center)<=self.table_touch_radius)

  def attach(self,name,measured=None):
    """Payload pose relative to the gripper. Sim: exact. Camera: from `measured` (the grasp-time
    detection) or, when given none, a fresh view; otherwise the existing attachment is kept."""
    self.held_baseline={}
    self._cloud_cache=None
    if self.world is not None:
      d=self.world.data
      gr=self.m.body("gripper").id; ob=self.world.obj_body[name]
      r=d.xmat[gr].reshape(3,3)
      self.held_transform=(r.T@(d.xpos[ob]-d.xpos[gr]),r.T@d.xmat[ob].reshape(3,3))
      # Contact penetration the payload already has at the measured state (e.g. a grasp that
      # tilted it against its support). Later poses may not press it in any further.
      for hg in (i for i in self.env_geoms if self.m.geom_bodyid[i]==ob):
        for e in self.env_geoms:
          if self.m.geom_bodyid[e]!=ob:
            dist=mujoco.mj_geomDistance(self.m,d,int(hg),int(e),0.01,None)
            if dist<0:self.held_baseline[(int(hg),int(e))]=float(dist)
    else:
      ob=measured if measured is not None else self.source.snapshot().objects.get(name)
      if measured is None and (ob is None or self.source.now()-self.source.snapshot().seen.get(name,-1e9)>self.max_age):
        # A held object is usually hidden from the head camera: keep the grasp-time attachment.
        if self.held_transform is None:
          raise SceneUnavailable(f"{name!r} not observed; cannot attach payload geometry")
        return
      self.arm.geom(self.arm.q6()[0])
      d=self.arm.kin.d; gr=self.arm.kin.body
      r=d.xmat[gr].reshape(3,3); a=np.radians(ob.yaw_deg)
      ro=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
      self.held_transform=(r.T@(ob.center-d.xpos[gr]),r.T@ro)

  def payload_geometry(self,q6):
    self.arm.geom(q6)
    d=self.arm.kin.d;gr=self.arm.kin.body
    r=d.xmat[gr].reshape(3,3);offset,rotation=self.held_transform
    center=d.xpos[gr]+r@offset
    orient=r@rotation
    half=np.array(self.held_size)/2
    extent=np.abs(orient)@half
    if self.world is not None:
      geom=self.m.body_geomadr[self.world.obj_body[self.held_name]]
      kind=self.m.geom_type[geom]
      if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        extent=np.full(3,self.m.geom_size[geom,0])
      elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis=orient[:,2];size=self.m.geom_size[geom]
        extent=size[0]*np.sqrt(np.maximum(0,1-axis**2))+size[1]*np.abs(axis)
    return center,extent

  def _table_clearance(self,g,pos,rot,kind,size):
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
      low=pos[2]-np.abs(rot[2])@size
    elif int(g) in self.mesh_samples:
      low=float((pos+self.mesh_samples[int(g)]@rot.T)[:,2].min())
    else:
      low=pos[2]-self.m.geom_rbound[g]
    return float(low-self.source.table_z)

  def set_table_baseline(self,q6):
    """Camera mode: per-link table clearance at the start of a move (see _check)."""
    if self.world is not None:
      return
    self.d.qpos[self.qadr]=q6
    mujoco.mj_forward(self.m,self.d)
    base={}
    for g in self.robot_geoms:
      c=self._table_clearance(g,self.d.geom_xpos[g],self.d.geom_xmat[g].reshape(3,3),self.m.geom_type[g],self.m.geom_size[g])
      if c<self.margin+0.010:
        base[int(g)]=c
    for i,(b,c,h,_) in enumerate(self.attached):
      rot=self.d.xmat[b].reshape(3,3)
      low=self.d.xpos[b][2]+(rot@c)[2]-np.abs(rot[2])@h-self.source.table_z
      if low<self.margin+0.010:
        base[("att",i)]=float(low)
    self.table_baseline=base or None

  def _cloud_without_payload(self,s):
    """Observed cloud minus the held object's own surface, located at the measured arm pose."""
    if not (self.held_name and self.held_transform is not None and len(s.cloud)):
      return s.cloud
    # A depth frame can lag the arm by several control ticks. Remove the payload
    # where that frame saw it, rather than at its later position or a future goal.
    q=getattr(s,"joints",None)
    if q is None:q=self._q_meas if self._q_meas is not None else self.arm.q6()[0]
    key=(id(s),s.stamp,q.tobytes())
    if self._cloud_cache is None or self._cloud_cache[0]!=key:
      center,extent=self.payload_geometry(q)
      self._cloud_cache=(key,s.cloud[~np.all(np.abs(s.cloud-center)<=extent+0.012,axis=1)])
    return self._cloud_cache[1]

  def _allowed(self,g,e):
    if self.world is None:
      return False
    bid=int(self.m.geom_bodyid[e])
    names=[self.grasp_contact,self.held_name]
    return g in self.jaw_geoms and any(n in self.world.obj_body and self.world.obj_body[n]==bid for n in names if n)

  def check(self,q6,extra_margin=0.0,table_extra=None):
    """Check one prospective pose against the latest available geometry.

    extra_margin enlarges the required clearance (e.g. by measured path-tracking error);
    table_extra (default: extra_margin) is the part that applies to table clearance."""
    base=self.margin
    self.margin=base+max(0.0,float(extra_margin))
    self._table_margin=base+max(0.0,float(extra_margin if table_extra is None else table_extra))
    try:
      return self._check(q6)
    finally:
      self.margin=base
      self._table_margin=base

  def check_hard(self,q6):
    """Geometry check with zero clearance margin: nothing may touch the table or observations."""
    base=self.margin;self.margin=0.0
    try:
      return self.check(q6)
    finally:
      self.margin=base

  def _check(self,q6):
    self.checks+=1
    if self.world is not None:
      self.d.qpos[:]=self.world.data.qpos
    self.d.qpos[self.qadr]=q6
    mujoco.mj_forward(self.m,self.d)
    held_geoms=set()
    if self.world is not None and self.held_name and self.held_transform is not None:
      # The payload follows prospective robot poses; its old world position is not an obstacle.
      gr=self.m.body("gripper").id; ob=self.world.obj_body[self.held_name]
      r=self.d.xmat[gr].reshape(3,3); offset,rotation=self.held_transform
      adr=self.m.jnt_qposadr[self.m.body_jntadr[ob]]
      self.d.qpos[adr:adr+3]=self.d.xpos[gr]+r@offset
      quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(r@rotation).ravel())
      self.d.qpos[adr+3:adr+7]=quat
      mujoco.mj_forward(self.m,self.d)
      held_geoms={i for i in self.env_geoms if self.m.geom_bodyid[i]==ob}
    # MuJoCo's pair exclusions remove adjacent-link contacts by construction.
    for c in self.d.contact:
      a,b=int(c.geom1),int(c.geom2)
      if a in self.all_robot and b in self.all_robot and c.dist < -0.0005:
        return "self collision"
    if self.world is not None:
      # Broad phase before exact geom distances keeps the per-tick shield inexpensive.
      for g in self.robot_geoms:
        gp=self.d.geom_xpos[g]
        nearby=self.env_geoms[(np.linalg.norm(self.d.geom_xpos[self.env_geoms]-gp,axis=1)
                  <=self.m.geom_rbound[self.env_geoms]+self.m.geom_rbound[g]+self.margin)
                  |(self.m.geom_type[self.env_geoms]==mujoco.mjtGeom.mjGEOM_PLANE)]
        for e in nearby:
          if self._allowed(g,e):
            continue
          dist=mujoco.mj_geomDistance(self.m,self.d,int(g),int(e),self.margin,None)
          table_touch=self.m.geom_type[e]==mujoco.mjtGeom.mjGEOM_PLANE and self._jaw_touch(g,gp)
          limit=0.0 if table_touch else self.margin
          if dist < limit-1e-6:
            return f"collision clearance: {self.m.body(self.m.geom_bodyid[g]).name} / {self.m.geom(e).name or 'scene'} ({1000*dist:.1f} mm)"
      for hg in held_geoms:
        for e in self.env_geoms:
          if e in held_geoms:
            continue
          dist=mujoco.mj_geomDistance(self.m,self.d,int(hg),int(e),self.margin,None)
          # A newly grasped payload may start touching its support, but cannot penetrate it
          # deeper than it already did when measured at attach time.
          if dist < min(-0.002,self.held_baseline.get((int(hg),int(e)),0.0)-0.001):
            return f"held object would collide with {self.m.geom(e).name}"
      return None
    s=self._snapshot or self.source.snapshot()
    cloud=self._cloud_without_payload(s)
    for g in self.robot_geoms:
      pos,rot=self.d.geom_xpos[g],self.d.geom_xmat[g].reshape(3,3)
      size=self.m.geom_size[g]
      kind=self.m.geom_type[g]
      # All collision geometry, including wrist-camera mounting, must clear the table. A move
      # that starts closer than the margin may not bring any link closer than at its start.
      clearance=self._table_clearance(g,pos,rot,kind,size)
      allowed=self._table_margin if self.table_baseline is None else min(self._table_margin,self.table_baseline.get(int(g),np.inf)-0.0005)
      if self._jaw_touch(g,pos):allowed=0.0
      if clearance<allowed:
        return "full-link geometry would hit the table"
      if not len(cloud):
        continue
      points=cloud
      if g in self.jaw_geoms:
        for name in (self.grasp_contact,self.held_name):
          # Both cloud and detection use head-camera coordinates. A wrist
          # correction compensates robot pose error; it does not move the fruit.
          ob=s.objects.get(name) if name else None
          if ob is None:ob=self.contact_override.get(name)
          if ob is not None:
            mask=np.all(np.abs(points-ob.center)<np.array(ob.size)/2+0.012,axis=1)
            points=points[~mask]
      nearby=points[np.linalg.norm(points-pos,axis=1)<self.m.geom_rbound[g]+0.015]
      if not len(nearby):
        continue
      if int(g) in self.mesh_samples:
        from scipy.spatial import cKDTree
        surf=pos+self.mesh_samples[int(g)]@rot.T
        # Half the 3 mm sample spacing is added so gaps between samples cannot hide contact.
        dd=cKDTree(surf).query(nearby)[0]
        if np.min(dd)<self.margin+0.0015:
          self.last_block=(self.m.body(self.m.geom_bodyid[g]).name,nearby[dd<self.margin+0.0015],np.array(q6))
          return f"observed obstacle intersects {self.m.body(self.m.geom_bodyid[g]).name}"
      elif np.min(dd:=point_distance(nearby,pos,rot,kind,size,self.m.geom_rbound[g]))<self.margin:
        self.last_block=(self.m.body(self.m.geom_bodyid[g]).name,nearby[dd<self.margin],np.array(q6))
        return f"observed obstacle intersects {self.m.body(self.m.geom_bodyid[g]).name}"
    for i,(b,c,h,name) in enumerate(self.attached):
      rot=self.d.xmat[b].reshape(3,3);center=self.d.xpos[b]+rot@c
      low=center[2]-np.abs(rot[2])@h-self.source.table_z
      allowed=self._table_margin if self.table_baseline is None else min(self._table_margin,self.table_baseline.get(("att",i),np.inf)-0.0005)
      if low<allowed:
        return f"{name} would hit the table"
      if len(cloud):
        local=np.abs((cloud-center)@rot)
        hit=np.all(local<=h+self.margin,axis=1)
        if np.any(hit):
          self.last_block=(name,cloud[hit],np.array(q6))
          return f"observed obstacle intersects {name}"
    # Conservative held-object volume (translation model); do not forget payload clearance.
    if self.held_name and self.held_size is not None:
      center,extent=self.payload_geometry(np.asarray(q6))
      points=cloud
      ob=s.objects.get(self.held_name)
      if ob is not None:
        points=points[~np.all(np.abs(points-ob.center)<np.array(ob.size)/2+0.012,axis=1)]
      zone=getattr(self,"release_zone",None)
      if zone is not None and len(points):
        # Placing into an open container: its rim and contents may not block the held object
        # (user request 2026-11-08: dropping onto cubes already in the box is fine). The arm
        # links and jaws are still checked against them above.
        points=points[~self._in_release_zone(points,zone)]
      if len(points):
        hit=np.all(np.abs(points-center)<extent+self.margin,axis=1)
        if np.any(hit):
          self.last_block=("payload",points[hit],np.array(q6))
          return "held-object clearance blocked"
    return None

  def before_command(self,next_q,predicted_path=None):
    """Check the motion from the measured pose to the next predicted pose.

    predicted_path: recent nominal-plan points (arm.track_points). The measured deviation from
    that path is added to the clearance; beyond max_tracking the prediction is not trusted."""
    q,raw=self.arm.q6()
    self._q_meas=q
    self.source.set_joints(q)
    s=self.fresh_snapshot();self._snapshot=s
    now=self.source.now()
    reason=None
    deviation=0.0
    below=0.0
    if s.error or now-s.stamp>self.max_age:
      reason=s.error or "scene observation expired"
    if getattr(s,"hands",None):
      self.last_hand_time=now
      if reason is None:
        reason=f"hand in workspace ({len(s.hands)})"
    if reason is None and self.active:
      try:
        ob=fresh_object(s,self.active,now,self.max_age,self.allow_occluded)
        if ob.extra.get("occluded_s") is not None:
          self.occluded_uses+=1
        if self.anchor is not None and np.linalg.norm(ob.center-self.anchor)>0.012:
          reason=f"target moved: {self.active}"
      except SceneUnavailable as e:
        reason=str(e)
    if reason is None and self.table_touch_center is not None and self.touch_loads and now-self._touch_sample>=.08:
      self._touch_sample=now
      gp=self.arm.geom(q)['gp'];next_gp=self.arm.geom(next_q)['gp']
      if (gp[2]-self.source.table_z<.08 and next_gp[2]<gp[2]-.0001 and
          np.linalg.norm(gp[:2]-self.table_touch_center)<=self.table_touch_radius and
          np.linalg.norm(np.asarray(next_q)[:4]-q[:4])>1e-5):
        loads={n:abs(self.arm.bus.read(n,'Present_Load')) for n in self.touch_loads}
        if any(loads[n]-self.touch_loads[n]>100 for n in loads):
          self.hold()
          raise TableContact('descent stopped on servo load rise; no further downward command')
    if reason is None and getattr(self,"descent",None) is not None:
      self._descent_check(q,raw,now)
    if reason is None and predicted_path is not None and len(predicted_path):
      here=self.arm.track_points(q)
      dists=np.max(np.linalg.norm(predicted_path-here,axis=2),axis=1)
      k=int(np.argmin(dists))
      deviation=float(dists[k])
      # Friction makes the servos lag *above* a descending path; only a measured position below
      # the predicted one reduces table clearance.
      below=float(max(0.0,np.max(predicted_path[k][:,2]-here[:,2])))
      self.max_deviation=max(self.max_deviation,deviation)
      self.move_peak_deviation=max(self.move_peak_deviation,deviation)
      self.dev_samples.append(deviation)
      self.recent_deviation=deviation
      if deviation>self.max_tracking:
        reason=f"arm left its predicted path by {1000*deviation:.1f} mm; collision prediction not trusted"
    if reason is None:
      # Sweep each command, not just its endpoint. Includes RL command increments.
      n=max(1,int(np.ceil(np.max(np.abs(next_q[:5]-q[:5]))/0.04)))
      for a in np.linspace(0,1,n+1):
        reason=self.check(q+a*(next_q-q),deviation,table_extra=below)
        if reason:
          if deviation>0.004 and self.check(q+a*(next_q-q)) is None:
            reason+=f" (only with the {1000*deviation:.1f} mm measured path deviation added; the arm may be in contact)"
          break
    if reason:
      if self._checking_paused:raise MotionInterrupted(reason,hard_stop=True)
      if self.is_temporary(reason) and self.auto_pause_timeout is not None and self.check(q) is None:
        self._wait_obstacle(next_q,predicted_path,reason)
        return
      self.interruptions.append({"time":now,"reason":reason})
      self.hold()
      raise MotionInterrupted(reason,hard_stop=True)

  def hand_recent(self,window=5.0):
    return self.source.now()-self.last_hand_time<=window

  def is_temporary(self,reason):
    """Conditions a person causes and resolves: pause and wait, do not abort."""
    if reason.startswith(("observed obstacle","held-object clearance blocked","hand in workspace")):
      return True
    # A hand over the target hides it (or corrupts its depth) only for as long as it is there.
    return ("not freshly observed" in reason or "target moved" in reason) and self.hand_recent()

  def _pause_tick(self,reason):
    self.pause_reason=reason
    for _ in range(5):self.arm.plant.wait()  # the arm holds; the preview keeps updating

  def _wait_obstacle(self,next_q,predicted_path,reason):
    """Pause (arm held) until the cause is gone, then re-check everything before resuming.

    Phase 1: no hand and a clear pending sweep for 1 s (target checks off: a hand may hide or
    shift the target). Phase 2: the target is checked again; still hidden -> keep waiting,
    really moved -> raise "target moved" (the skill re-plans), otherwise resume."""
    started=self.source.now();clear_since=None
    active,anchor,allow=self.active,self.anchor,self.allow_occluded
    self.hold()
    self.pause_events.append({'time':started,'reason':reason,'state':'paused'})
    print('PAUSED:',reason,flush=True)
    self._checking_paused=True
    def timed_out():
      return self.auto_pause_timeout and self.source.now()-started>=self.auto_pause_timeout
    try:
      self.active=None;self.anchor=None
      while True:
        self._pause_tick(reason)
        try:
          self.before_command(next_q,None)
        except MotionInterrupted as e:
          if not self.is_temporary(str(e)) and not str(e).startswith(("scene stale","scene observation expired")):raise
          clear_since=None;reason=str(e)
        else:
          if clear_since is None:clear_since=self.source.now()
          if self.source.now()-clear_since>=1.0:break
        if timed_out():
          raise MotionInterrupted('pause timed out ('+reason+'); arm remains held',hard_stop=True)
      self.active,self.anchor,self.allow_occluded=active,anchor,allow
      while True:
        try:
          self.before_command(next_q,None)
          break
        except MotionInterrupted as e:
          text=str(e)
          if "target moved" in text and not self.hand_recent(1.0):
            raise  # it really moved: re-plan from fresh observations, never resume stale
          if not self.is_temporary(text) and "not freshly observed" not in text:raise
          reason=text
        if timed_out():
          raise MotionInterrupted('pause timed out ('+reason+'); arm remains held',hard_stop=True)
        self._pause_tick(reason)
      self.pause_events.append({'time':self.source.now(),'state':'resumed','wait_s':self.source.now()-started})
      print('RESUMING: clear for 1 s, target re-checked',flush=True)
      if predicted_path is not None:
        raise ReplanAfterPause('pause cleared; replan from measured pose')
    finally:
      self.active,self.anchor,self.allow_occluded=active,anchor,allow
      self._checking_paused=False;self.pause_reason=None

  def wait_until_fresh(self,name,allow_occluded=0.0,why="target not observed"):
    """Wait (arm held) for `name` to be freshly observed while a hand is or was just present."""
    started=self.source.now();announced=False
    while True:
      s=self.fresh_snapshot()
      if getattr(s,"hands",None):self.last_hand_time=self.source.now()
      try:
        if s.hands:raise SceneUnavailable(f"hand in workspace ({len(s.hands)})")
        return fresh_object(s,name,self.source.now(),allow_occluded=allow_occluded)
      except SceneUnavailable as e:
        if self.auto_pause_timeout is None or not (s.hands or self.hand_recent()):raise
        if not announced:
          print('PAUSED:',e,flush=True);announced=True
          self.pause_events.append({'time':started,'reason':str(e),'state':'paused'})
        if self.auto_pause_timeout and self.source.now()-started>=self.auto_pause_timeout:
          raise SceneUnavailable(f"pause timed out ({e})") from e
        self._pause_tick(str(e))
      finally:
        if announced and self.pause_reason and not getattr(s,"hands",None):pass

  def fresh_snapshot(self,timeout=1.0):
    """Latest scene; if it is older than max_age, wait (no new command: the arm holds) for a
    fresh one, up to `timeout`. A stale result is still returned and then refused by callers."""
    s=self.source.snapshot()
    if not s.error and self.source.now()-s.stamp>self.max_age and hasattr(self.source,"wait_fresh"):
      t0=self.source.now()
      s=self.source.wait_fresh(self.max_age*0.75,timeout)
      self.fresh_waits.append(self.source.now()-t0)
    return s

  def hold(self):
    # Preserve the gripper goal so a failure does not release a carried object.
    try:
      raw=self.arm.bus.sync_read("Present_Position")
      self.arm.bus.sync_write("Goal_Position",{n:self.arm.plant._clamp(n,raw[n]) for n in JOINT_ORDER[:5]})
    except OSError as e:
      return f"hold could not be confirmed over the servo bus: {e}"
    return None

  def check_actual_contacts(self):
    if self.world is None:
      return
    for c in self.world.data.contact:
      a,b=int(c.geom1),int(c.geom2)
      if c.dist>=-0.0005 or not (a in self.all_robot or b in self.all_robot):
        continue
      if b in self.all_robot and a not in self.all_robot:
        a,b=b,a
      if self._allowed(a,b):
        continue
      if (self.m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE and
          self._jaw_touch(a,self.world.data.geom_xpos[a]) and c.dist>=-.001):continue
      event={"time":self.source.now(),"geom1":self.m.geom(a).name or self.m.body(self.m.geom_bodyid[a]).name,
             "geom2":self.m.geom(b).name or self.m.body(self.m.geom_bodyid[b]).name,"penetration_m":-float(c.dist)}
      self.actual_contacts.append(event)
      self.hold()
      raise MotionInterrupted(f"unexpected physical contact: {event['geom1']} / {event['geom2']}",hard_stop=True)


class Motion:
  def __init__(self,arm,source,controller="ik",speed=0.14):
    self.arm,self.source=arm,source
    self.guard=SceneGuard(arm,source)
    self.arm.path_guard=self.guard.check
    self.arm.aim_guard=self.guard.check_hard
    self.arm.execution_guard=self.guard.before_command
    self.arm.travel_controller=controller
    self.arm.speed=speed
    self.arm.settle_steps=8
    self.controller=controller
    self.moves=[]

  def observe_pose(self,max_replans=5):
    """arm.observe_pose with the same pause handling as goto: a pause during the move (hand,
    obstacle) ends with ReplanAfterPause, and the move is re-planned from the measured pose.
    Called directly, it escaped and ended a run after a hand moved the target (2026-11-08)."""
    for _ in range(max_replans+1):
      try:
        r=self.arm.observe_pose(self.source.table_z)
      except ReplanAfterPause:
        self.moves.append({'target':'observe_pose','status':'replan_after_pause','controller':self.controller})
        continue
      self.moves.append({'target':'observe_pose','status':r,'controller':self.controller})
      return r
    raise MotionInterrupted('too many pause/replan cycles on the way to the observation pose; arm held',hard_stop=True)

  def goto(self,p,tilt=0.0,yaw=None,watch=None,allow_occluded=0.0,tolerance=None,pause_on_block=True):
    """pause_on_block: if any observed obstacle (a new object, a hand) blocks the planned path,
    hold and re-plan until it is gone (auto-pause on); otherwise raise so callers can re-route."""
    if tolerance is not None:
      old=self.arm.residual_tol;self.arm.residual_tol=tolerance
      try:
        return self.goto(p,tilt,yaw,watch,allow_occluded,pause_on_block=pause_on_block)
      finally:
        self.arm.residual_tol=old
    p=np.array(p,float)
    s=self.guard.fresh_snapshot()
    self.guard._q_meas=self.arm.q6()[0]
    self.guard.set_table_baseline(self.guard._q_meas)
    anchor=self.guard.wait_until_fresh(watch,allow_occluded).center if watch else None
    self.guard.watch(watch,anchor,allow_occluded)
    self.guard._snapshot=self.guard.fresh_snapshot()
    start=time.perf_counter(); sim_start=self.source.now()
    paused_at=None
    replans=0
    while True:
      try:
        r=self.arm.move_to(100*p[0],100*p[1],100*(p[2]-self.source.table_z),tilt,yaw)
      except ReplanAfterPause:
        replans+=1
        if replans>5:raise MotionInterrupted('too many pause/replan cycles; arm held',hard_stop=True)
        g=self.guard;g._q_meas=self.arm.q6()[0];g.set_table_baseline(g._q_meas)
        g._snapshot=g.fresh_snapshot()
        if watch:
          current=g.wait_until_fresh(watch,allow_occluded)
          if np.linalg.norm(current.center-anchor)>.012:raise MotionInterrupted('target moved: '+watch)
        self.moves.append({'target_m':p.tolist(),'status':'replan_after_pause','controller':self.controller})
        continue
      g=self.guard
      blocked=not r.startswith("ok") and ("observed obstacle" in r or "held-object clearance" in r) and (pause_on_block or g.hand_recent())
      if not blocked or g.auto_pause_timeout is None:
        break
      # Something observed (a new object, a hand) blocks the planned path: hold and re-plan.
      if paused_at is None:
        paused_at=self.source.now();print("PAUSED: planned path blocked:",r,flush=True)
        g.pause_events.append({'time':paused_at,'reason':r,'state':'paused'})
      if g.auto_pause_timeout and self.source.now()-paused_at>=g.auto_pause_timeout:
        r="pause timed out: "+r;break
      g._pause_tick(r)
      g.pause_reason=None
      g._snapshot=g.fresh_snapshot()
      if getattr(g._snapshot,"hands",None):g.last_hand_time=self.source.now()
    self.moves.append({"target_m":p.tolist(),"status":r,"wall_s":time.perf_counter()-start,"residual_mm":None if self.arm.last_residual is None else 1000*self.arm.last_residual,
                       "motion_s":self.source.now()-sim_start,"controller":self.controller})
    if not r.startswith("ok"):
      self.guard.hold()
      raise MotionInterrupted(r)

  def transfer(self,p,clearance,yaw=None,watch=None,allow_occluded=0.0):
    """Checked up/across/down candidates, then a joint-space detour, so the arm goes around a
    new obstacle when it can. If every route is blocked by observed obstacles and auto-pause is
    on, hold and retry until they are gone (or the pause times out)."""
    started=None
    while True:
      try:
        return self._transfer_once(p,clearance,yaw,watch,allow_occluded)
      except MotionInterrupted as e:
        g=self.guard;text=str(e)
        if e.hard_stop or g.auto_pause_timeout is None or not ("observed obstacle" in text or "held-object clearance" in text):
          raise
        if started is None:
          started=self.source.now();print("PAUSED: every route blocked by an observed obstacle; waiting",flush=True)
          g.pause_events.append({'time':started,'reason':text,'state':'paused'})
        if g.auto_pause_timeout and self.source.now()-started>=g.auto_pause_timeout:
          raise MotionInterrupted("pause timed out: "+text,hard_stop=True) from e
        for _ in range(4):g._pause_tick("route blocked: "+text[:80])
        g.pause_reason=None

  def _transfer_once(self,p,clearance,yaw=None,watch=None,allow_occluded=0.0):
    here=self.arm.geom(self.arm.q6()[0])["gp"]
    p=np.array(p,float)
    last=None
    for extra in (0.0,0.04,0.08):
      h=max(clearance+extra,here[2],p[2])
      try:
        if here[2]<h-0.006:
          # Rise without reorienting: wrist turns swing the side-mounted D405 (measured: towards the
          # wall from the observation pose). Orientation changes happen on the horizontal leg.
          self.goto([*here[:2],h],tilt=None,yaw=None,watch=watch,allow_occluded=allow_occluded,pause_on_block=False)
        self.goto([*p[:2],h],yaw=yaw,watch=watch,allow_occluded=allow_occluded,pause_on_block=False)
        if p[2]<h-0.006:
          self.goto(p,yaw=yaw,watch=watch,allow_occluded=allow_occluded,pause_on_block=False)
        return
      except MotionInterrupted as e:
        last=e
        if e.hard_stop:
          raise
        text=str(e)
        # Observation failures end the attempt; an observed obstacle tries the next route.
        if "target moved" in text or "not freshly observed" in text or "scene" in text or "fresh" in text:
          raise
        here=self.arm.geom(self.arm.q6()[0])["gp"]
    # Height candidates are fast; bounded joint-space planning handles lateral detours.
    for _ in range(4):
      try:
        self.joint_detour(p,yaw,watch,allow_occluded=allow_occluded)
        return
      except ReplanAfterPause:
        continue  # paused on the detour; plan a new one from the measured pose
      except MotionInterrupted as e:
        raise MotionInterrupted(f"no checked transfer path: {last}; detour: {e}",e.hard_stop) from e
    raise MotionInterrupted('too many pause/replan cycles on the detour; arm held',hard_stop=True)

  def _detour_goal(self,p,yaw,start):
    goal,err=self.arm._pose(p,0,yaw,start)
    if err<=.003 and self.guard.check(goal) is None:return goal
    # An unconstrained carry may turn the wrist. Raising the same blocked
    # camera orientation repeatedly cannot clear a wall beside the box.
    if yaw is None:
      lo,hi=self.arm.kin.m.jnt_range[self.arm.kin.m.joint('wrist_roll').id]
      choices=[]
      for offset in (30,-30,60,-60,90,-90,180,-180):
        seed=start.copy();seed[4]+=np.radians(offset)
        if not lo<=seed[4]<=hi:continue
        q,e=self.arm._pose(p,0,None,seed)
        if e<=.003 and self.guard.check(q) is None:
          choices.append(q)
      if choices:return min(choices,key=lambda q:float(np.linalg.norm(q[:5]-start[:5])))
    raise MotionInterrupted('destination pose is blocked or unreachable')

  def joint_detour(self,p,yaw=None,watch=None,budget_s=2.0,allow_occluded=0.0):
    from so101_reach.real.grasp_replay import ReplayPlan,joint_plan
    start=self.arm.q6()[0]
    goal=self._detour_goal(p,yaw,start)
    lo=np.r_[self.arm.kin.lo,self.arm.kin.m.jnt_range[self.arm.kin.m.joint("wrist_roll").id,0]]
    hi=np.r_[self.arm.kin.hi,self.arm.kin.m.jnt_range[self.arm.kin.m.joint("wrist_roll").id,1]]
    rng=np.random.default_rng(0);deadline=time.perf_counter()+budget_s
    trees=[([start[:5].copy()],[-1]),([goal[:5].copy()],[-1])]
    def edge(a,b):
      for v in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/.05))+1)):
        q=start.copy();q[:5]=a+v*(b-a)
        if self.guard.check(q):return False
      return True
    def extend(tree,target):
      nodes,parents=tree
      i=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
      diff=target-nodes[i];dist=np.linalg.norm(diff)
      q=nodes[i]+diff*min(1,.25/max(dist,1e-9))
      if not edge(nodes[i],q):return None
      nodes.append(q);parents.append(i);return len(nodes)-1
    def trace(tree,i):
      out=[]
      while i>=0:out.append(tree[0][i]);i=tree[1][i]
      return out[::-1]
    path=None
    for k in range(800):
      if time.perf_counter()>deadline:break
      a=k%2;b=1-a
      sample=trees[b][0][0] if rng.random()<.2 else rng.uniform(lo,hi)
      ia=extend(trees[a],sample)
      if ia is None:continue
      qa=trees[a][0][ia]
      for _ in range(40):
        ib=extend(trees[b],qa)
        if ib is None:break
        if np.linalg.norm(trees[b][0][ib]-qa)<1e-5:
          pa,pb=trace(trees[a],ia),trace(trees[b],ib)
          path=pa+pb[::-1][1:] if a==0 else pb+pa[::-1][1:]
          break
      if path is not None:break
    if path is None:raise MotionInterrupted("joint-space planning budget exhausted; holding")
    # Shortcut only collision-free edges. No Cartesian-path assumptions are made.
    i=0
    while i<len(path)-2:
      j=len(path)-1
      while j>i+1 and not edge(path[i],path[j]):j-=1
      path=path[:i+1]+path[j:];i+=1
    parts=[];offset=0.0
    for a,b in zip(path[:-1],path[1:]):
      qa,qb=start.copy(),start.copy();qa[:5],qb[:5]=a,b
      duration=max(.3,float(np.max(np.abs(b-a)))/.65)
      segment=joint_plan(self.arm.kin,qa,qb,duration,self.arm.grip_goal,self.arm.grip_goal)
      parts.append((segment,offset));offset+=segment.t[-1]+.02
    if not parts:return
    plan=ReplayPlan(np.concatenate([s.t+t for s,t in parts]),np.concatenate([s.q for s,_ in parts]),
                    np.concatenate([s.gripper_goal for s,_ in parts]),np.zeros(sum(len(s.t) for s,_ in parts)),
                    np.zeros(sum(len(s.t) for s,_ in parts)),np.array([p]))
    snapshot=self.source.snapshot()
    self.guard.watch(watch,fresh_object(snapshot,watch,self.source.now(),allow_occluded=allow_occluded).center if watch else None,
                     allow_occluded=allow_occluded)
    r=self.arm._play(plan,np.asarray(p))
    self.moves.append({"target_m":np.asarray(p).tolist(),"status":r,"controller":"rrt_ik_detour","waypoints":len(path)})
    if not r.startswith("ok"):raise MotionInterrupted(r)
