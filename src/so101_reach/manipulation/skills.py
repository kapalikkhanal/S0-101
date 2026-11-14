"""Geometric pick/place skills with explicit preconditions and measured postconditions."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from so101_reach.manipulation.motion import MotionInterrupted,TableContact
from so101_reach.manipulation.scene import SceneUnavailable, footprint_contains, fresh_object
from so101_reach.manipulation.destination import prepare_target,copy_detection,packed_release_point
from so101_reach.manipulation.relations import placement_relation

# Grasp candidates must clear neighbours by 7 mm (execution keeps 3 mm plus measured tracking error).
GRASP_PLANNING_MARGIN = 0.007
# A pick target hidden by the arm itself may be approached on its last fresh measurement for
# at most this long (head-camera geometry hides it under the gripper). Never applies to placement.
OCCLUSION_ALLOWANCE_S = 8.0  # head-camera trials: target hidden for most of the approach (3-6 s wall)
# With the wrist camera, which re-observes the target before the descent.
WRIST_CONFIRMED_ALLOWANCE_S = 30.0
# Wrist check at grasp height: tolerated object offset along the jaw length, image pixels.
LONGITUDINAL_TOL_PX = 35.0
# Re-plans after the target was observed to move (each re-observes it first); not failures.
MAX_MOVED_REPLANS = 8
# Jaw contact pads extend ~10 mm either side of the grip point across the closing axis
# (fixed_jaw_box6/7 half-size 9-10 mm in the gripper frame).
JAW_PAD_HALF = 0.010
# Model wrist-roll window (deg) of the verified real grasps. 2026-11-07 state-machine trials on the
# red cube: 4/4 verified at roll +14..+34 deg; 0/4 at -57..-148 deg (every contact stop and empty
# close happened there; shoulder +7..+9 / elbow 16..19 deg vs -2..-7 / 30..37 deg). Likely the same
# kind of branch-dependent height error as the elbow-reversed branch. Equivalent grasps 90 deg
# apart in roll are preferred inside the window; outside it is still allowed when nothing else fits.
ROLL_VALIDATED_DEG = (-20.0, 70.0)  # logged only; arm shape (below) is the better predictor
# Arm shape is the better predictor. Across all 14 hardware grasp attempts on 2026-11-07, every
# physical failure had shoulder lift >= +2.9 deg and elbow <= 25 deg (an extended arm). Every
# verified grasp but one had lift <= -2 deg and elbow >= 31 deg; the exception (pick_place ik_1)
# had elbow 22 deg and also needed a contact stop. The RL pick-and-place rl_1 had an in-window roll
# but elbow 17 deg and a 16.8 mm tracking deviation. The object's distance sets most of it: the
# best candidate's elbow is ~28 deg at x=0.18 m and ~9 deg at x=0.22 m (y=-0.13). Equivalent
# grasps are ranked by elbow, and a grasp below this value is announced.
BENT_ELBOW_MIN_DEG = 22.0


class Executor:
  def __init__(self,motion,log_dir,retries=2):
    self.motion,self.arm,self.source=motion,motion.arm,motion.source
    self.log_dir=Path(log_dir);self.log_dir.mkdir(parents=True,exist_ok=True)
    self.retries=retries
    self.parked=False
    self.wrist=None  # optional WristServo (real hardware): final centring and grasp evidence
    self.events=[]
    self.phase="idle"
    self.targets={}
    self.placed_objects={}
    offset_file=Path(__file__).resolve().parents[3]/"configs/grasp_offsets.json"
    offsets=json.loads(offset_file.read_text()) if offset_file.exists() else {}
    self.longitudinal_bias_m=float(offsets.get("longitudinal_bias_m",.005))
    self.grasp_position=offsets.get("grasp_position","center")
    self.grasp_offset_m=None
    self.grasp_shift_xy=np.zeros(2)
    self.local_retries=2
    self.table_contact=True
    # 'fsm': bounded grasp state machine (manipulation/grasp_fsm.py) on the wrist-camera path.
    # 'legacy': the earlier iterative centring loop (_wrist_centre/_close_near/_wrist_descend).
    self.grasp_controller="fsm"
    self.grasp_score_threshold=0.60
    self.fsm_local_retries=1
    self.grasp_stats=None
    if not np.isfinite(self.longitudinal_bias_m) or abs(self.longitudinal_bias_m)>.010:
      raise ValueError("grasp longitudinal bias must be within 10 mm")

  def event(self,kind,**data):
    ev={"kind":kind,"phase":self.phase,"time":self.source.now(),**data}
    self.events.append(ev)
    with (self.log_dir/"events.jsonl").open("a") as f:
      f.write(json.dumps(ev,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+"\n")
    if type(self.arm.plant).__name__=="RealPlant":
      from so101_reach.manipulation.payload_state import record_payload,STATE
      if kind=="payload_attached":record_payload(ev,self.log_dir)
      elif kind=="resume_held":record_payload(data['attachment'],self.log_dir)
      elif kind=="placement_verified" and data.get('success'):STATE.unlink(missing_ok=True)
    if kind in ("phase","recovery","failure"):
      print(f"[{self.phase}] {data}",flush=True)

  def set_phase(self,name,**data):
    if name in ('approach','descend','carry','lift','retreat','local_retry'):self.parked=False
    self.phase=name;self.event("phase",**data)

  def observe(self,name,allow_occluded=0.0):
    # Waits (arm held) while a hand hides the object; otherwise behaves like fresh_object.
    d=self.motion.guard.wait_until_fresh(name,allow_occluded)
    if d.extra.get("occluded_s") is not None:
      self.event("hidden_by_arm",object=name,last_measured_s_ago=d.extra["occluded_s"])
    return d

  def wait(self,seconds):
    for _ in range(max(1,int(seconds/self.arm.plant.dt))):
      self.arm.plant.wait()

  def clearance(self,extra=0.04):
    ds=self.source.snapshot().objects.values()
    return max([self.source.table_z+0.10]+[d.top_z+extra for d in ds if d.name!=self.motion.guard.held_name])

  def _grasp_xy(self,ob,along):
    a=np.radians(ob.yaw_deg)
    return ob.center[:2]+along*np.array([np.cos(a),np.sin(a)])+self.grasp_shift_xy

  def _hidden_allowance(self):
    """How long a pick target hidden by the arm may be approached on its last head measurement.
    With the wrist camera, the object is re-observed at the alignment pose (2 cm limit) before any
    motion near it, so the head allowance only has to cover the slow approach (hardware: the
    cross-table move takes 12-14 s and the cube is hidden for part of it)."""
    return WRIST_CONFIRMED_ALLOWANCE_S if self.wrist is not None else OCCLUSION_ALLOWANCE_S

  def _choose_grasp_point(self,ob):
    here=self.arm.geom(self.arm.q6()[0])["gp"][:2]
    forward=ob.center[:2]-here
    forward/=max(np.linalg.norm(forward),1e-9)
    left=np.array([-forward[1],forward[0]])
    directions={"center":np.zeros(2),"far":forward,"near":-forward,"left":left,"right":-left}
    if self.grasp_position not in directions:raise ValueError("unknown grasp position")
    distance=.25*min(ob.size[:2]) if self.grasp_offset_m is None else self.grasp_offset_m
    if self.grasp_position!="center" and distance>min(ob.size[:2])/2-.004:
      raise ValueError("requested grasp offset would lie outside the object")
    self.grasp_shift_xy=directions[self.grasp_position]*distance
    self.event("grasp_position",position=self.grasp_position,offset_m=self.grasp_shift_xy,
               object_center_m=ob.center[:2],grasp_point_m=ob.center[:2]+self.grasp_shift_xy)

  def _select_grasp(self,ob,jaw,elongated,curved):
    """First collision-free candidate in preference order: centred before offset, default wrist
    roll before the flipped arrangement. Elongated objects may be gripped off-centre along their
    long axis while the jaw pads stay at least 2.5 mm inside the object's ends."""
    g=self.motion.guard
    yaws=[jaw] if elongated else [jaw,jaw+90]
    offsets=[0.0]
    if elongated:
      limit=ob.size[0]/2-JAW_PAD_HALF-0.0025
      for o in (0.005,0.010,0.015,0.020):
        if o<=limit+1e-9:offsets+= [-o,o]
    low=ob.support_z+(.55 if curved or self.wrist is not None else .90)*ob.size[2]
    current=self.arm.q6()[0]
    flipped=(current[4]+2*np.pi)%(2*np.pi)-np.pi
    old_margin=g.margin
    min_err,last_reason=np.inf,None
    try:
      # Reserve extra clearance for the servo tracking error during contact approach.
      g.margin=max(g.margin,GRASP_PLANNING_MARGIN)
      fallback=None
      for along in offsets:
        normal=[]
        for j in yaws:
          for roll in (None,flipped):
            self.arm.preferred_roll=roll
            for z in np.arange(low,ob.top_z+0.005,0.002):
              qs,err=self.arm._pose(np.array([*self._grasp_xy(ob,along),z]),0,j,current)
              min_err=min(min_err,err)
              reason=self.arm._check(qs) if err<0.003 else None
              if err<0.003 and reason is not None:last_reason=reason
              if err<0.003 and reason is None:
                # Prefer the normal arm branch (elbow >= 0). On hardware the elbow-reversed branch
                # (shoulder ~+40 deg, elbow ~-31 deg, chosen for jaw yaws of +12..+22 deg) left the
                # real jaws above the cube top: 0/6 grasps there vs 2/2 in the normal branch.
                if qs[2]>=-0.05:
                  normal.append((j,roll,along,np.degrees(qs).round(1)))
                elif fallback is None:
                  fallback=(j,roll,along,np.degrees(qs).round(1))
                break
        if normal:
          # Among equivalent normal-branch grasps (a cube's yaw is modulo 90 deg; default/flipped
          # roll), prefer the most bent arm (largest elbow flex); see BENT_ELBOW_MIN_DEG.
          best=max(normal,key=lambda c:c[3][2])
          j,roll,along_c,qd=best
          inside=ROLL_VALIDATED_DEG[0]<=qd[4]<=ROLL_VALIDATED_DEG[1]
          self.event("grasp_branch",branch="normal",q_deg=qd,roll_in_validated_window=bool(inside),
                     extended_arm=bool(qd[2]<BENT_ELBOW_MIN_DEG),
                     candidates=[{"jaw_yaw_deg":round(float(c[0]),1),"roll":"default" if c[1] is None else "flipped",
                                  "wrist_roll_deg":float(c[3][4]),"shoulder_lift_deg":float(c[3][1]),
                                  "elbow_deg":float(c[3][2])} for c in normal])
          if qd[2]<BENT_ELBOW_MIN_DEG:
            print(f"[grasp] warning: {ob.name} at ({100*ob.center[0]:.1f}, {100*ob.center[1]:.1f}) cm needs an extended arm "
                  f"(elbow {qd[2]:.0f} deg; every verified grasp used >= {BENT_ELBOW_MIN_DEG:.0f}). Tracking error is larger here; "
                  "if the grasp fails, move the object closer to the base.",flush=True)
          return j,roll,along_c
        if fallback is not None and along!=0.0:
          break
      if fallback is not None:
        self.event("grasp_branch",branch="elbow_reversed",q_deg=fallback[3])
        return fallback[:3]
    finally:
      g.margin=old_margin
      self.arm.preferred_roll=None
    if min_err>=0.003:
      # Not a collision: the arm cannot put the grip point there (about 0.41 m from the shoulder
      # axis at table height). Re-approaching cannot help.
      xy=self._grasp_xy(ob,0.0)
      raise MotionInterrupted(f"{ob.name} at ({100*xy[0]:.1f}, {100*xy[1]:.1f}) cm is out of reach: the closest "
                              f"reachable grasp pose is {1000*min_err:.0f} mm away",hard_stop=True)
    raise MotionInterrupted(f"no face-aligned grasp clears surrounding geometry (last: {last_reason})")

  def _grasp(self,step):
    g=self.motion.guard;floor=self.arm.floor
    old=(g.table_touch_center,g.table_touch_radius,g.touch_loads)
    try:return self._grasp_impl(step)
    finally:
      self.arm.floor=floor
      g.table_touch_center,g.table_touch_radius,g.touch_loads=old

  def _grasp_impl(self,step):
    g=self.motion.guard
    attempt=0;moved=0
    while attempt<=self.retries:
      self.set_phase("approach",object=step.pick,attempt=attempt+1,replans_after_move=moved)
      ob=self.observe(step.pick)
      self._choose_grasp_point(ob)
      width=min(ob.size[:2])
      if width>0.075 or min(ob.size)<=0.006:
        raise ValueError("object geometry is outside the validated grasp domain (6–75 mm width)")
      # Reject picking the bottom of a stack.
      for n,d in self.source.snapshot().objects.items():
        if n!=step.pick and abs(d.support_z-ob.top_z)<0.012 and footprint_contains(ob,d.center[:2]):
          raise ValueError(f"cannot pick {step.pick}: {n} rests on it")
      g.grasp_contact=step.pick
      g.watch(step.pick,ob.center)
      opening=self.arm.percent_for_gap(width+0.024)
      self.arm.gripper(opening,"medium")
      elongated=ob.size[0]>1.3*width
      jaw=ob.yaw_deg+90 if elongated else ob.yaw_deg
      try:
        curved=any(w in step.pick.lower().split() for w in ("ball","sphere","orange","apple"))
        if curved:jaw=0.0  # spherical objects have no meaningful min-area-rectangle yaw
        jaw,roll,along=self._select_grasp(ob,jaw,elongated,curved)
        self.arm.preferred_roll=roll
        self.event("grasp_candidate",jaw_yaw_deg=jaw,roll="default" if roll is None else "flipped",along_mm=1000*along)
        pre=np.array([*self._grasp_xy(ob,along),ob.top_z+0.045])
        g.move_peak_deviation=0.0
        self.motion.transfer(pre,self.clearance(),yaw=jaw,watch=step.pick,allow_occluded=self._hidden_allowance())
        # Reacquire after travel; object motion causes a bounded restart, not a stale descent.
        # A head camera loses the target under the gripper: its last measurement may be used for
        # OCCLUSION_ALLOWANCE_S, flagged; a moved target then yields no contact and a retry.
        ob_start=ob
        ob=self.observe(step.pick,allow_occluded=self._hidden_allowance())
        if self.wrist is not None and (ob.extra.get("occluded_s") is not None or ob.extra.get("partial") is not None
                                       or min(ob.size[:2])<0.8*min(ob_start.size[:2])):
          # Height/size do not change during the approach; the head's latest measurement is taken
          # with the arm partly covering the object (measured: 20-25 mm width, low top, for a 34 mm
          # cube). Plan heights from the unoccluded start measurement; position comes from the wrist.
          from dataclasses import replace as _replace
          self.event("height_from_start_measurement",start_top_mm=1000*(ob_start.top_z-self.source.table_z),
                     latest_top_mm=1000*(ob.top_z-self.source.table_z),start_width_mm=1000*min(ob_start.size[:2]),latest_width_mm=1000*min(ob.size[:2]))
          ob=_replace(ob,top_z=ob_start.top_z,support_z=ob_start.support_z,size=ob_start.size)
        grasp_xy=self._grasp_xy(ob,along)
        g.contact_override.pop(step.pick,None)
        if self.wrist is not None and self.grasp_controller=="fsm":
          self._grasp_fsm(step,ob,jaw,grasp_xy)
        else:
          if self.wrist is not None:
            head_xy=np.array(grasp_xy,float)
            grasp_xy=self._wrist_centre(step,ob,grasp_xy,jaw)
            # The jaws may touch the object where the wrist found it, not where the head measured it.
            from dataclasses import replace as _replace
            shift=np.array([*(grasp_xy-head_xy),0.0])
            g.contact_override[step.pick]=_replace(ob,center=ob.center+shift)
          if self.table_contact:
            self.arm.floor=0.0
            g.allow_jaw_table_touch(ob)
          q,_=self.arm.q6()
          candidate=None
          low=ob.support_z+(.55 if curved or self.wrist is not None else .90)*ob.size[2]
          old_margin=g.margin
          # Execution requires 3 mm + the measured deviation; plan the descent with what this arm
          # actually showed during the approach (hardware: up to ~9 mm vs ~3-4 mm in simulation).
          dev=g.move_peak_deviation
          if self.wrist is not None and len(g.dev_samples)>=10:
            # Alignment moves near the object; their transients peak at 8-10 mm on hardware while
            # execution checks the instantaneous deviation every tick anyway.
            dev=float(np.percentile(g.dev_samples,75))
          descent_margin=max(GRASP_PLANNING_MARGIN,old_margin+dev)
          self.event("grasp_height_margin",margin_mm=1000*descent_margin,approach_peak_deviation_mm=1000*g.move_peak_deviation)
          try:
            g.margin=descent_margin
            for z in np.arange(low,ob.top_z+0.005,0.002):
              p=np.array([*grasp_xy,z])
              qs,err=self.arm._pose(p,0,jaw,q)
              if err<0.003 and self.arm._check(qs) is None:
                candidate=p;break
          finally:
            g.margin=old_margin
          if candidate is None:
            g.margin=descent_margin
            try:
              reason=self.arm._check(qs) if err<0.003 else f"IK error {1000*err:.1f} mm"
            finally:
              g.margin=old_margin
            self.event("grasp_height_rejected",last_reason=reason,margin_mm=1000*descent_margin)
            raise MotionInterrupted(f"no collision-free grasp pose with current jaw opening (last: {reason})")
          self.set_phase("descend",target_m=candidate)
          # _wrist_descend grabs a fresh frame itself. A planning delay must not
          # silently bypass visual verification and local retries.
          wrist_fresh=self.wrist is not None
          # Near the table the P-servos stop short (hardware: 6-14 mm above low targets). The jaws
          # straddle the object either way; the grasp is then judged by contact and lift evidence.
          if wrist_fresh:
            candidate=self._close_near(step,ob,candidate,jaw)
            # Fine centring at grasp height can add another correction after alignment.
            # Carry the same corrected pose into both contact allowance and payload attachment.
            from dataclasses import replace as _replace
            shift=np.array([*(candidate[:2]-head_xy),0.0])
            g.contact_override[step.pick]=_replace(ob,center=ob.center+shift)
          else:
            self.motion.goto(candidate,yaw=jaw,watch=step.pick,allow_occluded=OCCLUSION_ALLOWANCE_S)
            self.set_phase('close');g.watch(None)
            r=self.arm.gripper(0,step.force);self.event('gripper',result=r)
        if not self.arm.state()["holding_signal"]:
          self._wrist_blob(step,"miss")
          raise MotionInterrupted("gripper did not establish contact")
        self._wrist_before_lift=self._wrist_blob(step,"closed")
        gp=self.arm.geom(self.arm.q6()[0])["gp"]
        below=float(gp[2]-ob.support_z)
        self.arm.held={"name":step.pick,"below_gp":below,"size":ob.size}
        g.held_name,g.held_below,g.held_size=step.pick,below,ob.size
        # The wrist measured 47 mm on the failing orange, versus 37 mm from
        # the head. Use that uncertainty in BOTH payload collision bounds and
        # self-observation filtering, rather than modelling the carried fruit too small.
        wrist_width=float(getattr(self,'wrist_width',0) or 0)
        if getattr(self.source,'world',None) is None and 0<wrist_width<=1.6*max(ob.size[:2]):
          g.held_size=(max(ob.size[0],wrist_width),max(ob.size[1],wrist_width),ob.size[2])
          self.arm.held['size']=g.held_size
          self.event('payload_envelope',measured_size_m=ob.size,collision_size_m=g.held_size,wrist_width_m=wrist_width)
        g.attach(step.pick,g.contact_override.get(step.pick,ob))
        self.event("payload_attached",object=step.pick,size_m=ob.size,below_gp_m=below,
                   joints_rad=self.arm.q6()[0],offset_m=g.held_transform[0],rotation=g.held_transform[1],collision_size_m=g.held_size)
        self.set_phase("lift")
        # A short vertical lift is a physical test of the grasp, not a language-model claim.
        self.motion.goto([*gp[:2],max(gp[2]+0.045,self.clearance()+below)],tilt=None,yaw=jaw)
        self.wait(0.12)
        self._verify_lift(step,ob)
        return ob,below
      except (MotionInterrupted,SceneUnavailable) as e:
        self.event("recovery",reason=str(e))
        if isinstance(e,SceneUnavailable) or (getattr(e,"hard_stop",False) and "target moved" not in str(e)):
          raise
        if self.arm.state()["holding_signal"]:
          raise  # Do not automatically open a potentially held object after failure.
        # A target that was really moved is re-acquired and approached again; this follows it
        # (bounded) without using up the failure budget.
        moved_replan="target moved" in str(e) and moved<MAX_MOVED_REPLANS
        if not moved_replan and attempt==self.retries:
          raise
        if moved_replan:
          moved+=1
        else:
          attempt+=1
        if self.grasp_stats is not None:
          self.grasp_stats["re_approaches"]+=1
          self.grasp_stats.setdefault("re_approach_reasons",[]).append(str(e))
        g.watch(None)
        if self.wrist is not None:
          # Look again from a pose that does not hide the object from the head camera.
          r=self.motion.observe_pose()
          self.event("observation_pose",result=r,reason="retry")
          if not r.startswith("ok"):
            raise MotionInterrupted(f"could not reach the observation pose for a retry: {r}")
          self.wait(1.0)
        self.wait(0.10)
    raise AssertionError("unreachable")

  def _grasp_fsm(self,step,ob,jaw,grasp_xy):
    """Align once, descend, score, close (manipulation/grasp_fsm.py). Leaves the jaws closed on
    the object with contact, or raises: ObjectLost lets _grasp_impl re-approach; anything else
    is a hard stop with a specific diagnosis."""
    from so101_reach.manipulation.grasp_fsm import GraspFSM
    g=self.motion.guard
    if self.table_contact:
      # Jaws may touch the table next to the object (geometric permission). The effort limit is
      # the explicit per-descent monitor, not the legacy one-baseline load check.
      self.arm.floor=0.0
      g.allow_jaw_table_touch(ob,sample_loads=False)
    fsm=GraspFSM(self,self.grasp_score_threshold,self.fsm_local_retries,stats=self.grasp_stats)
    self.grasp_stats=fsm.stats
    fsm.run(step,ob,jaw,grasp_xy)

  def _close_near(self,step,ob,candidate,jaw):
    """Retry an empty/poorly aligned grasp locally; don't repeat cross-table travel."""
    candidate=np.array(candidate,float)
    for attempt in range(self.local_retries+1):
      reason='empty close'
      try:
        self.set_phase('descend',local_attempt=attempt+1)
        candidate=self._wrist_descend(step,candidate,jaw)
        self.set_phase('close');self.motion.guard.watch(None)
        result=self.arm.gripper(0,step.force);self.event('gripper',result=result)
        if self.arm.state()['holding_signal']:return candidate
        self._wrist_blob(step,'miss')
      except MotionInterrupted as e:
        if e.hard_stop:raise
        reason=str(e)
      # Never open or reposition an object that may have been caught.
      if self.arm.state()['holding_signal']:raise MotionInterrupted(reason,hard_stop=True)
      gp=self.arm.geom(self.arm.q6()[0])['gp']
      self.set_phase('local_retry',attempt=attempt+1,reason=reason,lift_mm=12)
      self.motion.goto([*gp[:2],gp[2]+.012],tilt=None,yaw=jaw,tolerance=.015)
      if attempt==self.local_retries:
        raise MotionInterrupted('local grasp retries exhausted; stopped above the object',hard_stop=True)
      width=max(min(ob.size[:2]),min(float(getattr(self,'wrist_width',0) or 0),min(ob.size[:2])+.006))
      self.arm.gripper(self.arm.percent_for_gap(width+.012),'medium')
      current=self.arm.geom(self.arm.q6()[0])['gp']
      self._align_z=current[2]
      candidate[:2]=current[:2]
      # One millimetre deeper only after an empty close; the jaw/table monitor
      # still forbids penetration and stops descent on a load rise.
      if reason=='empty close':candidate[2]=max(self.source.table_z+.015,candidate[2]-.001)
      # Reopening changes the hinged jaw's lowest point. Replan the local
      # height for this opening rather than descend to an infeasible old pose.
      q=self.arm.q6()[0];planned=None
      for z in np.arange(candidate[2],max(candidate[2]+.002,ob.top_z+.005),.002):
        point=np.array([*candidate[:2],z])
        qs,error=self.arm._pose(point,0,jaw,q)
        if error<.003 and self.arm._check(qs) is None:
          planned=point;break
      if planned is None:raise MotionInterrupted('no feasible nearby retry pose at this jaw opening',hard_stop=True)
      candidate=planned
      self.wait(.1)
    raise AssertionError('unreachable')

  def _verify_lift(self,step,ob):
    """The object must demonstrably rise with the gripper. Preferred evidence: a fresh
    measurement of the object >= 2.5 cm higher. If the arm hides it from the head camera, all of:
    jaw contact held, jaw gap within 10 mm of the measured width, and >= 10 visible samples of
    the old footprint (>= 90 % reading the support surface) in a depth frame <= 0.8 s old.
    Gripper closure alone is never enough."""
    if not self.arm.state()["holding_signal"]:
      raise MotionInterrupted("lift verification failed: jaws lost contact")
    wrist=self._wrist_blob(step,"lifted")
    before=getattr(self,"_wrist_before_lift",None)
    if wrist and before:
      # The detected box keeps its place and size between the jaws while the table recedes
      # (pixel area of the colour mask is not used: shading changes it, measured 0.54 for a held cube).
      shift=float(np.linalg.norm(np.subtract(wrist["centroid"],before["centroid"])))
      bw=lambda b:(b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])
      ratio=float(np.sqrt(bw(wrist)/max(bw(before),1)))
      if 0.75<=ratio<=1.33 and shift<=30:
        # The object stays the same size and place between the jaws while the table recedes.
        self.event("grasp_verified",lift_evidence="wrist_object_between_jaws",area_ratio=ratio,shift_px=shift)
        self._held_forget(step.pick)
        return
      self.event("wrist_lift_evidence_insufficient",area_ratio=ratio,shift_px=shift)
    try:
      lifted=self.observe(step.pick)
      # The measured top rises with a lifted rigid object (a camera cannot see under it, so its
      # measured support stays on the table; ground truth reports the same rise for both).
      if lifted.top_z<ob.top_z+0.025:
        # A static/partial tracker can still report its old tabletop hypothesis.
        # Try independent vacated-footprint evidence before rejecting a real lift.
        raise SceneUnavailable("head geometry does not establish that the object rose")
      self.event("grasp_verified",lift_m=lifted.top_z-ob.top_z,lift_evidence="observed")
      self._held_forget(step.pick)
      return
    except SceneUnavailable as e:
      if not hasattr(self.source,"vacated"):
        raise
      reason=str(e)
    gap=self.arm.geom(self.arm.q6()[0])["gap"]
    width=float(self.wrist_width) if getattr(self,"wrist_width",None) else float(min(ob.size[:2]))
    vac=self.source.vacated(ob)
    # A rigid object still resting at its old pose occupies every footprint sample: >= 10 visible
    # samples (of 49, footprint shrunk by 3 mm) nearly all reading the support surface refute that.
    ok=abs(gap-width)<=0.010 and vac.get("visible_n",0)>=10 and vac["vacated"]>=0.9 and vac["age_s"] is not None and vac["age_s"]<=0.8
    self.event("grasp_verified" if ok else "lift_evidence_insufficient",lift_evidence="vacated_footprint+jaw_width",
               object_hidden=reason,jaw_gap_m=gap,object_width_m=width,**{f"footprint_{k}":x for k,x in vac.items()})
    if not ok:
      raise MotionInterrupted("lift verification failed: object hidden and footprint/jaw evidence insufficient")
    self._held_forget(step.pick)

  def _wrist_descend(self,step,candidate,jaw):
    """Descend, then centre at grasp height where the object and the jaw tips share a depth (no
    parallax; the alignment height left ~7 mm of bias toward the moving jaw, which then landed on
    the cube's edge). Off centre: rise to the alignment height before moving sideways, so the jaws
    cannot push the object, correct, descend again."""
    candidate=np.array(candidate,float);z_align=self._align_z;start_xy=candidate[:2].copy()
    for k in range(3):
      try:self.motion.goto(candidate,yaw=jaw,tolerance=0.030)
      except TableContact as e:
        candidate=self.arm.geom(self.arm.q6()[0])['gp'].copy()
        self.event('table_contact_stop',reason=str(e))
      self.wait(0.15)
      fr=self.wrist.fresh();q,_=self.arm.q6()
      st=self.wrist.measure(q,step.pick,frame=fr,width_m=float(min(self._wrist_obj_size[:2])))
      if st is None:
        raise MotionInterrupted("wrist camera does not see the object at grasp height")
      up,lo=np.array(st.tips["upper"]),np.array(st.tips["lower"])
      axis=(lo-up)/max(np.linalg.norm(lo-up),1e-9);span=float(np.linalg.norm(lo-up))
      along=np.array([axis[1],-axis[0]])
      goal=self.wrist.grip_pixel(q,st.tips,bias_m=getattr(self,"longitudinal_bias_m",.005)) if hasattr(self.wrist,"grip_pixel") else (up+lo)/2
      depth=None
      if any(w in step.pick.lower().split() for w in ("orange","oranges","ball","sphere")) and hasattr(self.wrist,"fruit_depth"):
        depth=self.wrist.fruit_depth(q,fr,st.blob,float(max(self._wrist_obj_size[:2])))
      center=np.array(st.blob.get("centroid",(up+lo)/2)) if depth is None else depth['center_px']
      point_shift=self.wrist.offset_pixels(q,center,np.r_[getattr(self,"grasp_shift_xy",np.zeros(2)),0.0]) if hasattr(self.wrist,"offset_pixels") else np.zeros(2)
      center=center+point_shift
      st.blob['grasp_point_px']=center.tolist()
      longitudinal=float((center-goal)@along)
      st.target_px=goal
      self.wrist.annotate(fr.color,st,self.log_dir/f"wrist_grasp_{k}.jpg")
      if hasattr(fr,"depth"):
        np.savez_compressed(self.log_dir/f"wrist_depth_{time.monotonic():.3f}.npz",depth=fr.depth,joints=q,
                            tips=np.array([up,lo]),bbox=np.array(st.blob['bbox']),goal_px=goal,
                            K=np.r_[self.wrist.f,self.wrist.c])
      b=st.blob["bbox"];corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]],float)
      proj=(corners-up)@axis
      off=float((proj.min()+proj.max())/2-span/2)  # object centre vs jaw midpoint, along the closing axis
      off+=float(point_shift@axis)
      height_err=float(candidate[2]-self.arm.geom(q)["gp"][2])
      try:
        bus=self.arm.bus;servo_pos=bus.sync_read("Present_Position");servo_goal=bus.sync_read("Goal_Position")
        from so101_reach.real.feetech_bus import JOINT_ORDER as _JO
        self.event("grasp_height_servo_state",goal_minus_present_ticks={n:int(servo_goal[n]-servo_pos[n]) for n in _JO[:4]},
                   load={n:int(bus.read(n,"Present_Load")) for n in _JO[:4]},correction_log=getattr(self.arm,"correction_log",None))
      except Exception:
        pass
      self.event("wrist_grasp_check",round=k,error_px=st.error_px,closing_axis_offset_px=off,object_span_px=[float(proj.min()),float(proj.max())],
                 jaw_span_px=span,height_error_mm=1000*height_err,correction_mm=1000*st.correction_m[:2],
                 longitudinal_error_px=longitudinal,longitudinal_bias_mm=1000*getattr(self,"longitudinal_bias_m",.005),depth_evidence=depth)
      # Across the jaws (closing axis) decides whether the object ends up between them: 12 px.
      # Along the jaws the pads tolerate several mm, and the object's centroid there is biased by
      # its visible side faces (47 px on a known-good grasp frame): accept up to 35 px (~8 mm).
      # 'center' guides the reach. Accept a capture region rather than demand
      # pixel-perfect centring of a noisy visible-face bounding box.
      pixels_per_m=(proj.max()-proj.min())/max(min(self._wrist_obj_size[:2]),.006)
      slack=min(12.0,max(6.0,.002*pixels_per_m))
      centred=abs(longitudinal)<=LONGITUDINAL_TOL_PX and proj.min()>=-slack and proj.max()<=span+slack
      if centred:
        geom=self.arm.geom(q)
        top=getattr(self,'_wrist_top_z',None)
        engagement=-np.inf if top is None else top-geom.get('lowest',np.inf)
        if engagement>=.004:
          self.event('grasp_capture_region',engagement_mm=1000*engagement,slack_px=slack,offset_px=off)
          return candidate
        if depth is not None and -8<=depth['insertion_mm']<=35:
          self.event("grasp_pose_candidate",evidence="wrist_depth_and_jaw_geometry",**depth)
          return candidate
        if depth is None and height_err>-.012:return candidate
        if depth is not None and height_err>-.012:
          raise MotionInterrupted("wrist depth does not establish fruit insertion between the pads",hard_stop=True)
        # The wide approach tolerance can accept servo stiction above the goal.
        # Lateral centring is already correct: an up/side/down cycle repeats the
        # same height error. Let the existing guarded sag corrector finish the
        # ORIGINAL physical height, with a tighter measured residual tolerance.
        self.event("grasp_height_correction",height_error_mm=1000*height_err)
        try:self.motion.goto(candidate,yaw=jaw,tolerance=.008)
        except MotionInterrupted as e:
          raise MotionInterrupted(f"grasp height could not be established: {e}",hard_stop=True) from e
        continue
      if k==2:
        break
      # Both directions matter: tip-midpoint-only alignment biases the grasp
      # toward the camera-facing edge. Correct toward the pad centre in jaw coordinates.
      # Full correction across the jaws; along them only half of any excess beyond the tolerance
      # (the biased estimate dominated the step and the closing-axis error never converged).
      excess=np.sign(longitudinal)*max(0.0,abs(longitudinal)-LONGITUDINAL_TOL_PX)
      c=self.wrist.px_to_base(q,off*axis+0.5*excess*along)
      c*=min(1.0,0.010/max(np.linalg.norm(c),1e-9))
      gp=self.arm.geom(self.arm.q6()[0])['gp']
      local_z=min(z_align,gp[2]+.012)
      self.motion.goto([*candidate[:2],local_z],yaw=jaw,tolerance=0.015)
      candidate[:2]+=c
      if np.linalg.norm(candidate[:2]-start_xy)>.020:
        raise MotionInterrupted("final wrist correction exceeded 20 mm",hard_stop=True)
      self.motion.goto([*candidate[:2],local_z],yaw=jaw,tolerance=0.015)
    raise MotionInterrupted("could not centre the object between the jaws at grasp height")

  def _wrist_between_jaws(self,step):
    """At grasp height the object and the jaw tips are at about the same depth: its extent along
    the closing axis must lie between the tips, or closing would miss or push it."""
    fr=self.wrist.fresh()
    q,_=self.arm.q6()
    st=self.wrist.measure(q,step.pick,frame=fr,width_m=float(min(self._wrist_obj_size[:2])))
    if st is not None:
      self.wrist.annotate(fr.color,st,self.log_dir/"wrist_preclose.jpg")
    if st is None:
      raise MotionInterrupted("wrist camera does not see the object at grasp height")
    up,lo=np.array(st.tips["upper"]),np.array(st.tips["lower"])
    axis=(lo-up)/max(np.linalg.norm(lo-up),1e-9)
    b=st.blob["bbox"]
    corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]],float)
    proj=(corners-up)@axis;span=float(np.linalg.norm(lo-up))
    inside=bool(proj.min()>=-5 and proj.max()<=span+5)
    self.event("wrist_preclose",inside=inside,object_span_px=[float(proj.min()),float(proj.max())],jaw_span_px=span)
    if not inside:
      raise MotionInterrupted("object is not between the jaws at grasp height")

  def _wrist_blob(self,step,tag):
    if self.wrist is None:
      return None
    fr=self.wrist.fresh()
    q,_=self.arm.q6()
    st=self.wrist.measure(q,step.pick,frame=fr,width_m=float(min(self._wrist_obj_size[:2])))
    if st is not None:
      self.wrist.annotate(fr.color,st,self.log_dir/f"wrist_{tag}.jpg")
    return st.blob if st is not None else None

  def _wrist_centre(self,step,ob,xy,jaw):
    """Lower until the jaw tips are ~12 mm above the object, then centre it between the jaws
    in the wrist image (relative measurement, iterated). Returns the corrected grasp xy."""
    self._wrist_obj_size=ob.size
    self._wrist_top_z=ob.top_z
    self.wrist_width=None
    z=ob.top_z+0.012+0.015  # grip point 15 mm above the tips
    self._align_z=z
    xy=np.array(xy,float);start=xy.copy()
    self.set_phase("align")
    # A sensing pose: the wrist measures relative to the jaws, so 15 mm absolute accuracy suffices.
    # The tips stay 12 mm above the object's top (no contact even if it moved) and the wrist
    # confirms it immediately after, so head-camera target freshness is not required here.
    # The descent margin is then taken from these short, slow moves near the object, which match
    # the descent better than the long transfer did.
    self.motion.guard.move_peak_deviation=0.0
    self.motion.guard.dev_samples=[]
    self.motion.goto([*xy,z],yaw=jaw,tolerance=0.015)
    # From here the wrist camera observes the target directly at every step (the head camera
    # usually cannot): target freshness comes from the wrist; obstacle checks stay on the head cloud.
    for i in range(3):
      self.wait(0.15)
      fr=self.wrist.fresh()
      q,_=self.arm.q6()
      st=self.wrist.measure(q,step.pick,frame=fr,width_m=float(min(self._wrist_obj_size[:2])))
      if st is None:
        raise MotionInterrupted("wrist camera does not see the object at the alignment pose")
      shift=self.wrist.offset_pixels(q,np.array(st.blob['centroid']),np.r_[self.grasp_shift_xy,0.0])
      st.blob['grasp_point_px']=(np.array(st.blob['centroid'])+shift).tolist()
      st.error_px=tuple(np.array(st.error_px)+shift)
      st.correction_m=np.r_[self.wrist.px_to_base(q,st.error_px),0.0]
      self.wrist.annotate(fr.color,st,self.log_dir/f"wrist_align_{i}.jpg")
      err=float(np.hypot(*st.error_px))
      self.event("wrist_align",iteration=i,error_px=st.error_px,correction_mm=1000*st.correction_m[:2],tips=st.tips,blob=st.blob)
      if err<25 or i==2:
        break
      c=st.correction_m[:2]
      c*=min(1.0,0.020/max(np.linalg.norm(c),1e-9))  # at most 2 cm per step
      xy=xy+c
      if np.linalg.norm(xy-start)>0.035:
        raise MotionInterrupted("wrist centring disagrees with the head camera by more than 35 mm")
      self.motion.goto([*xy,z],yaw=jaw,tolerance=0.015)
    self.wrist_seen_at=self.source.now()
    # Width across the jaws from the wrist (box extent along the closing direction at the object's
    # depth): the head camera underestimated it when partly hidden (22 mm for a 34 mm cube).
    b=st.blob["bbox"];tips=st.tips
    axis=np.subtract(tips["lower"],tips["upper"]);axis=axis/max(np.linalg.norm(axis),1e-9)
    # Spread of the object's own pixels along the closing axis (the detector's box is padded and
    # includes side faces: 39 mm for a 34 mm cube).
    import cv2 as _cv2
    crop=_cv2.cvtColor(fr.color[b[1]:b[3],b[0]:b[2]],_cv2.COLOR_BGR2HSV)
    obj=(crop[...,1]<=90)&(crop[...,2]>=130) if "white" in step.pick.lower().split() else np.ones(crop.shape[:2],bool)
    ys,xs=np.nonzero(obj)
    if len(xs)>200:
      proj=xs*axis[0]+ys*axis[1]
      ext=float(np.percentile(proj,95)-np.percentile(proj,5))/0.9
    else:
      ext=abs(axis[0])*(b[2]-b[0])+abs(axis[1])*(b[3]-b[1])
    zt=float(np.mean([self.wrist._tip_cam(q,sd)[2] for sd in ("upper","lower")]))+0.012
    self.wrist_width=float(ext*zt/self.wrist.f[1])
    # Leave 12 mm per side for approach. Wider opening changes the moving jaw's
    # table clearance, so the descent is still planned at this actual opening.
    # It does not remove the need to centre the object before closing.
    # Wide approach, then narrow above the object: the hinged moving jaw drops
    # as it opens, preventing a low central grasp if kept at the approach gap.
    closing=self.arm.geom(q)["axis"][:2]
    closing=closing/max(np.linalg.norm(closing),1e-9)
    cross_offset=abs(float(self.grasp_shift_xy@closing))
    # Width: the head's unoccluded start measurement (33.6-34.8 mm for the 34 mm cube); the wrist
    # over-reads (37-40 mm) and only widens the opening when it sees a clearly larger object.
    # Hardware: openings of 34-48 mm held the cube 4/4; 51-60 mm 0/8 (the fixed jaw tip dips and the
    # moving tip rises above ~47 mm, so one jaw lands on the cube's edge and the other clears its top).
    w_head=float(min(ob.size[:2]))
    w=w_head if self.wrist_width<=w_head+0.006 else self.wrist_width
    need=w+0.012+2*cross_offset
    self.arm.gripper(self.arm.percent_for_gap(need),"medium")
    self.event("wrist_aligned",wrist_width_mm=1000*self.wrist_width,head_width_mm=1000*float(min(ob.size[:2])),opening_gap_mm=1000*need,offset_from_head_mm=1000*(xy-start),final_error_px=st.error_px)
    return xy

  def _held_forget(self,name):
    if hasattr(self.source,"forget"):
      self.source.forget(name)  # its old table pose is no longer valid; re-detect after release

  def _geometry_at_release(self,step,ob,payload_center,why):
    if not hasattr(self.source,"measure_at"):
      raise SceneUnavailable(why)
    m=self.source.measure_at(payload_center,ob.size)
    if m is None or m["age_s"]>self.motion.guard.max_age:
      raise SceneUnavailable(f"{why}; no fresh depth at the release location")
    from so101_reach.agent.perception import Detection
    return Detection(0,step.pick,0.0,m["center"],m["top"],m["support"],m["size"],m["yaw"],"head",extra={"evidence":"geometry_at_release"})

  def _observe_result(self,name):
    """Placement evidence needs a fresh view. If the arm hides the result, park it out of the
    head camera's view (explicit observation pose) and look again."""
    try:
      return self.observe(name)
    except SceneUnavailable:
      if not hasattr(self.source,"vacated") or self.parked:
        raise
    self._park_for_view()
    return self._await_fresh(name,3.0)

  def _park_for_view(self):
    self.set_phase("observe")
    self.motion.guard.watch(None)
    r=self.motion.observe_pose()
    self.event("observation_pose",result=r)
    if not r.startswith("ok"):
      raise MotionInterrupted(f"could not reach the observation pose: {r}")
    self.parked=True
    self.set_phase("verify")

  def _verify_camera(self,step,ob,tgt,payload_center):
    """Camera placement evidence: fresh depth geometry where the payload was released, which
    needs no detector identity (white-on-white stacks merge into one detection). The destination
    is a static fixture measured before release: once loaded, its region includes the new
    object, so the pre-release geometry is the height reference and a fresh view only has to
    confirm that it did not move."""
    if not self.parked:self._park_for_view()
    def geometry(timeout):
      end=self.source.now()+timeout
      while True:
        try:
          if hasattr(self.source,"measure_placed"):
            d=self.source.measure_placed(tgt,ob)
            if d is None:raise SceneUnavailable("placed object not separable from the destination")
            return d
          return self._geometry_at_release(step,ob,payload_center,"release region not measurable")
        except SceneUnavailable:
          if self.source.now()>end:
            raise
          self.wait(0.1)
    try:
      final=geometry(1.0)
    except SceneUnavailable:
      self._park_for_view()
      final=geometry(3.0)
    first=final.center.copy()
    self.wait(0.30)
    final=geometry(1.0)
    shift=float(np.linalg.norm(final.center-first))
    if shift>=0.004:
      # The retreated arm still hangs over the release point and partly hides it; the region
      # measurement then jitters (pick_place_fsm_real/ik_1: cube visibly in the box, inside and
      # height ok, "unstable"). Judge stability from the observation pose instead.
      self.event("placement_reobserve",reason="unstable while the arm may occlude the release point",shift_mm=1000*shift)
      if not self.parked:self._park_for_view()
      else:self.wait(.30)
      final=geometry(3.0)
      first=final.center.copy()
      self.wait(0.30)
      final=geometry(1.0)
    self.event("placement_stability",shift_mm=1000*float(np.linalg.norm(final.center-first)),parked=bool(self.parked))
    try:
      now=self.observe(step.target)
      moved=float(np.linalg.norm(now.center[:2]-tgt.center[:2]))
    except SceneUnavailable:
      if not self.parked:self._park_for_view()
      now=self._await_fresh(step.target,3.0)
      moved=float(np.linalg.norm(now.center[:2]-tgt.center[:2]))
    self.event("destination_check",reobserved=moved is not None,shift_m=moved)
    if moved is not None and moved>0.010:
      raise MotionInterrupted("destination moved during placement")
    return final,first,tgt,"geometry_at_release"

  def _prepare_destination(self,step,payload=None):
    """Validate the complete placement while the gripper is still empty."""
    self.set_phase("prepare_destination",target=step.target)
    payload=payload or self.observe(step.pick)
    try:
      target=self.source.inspect_target(step.target) if hasattr(self.source,"inspect_target") else self.observe(step.target)
      if target.extra.get("partial") is not None or target.extra.get("occluded_s") is not None:
        raise SceneUnavailable("destination only partially observed")
    except SceneUnavailable:
      if not hasattr(self.source,"inspect_target"):raise
      self.event("destination_reobserve",reason="establish destination before grasp")
      self._park_for_view()
      self._await_fresh(step.target,4.0)
      target=self.source.inspect_target(step.target)
    plan=prepare_target(step,payload,target)
    self.event("destination_inspected",center_m=target.center,size_m=target.size,
               payload_size_m=payload.size,geometry=target.extra)
    if hasattr(self.source,"placement_point"):
      if self.placed_objects:
        plan.release_xy=self.source.placement_point(plan.detection,payload,occupants=list(self.placed_objects.values()))
      else:plan.release_xy=self.source.placement_point(plan.detection,payload)
    elif self.placed_objects and plan.detection.extra.get('container'):
      plan.release_xy=packed_release_point(plan.detection,payload,list(self.placed_objects.values()))
    self.targets[(step.pick,step.target)]=plan
    if hasattr(self.source,"pin_target"):
      self.source.pin_target(step.target,plan.detection)
    self.event("destination_prepared",requested_relation=step.relation,effective_relation=plan.relation,
               center_m=plan.detection.center,top_z=plan.detection.top_z,size_m=plan.detection.size,
               support_z=plan.detection.support_z,yaw_deg=plan.detection.yaw_deg,geometry=plan.detection.extra)
    if plan.release_xy is not None:self.event("placement_point",xy_m=plan.release_xy)
    if plan.relation!=step.relation:
      print(f"[placement] {step.target} has an open interior; the object will be deposited inside." if plan.relation=="in" else
            f"[placement] {step.target} has no verified open interior; the object will be placed on its measured top.",flush=True)
    return plan

  def _await_fresh(self,name,timeout):
    """Wait (arm still) up to `timeout` for the camera worker to re-detect and measure `name`."""
    end=self.source.now()+timeout
    while True:
      self.wait(0.1)
      try:
        return self.observe(name)
      except SceneUnavailable:
        if self.source.now()>end:
          raise

  def _place(self,step,ob,below):
    plan=self.targets.get((step.pick,step.target))
    if plan is not None and plan.relation=="in":
      # Filling an open container: cubes already inside (and the rim) do not block the held
      # object; it is released 12 mm above the rim and may land on them (user request 2026-11-08).
      self.motion.guard.set_release_zone(plan.detection)
      self.event("release_zone",center_m=plan.detection.center[:2],size_m=plan.detection.size[:2],rim_z=plan.detection.top_z)
    try:
      return self._place_impl(step,ob,below)
    finally:
      self.motion.guard.set_release_zone(None)

  def _place_impl(self,step,ob,below):
    g=self.motion.guard
    self.set_phase("prepare_place",target=step.target)
    plan=self.targets.get((step.pick,step.target))
    if plan is None:
      raise MotionInterrupted("placement was not validated before the grasp")
    relation=plan.relation
    # The lifted arm often hides the destination from the head camera; a fixture hidden by the arm
    # is used on its last fresh measurement (bounded), and placement is re-verified afterwards.
    try:
      tgt=self.observe(step.target,allow_occluded=OCCLUSION_ALLOWANCE_S)
    except SceneUnavailable as e:
      if not hasattr(self.source,"inspect_target"):
        raise
      # The reaching arm hid the destination for long (hardware: 75-82 % for >10 s). Carry the
      # payload to the observation pose and look again; never place on a stale destination.
      self.event("destination_reobserve",reason=str(e))
      self._park_for_view()
      self.parked=False
      tgt=self._await_fresh(step.target,4.0)
    if np.linalg.norm(tgt.center[:2]-plan.detection.center[:2])>.012:
      raise MotionInterrupted("destination moved since placement preflight")
    # Classification and rim height came from a clear pre-grasp view. A partially hidden
    # interior must not turn a known container into a solid surface after a successful lift.
    tgt=copy_detection(plan.detection)
    place_xy=tgt.center[:2].copy() if plan.release_xy is None else plan.release_xy.copy()
    release_bottom=tgt.top_z+plan.release_clearance
    self.set_phase("carry",target=step.target)
    p=np.array([*place_xy,release_bottom+below])
    # This target can move too. Reacquire and replan the transport, never release at an old goal.
    carry_reobservations=0
    for attempt in range(self.retries+1):
      try:
        hover=p.copy();hover[2]=max(self.clearance()+below,p[2]+0.045)
        self.motion.transfer(hover,hover[2],watch=step.target,allow_occluded=self._hidden_allowance())
        break
      except (MotionInterrupted,SceneUnavailable) as e:
        if getattr(e,'hard_stop',False):raise
        # Once a hidden track expires entirely, CameraScene no longer has the
        # occlusion label. Reacquire a stale destination in either case; only
        # the fresh, position-checked observation may authorize another carry.
        stale_target=f'{step.target!r} not freshly observed' in str(e)
        if stale_target and carry_reobservations<1 and attempt<self.retries and hasattr(self.source,'inspect_target'):
          carry_reobservations+=1
          self.event('destination_reobserve',reason=str(e),during='carry')
          self._park_for_view()
          current=self._await_fresh(step.target,4.0)
          if np.linalg.norm(current.center[:2]-plan.detection.center[:2])>.012:
            raise MotionInterrupted('destination moved during carry reobservation')
          self.set_phase('carry',target=step.target)
          continue
        if "target moved" not in str(e) or attempt==self.retries:
          raise
        self.event("recovery",reason=str(e))
        tgt=self.observe(step.target,allow_occluded=OCCLUSION_ALLOWANCE_S)
        p=np.array([*tgt.center[:2],tgt.top_z+plan.release_clearance+below])
    # Re-estimate the attachment after carrying: a grasp may rotate slightly under load.
    g.attach(step.pick)
    q=self.arm.q6()[0]
    for _ in range(6):
      qs,err=self.arm._pose(p,0,None,q)
      if err>0.003:
        raise MotionInterrupted("payload placement pose is unreachable")
      center,extent=g.payload_geometry(qs)
      delta=np.array([*(place_xy-center[:2]),release_bottom-(center[2]-extent[2])])
      p+=delta;q=qs
      if np.linalg.norm(delta)<0.001:
        break
    self.motion.goto(p,watch=step.target,allow_occluded=self._hidden_allowance())
    if not self.arm.state()["holding_signal"]:
      raise MotionInterrupted("grasp lost during transport")
    # Equal-size stacking has almost no lateral margin. The generic 8 mm motion
    # tolerance and 10 mm release check are too loose for a 34 mm cube-on-cube task.
    release_tol=min(.010,max(.0025,(min(tgt.size[:2])-max(ob.size[:2]))/2+.003))
    aim=p.copy()
    for k in range(3):
      center,_=g.payload_geometry(self.arm.q6()[0])
      error=place_xy-center[:2]
      self.event("release_alignment",attempt=k,error_mm=1000*error,tolerance_mm=1000*release_tol)
      if np.linalg.norm(error)<=release_tol:break
      aim[:2]+=error
      self.motion.goto(aim,watch=step.target,allow_occluded=self._hidden_allowance())
    self.set_phase("release")
    gp=self.arm.geom(self.arm.q6()[0])["gp"]
    payload_center,_=g.payload_geometry(self.arm.q6()[0])
    if np.linalg.norm(payload_center[:2]-place_xy)>release_tol:
      raise MotionInterrupted("release position is not over destination")
    g.watch(None)
    before=self._wrist_blob(step,"release_before") if self.wrist is not None else None
    held_state=None if self.arm.held is None else dict(self.arm.held)
    width=max(max(ob.size[:2]),float(getattr(self,"wrist_width",0) or 0),self.arm.geom(self.arm.q6()[0])["gap"])
    release_gap=width+.024
    result=self.arm.gripper(self.arm.percent_for_gap(release_gap),"gentle")
    self.arm.held=held_state  # opening is a command, not evidence the object left
    measured_gap=self.arm.geom(self.arm.q6()[0])["gap"]
    if measured_gap<width+.018:
      result=self.arm.gripper(self.arm.percent_for_gap(release_gap+.008),"medium")
      self.arm.held=held_state
      measured_gap=self.arm.geom(self.arm.q6()[0])["gap"]
    self.event("release_gripper",requested_gap_mm=1000*release_gap,measured_gap_mm=1000*measured_gap,
               payload_width_mm=1000*width,result=result)
    if measured_gap<width+.018:
      raise MotionInterrupted("gripper did not open clear of the payload",hard_stop=True)
    self.wait(.25)
    after=self._wrist_blob(step,"release_after") if before is not None else None
    if before is not None and after is not None and relation=="in":
      area_ratio=after['area']/max(before['area'],1)
      shift=float(np.linalg.norm(np.array(after['centroid'])-before['centroid']))
      self.event("wrist_release",area_ratio=area_ratio,shift_px=shift)
      if .90<area_ratio<1.10 and shift<6:
        raise MotionInterrupted("payload still appears attached after opening; retaining held state",hard_stop=True)
    if hasattr(self.source,"released"):
      self.source.released(step.pick)
    # Keep the conservative payload model until the destination measurement
    # proves release. An open command can leave soft fruit wedged on the pads.
    # The source remains allowed jaw contact until the arm has risen away from it.
    self.set_phase("retreat")
    nowgp=self.arm.geom(self.arm.q6()[0])["gp"]
    try:
      self.motion.goto([*nowgp[:2],max(nowgp[2]+0.045,p[2]+0.035)],tilt=None,yaw=None,tolerance=.015)
    except MotionInterrupted as e:
      actual=self.arm.geom(self.arm.q6()[0])['gp']
      residual_only=str(e).startswith('stopped: final grip-point error') and not e.hard_stop
      if not residual_only or actual[2]-nowgp[2]<.020:raise
      self.event('retreat_reobserve',reason=str(e),measured_rise_m=float(actual[2]-nowgp[2]))
      self._park_for_view()
    g.grasp_contact=None
    self.wait(0.45)
    self.set_phase("verify")
    # _park_for_view above may already have established the verification pose.
    if hasattr(self.source,"measure_at"):
      final,first_center,target,evidence=self._verify_camera(step,ob,tgt,payload_center)
    else:
      final=self._observe_result(step.pick)
      first_center=final.center.copy()
      self.wait(0.30)
      final=self.observe(step.pick)
      evidence="detected"
      target=self.observe(step.target)
    stable=bool(np.linalg.norm(final.center-first_center)<0.004)
    half=min(final.size[:2])/2
    if relation=="in":
      inside=footprint_contains(target,final.center[:2],half,margin=0.003)
      height_ok=final.support_z<target.top_z-0.005 and final.support_z>=self.source.table_z-0.003
    else:
      inside=footprint_contains(target,final.center[:2],max(final.size[:2])/2,margin=-0.004)
      height_ok=abs(final.support_z-target.top_z)<0.010
    if evidence=="geometry_at_release":
      # Region measurement: its top must be one object-height above the destination surface
      # (inside a container: at most the rim, at least half an object above the table).
      if relation=="on":
        height_ok=abs(final.top_z-(target.top_z+ob.size[2]))<0.010
      else:
        height_ok=self.source.table_z+0.5*ob.size[2]<final.top_z<target.top_z+ob.size[2]-0.005
    ok=bool(inside and height_ok and stable and not self.arm.state()["holding_signal"])
    self.event("placement_verified",success=ok,object_center_m=final.center,object_bottom_m=final.support_z,
               target_top_m=target.top_z,inside=bool(inside),height_ok=bool(height_ok),stable=stable,evidence=evidence)
    if not ok:
      raise MotionInterrupted("placement verification failed")
    occupied=copy_detection(final)
    occupied.size=tuple(g.held_size or ob.size)
    self.placed_objects[step.pick]=occupied
    self.arm.held=None
    g.held_name=None;g.held_size=None

  def _already_satisfied(self,step,payload):
    """A command describes a desired relation, not a requirement to repeat a pick."""
    try:
      target=self.source.inspect_target(step.target) if hasattr(self.source,'inspect_target') else self.observe(step.target)
      first=placement_relation(payload,target,step.relation,self.source.table_z)
      if not first['satisfied']:return False
      self.wait(.30)
      again=self.observe(step.pick)
      now=self.observe(step.target)
      second=placement_relation(again,target,step.relation,self.source.table_z)
      stable=np.linalg.norm(again.center-payload.center)<.004
      fixture_static=np.linalg.norm(now.center[:2]-target.center[:2])<.010
      if not(second['satisfied'] and stable and fixture_static):return False
    except SceneUnavailable:return False
    self.set_phase('already_satisfied',object=step.pick,target=step.target,
                   message=f'{step.pick} is already {second["effective_relation"]} {step.target}; no pick needed')
    self.event('goal_satisfied',evidence='two_fresh_named_geometry_measurements',**second)
    return True

  def _progress(self,steps,results):
    data={'total_steps':len(steps),'completed_steps':len(results),'steps':results,
          'remaining_steps':[asdict(s) for s in steps[len(results):]],'phase':self.phase}
    (self.log_dir/'progress.json').write_text(json.dumps(data,indent=2)+'\n')
    if hasattr(self.source,'names') and data['remaining_steps']:
      self.source.names=list(dict.fromkeys(n for step in data['remaining_steps'] for n in (step['pick'],step['target'])))

  def run(self,steps,home=False,grasp_only=False):
    if not steps:
      raise ValueError("cannot execute an empty plan")
    t0=time.perf_counter();motion_start=self.source.now();results=[]
    if hasattr(self.source,'names'):
      self.source.names=list(dict.fromkeys(n for step in steps for n in ((step.pick,) if grasp_only else (step.pick,step.target))))
    try:
      for i,step in enumerate(steps):
        self.event("step",index=i,total=len(steps),**asdict(step))
        print(f'Step {i+1}/{len(steps)}: {step.pick}',flush=True)
        payload=self.observe(step.pick)
        if not grasp_only:
          if self._already_satisfied(step,payload):
            results.append({**asdict(step),'success':True,'status':'already_satisfied'})
            self._progress(steps,results)
            continue
          self.observe(step.target)
          self._prepare_destination(step)
        grasp_t0=time.perf_counter()
        try:
          ob,below=self._grasp(step)
        finally:
          self.grasp_wall_s=time.perf_counter()-grasp_t0
        if grasp_only:
          results.append({'pick':step.pick,'action':'grasp_and_hold','success':True})
          break
        self._place(step,ob,below)
        results.append({**asdict(step),"effective_relation":self.targets[(step.pick,step.target)].relation,"success":True})
        self._progress(steps,results)
        print(f'Verified {i+1}/{len(steps)} steps',flush=True)
      if home or not grasp_only:
        # Pick-and-place runs always finish at rest (user request 2026-11-08); grasp-only keeps
        # holding its object where it verified the lift.
        self.set_phase("home")
        g=self.motion.guard;g.watch(None)
        r='ok' if self.parked else self.motion.observe_pose()
        if not r.startswith("ok"):
          raise MotionInterrupted(r)
      self.set_phase("complete")
      result={"success":True,"steps":results,'object_may_be_held':bool(grasp_only)}
    except Exception as e:
      hold_error=self.motion.guard.hold()
      self.event("failure",error=str(e))
      failed_phase=self.phase
      end=self._end_safely(e)
      try:
        holding=bool(self.arm.state()["holding_signal"] or self.arm.held or self.motion.guard.held_name)
      except OSError:
        holding=bool(self.arm.held)
      result={"success":False,"steps":results,"error":str(e),"failed_phase":failed_phase,
               "error_type":type(e).__name__,"hold_error":hold_error,"object_may_be_held":holding,"end_of_run":end}

    result.update(wall_s=time.perf_counter()-t0,motion_s=self.source.now()-motion_start,
                  total_steps=len(steps),completed_steps=len(results),remaining_steps=[asdict(s) for s in steps[len(results):]],
                  controller=self.motion.controller,collision_checks=self.motion.guard.checks,
                  interruptions=self.motion.guard.interruptions,moves=self.motion.moves,
                  max_path_deviation_mm=1000*self.motion.guard.max_deviation,
                  observation_waits=len(self.motion.guard.fresh_waits),
                  observation_wait_s=float(sum(self.motion.guard.fresh_waits)),
                  occluded_target_commands=self.motion.guard.occluded_uses,
                  perception="sim_ground_truth" if getattr(self.source,"world",None) is not None else "rgbd_dino_flow")
    if getattr(self,"grasp_wall_s",None) is not None:
      result["grasp_wall_s"]=self.grasp_wall_s  # approach start to verified lift (or failure)
    if self.grasp_stats is not None:
      s=self.grasp_stats
      s["grasp_verified"]=any(e["kind"]=="grasp_verified" for e in self.events)
      result["grasp_stats"]=s
      self.event("grasp_summary",**{k:v for k,v in s.items() if k not in ("assessments","states")},
                 scores=[a["score"] for a in s["assessments"]],decisions=[a["decision"] for a in s["assessments"]])
    (self.log_dir/"result.json").write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x))+"\n")
    return result

  def _end_safely(self,error):
    """After any failure: put a held object down, leave the gripper open and empty, and return to
    the rest pose (user request 2026-11-08). Every move stays monitored (obstacles and hands still
    pause it). A stage that cannot be done safely leaves the arm held, and the reason is reported.
    Not used for Ctrl-C / Esc, which stop and hold."""
    g=self.motion.guard
    out={"object_released":None,"gripper_open_percent":None,"rest":None}
    try:
      self.set_phase("end_safely",after=str(error)[:160])
      g.watch(None);g.set_release_zone(None);g.disarm_descent()
      held=bool(self.arm.state()["holding_signal"] or self.arm.held or g.held_name)
      if held:
        out["object_released"]=self._put_down_held()
        if not out["object_released"]:
          raise MotionInterrupted("held object could not be put down safely; arm held with it")
      st=self.arm.state()
      if st["gripper_open_percent"]<15:
        self.arm.gripper(self.arm.percent_for_gap(0.050),"gentle")
      out["gripper_open_percent"]=self.arm.state()["gripper_open_percent"]
      out["rest"]=self.motion.observe_pose()
    except Exception as e:
      out["error"]=f"{type(e).__name__}: {e}"
      g.hold()
    self.event("end_of_run",**out)
    print("[end_of_run]",out,flush=True)
    return out

  def _put_down_held(self):
    """Release a held object: into its destination container if it is already over it,
    otherwise lowered straight down to just above the table. Never let go higher than 60 mm."""
    from types import SimpleNamespace
    g,arm=self.motion.guard,self.arm
    name=g.held_name or (arm.held or {}).get("name") or "object"
    below=float((arm.held or {}).get("below_gp") or g.held_below or 0.030)
    size=tuple(g.held_size or (arm.held or {}).get("size") or (0.034,)*3)
    q=arm.q6()[0]
    plan=next((p for (pick,_),p in self.targets.items() if pick==name),None)
    over_box=False
    if plan is not None and plan.relation=="in" and g.held_transform is not None and g.held_size is not None:
      c,_=g.payload_geometry(q)
      over_box=footprint_contains(plan.detection,c[:2],max(size[:2])/2,margin=.003)
    if not over_box:
      gp=arm.geom(q)["gp"]
      target_z=self.source.table_z+below+.006
      if gp[2]>target_z+.003:
        self.set_phase("put_down",object=name)
        try:
          self.motion.goto([*gp[:2],target_z],tilt=None,yaw=None,tolerance=.015)
        except MotionInterrupted as e:
          self.event("put_down_lowering_stopped",reason=str(e))
    gp=arm.geom(arm.q6()[0])["gp"]
    drop=float(gp[2]-below-self.source.table_z)
    if not over_box and drop>.060:
      self.event("put_down_refused",object_bottom_mm=1000*drop)
      return False
    probe=SimpleNamespace(pick=name)
    can_see=self.wrist is not None and getattr(self,"_wrist_obj_size",None) is not None
    before=self._wrist_blob(probe,"put_down_before") if can_see else None
    held_state=None if arm.held is None else dict(arm.held)
    r=arm.gripper(arm.percent_for_gap(max(size[:2])+.024),"gentle")
    self.wait(.25)
    after=self._wrist_blob(probe,"put_down_after") if before is not None else None
    if before is not None and after is not None:
      ratio=after['area']/max(before['area'],1)
      shift=float(np.linalg.norm(np.array(after['centroid'])-before['centroid']))
      if .90<ratio<1.10 and shift<6:
        arm.held=held_state
        self.event("put_down_failed",reason="object still between the jaws after opening",area_ratio=ratio,shift_px=shift)
        return False
    self.event("put_down",object=name,into_destination=bool(over_box),drop_mm=1000*drop,gripper=r)
    arm.held=None;g.held_name=None;g.held_size=None;g.grasp_contact=None
    if hasattr(self.source,"released"):self.source.released(name)
    if type(arm.plant).__name__=="RealPlant":
      from so101_reach.manipulation.payload_state import STATE
      STATE.unlink(missing_ok=True)
    gp=arm.geom(arm.q6()[0])["gp"]
    try:
      self.motion.goto([*gp[:2],gp[2]+.040],tilt=None,yaw=None,tolerance=.015)
    except MotionInterrupted as e:
      self.event("put_down_rise_stopped",reason=str(e))
    return True
