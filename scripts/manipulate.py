"""Run offline prompt-to-skill manipulation. Simulation is the default.

  .venv/bin/python scripts/manipulate.py "Pick the white cube and place it on the white box"
  .venv/bin/python scripts/manipulate.py "Put the red cube into the white box" --scene table
  .venv/bin/python scripts/manipulate.py "Put the white cube on the white box" --real --preview
"""
from __future__ import annotations

import argparse
import base64
import contextlib
from dataclasses import replace
from datetime import datetime
import json
import os
import re
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")

import cv2
import numpy as np

from so101_reach.agent.arm import AgentArm
from so101_reach.manipulation.planning import LocalPlanner,RulesPlanner,Step,plan_dict,ensure_local_server
from so101_reach.manipulation.motion import Motion
from so101_reach.manipulation.scene import CameraScene,SimScene
from so101_reach.manipulation.skills import Executor
from so101_reach.real.controller import ControllerCfg,ReachController
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.sim import so101
from so101_reach.settings import AGENT_POLICY_DIR as POLICY_DIR, require_hardware_config


def demo_scene(seed=0,clutter=False,perturb="none"):
  from so101_reach.agent.sim_world import WorldCfg,StaticBox
  from so101_reach.sim.grasp_scene import ObjectCfg
  rng=np.random.default_rng(seed);tz=-0.0132
  side=0.034+float(rng.uniform(-0.003,0.003))
  xy=np.array([0.23,-0.07])+rng.uniform(-0.012,0.012,2)
  cube=ObjectCfg("box",(side/2,)*3,(*xy,tz+side/2),mass=0.035,rgba=(0.95,0.95,0.95,1))
  cfg=WorldCfg(tz,{"white cube":cube},[StaticBox("white box",(0.29,-0.16,tz+0.022),(0.05,0.05,0.022))])
  if clutter:
    cfg.platforms.append(StaticBox("obstacle",(0.28,-0.07,tz+0.055),(0.018,0.018,0.055),(0.2,0.3,0.8,1)))
  if perturb=="obstacle":
    cfg.objects["moving obstacle"]=ObjectCfg("box",(0.025,0.025,0.10),(0.50,-0.40,tz+0.10),mass=5,rgba=(0.2,0.3,0.8,1))
  return cfg


def novel_scene(kind,seed):
  from so101_reach.sim.grasp_scene import ObjectCfg
  cfg=demo_scene(seed)
  xy=next(iter(cfg.objects.values())).pos[:2]
  definitions={
    "cylinder":("blue cylinder","cylinder",(.018,.025),.025,(.1,.2,.9,1)),
    "sphere":("orange ball","sphere",(.025,),.025,(1,.45,.02,1)),
    "bar":("green bar","box",(.0275,.014,.0175),.0175,(.1,.7,.2,1)),
  }
  name,shape,size,half_z,color=definitions[kind]
  cfg.objects={name:ObjectCfg(shape,size,(*xy,cfg.table_z+half_z),mass=.035,rgba=color)}
  return cfg


def get_args(argv=None):
  p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("prompt",nargs="?",help="natural-language manipulation instruction")
  p.add_argument("--real",action="store_true",help="explicitly operate the connected SO-101 (default: simulation)")
  p.add_argument("--scene",default="demo",choices=["demo","clutter","table","container","sorting","on_box","stack","tower","cylinder","sphere","bar"])
  p.add_argument("--planner",default="ollama",choices=["ollama","rules"])
  p.add_argument("--perception",default="truth",choices=["truth","rendered"],
                 help="simulation only: 'truth' = labelled ground truth; 'rendered' = the hardware RGB-D/DINO worker "
                      "on rendered head-camera images, paced to wall-clock time")
  p.add_argument("--model",default="qwen3-vl:8b-instruct")
  p.add_argument("--ollama-url",default="http://127.0.0.1:11434")
  p.add_argument("--controller",default="ik",choices=["ik","rl"])
  p.add_argument("--speed",type=float,default=0.14,help="IK translation speed in m/s; validated range 0.02–0.20")
  p.add_argument("--rl-speed",type=float,default=0.5)
  p.add_argument("--seed",type=int,default=0)
  p.add_argument("--perturb",default="none",choices=["none","target","obstacle","occlusion"])
  p.add_argument("--preview",action="store_true",help="live camera/simulation window; Esc stops and holds")
  p.add_argument("--video",type=Path,help="record simulation video; rendering time is included in wall time")
  p.add_argument("--vision-plan",action="store_true",help="include one image in local planning (adds latency)")
  p.add_argument("--select",help="rank multiple instances by a visible criterion using the local VLM before grasp")
  p.add_argument("--grasp-position",choices=["center","left","right","near","far"],
                 help="grasp point relative to the gripper's approach; default from configs/grasp_offsets.json")
  p.add_argument("--grasp-offset-mm",type=float,help="distance from object centre for a non-centred grasp (default: quarter width)")
  p.add_argument('--grasp-only',action='store_true',help='pick, verify lift and hold; skip destination and placement')
  p.add_argument('--local-retries',type=int,default=None,
                 help='nearby retries after an empty close (0–4; default 1 for the fsm controller, 2 for legacy)')
  p.add_argument('--grasp-controller',choices=['fsm','legacy'],default='fsm',
                 help='wrist-camera grasp: bounded state machine (default) or the earlier iterative centring loop')
  p.add_argument('--grasp-score-threshold',type=float,default=0.60,
                 help='fsm: close when the grasp score reaches this (an uncalibrated geometric score, not a probability)')
  p.add_argument('--table-contact',choices=['light','avoid'],default='light',help='allow nominal jaw/table touch near the grasp; other links retain clearance')
  p.add_argument("--home",action=argparse.BooleanOptionalAction,default=None,help="return to rest after all placements (default); --no-home skips final return")
  p.add_argument("--start-pose",choices=["rest","current"],default=None,
                 help="real arm: go to the monitored home/rest pose first (default with --real); simulation default: current")
  p.add_argument("--empty-gripper",action="store_true",help="assert physically empty gripper and clear an unresolved payload record")
  p.add_argument("--no-hand-watch",action="store_true",help="do not detect hands (hands then pause only when they block the arm's path)")
  p.add_argument("--pause-timeout",type=float,default=60,help="IK: wait for temporary observed obstacles; seconds, 0 = unlimited")
  p.add_argument("--log-dir",type=Path)
  p.add_argument("--diagnose",action="store_true",help="check prerequisites without touching hardware")
  args=p.parse_args(argv)
  if not args.diagnose and not args.prompt:
    p.error("provide a prompt")
  if not .02<=args.speed<=.20 or not .05<=args.rl_speed<=1:
    p.error("speed must be 0.02–0.20 m/s and rl-speed 0.05–1")
  if args.real and args.perturb!="none":
    p.error("perturbations are simulation-only")
  if args.real and args.perception!="truth":
    p.error("--perception selects a simulation mode; hardware always uses the RGB-D worker")
  if args.grasp_offset_mm is not None and not 0<=args.grasp_offset_mm<=20:
    p.error("grasp offset must be 0–20 mm and fit inside the object")
  if args.pause_timeout<0:p.error("pause timeout cannot be negative")
  if args.local_retries is None:args.local_retries=1 if args.grasp_controller=='fsm' else 2
  if not 0<=args.local_retries<=4:p.error('local retries must be 0–4')
  if not 0<args.grasp_score_threshold<=1:p.error('grasp score threshold must be in (0, 1]')
  if args.home is None:args.home=not args.grasp_only
  if args.grasp_only and args.home:p.error('grasp-only holds the object; --home is for completed placement')
  if args.start_pose is None:
    args.start_pose="rest" if args.real else "current"
  return args


def instruction_plan(args,planner,available=None,image=None):
  if args.grasp_only:
    match=re.fullmatch(r'(?:pick(?: up)?|grasp)\s+(?:the\s+)?(.+)',args.prompt.strip(),re.I)
    if match:
      name=re.split(r'\s+and\s+',match[1],maxsplit=1,flags=re.I)[0].strip()
      planner.metrics={'backend':'grasp_object','wall_s':0.0}
      return [Step(name,'','on','medium')]
  return planner.plan(args.prompt,available,image)


def wait_for_objects(source,plant,guard,pause_timeout,timeout=20.0):
  """Startup: every task object freshly measured. The arm holds (plant.wait keeps the preview
  live). A hand in view extends the wait up to the pause timeout instead of failing."""
  from so101_reach.manipulation.scene import SceneUnavailable,fresh_object
  t0=source.now();announced=False
  while True:
    s=source.snapshot()
    try:
      if s.error:raise SceneUnavailable(s.error)
      for n in source.names:fresh_object(s,n,source.now())
      return
    except SceneUnavailable as e:
      waited=source.now()-t0
      if s.hands:
        guard.last_hand_time=source.now()
        if not announced:print("PAUSED: hand in view at startup; waiting",flush=True);announced=True
        guard.pause_reason=f"hand in view ({len(s.hands)})"
        if pause_timeout and waited>=pause_timeout:raise SceneUnavailable(f"pause timed out: {e}") from e
      elif waited>=timeout:
        gates=getattr(source,"gate_rejections",{})
        if gates:
          raise SceneUnavailable(f"{e}; detections rejected as pick targets: "+"; ".join(f"{n}: {r}" for n,r in gates.items())) from e
        raise
      else:
        guard.pause_reason=None
    plant.wait()


def save_camera_debug(source,path,block=None):
  """Last head frame with the arm mask (red), tracked/measured boxes (green: fresh, yellow: not)
  and the cloud points behind the latest observed-obstacle refusal (magenta)."""
  from so101_reach.agent.perception import View
  snap=source.snapshot()
  if snap.image is None:return
  cam,calib=source.per.cams["head"];R,t=calib.pose();k=cam.intrinsics
  img=snap.image.copy()
  mask=source.per._arm_mask(View(img,R,t,k.fx,k.fy,k.cx,k.cy),img.shape[:2])
  img[mask]=(0.5*img[mask]+[0,0,127]).astype(np.uint8)
  for n,d in snap.objects.items():
    if d.bbox is None:continue
    fresh=source.now()-snap.seen.get(n,-1e9)<=0.8
    b=d.bbox.astype(int);col=(0,220,0) if fresh else (0,220,220)
    cv2.rectangle(img,tuple(b[:2]),tuple(b[2:]),col,2)
    cv2.putText(img,f"{n} {'fresh' if fresh else 'stale'}",(b[0],b[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.6,col,2)
  if block is not None:
    view=View(img,R,t,k.fx,k.fy,k.cx,k.cy)
    for p3 in block[1]:
      pp=view.project(p3)
      if pp is not None:cv2.circle(img,pp,4,(255,0,255),-1)
  cv2.imwrite(str(path),img)


def preflight_hardware():
  """Read-only presence checks before any camera stream or servo bus is opened."""
  require_hardware_config(cameras=True, servo_fit=True)
  from so101_reach.real.cameras import CAMERAS
  from so101_reach.real.joint_map import FOLLOWER_PORT
  import pyrealsense2 as rs
  missing=[]
  if not Path(FOLLOWER_PORT).exists():
    missing.append(f"SO-101 follower servo bus ({FOLLOWER_PORT})")
  serials=[d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
  if CAMERAS["head"] not in serials:
    missing.append(f"head D435i SN {CAMERAS['head']} (enumerated RealSense: {serials or 'none'})")
  if missing:
    raise RuntimeError("hardware not connected: "+"; ".join(missing)+". Nothing was opened or moved.")
  return serials


def diagnose(args):
  """Read-only prerequisite report: never opens the servo bus or engages motors."""
  import importlib.util
  import shutil
  status={"policy_present":(POLICY_DIR/"policy.onnx").exists(),
          "modules":{n:importlib.util.find_spec(n) is not None for n in ("mujoco","onnxruntime","scipy","cv2","pyrealsense2","transformers","scservo_sdk")},
          "ollama_binary":shutil.which("ollama")}
  try:
    status["local_model"]=LocalPlanner(args.model,args.ollama_url).check_model().get("name")
  except Exception as e:
    status["local_model_error"]=str(e)
  from so101_reach.real.joint_map import FOLLOWER_PORT
  status["servo_port_present"]=Path(FOLLOWER_PORT).exists()
  try:
    import pyrealsense2 as rs
    status["realsense_serials"]=[d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
  except Exception as e:
    status["realsense_error"]=str(e)
  hf=Path.home()/".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny"
  status["grounding_dino_cached"]=hf.exists()
  status["ready_for_simulation"]=bool(status["policy_present"] and status.get("local_model"))
  from so101_reach.real.cameras import CAMERAS
  status["ready_for_hardware"]=bool(status["ready_for_simulation"] and status["servo_port_present"]
                                    and CAMERAS["head"] in status.get("realsense_serials",[]) and status["grounding_dino_cached"])
  print(json.dumps(status,indent=2));return status


def run(args):
  if args.diagnose:
    return diagnose(args)
  log=args.log_dir or so101.PROJECT_ROOT/"logs/manipulation"/f"{datetime.now():%Y%m%d_%H%M%S_%f}"
  log.mkdir(parents=True,exist_ok=True)
  state={"stage":"setup","motors_engaged":False,"planner":None}
  try:
    return _run(args,log,state)
  except BaseException as e:
    # Setup/planning failures and user interrupts are logged too, with whether motors were engaged.
    if not (log/"result.json").exists():
      (log/"result.json").write_text(json.dumps({"success":False,"error":f"{type(e).__name__}: {e}","failed_phase":state["stage"],
        "error_type":type(e).__name__,"motors_engaged":state["motors_engaged"],"prompt":args.prompt,"real":args.real,
        "planner":getattr(state["planner"],"metrics",None)},indent=2)+"\n")
    raise


def _run(args,log,state):
  task_start=time.perf_counter()
  if args.real:
    from so101_reach.manipulation.payload_state import require_empty
    require_empty(args.empty_gripper)
  if args.real or args.perception=="rendered":
    # The camera worker shares the interpreter with CPU-bound planning; a short GIL switch
    # interval keeps its observation loop from being starved (measured: 1.0 s -> see results).
    sys.setswitchinterval(0.0005)
  state["stage"]="local_model_server"
  if args.planner=="ollama":ensure_local_server(args.ollama_url,log)
  planner=LocalPlanner(args.model,args.ollama_url) if args.planner=="ollama" else RulesPlanner()
  state["planner"]=planner
  if args.planner=="ollama":planner.check_model()
  with contextlib.ExitStack() as stack:
    ctrl=ReachController(ControllerCfg(speed_factor=args.rl_speed),POLICY_DIR)
    world=None;writer=None;source=None;last_visual=-1e9;preview=None
    cam=pacer=truth_source=None
    if args.preview:
      from so101_reach.manipulation.preview import Preview
      preview=Preview();stack.callback(preview.close)
    if not args.real:
      from so101_reach.agent.sim_world import SimWorld,scene
      from so101_reach.real.robot import SimRealPlant
      if args.scene in ("demo","clutter"):
        cfg=demo_scene(args.seed,args.scene=="clutter",args.perturb)
      elif args.scene in ("cylinder","sphere","bar"):
        cfg=novel_scene(args.scene,args.seed)
      elif args.scene=="container":
        cfg=scene("table")
        cube=cfg.objects.pop("red cube");cube.rgba=(.95,.95,.95,1)
        cfg.objects["white cube"]=cube
      elif args.scene=='sorting':
        from so101_reach.sim.grasp_scene import ObjectCfg
        cfg=scene('table')
        cfg.objects={name:ObjectCfg('box',(.017,)*3,(x,y,cfg.table_z+.017),mass=.035,rgba=color)
          for name,x,y,color in [('white cube',.20,-.04,(.95,.95,.95,1)),
                                 ('red cube',.25,-.08,(.9,.1,.1,1)),
                                 ('black cube',.20,-.16,(.08,.08,.08,1))]}
      else:
        cfg=scene(args.scene)
      if args.scene not in ("demo","clutter","cylinder","sphere","bar") and args.seed:
        rng=np.random.default_rng(args.seed)
        common=rng.uniform(-.006,.006,2)
        for obj in cfg.objects.values():
          offset=common if args.scene=="tower" else rng.uniform(-.006,.006,2)
          obj.pos=tuple(np.array(obj.pos)+[*offset,0])
          if args.scene!="tower":
            yaw=float(rng.uniform(-.15,.15));obj.quat=(np.cos(yaw/2),0,0,np.sin(yaw/2))
      world=SimWorld(cfg);stack.callback(world.close_renderers)
      world.set_arm({**so101.HOME_JOINT_POS,**so101.FIXED_JOINT_TARGETS,"wrist_roll":world.jmap.to_rad("wrist_roll",2048)})
      plant=SimRealPlant(world,world.jmap,world.fcal);plant.engage()
      for _ in range(20): plant.wait()
      available=world.labels+[s["label"] for s in world.surfaces()]
      image=base64.b64encode(cv2.imencode(".jpg",cv2.resize(world.render("head"),(640,360)))[1]).decode() if args.vision_plan else None
      state["stage"]="planning"
      steps=instruction_plan(args,planner,available,image)
      names=list(dict.fromkeys(n for s in steps for n in ((s.pick,) if args.grasp_only else (s.pick,s.target))))
      truth_source=source=SimScene(world,names,args.perturb)
      if args.perception=="rendered":
        # The hardware perception stack on rendered images; ground truth only injects perturbations
        # and independently records physical contacts.
        from so101_reach.agent.perception import RealPerception
        from so101_reach.manipulation.rendered_camera import RealtimePacer,RenderedRGBD
        from so101_reach.manipulation.scene import SceneUnavailable,fresh_object
        from so101_reach.real.orange import OrangeDetector
        state["stage"]="perception_ready"
        cam=RenderedRGBD(world);stack.callback(cam.close)
        cam.capture(force=True)
        tz=OrangeDetector(cam,cam.calib).estimate_table_z(cam.read())
        if tz is None:raise RuntimeError("cannot measure table height from rendered depth")
        per=RealPerception(cam,None,cam.calib,None,tz,memory_frames=0)
        source=CameraScene(per,names);stack.callback(source.close);source.watch_hands=not args.no_hand_watch
        raw=world.sync_read("Present_Position")
        from so101_reach.real.feetech_bus import JOINT_ORDER
        source.set_joints([world.jmap.to_rad(n,raw[n]) for n in JOINT_ORDER])
        pacer=RealtimePacer(world);end=time.monotonic()+90
        while True:
          plant.wait();cam.capture();pacer.wait()
          snap=source.snapshot()
          if snap.error:raise SceneUnavailable(snap.error)
          try:
            for n in names:fresh_object(snap,n,source.now())
            break
          except SceneUnavailable:
            if time.monotonic()>end:raise
    else:
      # Plan before engaging motors. No API key or cloud route exists in this command.
      state["stage"]="planning"
      steps=None if args.vision_plan else instruction_plan(args,planner)
      state["stage"]="hardware_setup"
      preflight_hardware()
      from so101_reach.agent.perception import RealPerception
      from so101_reach.real.cameras import open_camera
      from so101_reach.real.handeye import HeadCameraCalibration
      from so101_reach.real.orange import OrangeDetector
      from so101_reach.real.feetech_bus import FeetechBus,load_calibration
      from so101_reach.real.joint_map import FOLLOWER_PORT,JointMap
      from so101_reach.real.robot import RealPlant
      head=stack.enter_context(open_camera("head"));head.warmup(1)
      hcal=HeadCameraCalibration.load()
      tz=OrangeDetector(head,hcal).estimate_table_z(head.read())
      if tz is None: raise RuntimeError("cannot measure table height")
      if steps is None:
        img=head.read().color
        image=base64.b64encode(cv2.imencode(".jpg",cv2.resize(img,(640,360)))[1]).decode()
        steps=instruction_plan(args,planner,image=image)
      names=list(dict.fromkeys(n for s in steps for n in ((s.pick,) if args.grasp_only else (s.pick,s.target))))
      per=RealPerception(head,None,hcal,None,tz,memory_frames=0)
      source=CameraScene(per,names);stack.callback(source.close);source.watch_hands=not args.no_hand_watch
      # A pick target must lie in the grasp domain (<= 75 mm wide, as the executor enforces).
      # Without this, a grasp-only run (no destination name to compete) assigned the open white
      # box (149 x 109 mm, DINO 0.52) to "white cube" over the real cube (0.35).
      source.max_width={s.pick:0.075 for s in steps}
      # ...and within reach: near table height the grip point reaches ~0.41 m from the shoulder
      # pan axis in every direction (IK, 2026-11-07); 0.42 m leaves the object's half-width.
      from so101_reach.real.grasp_replay import ArmKinematics as _AK
      _k=_AK();_k.fk(np.zeros(6))
      source.max_reach={s.pick:(_k.d.xpos[_k.m.body("shoulder").id][:2].copy(),0.42) for s in steps}
      jmap,fcal=JointMap.load(),load_calibration()
      bus=stack.enter_context(FeetechBus(FOLLOWER_PORT,calib=fcal))
      plant=RealPlant(bus,jmap,fcal,torque_limit=400)
      # Get joint geometry before first camera mask; no motion command here.
      raw=bus.sync_read("Present_Position")
      if raw['gripper']>fcal['gripper'].range_min+70 and abs(bus.read('gripper','Present_Load'))>80:
        raise RuntimeError("gripper appears loaded; recover/remove the payload before starting a new pick")
      from so101_reach.real.feetech_bus import JOINT_ORDER
      source.set_joints([jmap.to_rad(n,raw[n]) for n in JOINT_ORDER])
      state["stage"]="perception_ready"
      initial=source.wait_fresh(.5,15)  # rest first: the current arm may hide task objects
      if initial.error or source.now()-initial.stamp>.5:raise RuntimeError('no fresh obstacle scene for startup')
      state["stage"]="engage"
      plant.engage();state["motors_engaged"]=True
    print("Plan:",json.dumps(plan_dict(steps)),flush=True)
    (log/"plan.json").write_text(json.dumps({"prompt":args.prompt,"steps":plan_dict(steps),"planner":planner.metrics},indent=2)+"\n")
    arm=AgentArm(plant,ctrl,SafeTargets(),source.table_z,speed_cm_s=args.speed*100)
    if args.real:
      arm._compensated_hold()  # engage() holds goal = present, which lets the P-servos sag
    source.set_joints(arm.q6()[0])
    motion=Motion(arm,source,args.controller,args.speed)
    executor=Executor(motion,log)
    executor.local_retries=args.local_retries
    executor.fsm_local_retries=args.local_retries
    executor.grasp_controller=args.grasp_controller
    executor.grasp_score_threshold=args.grasp_score_threshold
    executor.table_contact=args.table_contact=='light'
    if args.grasp_position is not None:executor.grasp_position=args.grasp_position
    if args.grasp_offset_mm is not None:executor.grasp_offset_m=args.grasp_offset_mm/1000
    if args.real:
      from so101_reach.real.grasp_replay import ArmKinematics
      from so101_reach.real.wrist_servo import WristServo
      wcam=stack.enter_context(open_camera("wrist"));wcam.warmup(0.5)
      executor.wrist=WristServo(wcam,ArmKinematics())
    if args.video:
      if world is None: raise ValueError("--video currently records simulation; --preview shows hardware")
      args.video.parent.mkdir(parents=True,exist_ok=True)
      writer=cv2.VideoWriter(str(args.video),cv2.VideoWriter_fourcc(*"mp4v"),10,(640,360))
      if not writer.isOpened(): raise RuntimeError("could not open video writer")
      stack.callback(writer.release)
    state["stage"]="execution"
    original_wait=plant.wait
    contact_guard=None
    if cam is not None:
      from so101_reach.manipulation.motion import SceneGuard
      contact_guard=SceneGuard(arm,truth_source)
    sim_start=float(world.data.time) if world is not None else None
    def wait():
      nonlocal last_visual
      original_wait()
      if truth_source is not None: truth_source.after_step()
      if cam is not None:
        cam.capture();pacer.wait()
        g=motion.guard
        contact_guard.grasp_contact,contact_guard.held_name=g.grasp_contact,g.held_name
        contact_guard.check_actual_contacts()
      motion.guard.check_actual_contacts()
      if (args.preview or writer is not None) and source.now()-last_visual>=0.1:
        last_visual=source.now()
        img=world.render("head") if world is not None else source.snapshot().image
        if img is not None:
          from so101_reach.manipulation.preview import annotate_scene
          img=annotate_scene(img,source.snapshot(),source.now(),source.table_z,executor.phase,
                             getattr(motion.guard,'pause_reason',None),
                             wrist=getattr(executor.wrist,"last_frame",None) if executor.wrist is not None else None)
          img=cv2.resize(img,(640,360))
          cv2.putText(img,f"{executor.phase} | {args.controller} | {source.now():.2f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,200,255),2)
          if writer is not None: writer.write(img)
          if args.preview:
            preview.show(img)
    plant.wait=wait
    motion.guard.auto_pause_timeout=args.pause_timeout if isinstance(source,CameraScene) and args.controller=='ik' else None
    try:
      if args.start_pose=="rest":
        state['stage']='startup_rest';executor.set_phase('startup_rest')
        status=motion.observe_pose()
        executor.event('startup_rest',result=status)
        if not status.startswith('ok'):raise RuntimeError('startup rest failed: '+status)
        executor.parked=True
      if args.real:
        ranked=args.select or (args.prompt if any(w in args.prompt.lower() for w in
                    ('most bad','most spoiled','most rotten','worst orange','most deteriorated')) else None)
        if ranked:
          if args.planner!='ollama' or len(steps)!=1:raise ValueError('visual ranking needs local planner and one step')
          from so101_reach.manipulation.selection import select_instance
          noun=steps[0].pick.lower().split()[-1];category={'oranges':'orange','fruits':'fruit'}.get(noun,noun)
          chosen,state['selection']=select_instance(per,head.read(),category,ranked,planner,log)
          old=steps[0].pick
          steps=[replace(steps[0],pick=chosen.name,force='gentle' if category=='orange' else steps[0].force)]
          source.forget(old);source.names=[steps[0].pick,steps[0].target]
          (log/'plan.json').write_text(json.dumps({'prompt':args.prompt,'steps':plan_dict(steps),'planner':planner.metrics},indent=2)+'\n')
        wait_for_objects(source,plant,motion.guard,args.pause_timeout)
      state['stage']='execution'
      result=executor.run(steps,args.home,grasp_only=args.grasp_only)
    finally:
      motion.guard.hold()
    result.update(prompt=args.prompt,planner=planner.metrics,total_wall_s=time.perf_counter()-task_start,
                  scene=args.scene if world else "real",seed=args.seed,log=str(log),perturbation=args.perturb,
                  unexpected_contacts=motion.guard.actual_contacts,real=args.real,motors_engaged=state["motors_engaged"])
    result['pause_events']=motion.guard.pause_events
    result['grasp_position']=executor.grasp_position
    result['start_pose']=args.start_pose
    if args.real:
      result["plant_overruns"]=plant.overruns
      if state.get("selection"):result["visual_selection"]=state["selection"]
    if world is not None:
      result["ground_truth"]={n:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in world.object_truth(n).items()} for n in world.labels}
      result["surfaces"]=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()} for s in world.surfaces()]
      result["perturbation_events"]=truth_source.events
      result["sim_motion_s"]=float(world.data.time)-sim_start
    if cam is not None:
      result.update(perception="sim_rendered_rgbd",measured_table_z=source.table_z,true_table_z=world.cfg.table_z,
                    unexpected_contacts=contact_guard.actual_contacts,frames_rendered=cam.frames_rendered,
                    frames_reserved=cam.frames_reserved,max_pacing_lag_s=pacer.max_lag)
    if isinstance(source,CameraScene):
      save_camera_debug(source,log/"last_head.jpg",motion.guard.last_block)
      if motion.guard.last_block is not None:
        part,pts,qb=motion.guard.last_block
        snap=source.snapshot()
        np.savez_compressed(log/"collision_evidence.npz",points=pts,checked_joints=qb,
                            cloud=snap.cloud,frame_joints=snap.joints,
                            frame_stamp=snap.stamp,measured_joints=motion.guard._q_meas)
        result["last_block"]={"part":part,"n":int(len(pts)),"min_m":pts.min(0).round(3).tolist(),"max_m":pts.max(0).round(3).tolist(),
                              "height_above_table_mm":[round(1000*(float(pts[:,2].min())-source.table_z)),round(1000*(float(pts[:,2].max())-source.table_z))],
                              "pose_deg":np.degrees(qb).round(1).tolist()}
      ls=np.array(source.loop_s or [np.nan])
      result["camera_worker"]={"loops":len(source.loop_s),"loop_s_median":float(np.median(ls)),"loop_s_p95":float(np.percentile(ls,95)),
                               "loop_s_max":float(ls.max()),"assignment":source.assignment,"self_filtered_points":source.self_filtered,
                               "timing_median_s":{k:float(np.median(v)) for k,v in source.timing.items() if v},
                               "timing_max_s":{k:float(np.max(v)) for k,v in source.timing.items() if v},
                               "timeline":source.timeline}
    jdefault=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)
    (log/"result.json").write_text(json.dumps(result,indent=2,default=jdefault)+"\n")
    print("RESULT",json.dumps({k:v for k,v in result.items() if k not in ("ground_truth","moves","grasp_stats","camera_worker")},default=jdefault),flush=True)
    if result.get("grasp_stats"):
      s=result["grasp_stats"]
      print("GRASP",json.dumps({"verified":s.get("grasp_verified"),"closes":s["closes"],"corrections":s["corrections"],
                                "align_moves":s["align_moves"],"contact_stops":s["contact_stops"],"local_retries":s["local_retries"],
                                "re_approaches":s["re_approaches"],"grasp_wall_s":round(result.get("grasp_wall_s") or 0,2),
                                "scores":[a["score"] for a in s["assessments"]],"decisions":[a["decision"] for a in s["assessments"]]}),flush=True)
    return result


if __name__=="__main__":
  args=get_args()
  try:
    result=run(args)
    if args.diagnose:sys.exit(0 if result.get("ready_for_simulation") else 1)
    sys.exit(0 if result.get("success") else 2)
  except KeyboardInterrupt:
    print("Interrupted; arm held.",file=sys.stderr);sys.exit(130)
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}",file=sys.stderr);sys.exit(1)
