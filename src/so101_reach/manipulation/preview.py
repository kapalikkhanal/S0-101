"""GLFW image preview compatible with the project's headless OpenCV build."""
import cv2
import glfw
import mujoco


class Preview:
  def __init__(self):
    if not glfw.init():
      raise RuntimeError("GLFW cannot open a display; omit --preview on a headless machine")
    self.window=glfw.create_window(960,540,"SO-101 offline manipulation | Esc: hold",None,None)
    if not self.window:
      glfw.terminate();raise RuntimeError("could not create preview window")
    glfw.make_context_current(self.window)
    glfw.swap_interval(0)
    self.model=mujoco.MjModel.from_xml_string("<mujoco/>")
    self.ctx=mujoco.MjrContext(self.model,mujoco.mjtFontScale.mjFONTSCALE_150)

  def show(self,img):
    glfw.make_context_current(self.window)
    w,h=glfw.get_framebuffer_size(self.window)
    rgb=cv2.cvtColor(cv2.resize(img,(w,h)),cv2.COLOR_BGR2RGB)[::-1].copy()
    mujoco.mjr_drawPixels(rgb.ravel(),None,mujoco.MjrRect(0,0,w,h),self.ctx)
    glfw.swap_buffers(self.window);glfw.poll_events()
    if glfw.window_should_close(self.window) or glfw.get_key(self.window,glfw.KEY_ESCAPE)==glfw.PRESS:
      raise KeyboardInterrupt()

  def close(self):
    self.ctx.free()
    glfw.destroy_window(self.window);glfw.terminate()


def annotate_scene(image,snapshot,now,table_z,phase,pause_reason=None,wrist=None):
  """Names, measured XYZ and freshness on the live head view; detected hands (red); the latest
  wrist-camera frame as an inset."""
  out=image.copy()
  for h in getattr(snapshot,"hands",[]) or []:
    x0,y0,x1,y1=h["bbox"]
    cv2.rectangle(out,(x0,y0),(x1,y1),(0,0,255),3)
    cv2.putText(out,f'HAND {h["score"]:.2f}',(x0,max(20,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
  for name,d in snapshot.objects.items():
    if d.bbox is None:continue
    fresh=now-snapshot.seen.get(name,-1e9)<=.8
    color=(0,220,0) if fresh else (0,210,255)
    x0,y0,x1,y1=map(int,d.bbox)
    cv2.rectangle(out,(x0,y0),(x1,y1),color,2)
    text=f'{name} {"fresh" if fresh else "stale"} | {d.center[0]:.3f}, {d.center[1]:.3f}, {d.center[2]-table_z:.3f} m'
    cv2.putText(out,text,(max(0,x0),max(20,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
  if wrist is not None:
    h,w=out.shape[:2];iw=w//3;ih=int(iw*wrist.shape[0]/wrist.shape[1])
    out[h-ih-10:h-10,w-iw-10:w-10]=cv2.resize(wrist,(iw,ih))
    cv2.rectangle(out,(w-iw-10,h-ih-10),(w-10,h-10),(255,255,255),2)
    cv2.putText(out,"wrist",(w-iw-4,h-ih+12),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
  if pause_reason:
    cv2.rectangle(out,(0,0),(out.shape[1],62),(0,0,100),-1)
    cv2.putText(out,'PAUSED: '+pause_reason[:100],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    cv2.putText(out,'Waiting for fresh clear observations; Esc stops.',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
  return out
