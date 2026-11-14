"""Open-vocabulary object detection (Grounding DINO tiny) for picking any roughly round object.

A text prompt ("orange", "red apple", "peach") gives image boxes; each box becomes a sphere
candidate (centre = box centre, radius = mean half side) and is located in 3-D exactly like
the orange (`OrangeDetector.locate`: resting on the table plane, size from the angular size
and depth). Boxes touching the image border (cut off) or far from square are dropped.

Runs on the GPU in half precision at 480 px (~80 ms per frame on the RTX PRO 5000); one model
is shared by all camera threads. cuDNN is disabled for it: the installed torch build cannot
load its runtime-compiled cuDNN engines (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED).
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from so101_reach.real.cameras import Frame, RGBDCamera
from so101_reach.real.orange import CameraPose, OrangeDetector

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class DinoModel:
  _instance: DinoModel | None = None
  _create_lock = threading.Lock()

  def __init__(self, model_id: str = MODEL_ID, size: int = 480):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.backends.cudnn.enabled = False
    self.torch = torch
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.half = self.device == "cuda"
    self.proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
    self.model = model.half() if self.half else model
    self.size = {"shortest_edge": size, "longest_edge": int(size * 16 / 9)}
    self.lock = threading.Lock()

  @classmethod
  def shared(cls) -> DinoModel:
    with cls._create_lock:
      if cls._instance is None:
        cls._instance = cls()
      return cls._instance

  def detect(self, bgr: np.ndarray, prompt: str, box_threshold: float, text_threshold: float) -> list[tuple[float, np.ndarray]]:
    """[(score, box x0 y0 x1 y1 [px])], best first."""
    text = prompt.strip().lower().rstrip(".") + "."
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with self.lock, self.torch.no_grad():
      inp = self.proc(images=rgb, text=text, return_tensors="pt", size=self.size)
      inp = {k: v.to(self.device) for k, v in inp.items()}
      if self.half:
        inp["pixel_values"] = inp["pixel_values"].half()
      out = self.model(**inp)
      res = self.proc.post_process_grounded_object_detection(
        out, inp["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[bgr.shape[:2]]
      )[0]
    dets = [(float(s), b.float().cpu().numpy()) for s, b in zip(res["scores"], res["boxes"], strict=True)]
    return sorted(dets, key=lambda d: -d[0])


class TextObjectDetector(OrangeDetector):
  """OrangeDetector with Grounding DINO boxes instead of the orange colour mask."""

  def __init__(
    self,
    cam: RGBDCamera,
    calib: CameraPose,
    prompt: str,
    table_z: float | None = None,
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    radius_range_m: tuple[float, float] = (0.015, 0.06),
    max_aspect: float = 1.6,
    border_px: int = 3,
    allow_lifted: bool = False,
  ):
    super().__init__(cam, calib, table_z=table_z, radius_range_m=radius_range_m, allow_lifted=allow_lifted)
    self.prompt = prompt
    self.box_threshold, self.text_threshold = box_threshold, text_threshold
    self.max_aspect, self.border_px = max_aspect, border_px
    self.model = DinoModel.shared()

  def candidates(self, frame: Frame) -> list[tuple[float, float, float, float, str]]:
    h, w = frame.color.shape[:2]
    out = []
    for score, (x0, y0, x1, y1) in self.model.detect(frame.color, self.prompt, self.box_threshold, self.text_threshold):
      if min(x0, y0) < self.border_px or x1 > w - self.border_px or y1 > h - self.border_px:
        continue  # Cut off by the image edge: centre and size unreliable.
      bw, bh = x1 - x0, y1 - y0
      if max(bw, bh) > self.max_aspect * max(min(bw, bh), 1.0):
        continue  # Far from round (partly hidden, or not a fruit-like object).
      out.append(((x0 + x1) / 2, (y0 + y1) / 2, (bw + bh) / 4, min(bw, bh) / max(bw, bh), f"{self.prompt} {score:.2f}"))
    return out

  def mask(self, bgr: np.ndarray) -> np.ndarray:
    # Only used to exclude object pixels from the table-height estimate, which already keeps
    # points within a few cm of the base plane.
    return np.zeros(bgr.shape[:2], np.uint8)


def make_detector(cam: RGBDCamera, calib: CameraPose, obj: str | None = None, **kw) -> OrangeDetector:
  """Colour orange detector (obj None) or a Grounding DINO detector for the text `obj`."""
  return OrangeDetector(cam, calib, **kw) if obj is None else TextObjectDetector(cam, calib, obj, **kw)
