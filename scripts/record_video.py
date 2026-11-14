"""Record an MP4 of consecutive reaching motions (offscreen EGL rendering, 50 fps).

Each frame shows the arm, the target (orange; green once within tolerance), the TCP
(blue) and an overlay with target, TCP, error, elapsed time and status.

Examples:
  python scripts/record_video.py --num-targets 8
  python scripts/record_video.py --targets "0.20,0.10,0.15;0.25,-0.12,0.05;0.0,0.25,0.30"
  python scripts/record_video.py --controller ik_joint --out videos/ik_joint.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import mediapy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from so101_reach.sim import so101  # noqa: E402
from so101_reach.viz.session import make_session  # noqa: E402


def overlay(frame: np.ndarray, lines: list[tuple[str, str]], title: str) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img, "RGBA")
  try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
    font_b = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 20)
  except OSError:
    font = font_b = ImageFont.load_default()
  h = 34 + 24 * len(lines)
  draw.rectangle([10, 10, 620, 10 + h], fill=(0, 0, 0, 150))
  draw.text((20, 16), title, font=font_b, fill=(255, 255, 255, 255))
  state = lines[0][1]
  colors = {"REACHED": (80, 220, 120), "MOVING": (120, 180, 255)}
  for i, (k, v) in enumerate(lines):
    c = colors.get(state.split()[0], (255, 110, 110)) if i == 0 else (235, 235, 235)
    draw.text((20, 44 + 24 * i), f"{k:<13s} {v}", font=font, fill=(*c, 255))
  return np.asarray(img)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--controller", choices=["native", "nvidia", "ik", "ik_joint"], default="native")
  p.add_argument("--checkpoint", type=Path, default=None)
  p.add_argument("--targets", type=str, default=None, help='"x,y,z;x,y,z;..." in metres (base frame)')
  p.add_argument("--num-targets", type=int, default=6, help="random eval-workspace targets if --targets is not given")
  p.add_argument("--seed", type=int, default=7)
  p.add_argument("--max-time", type=float, default=5.0, help="max seconds per target")
  p.add_argument("--dwell", type=float, default=0.6, help="seconds to keep recording after reaching")
  p.add_argument("--tol-mm", type=float, default=10.0)
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  sess = make_session(args.controller, args.checkpoint, args.device, args.tol_mm, render=True, width=args.width, height=args.height)
  env = sess.env
  if args.targets:
    targets = [np.array([float(v) for v in t.split(",")], dtype=np.float32) for t in args.targets.split(";")]
  else:
    rng = np.random.default_rng(args.seed)
    bank = sess.bank_tcp.cpu().numpy()
    targets = [bank[i] for i in rng.choice(len(bank), args.num_targets, replace=False)]

  out = args.out or so101.PROJECT_ROOT / "videos" / f"reach_{args.controller}.mp4"
  out.parent.mkdir(parents=True, exist_ok=True)
  title = {"native": "SO-101 reach - MjLab PPO policy", "nvidia": "SO-101 reach - nvidia/so101-reach (ONNX)",
           "ik": "SO-101 reach - DLS IK baseline", "ik_joint": "SO-101 reach - IK + joint interpolation"}[args.controller]
  frames, results = [], []
  for k, xyz in enumerate(targets):
    sess.command(xyz)
    reached_at = None
    for _ in range(int(args.max_time / env.step_dt)):
      with torch.no_grad():
        env.step(sess.policy(env.get_observations()))
      st = sess.status()
      frames.append(overlay(env.render(), st.lines(), f"{title}   [target {k + 1}/{len(targets)}]"))
      if st.state == "REACHED" and reached_at is None:
        reached_at = st.elapsed_s
      if reached_at is not None and st.elapsed_s - reached_at >= args.dwell:
        break
    st = sess.status()
    results.append((xyz, st.state, st.reach_time_s, st.error_mm))
    print(f"[video] target {k + 1}: {np.round(xyz, 3)} -> {st.state}, time {st.reach_time_s}, final error {st.error_mm:.1f} mm")
  mediapy.write_video(str(out), frames, fps=round(1.0 / env.step_dt))
  print(f"[video] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
  main()
