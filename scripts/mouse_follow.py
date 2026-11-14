"""MuJoCo window: the SO-101 TCP follows the mouse pointer.

  python scripts/mouse_follow.py                         # latest trained checkpoint
  python scripts/mouse_follow.py --checkpoint logs/rsl_rl/so101_reach/<run>/model_6000.pt
  python scripts/mouse_follow.py --controller ik         # compare with the DLS IK baseline

Move the mouse: the target follows the cursor on a horizontal plane (scroll changes its
height). See so101_reach/viz/mouse_follow.py for all controls.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"
os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from so101_reach.viz.mouse_follow import MouseFollowApp  # noqa: E402
from so101_reach.viz.session import make_session  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--controller", choices=["native", "nvidia", "ik", "ik_joint"], default="native")
  p.add_argument("--checkpoint", type=Path, default=None)
  p.add_argument("--tol-mm", type=float, default=10.0)
  p.add_argument("--height", type=float, default=0.12, help="initial target-plane height [m]")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  sess = make_session(args.controller, args.checkpoint, args.device, args.tol_mm)
  MouseFollowApp(sess, plane_z=args.height).run()


if __name__ == "__main__":
  main()
