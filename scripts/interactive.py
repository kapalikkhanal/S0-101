"""Interactive SO-101 reaching: type a coordinate, the controller moves the TCP there.

Examples:
  # Browser GUI (works headless/remote): open http://localhost:8080
  python scripts/interactive.py
  # MuJoCo desktop window, targets typed in this terminal
  python scripts/interactive.py --viewer native
  # Other controllers: the NVIDIA checkpoint or the IK baselines
  python scripts/interactive.py --controller nvidia
  python scripts/interactive.py --controller ik_joint --checkpoint none
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from so101_reach.viz.interactive import ReachNativeViewer, ReachViserViewer  # noqa: E402
from so101_reach.viz.session import make_session  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--controller", choices=["native", "nvidia", "ik", "ik_joint"], default="native")
  p.add_argument("--checkpoint", type=Path, default=None, help="native checkpoint (.pt); default: latest")
  p.add_argument("--viewer", choices=["viser", "native"], default="viser")
  p.add_argument("--tol-mm", type=float, default=10.0, help="settled-success tolerance shown in the UI")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  sess = make_session(args.controller, args.checkpoint, args.device, args.tol_mm)
  if args.viewer == "viser":
    print("[interactive] open http://localhost:8080 ('Reach target' panel)")
    ReachViserViewer(sess.env, sess.policy, sess).run()
  else:
    ReachNativeViewer(sess.env, sess.policy, sess).run()


if __name__ == "__main__":
  main()
