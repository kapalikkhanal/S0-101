#!/usr/bin/env bash
# Keep LeRobot's dependencies separate from the MjLab training environment.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv-lerobot --python 3.12
uv pip install --python .venv-lerobot/bin/python 'lerobot[feetech,intelrealsense]==0.4.4'
.venv-lerobot/bin/python -c 'import importlib.metadata as m; print("LeRobot", m.version("lerobot"))'
