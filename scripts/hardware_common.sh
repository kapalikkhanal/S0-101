# Source from the hardware scripts. Load .env.local explicitly before calling them.
set -euo pipefail
SO101_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SO101_LEROBOT_PYTHON=${SO101_LEROBOT_PYTHON:-$SO101_REPO/.venv-lerobot/bin/python}
SO101_FOLLOWER_PORT=${SO101_FOLLOWER_PORT:-/dev/ttyACM0}
SO101_LEADER_PORT=${SO101_LEADER_PORT:-/dev/ttyACM1}
SO101_FOLLOWER_ID=${SO101_FOLLOWER_ID:-my_follower}
SO101_LEADER_ID=${SO101_LEADER_ID:-my_leader}
export PATH="$(dirname "$SO101_LEROBOT_PYTHON"):$PATH"
unset PYTHONPATH
CAMERAS=${SO101_RECORD_CAMERAS:-'{}'}
if [[ -z ${SO101_RECORD_CAMERAS:-} && -n ${SO101_HEAD_CAMERA_SERIAL:-} && -n ${SO101_WRIST_CAMERA_SERIAL:-} ]]; then
  CAMERAS="{head: {type: intelrealsense, serial_number_or_name: '$SO101_HEAD_CAMERA_SERIAL', width: 640, height: 480, fps: 30, warmup_s: 3}, gripper: {type: intelrealsense, serial_number_or_name: '$SO101_WRIST_CAMERA_SERIAL', width: 640, height: 480, fps: 30, warmup_s: 3}}"
fi
if [[ ! -x "$SO101_LEROBOT_PYTHON" ]]; then
  echo 'Run bash scripts/setup_recording.sh first.' >&2
  exit 1
fi
require_follower() { [[ -e "$SO101_FOLLOWER_PORT" ]] || { echo 'Follower port missing. Set SO101_FOLLOWER_PORT.' >&2; exit 1; }; }
require_leader() { [[ -e "$SO101_LEADER_PORT" ]] || { echo 'Leader port missing. Set SO101_LEADER_PORT.' >&2; exit 1; }; }
