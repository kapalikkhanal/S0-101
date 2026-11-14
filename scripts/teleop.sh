#!/usr/bin/env bash
# Port/calibration handling adapted from So-101_Arm/so101/teleop.sh.
source "$(dirname "$0")/hardware_common.sh"
require_follower
require_leader
exec "$SO101_LEROBOT_PYTHON" "$SO101_REPO/scripts/lerobot_run.py" teleoperate \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID" \
  --display_data="${DISPLAY_DATA:-false}" "$@"
