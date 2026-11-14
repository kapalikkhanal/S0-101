#!/usr/bin/env bash
source "$(dirname "$0")/hardware_common.sh"
case "${1:-}" in
  follower) require_follower; exec "$SO101_LEROBOT_PYTHON" -m lerobot.scripts.lerobot_calibrate --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" ;;
  leader) require_leader; exec "$SO101_LEROBOT_PYTHON" -m lerobot.scripts.lerobot_calibrate --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID" ;;
  *) echo 'Usage: bash scripts/calibrate_motors.sh follower|leader' >&2; exit 2 ;;
esac
