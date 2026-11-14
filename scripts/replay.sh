#!/usr/bin/env bash
# Moves the follower, including to the episode start pose. Leader is not required.
source "$(dirname "$0")/hardware_common.sh"
[[ $# -eq 2 ]] || { echo 'Usage: bash scripts/replay.sh DATASET_NAME EPISODE_NUMBER' >&2; exit 2; }
[[ "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && "$2" =~ ^[0-9]+$ ]] || { echo 'Invalid dataset name or episode number.' >&2; exit 2; }
require_follower
exec "$SO101_LEROBOT_PYTHON" "$SO101_REPO/scripts/lerobot_run.py" replay \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --dataset.repo_id="local/$1" --dataset.root="$SO101_REPO/data/recordings/$1" --dataset.episode="$2"
