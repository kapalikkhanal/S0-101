#!/usr/bin/env bash
# Adapted from So-101_Arm/so101/record.sh: local data only, no automatic upload.
# bash scripts/record.sh --task "Pick up the red cube" --dataset red_cube --episodes 10
source "$(dirname "$0")/hardware_common.sh"
TASK=
DATASET=so101_demos
EPISODES=10
RESUME=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task|--dataset|--episodes) [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; } ;;
  esac
  case "$1" in
    --task) TASK=$2; shift 2 ;;
    --dataset) DATASET=$2; shift 2 ;;
    --episodes) EPISODES=$2; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help) echo 'Usage: bash scripts/record.sh --task "instruction" [--dataset NAME] [--episodes N] [--resume]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n ${TASK//[[:space:]]/} ]] || { echo 'Provide a nonempty --task instruction.' >&2; exit 2; }
[[ "$DATASET" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo 'Dataset name must contain only letters, digits, _ and -.' >&2; exit 2; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'Episodes must be a positive integer.' >&2; exit 2; }
DATASET_ROOT="$SO101_REPO/data/recordings/$DATASET"
require_follower
require_leader
exec "$SO101_LEROBOT_PYTHON" "$SO101_REPO/scripts/lerobot_run.py" record \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID" \
  --dataset.repo_id="local/$DATASET" --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s="${EPISODE_S:-120}" --dataset.reset_time_s=0 \
  --dataset.fps=30 --dataset.vcodec=h264 --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false --display_data="${DISPLAY_DATA:-false}" --resume="$RESUME"
