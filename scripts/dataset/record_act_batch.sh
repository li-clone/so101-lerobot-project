#!/usr/bin/env bash
set -euo pipefail

: "${FOLLOWER_PORT:?Set FOLLOWER_PORT}"
: "${LEADER_PORT:?Set LEADER_PORT}"
: "${HANDEYE_CAMERA:?Set HANDEYE_CAMERA}"
: "${ENVIRONMENT_CAMERA:?Set ENVIRONMENT_CAMERA}"

DATASET_ID="${DATASET_ID:-local/so101_pick_place_v2_supplement}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-$PWD/data}/so101_pick_place_v2_supplement}"
NUM_EPISODES="${NUM_EPISODES:-10}"
RESUME="${RESUME:-false}"
TASK="${TASK:-Pick up the fixed yellow cable bundle and place it in the black target area.}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower_main \
  --robot.cameras="{handeye: {type: opencv, index_or_path: $HANDEYE_CAMERA, width: 640, height: 480, fps: 30, fourcc: YUYV}, environment: {type: opencv, index_or_path: $ENVIRONMENT_CAMERA, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=so101_leader_main \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK" \
  --dataset.fps=20 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.rgb_encoder.preset=fast \
  --dataset.rgb_encoder.crf=23 \
  --display_data=false \
  --resume="$RESUME"
