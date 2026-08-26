#!/usr/bin/env bash
set -euo pipefail

: "${FOLLOWER_PORT:?Set FOLLOWER_PORT}"
: "${HANDEYE_CAMERA:?Set HANDEYE_CAMERA}"
: "${ENVIRONMENT_CAMERA:?Set ENVIRONMENT_CAMERA}"
: "${POLICY_PATH:?Set POLICY_PATH}"
: "${EVAL_DATASET_ID:?Set EVAL_DATASET_ID}"
: "${EVAL_DATASET_ROOT:?Set EVAL_DATASET_ROOT}"

N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_EPISODES="${NUM_EPISODES:-5}"
RESUME="${RESUME:-false}"
TASK="${TASK:-Pick up the fixed yellow cable bundle and place it in the black target area.}"

lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=sync \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower_main \
  --robot.cameras="{handeye: {type: opencv, index_or_path: $HANDEYE_CAMERA, width: 640, height: 480, fps: 30, fourcc: YUYV}, environment: {type: opencv, index_or_path: $ENVIRONMENT_CAMERA, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --dataset.repo_id="$EVAL_DATASET_ID" \
  --dataset.root="$EVAL_DATASET_ROOT" \
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
  --fps=20 \
  --task="$TASK" \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=true \
  --resume="$RESUME"
