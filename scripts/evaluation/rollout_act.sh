#!/usr/bin/env bash
set -euo pipefail

: "${FOLLOWER_PORT:?Set FOLLOWER_PORT}"
: "${HANDEYE_CAMERA:?Set HANDEYE_CAMERA}"
: "${ENVIRONMENT_CAMERA:?Set ENVIRONMENT_CAMERA}"
: "${POLICY_PATH:?Set POLICY_PATH to a pretrained_model directory}"

N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
DURATION="${DURATION:-60}"
TASK="${TASK:-Pick up the fixed yellow cable bundle and place it in the black target area.}"

lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower_main \
  --robot.cameras="{handeye: {type: opencv, index_or_path: $HANDEYE_CAMERA, width: 640, height: 480, fps: 30, fourcc: YUYV}, environment: {type: opencv, index_or_path: $ENVIRONMENT_CAMERA, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --fps=20 \
  --task="$TASK" \
  --duration="$DURATION" \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=true
