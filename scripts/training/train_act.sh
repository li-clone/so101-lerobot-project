#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v2}"
case "$VERSION" in
  v1)
    DATASET_NAME=so101_pick_place_v1
    RUN_NAME=act_so101_pick_place_v1_v1_50k
    ;;
  v2)
    DATASET_NAME=so101_pick_place_v2_70
    RUN_NAME=act_so101_pick_place_v2_70_v2_50k
    ;;
  *) echo "Usage: $0 {v1|v2}" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
mkdir -p "$OUTPUT_ROOT"
set -o pipefail

lerobot-train \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$DATA_ROOT/$DATASET_NAME" \
  --dataset.return_uint8=true \
  --dataset.eval_split=0.1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_ROOT/$RUN_NAME" \
  --job_name="$RUN_NAME" \
  --batch_size=16 \
  --steps=50000 \
  --num_workers=4 \
  --persistent_workers=false \
  --eval_steps=1000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --wandb.enable=false \
  2>&1 | tee "$OUTPUT_ROOT/${RUN_NAME}_console.log"
