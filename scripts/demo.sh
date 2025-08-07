#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/demo.sh
#   ./scripts/demo.sh --display_data=true   # example of passing extra flags

# Ensure we run in the right environment
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh" || true
  conda activate lerobot || true
fi

# Create a unique local dataset root per run
RUN_TS=$(date +%Y%m%d_%H%M%S)
DATASET_ROOT="/home/aditya/lerobot/local_eval_demo_${RUN_TS}"

python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30},
    "top":     {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30}
  }' \
  --display_data=false \
  --dataset.repo_id=local/eval_demo \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="Put the black box in the bowl" \
  --dataset.push_to_hub=false \
  --policy.path=adungus/PP-v1_act80k \
  "$@" 