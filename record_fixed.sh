#!/bin/bash

# Simple wrapper script for lerobot.record with SmolVLA statistics fix
# 
# Usage: ./record_fixed.sh
# 
# This script automatically uses your computed dataset statistics to fix the
# "mean is infinity" error in SmolVLA and runs your exact recording command.

set -e  # Exit on any error

echo "🚀 Starting LeRobot recording with SmolVLA statistics fix..."

# Check if dataset statistics exist
STATS_FILE="dataset_stats.pt"
if [ ! -f "$STATS_FILE" ]; then
    echo "❌ Dataset statistics not found: $STATS_FILE"
    echo "Please run: python compute_dataset_stats.py --dataset_path local_dataset"
    exit 1
fi

echo "📊 Found dataset statistics: $STATS_FILE"

# Use the wrapper script with your original command arguments
python record_with_fixed_stats.py \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58FD0171971 \
  --robot.cameras='{
    "top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},
    "side":     {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}
  }' \
  --display_data=true \
  --dataset.repo_id=adungus/eval_smolVLA_black \
  --dataset.single_task="Put the black box in the bowl" \
  --policy.path=lerobot/smolvla_base

echo "✅ Recording completed successfully!"