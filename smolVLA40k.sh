#!/bin/bash

# SmolVLA 40k Recording Script
# Runs lerobot.record with SmolVLA policy for collecting black box manipulation data
#
# Usage: ./smolVLA40k.sh
#
# This script uses the patched SmolVLA loading that automatically applies 
# dataset normalization statistics to avoid the "mean is infinity" error.

set -e  # Exit on any error

echo "🚀 Starting SmolVLA 40k Recording Session..."
echo "📋 Task: Put the black box in the bowl"
echo "🤖 Policy: adungus/smolVLA_PickPlace_40k"
echo "💾 Dataset: adungus/eval_smolVLA_40k-v4"
echo "⏱️  Episode Duration: 60 seconds"
echo "============================================"

# Check if we're in the correct directory
if [ ! -f "dataset_stats.pt" ]; then
    echo "⚠️  Warning: dataset_stats.pt not found in current directory"

# Run the recording command
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58FD0171971 \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}, "side": {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}}' \
  --dataset.repo_id=adungus/eval_smolVLA_40k-v4 \
  --dataset.single_task="Put the black box in the bowl" \
  --policy.path=adungus/smolVLA_PickPlace_40k \
  --log=true \
  --dataset.episode_time_s=60

echo "✅ Recording session completed!"