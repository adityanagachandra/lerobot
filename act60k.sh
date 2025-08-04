#!/bin/bash

# ACT PickPlace 100k Recording Script
# Runs lerobot.record with ACT policy for collecting black box manipulation data
#
# Usage: ./act60k.sh
#
# This script uses the ACT PickPlace_100k model for robotic manipulation recording.

set -e  # Exit on any error

echo "🚀 Starting ACT PickPlace 100k Recording Session..."
echo "📋 Task: Put the black box in the bowl"
echo "🤖 Policy: adungus/PickPlace_100k"
echo "💾 Eval Output : adungus/eval_smolVLA_40k-v4"
echo "⏱️  Episode Duration: 60 seconds"
echo "============================================"

# Run the recording command
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58FD0171971 \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}, "side": {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}}' \
  --dataset.repo_id=adungus/eval_smolVLA_40k-v4 \
  --dataset.single_task="Put the black box in the bowl" \
  --policy.path=adungus/PickPlace_100k \
  --log=true \
  --dataset.episode_time_s=60

echo "Recording session completed!"