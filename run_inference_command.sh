#!/bin/bash

# Real-world inference command for SO-100 robot with ACT policy
# This uses the existing lerobot.record command but configured for inference

echo "🤖 Running real-world inference with SO-100 robot"
echo "Policy: adungus/act_all"
echo "Robot: /dev/ttyACM0"
echo "Camera: /dev/video0 (OpenCV fallback)"
echo ""

# First, try with RealSense camera
echo "Attempting with RealSense camera first..."

python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so100_robot \
  --robot.cameras='{"main": {"type": "realsense", "serial_number_or_name": "auto", "width": 320, "height": 240, "fps": 30}}' \
  --dataset.repo_id=my_username/inference_test \
  --dataset.single_task="Pick objects and put them in the box" \
  --dataset.episode_time_s=30 \
  --dataset.num_episodes=1 \
  --dataset.push_to_hub=false \
  --policy.path=adungus/act_all

# If RealSense fails, the user can run this fallback command:
echo ""
echo "If RealSense camera fails, use this OpenCV fallback command:"
echo ""
echo "python -m lerobot.record \\"
echo "  --robot.type=so100_follower \\"
echo "  --robot.port=/dev/ttyACM0 \\"
echo "  --robot.id=my_so100_robot \\"
echo '  --robot.cameras='"'"'{"main": {"type": "opencv", "index_or_path": "/dev/video0", "width": 320, "height": 240, "fps": 30}}'"'"' \\'
echo "  --dataset.repo_id=my_username/inference_test \\"
echo '  --dataset.single_task="Pick objects and put them in the box" \\'
echo "  --dataset.episode_time_s=30 \\"
echo "  --dataset.num_episodes=1 \\"
echo "  --dataset.push_to_hub=false \\"
echo "  --policy.path=adungus/act_all" 