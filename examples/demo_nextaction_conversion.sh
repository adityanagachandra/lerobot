#!/bin/bash

# Demo script for converting LeRobot dataset to NextAction format
# This example uses the PP-v1 dataset as a demonstration

echo "=== LeRobot to NextAction Conversion Demo ==="
echo

# Set paths
LEROBOT_DATASET="$HOME/.cache/huggingface/lerobot/adungus/PP-v1"
NEXTACTION_OUTPUT="./nextaction_datasets/PP-v1_touch_demo"
TOUCH_START_FRAME=10

# Check if source dataset exists
if [ ! -d "$LEROBOT_DATASET" ]; then
    echo "Error: LeRobot dataset not found at $LEROBOT_DATASET"
    echo "Please download it first using:"
    echo "  python -c \"from lerobot.datasets import LeRobotDataset; LeRobotDataset('adungus/PP-v1')\""
    exit 1
fi

echo "Source dataset: $LEROBOT_DATASET"
echo "Output directory: $NEXTACTION_OUTPUT"
echo "Touch starts at frame: $TOUCH_START_FRAME"
echo

# Step 1: Convert dataset (first 3 episodes for demo)
echo "Step 1: Converting first 3 episodes..."
python examples/convert_lerobot_to_nextaction.py \
    --input-dir "$LEROBOT_DATASET" \
    --output-dir "$NEXTACTION_OUTPUT" \
    --touch-start-frame $TOUCH_START_FRAME \
    --camera-key observation.images.top \
    --max-episodes 3 \
    --include-state

if [ $? -ne 0 ]; then
    echo "Error: Conversion failed"
    exit 1
fi

echo
echo "Step 2: Verifying converted dataset..."
python examples/test_nextaction_dataset.py \
    --dataset-dir "$NEXTACTION_OUTPUT"

echo
echo "=== Conversion Complete ==="
echo
echo "NextAction dataset created at: $NEXTACTION_OUTPUT"
echo
echo "Directory structure:"
find "$NEXTACTION_OUTPUT" -type f -name "*.npy" -o -name "*.json" | head -20
echo
echo "To train NextAction with this dataset:"
echo "  1. Ensure you have the original NextAction training script"
echo "  2. Run: python train_nextaction.py --data-dir $NEXTACTION_OUTPUT --use-touch"
echo
echo "Or use LeRobot's training framework with the provided config."
