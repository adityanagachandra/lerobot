# LeRobot to NextAction Conversion Guide

This guide explains how to convert your existing LeRobot datasets to the NextAction format with synthetic touch data.

## Overview

The conversion process transforms LeRobot's video-based format into NextAction's frame-based format while adding synthetic touch observations that can be used for multimodal training.

### Format Comparison

**LeRobot Format (v2.1):**
```
dataset/
├── meta/
│   ├── info.json         # Dataset metadata
│   ├── episodes.jsonl    # Episode information
│   └── stats.json        # Dataset statistics
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet  # Episode data
└── videos/
    └── chunk-000/
        └── observation.images.*/
            └── episode_000000.mp4  # Video data
```

**NextAction Format:**
```
dataset/
├── seq_000/              # One sequence per episode
│   ├── frames/           # Individual JPEG frames
│   │   ├── 000000.jpg
│   │   └── ...
│   ├── actions.npy       # [T, action_dim + state_dim]
│   ├── touch.npy         # [T, 20, 20] touch maps
│   └── meta.json         # Sequence metadata
├── seq_001/
│   └── ...
└── dataset_info.json     # Overall dataset metadata
```

## Step 1: Convert Your Dataset

Use the conversion script to transform your LeRobot dataset:

```bash
python examples/convert_lerobot_to_nextaction.py \
    --input-dir ~/.cache/huggingface/lerobot/your_username/your_dataset \
    --output-dir ./nextaction_datasets/your_dataset \
    --touch-start-frame 10 \
    --camera-key observation.images.top \
    --max-episodes 10
```

### Parameters:
- `--input-dir`: Path to your LeRobot dataset
- `--output-dir`: Where to save the NextAction format dataset
- `--touch-start-frame`: Frame index when touch sensing begins (default: 10)
- `--camera-key`: Which camera view to use (default: observation.images.top)
- `--max-episodes`: Number of episodes to convert (omit for all)
- `--include-state/--no-include-state`: Whether to concatenate states with actions

### Touch Data Generation

The synthetic touch data simulates a tactile sensor with:
- 20x20 spatial resolution
- Values normalized to [0, 1]
- Realistic contact patterns based on gripper state
- Temporal variations and noise
- No signal before `touch_start_frame`

The touch generation considers:
1. **Gripper closure**: More intense touch when gripper is closed
2. **Action influence**: Secondary contact points based on movement
3. **Temporal dynamics**: Smooth variations over time
4. **Realistic noise**: Gaussian noise for sensor realism

## Step 2: Verify the Conversion

Test that the conversion was successful:

```bash
python examples/test_nextaction_dataset.py \
    --dataset-dir ./nextaction_datasets/your_dataset
```

This will display:
- Dataset metadata
- Data shapes and ranges
- Frame counts
- Touch activation timing
- Consistency checks

## Step 3: Train NextAction with Touch Data

The NextAction policy already supports multimodal inputs. To train with your converted dataset:

### Option A: Using the existing training script

```bash
python train_nextaction.py \
    --data-dir ./nextaction_datasets/your_dataset \
    --use-touch \
    --touch-start-frame 10 \
    --checkpoint-dir ./checkpoints/nextaction_touch
```

### Option B: Using LeRobot's training framework

1. Update the config file (`src/lerobot/configs/train_nextaction_touch.yaml`):
```yaml
dataset:
  repo_id: "your_username/your_dataset"
  root: "./nextaction_datasets/your_dataset"
  
policy:
  touch_start_frame: 10
  vision_backbone_dim: 256
  touch_backbone_dim: 128
```

2. Train using LeRobot's training script:
```bash
python src/lerobot/scripts/train.py \
    config_path=src/lerobot/configs/train_nextaction_touch.yaml
```

## Customizing Touch Generation

To modify how touch data is generated, edit the `generate_synthetic_touch` function in the conversion script. You can:

1. **Change touch patterns**: Modify the spatial distribution
2. **Adjust intensity mapping**: Change how gripper state maps to touch intensity
3. **Add different sensors**: Create multiple touch maps for different sensor locations
4. **Use real data**: Replace synthetic generation with actual touch sensor data

Example modification for pressure-based touch:

```python
def generate_synthetic_touch(action, state, frame_idx, episode_idx, touch_start_frame):
    # Your custom touch generation logic
    if frame_idx < touch_start_frame:
        return np.zeros((20, 20), dtype=np.float32)
    
    # Example: pressure increases with gripper force
    gripper_force = abs(action[-1])  # Last action dimension
    pressure_map = create_pressure_distribution(gripper_force)
    
    return pressure_map
```

## Troubleshooting

### Common Issues:

1. **Video extraction fails**: Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

2. **Memory issues with large datasets**: Use `--max-episodes` to process in batches

3. **Missing camera views**: Check available cameras with:
   ```bash
   ls ~/.cache/huggingface/lerobot/your_username/your_dataset/videos/chunk-000/
   ```

4. **Touch data shape mismatch**: Ensure your custom touch generation returns [20, 20] arrays

### Performance Tips:

- The conversion extracts individual frames which takes more disk space than videos
- For large datasets, consider converting only a subset for initial testing
- Touch generation is fast but can be parallelized if needed

## Next Steps

After successful conversion and training:

1. **Evaluate the policy**: Test if touch improves task performance
2. **Visualize touch maps**: Create visualization tools to inspect touch patterns
3. **Integrate real sensors**: Replace synthetic touch with actual sensor data
4. **Experiment with architectures**: Modify the touch CNN in NextAction for your needs

## Example Complete Workflow

```bash
# 1. Convert first 5 episodes with touch starting at frame 15
python examples/convert_lerobot_to_nextaction.py \
    --input-dir ~/.cache/huggingface/lerobot/adungus/PP-v1 \
    --output-dir ./nextaction_datasets/PP-v1_touch \
    --touch-start-frame 15 \
    --max-episodes 5

# 2. Verify conversion
python examples/test_nextaction_dataset.py \
    --dataset-dir ./nextaction_datasets/PP-v1_touch

# 3. Train NextAction (if you have the original training script)
python train_nextaction.py \
    --data-dir ./nextaction_datasets/PP-v1_touch \
    --use-touch \
    --epochs 100
```

For questions or issues, refer to the NextAction and LeRobot documentation.
