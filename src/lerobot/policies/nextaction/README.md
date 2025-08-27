# NextAction Policy

The NextAction policy predicts the next action from a sequence of:
- RGB image observations
- Touch sensor data
- Past actions

This policy is designed for robotics tasks where tactile feedback is crucial for precise manipulation.

## Architecture

The NextAction model consists of:

1. **Vision Encoder**: A lightweight CNN (`TinyCNN`) that processes RGB images
2. **Touch Encoder**: A specialized CNN (`TouchCNN`) that processes touch maps
3. **Sequence Modeling**: A GRU that processes the combined vision, touch, and action history
4. **Action Prediction Head**: Fully connected layers that output the next action

### Model Flow

```
Input:
├── RGB Images [B, T, 3, H, W]
├── Touch Maps [B, T, touch_H, touch_W]  
└── Past Actions [B, T, action_dim]

Processing:
├── Vision CNN → [B, T, vision_dim]
├── Touch CNN → [B, T, touch_dim]
└── Concatenate with actions → [B, T, vision_dim + touch_dim + action_dim]

Sequence Modeling:
└── GRU → [B, hidden_dim]

Output:
└── Action Head → [B, action_dim]
```

## Usage

### Training

Use the provided training script:

```bash
python examples/train_nextaction_policy.py \
    --config-path src/lerobot/configs/nextaction_example.yaml \
    --output-dir outputs/nextaction_training
```

### Using LeRobot's Training Infrastructure

You can also use LeRobot's standard training script:

```bash
python src/lerobot/scripts/train.py \
    --config-path src/lerobot/configs/nextaction_example.yaml
```

### Configuration

Key configuration parameters:

```yaml
policy:
  type: "nextaction"
  n_obs_steps: 5                # Number of observation steps to use
  vision_backbone_dim: 256      # Vision encoder output dimension
  touch_backbone_dim: 128       # Touch encoder output dimension
  hidden_dim: 512               # GRU hidden dimension
  gru_num_layers: 2             # Number of GRU layers
  img_size: 128                 # Input image size (square)
  touch_height: 20              # Touch map height
  touch_width: 20               # Touch map width
  tanh_output: false            # Whether to apply tanh to outputs
  use_amp: true                 # Use automatic mixed precision
  optimizer_lr: 1e-3            # Learning rate
  optimizer_weight_decay: 1e-5  # Weight decay
```

## Dataset Requirements

The NextAction policy expects datasets with the following features:

### Required Features

1. **Image observations**: `observation.images.{camera_name}`
   - Shape: `[3, H, W]` (RGB images)
   - Will be resized to `[3, img_size, img_size]`

2. **Touch observations**: `observation.touch`
   - Shape: `[touch_height, touch_width]` (touch maps)
   - Values should be normalized to [0, 1] range

3. **Actions**: `action`
   - Shape: `[action_dim]` (robot actions)

### Dataset Structure

Your LeRobot dataset should have the following structure:

```
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── stats.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/ (optional)
    └── chunk-000/
        ├── observation.images.camera_name/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.touch/ (if using video format)
```

### Converting Your Data

If you have data in the format used by your original training script, you can convert it to LeRobot format:

1. **Images**: Convert from individual frame files to LeRobot's video or parquet format
2. **Touch data**: Convert from `.npy` files to parquet format with proper feature definitions
3. **Actions**: Convert from `.npy` files to parquet format

Example conversion script:

```python
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Create dataset metadata
features = {
    "observation.images.camera": {
        "shape": [3, 480, 640],  # Original image size
        "names": ["C", "H", "W"],
        "dtype": "uint8"
    },
    "observation.touch": {
        "shape": [20, 20],
        "names": ["H", "W"], 
        "dtype": "float32"
    },
    "action": {
        "shape": [7],  # Your action dimension
        "names": ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", "dim_6"],
        "dtype": "float32"
    }
}

# Create empty dataset
dataset = LeRobotDataset.create(
    repo_id="my_dataset",
    fps=30,  # Your dataset frame rate
    features=features,
    root="path/to/output"
)

# Add episodes...
# (See LeRobot documentation for detailed conversion examples)
```

## Model Performance

The NextAction policy is designed for:
- **Real-time inference**: Lightweight CNN architectures for fast processing
- **Sequential reasoning**: GRU-based sequence modeling captures temporal dependencies
- **Multi-modal fusion**: Effective combination of vision, touch, and action history

### Typical Training Times

- **Small dataset** (~1K episodes): ~2-4 hours on single GPU
- **Medium dataset** (~10K episodes): ~8-12 hours on single GPU
- **Large dataset** (~100K episodes): ~1-2 days on single GPU

## Evaluation

After training, evaluate your model:

```bash
python src/lerobot/scripts/eval.py \
    --pretrained-policy-path outputs/nextaction_training \
    --eval-n-episodes 50
```

## Citation

If you use the NextAction policy in your research, please cite:

```bibtex
@misc{nextaction2024,
  title={NextAction: Multi-modal Action Prediction with Touch and Vision},
  author={Your Name},
  year={2024}
}
```
