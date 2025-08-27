# NextAction Policy Integration Guide

This guide explains how your NextAction model has been integrated into the LeRobot framework.

## What Was Created

### 1. Policy Implementation
- **Configuration**: `src/lerobot/policies/nextaction/configuration_nextaction.py`
- **Model**: `src/lerobot/policies/nextaction/modeling_nextaction.py`
- **Init**: `src/lerobot/policies/nextaction/__init__.py`
- **Documentation**: `src/lerobot/policies/nextaction/README.md`

### 2. Framework Integration
- **Factory Registration**: Updated `src/lerobot/policies/factory.py`
- **Policy Exports**: Updated `src/lerobot/policies/__init__.py`

### 3. Training & Examples
- **Training Script**: `examples/train_nextaction_policy.py`
- **Configuration Template**: `src/lerobot/configs/nextaction_example.yaml`
- **Dataset Converter**: `examples/convert_to_nextaction_dataset.py`

## Architecture Overview

Your original model has been preserved but adapted to LeRobot's conventions:

```python
# Original Model Components (preserved)
class TinyCNN(nn.Module):     # Vision encoder - unchanged
class TouchCNN(nn.Module):    # Touch encoder - unchanged  
class NextActionModel(nn.Module):  # Core model - slightly adapted

# New LeRobot Integration
class NextActionPolicy(PreTrainedPolicy):  # LeRobot policy wrapper
class NextActionConfig(PreTrainedConfig):  # Configuration management
```

## Key Features

✅ **Preserved**: Your exact CNN architectures and GRU-based sequence modeling
✅ **Enhanced**: Automatic normalization, device handling, and checkpointing  
✅ **Integrated**: Works with LeRobot's training infrastructure and dataset format
✅ **Configurable**: All hyperparameters configurable via YAML files
✅ **Extensible**: Easy to modify and experiment with

## How to Use

### Step 1: Convert Your Dataset

```bash
python examples/convert_to_nextaction_dataset.py \
    --input-dir /path/to/your/original/dataset_root \
    --output-dir /path/to/lerobot/dataset \
    --repo-id my_nextaction_dataset \
    --fps 30
```

This converts from your original format:
```
dataset_root/
├── seq_000/
│   ├── frames/000000.jpg, 000001.jpg, ...
│   ├── actions.npy  # [T, action_dim]
│   └── touch.npy    # [T, 20, 20]
├── seq_001/
└── ...
```

To LeRobot format with proper metadata and statistics.

### Step 2: Configure Training

Edit `src/lerobot/configs/nextaction_example.yaml`:

```yaml
# Key parameters from your original script
policy:
  type: "nextaction"
  n_obs_steps: 5          # Your sequence length
  vision_backbone_dim: 256
  touch_backbone_dim: 128 
  hidden_dim: 512
  img_size: 128           # Your IMG_SIZE
  touch_height: 20        # Your touch map size
  touch_width: 20
  optimizer_lr: 1e-3      # Your LR
  optimizer_weight_decay: 1e-5  # Your WEIGHT_DECAY
  use_amp: true           # Your USE_AMP

# Dataset
dataset:
  repo_id: "my_nextaction_dataset"  # Your converted dataset
  batch_size: 64          # Your BATCH_SIZE

# Training
steps: 20000              # Equivalent to NUM_EPOCHS * dataset_size / batch_size
```

### Step 3: Train the Model

```bash
python examples/train_nextaction_policy.py \
    --config-path src/lerobot/configs/nextaction_example.yaml \
    --output-dir outputs/nextaction_training
```

Or use LeRobot's standard training:

```bash
python src/lerobot/scripts/train.py \
    --config-path src/lerobot/configs/nextaction_example.yaml
```

## What's Different from Your Original Code

### Enhanced Features
1. **Automatic Normalization**: LeRobot handles input/output normalization based on dataset statistics
2. **Device Management**: Automatic GPU/CPU handling and mixed precision training
3. **Checkpointing**: Automatic model saving, resuming, and versioning
4. **Logging**: Integration with Weights & Biases and structured logging
5. **Evaluation**: Built-in evaluation during training (if environment provided)

### Code Changes
1. **Dataset Loading**: Now uses LeRobot's standardized dataset format instead of custom loaders
2. **Training Loop**: Uses LeRobot's training infrastructure instead of custom loop
3. **Configuration**: YAML-based configuration instead of hardcoded constants
4. **Model Interface**: Wrapped in `PreTrainedPolicy` for standardized inference API

### Maintained Compatibility
- ✅ Exact same CNN architectures (`TinyCNN`, `TouchCNN`)
- ✅ Same GRU-based sequence modeling
- ✅ Same input/output dimensions and flow
- ✅ Same loss function (SmoothL1Loss via LeRobot's training)
- ✅ Same optimization approach (AdamW)

## Advanced Usage

### Custom Configuration

Create your own config file:

```yaml
# my_nextaction_config.yaml
policy:
  type: "nextaction"
  n_obs_steps: 10                    # Longer sequences
  vision_backbone_dim: 512           # Larger vision encoder
  gru_num_layers: 3                  # Deeper GRU
  dropout: 0.2                       # More regularization
  tanh_output: true                  # Bounded actions
```

### Multiple Camera Support

The architecture supports multiple cameras (though your current model uses one):

```python
# In dataset, have multiple camera features:
# "observation.images.camera_left"
# "observation.images.camera_right" 
# The model will automatically handle multiple camera inputs
```

### Custom Touch Dimensions

Change touch map size in config:

```yaml
policy:
  touch_height: 32
  touch_width: 32
```

### Evaluation

After training, evaluate your model:

```bash
python src/lerobot/scripts/eval.py \
    --pretrained-policy-path outputs/nextaction_training/checkpoints/last \
    --eval-n-episodes 50
```

## Troubleshooting

### Common Issues

1. **Dataset Format Errors**
   - Ensure touch data is properly normalized [0, 1]
   - Check that image files exist and are readable
   - Verify action dimensions match across all episodes

2. **Memory Issues**
   - Reduce `batch_size` in config
   - Set `use_amp: true` for memory efficiency
   - Reduce `num_workers` if using too many CPU cores

3. **Training Convergence**
   - Check learning rate (try 5e-4 to 2e-3)
   - Verify dataset statistics are computed correctly
   - Monitor gradient norms (should be < 10)

### Performance Tips

1. **Faster Training**
   - Use `use_amp: true` for mixed precision
   - Increase `batch_size` if you have GPU memory
   - Use multiple workers: `num_workers: 8`

2. **Better Convergence**  
   - Start with smaller learning rate and gradually increase
   - Use learning rate scheduling
   - Add data augmentation if needed

## Next Steps

1. **Experiment**: Try different architectures by modifying the configuration
2. **Scale**: Train on larger datasets with the robust LeRobot infrastructure  
3. **Deploy**: Use the trained model with LeRobot's inference pipeline
4. **Share**: Push trained models to Hugging Face Hub with built-in support

The integration maintains your model's core strengths while adding production-ready features for scalable robotics research!
