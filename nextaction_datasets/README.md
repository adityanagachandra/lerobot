# NextAction Datasets

This directory contains datasets converted from LeRobot format to NextAction format.

## Dataset Structure

Each converted dataset follows this structure:
```
dataset_name/
├── seq_000/
│   ├── frames/           # Individual JPEG frames
│   ├── actions.npy      # Combined actions and states
│   ├── touch.npy        # Synthetic touch data
│   └── meta.json        # Sequence metadata
├── seq_001/
│   └── ...
└── dataset_info.json    # Overall dataset metadata
```

## Converted Datasets

### PP-v1_touch
- **Source**: `adungus/PP-v1` from HuggingFace
- **Episodes**: 3 (for testing)
- **Touch Start Frame**: 100
- **Camera**: observation.images.top
- **Size**: ~312MB

To convert more episodes or create your own dataset:
```bash
python examples/convert_lerobot_to_nextaction.py \
    --input-dir ~/.cache/huggingface/lerobot/your_username/your_dataset \
    --output-dir ./nextaction_datasets/your_dataset \
    --touch-start-frame 100 \
    --camera-key observation.images.top
```

## Note on Git

Large datasets are excluded from git via `.gitignore`. Only this README and sample metadata files are tracked. For sharing datasets, consider using:
- HuggingFace Hub
- Cloud storage (S3, GCS, etc.)
- Git LFS for smaller datasets
