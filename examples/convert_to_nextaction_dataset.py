#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert dataset from the original NextAction format to LeRobot format.

This script converts datasets from the format used in the original NextAction training script
(with seq_xxx/ directories containing frames/, actions.npy, and touch data) to LeRobot's
standardized dataset format.

Original format:
    dataset_root/
    ├── seq_000/
    │   ├── frames/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   ├── actions.npy  # [T, action_dim]
    │   └── touch.npy    # [T, 20, 20] or touch/ dir with .npy files
    ├── seq_001/
    └── ...

LeRobot format:
    dataset_output/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl  
    │   └── stats.json
    ├── data/
    └── videos/

Usage:
    python examples/convert_to_nextaction_dataset.py \
        --input-dir path/to/original/dataset_root \
        --output-dir path/to/lerobot/dataset \
        --repo-id my_nextaction_dataset \
        --fps 30
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def list_sequences(root: str) -> List[Path]:
    """List all sequence directories in the dataset root."""
    r = Path(root)
    seqs = []
    for p in sorted(r.iterdir()):
        if (p / "frames").is_dir() and (p / "actions.npy").exists():
            seqs.append(p)
    if not seqs:
        raise FileNotFoundError(f"No sequences found in {root}.")
    return seqs


def load_touch_data(seq_path: Path) -> np.ndarray:
    """Load touch data from sequence directory."""
    # Try touch.npy first (preferred format)
    touch_path = seq_path / "touch.npy"
    if touch_path.exists():
        arr = np.load(touch_path).astype(np.float32)  # [T, 20, 20]
    elif (seq_path / "touch").is_dir():
        # Load from individual .npy files
        fns = sorted((seq_path / "touch").glob("*.npy"))
        arr = np.stack([np.load(fn).astype(np.float32) for fn in fns], axis=0)
    else:
        raise FileNotFoundError(f"No touch data found in {seq_path} (need touch.npy or touch/ dir)")
    
    # Normalize to [0,1] if values look like 0..255
    if arr.max() > 1.5:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    
    return arr  # [T, 20, 20]


def get_dataset_info(sequences: List[Path]) -> dict:
    """Analyze sequences to determine dataset properties."""
    # Load first sequence to get dimensions
    first_seq = sequences[0]
    
    # Get action dimension
    actions = np.load(first_seq / "actions.npy")
    action_dim = actions.shape[1]
    
    # Get image dimensions from first frame
    frame_files = list((first_seq / "frames").glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {first_seq / 'frames'}")
    
    first_image = Image.open(frame_files[0])
    img_width, img_height = first_image.size
    
    # Get touch dimensions
    touch_data = load_touch_data(first_seq)
    touch_height, touch_width = touch_data.shape[1:]
    
    # Calculate total frames
    total_frames = 0
    for seq in sequences:
        actions = np.load(seq / "actions.npy")
        total_frames += len(actions)
    
    return {
        "action_dim": action_dim,
        "img_height": img_height,
        "img_width": img_width,
        "touch_height": touch_height,
        "touch_width": touch_width,
        "total_frames": total_frames,
        "num_episodes": len(sequences),
    }


def create_lerobot_dataset(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
) -> LeRobotDataset:
    """Create a LeRobot dataset from the original format."""
    
    # List all sequences
    sequences = list_sequences(input_dir)
    logging.info(f"Found {len(sequences)} sequences in {input_dir}")
    
    # Analyze dataset
    info = get_dataset_info(sequences)
    logging.info(f"Dataset info: {info}")
    
    # Define features
    features = {
        "observation.images.camera": {
            "shape": [3, info["img_height"], info["img_width"]],
            "names": ["C", "H", "W"],
            "dtype": "uint8"
        },
        "observation.touch": {
            "shape": [info["touch_height"], info["touch_width"]],
            "names": ["H", "W"],
            "dtype": "float32"
        },
        "action": {
            "shape": [info["action_dim"]],
            "names": [f"dim_{i}" for i in range(info["action_dim"])],
            "dtype": "float32"
        }
    }
    
    # Create empty LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        use_videos=True,  # Use video format for images
    )
    
    # Convert each sequence to an episode
    for episode_idx, seq_path in enumerate(tqdm(sequences, desc="Converting episodes")):
        logging.info(f"Converting episode {episode_idx}: {seq_path.name}")
        
        # Load data
        actions = np.load(seq_path / "actions.npy").astype(np.float32)
        touch_data = load_touch_data(seq_path)
        
        # Load images
        frame_files = sorted((seq_path / "frames").glob("*.jpg"))
        num_frames = len(actions)
        
        if len(frame_files) != num_frames:
            logging.warning(
                f"Mismatch in episode {episode_idx}: {len(frame_files)} frames vs {num_frames} actions"
            )
            # Use minimum to avoid index errors
            num_frames = min(len(frame_files), len(actions), len(touch_data))
        
        # Convert and add frames
        for frame_idx in range(num_frames):
            # Load and convert image
            img = Image.open(frame_files[frame_idx]).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            
            # Prepare frame data
            frame_data = {
                "observation.images.camera": img_tensor,
                "observation.touch": torch.from_numpy(touch_data[frame_idx]),
                "action": torch.from_numpy(actions[frame_idx]),
                "episode_index": episode_idx,
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
            }
            
            # Add frame to dataset
            dataset.add_frame(frame_data)
        
        # Consolidate episode
        dataset.consolidate_episode()
    
    # Finalize dataset
    dataset.close()
    logging.info(f"Dataset conversion complete: {output_dir}")
    
    return dataset


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert NextAction dataset to LeRobot format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to original dataset root containing seq_xxx/ directories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output LeRobot dataset directory")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate of the dataset (default: 30)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Convert dataset
    dataset = create_lerobot_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
    )
    
    logging.info("Conversion completed successfully!")
    logging.info(f"Dataset saved to: {args.output_dir}")
    logging.info(f"Episodes: {dataset.num_episodes}")
    logging.info(f"Total frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
