#!/usr/bin/env python

"""Test script to verify NextAction dataset loading after conversion.

This script loads a converted NextAction dataset and displays information
about the data structure, shapes, and sample values.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def test_dataset(dataset_dir: str):
    """Test loading a NextAction format dataset.
    
    Args:
        dataset_dir: Path to NextAction dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return
    
    # Load dataset metadata
    meta_file = dataset_path / "dataset_info.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            dataset_meta = json.load(f)
        print(f"Dataset Metadata:")
        print(f"  Episodes: {dataset_meta['num_episodes']}")
        print(f"  Camera: {dataset_meta['camera_key']}")
        print(f"  Touch start frame: {dataset_meta['touch_start_frame']}")
        print(f"  Includes state: {dataset_meta['includes_state']}")
        print()
    
    # Find all sequences
    sequences = sorted(dataset_path.glob("seq_*"))
    print(f"Found {len(sequences)} sequences")
    
    if not sequences:
        print("No sequences found!")
        return
    
    # Test first sequence
    print(f"\nTesting first sequence: {sequences[0].name}")
    seq_path = sequences[0]
    
    # Load actions
    actions_file = seq_path / "actions.npy"
    if actions_file.exists():
        actions = np.load(actions_file)
        print(f"  Actions shape: {actions.shape}")
        print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  First action: {actions[0]}")
    else:
        print(f"  Warning: actions.npy not found")
    
    # Load touch data
    touch_file = seq_path / "touch.npy"
    if touch_file.exists():
        touch = np.load(touch_file)
        print(f"\n  Touch shape: {touch.shape}")
        print(f"  Touch range: [{touch.min():.3f}, {touch.max():.3f}]")
        
        # Check when touch starts
        touch_start = -1
        for i in range(len(touch)):
            if touch[i].max() > 0.01:
                touch_start = i
                break
        print(f"  Touch starts at frame: {touch_start}")
        
        # Show touch statistics
        if touch_start >= 0:
            active_touch = touch[touch_start:]
            print(f"  Active touch mean: {active_touch.mean():.3f}")
            print(f"  Active touch std: {active_touch.std():.3f}")
    else:
        print(f"  Warning: touch.npy not found")
    
    # Check frames
    frames_dir = seq_path / "frames"
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("*.jpg"))
        print(f"\n  Frames: {len(frame_files)}")
        
        if frame_files:
            # Check first frame
            first_frame = Image.open(frame_files[0])
            print(f"  Frame size: {first_frame.size}")
            print(f"  Frame mode: {first_frame.mode}")
    else:
        print(f"  Warning: frames/ directory not found")
    
    # Load sequence metadata
    meta_file = seq_path / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            seq_meta = json.load(f)
        print(f"\n  Sequence metadata:")
        for key, value in seq_meta.items():
            print(f"    {key}: {value}")
    
    # Test data consistency
    print(f"\nData Consistency Check:")
    if 'actions' in locals() and 'touch' in locals():
        if len(actions) == len(touch):
            print(f"  ✓ Actions and touch have same length: {len(actions)}")
        else:
            print(f"  ✗ Length mismatch: actions={len(actions)}, touch={len(touch)}")
    
    if 'frame_files' in locals() and 'actions' in locals():
        if len(frame_files) == len(actions):
            print(f"  ✓ Frames and actions have same length: {len(actions)}")
        else:
            print(f"  ✗ Length mismatch: frames={len(frame_files)}, actions={len(actions)}")
    
    # Test loading multiple sequences
    print(f"\nTesting all sequences:")
    for i, seq_path in enumerate(sequences[:5]):  # Test first 5
        actions = np.load(seq_path / "actions.npy")
        touch = np.load(seq_path / "touch.npy")
        frames = len(list((seq_path / "frames").glob("*.jpg")))
        print(f"  {seq_path.name}: actions={actions.shape}, touch={touch.shape}, frames={frames}")


def main():
    parser = argparse.ArgumentParser(description="Test NextAction dataset loading")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to NextAction dataset directory")
    
    args = parser.parse_args()
    test_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
