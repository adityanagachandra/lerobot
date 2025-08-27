#!/usr/bin/env python

"""Create a new LeRobot dataset with touch data from the first 3 episodes of PP-v1."""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_synthetic_touch_data(action, frame_idx, episode_idx, touch_height=20, touch_width=20):
    """Create synthetic touch data based on gripper position and action."""
    # Use gripper position (last dimension of action) to create touch pattern
    gripper_pos = action[-1]  # gripper.pos
    
    # Create base touch map
    touch = np.zeros((touch_height, touch_width), dtype=np.float32)
    
    # Add touch intensity based on gripper closure
    # When gripper is closed (gripper_pos close to 0), add more touch
    touch_intensity = max(0.1, 1.0 - abs(gripper_pos))
    
    # Create a circular touch pattern in the center
    center_y, center_x = touch_height // 2, touch_width // 2
    radius = 3
    
    for y in range(touch_height):
        for x in range(touch_width):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist <= radius:
                # Add some noise and variation
                noise = np.random.normal(0, 0.1)
                touch[y, x] = touch_intensity * (1 - dist/radius) + noise
                touch[y, x] = np.clip(touch[y, x], 0, 1)
    
    # Add some temporal variation
    time_factor = np.sin(frame_idx * 0.1) * 0.2
    touch = touch * (1 + time_factor)
    touch = np.clip(touch, 0, 1)
    
    return touch


def copy_dataset_with_touch(
    source_path: str,
    target_path: str,
    repo_id: str,
    max_episodes: int = 3
):
    """Copy episodes from source dataset and add synthetic touch data."""
    
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    print(f"Creating new dataset with touch data...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Load source dataset metadata
    with open(source_path / "meta" / "info.json", 'r') as f:
        source_info = json.load(f)
    
    # Create new features with touch data
    new_features = source_info["features"].copy()
    new_features["observation.touch"] = {
        "dtype": "float32",
        "shape": [20, 20],
        "names": ["height", "width"]
    }
    
    # Remove target directory if it exists
    if target_path.exists():
        print(f"Removing existing directory: {target_path}")
        shutil.rmtree(target_path)
    
    # Create target directory
    target_path.mkdir(exist_ok=True, parents=True)
    
    # Create new dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source_info["fps"],
        features=new_features,
        root=target_path,
        use_videos=True,
    )
    
    # Load source episodes metadata
    episodes_data = []
    with open(source_path / "meta" / "episodes.jsonl", 'r') as f:
        for line in f:
            episodes_data.append(json.loads(line.strip()))
    
    print(f"Processing first {max_episodes} episodes...")
    
    # Process each episode
    for episode_idx in range(min(max_episodes, len(episodes_data))):
        episode_info = episodes_data[episode_idx]
        print(f"\nProcessing episode {episode_idx}: {episode_info}")
        
        # Load source episode data
        source_episode_file = source_path / f"data/chunk-000/episode_{episode_idx:06d}.parquet"
        if not source_episode_file.exists():
            print(f"Warning: Episode file not found: {source_episode_file}")
            continue
            
        # Load parquet data
        import pandas as pd
        df = pd.read_parquet(source_episode_file)
        
        print(f"Episode {episode_idx} has {len(df)} frames")
        
        # Load video data for images
        video_files = {
            "observation.images.gripper": source_path / f"videos/chunk-000/observation.images.gripper/episode_{episode_idx:06d}.mp4",
            "observation.images.top": source_path / f"videos/chunk-000/observation.images.top/episode_{episode_idx:06d}.mp4"
        }
        
        # Check if video files exist
        for key, video_file in video_files.items():
            if not video_file.exists():
                print(f"Warning: Video file not found: {video_file}")
                continue
        
        # Load video frames
        from lerobot.datasets.video_utils import VideoFrame, decode_video_frames_torchvision
        
        # Decode video frames
        frames_data = {}
        for key, video_file in video_files.items():
            if video_file.exists():
                frames = decode_video_frames_torchvision(
                    video_file,
                    timestamps=df['timestamp'].values,
                    tolerance_s=1e-4
                )
                frames_data[key] = frames
        
        # Process each frame
        for frame_idx in tqdm(range(len(df)), desc=f"Episode {episode_idx}"):
            row = df.iloc[frame_idx]
            
            # Extract action data
            action = np.array([
                row['action.shoulder_pan.pos'],
                row['action.shoulder_lift.pos'], 
                row['action.elbow_flex.pos'],
                row['action.wrist_flex.pos'],
                row['action.wrist_roll.pos'],
                row['action.gripper.pos']
            ], dtype=np.float32)
            
            # Extract state data
            state = np.array([
                row['observation.state.shoulder_pan.pos'],
                row['observation.state.shoulder_lift.pos'],
                row['observation.state.elbow_flex.pos'],
                row['observation.state.wrist_flex.pos'],
                row['observation.state.wrist_roll.pos'],
                row['observation.state.gripper.pos']
            ], dtype=np.float32)
            
            # Generate synthetic touch data
            touch_data = create_synthetic_touch_data(action, frame_idx, episode_idx)
            
            # Prepare frame data
            frame_data = {
                "action": torch.from_numpy(action),
                "observation.state": torch.from_numpy(state),
                "observation.touch": torch.from_numpy(touch_data),
                "episode_index": episode_idx,
                "frame_index": frame_idx,
                "timestamp": row['timestamp'],
            }
            
            # Add image data if available
            for key, frames in frames_data.items():
                if frame_idx < len(frames):
                    frame_data[key] = frames[frame_idx]
            
            # Add frame to dataset
            dataset.add_frame(frame_data)
        
        # Consolidate episode
        dataset.consolidate_episode()
    
    # Close dataset
    dataset.close()
    
    print(f"\nDataset creation complete!")
    print(f"New dataset saved to: {target_path}")
    print(f"Episodes: {min(max_episodes, len(episodes_data))}")
    
    return dataset


def main():
    source_path = "/Users/aditya/.cache/huggingface/lerobot/adungus/PP-v1"
    target_path = "/Users/aditya/lerobot/touch_dataset"  # Use local path instead
    repo_id = "touch_dataset"
    
    dataset = copy_dataset_with_touch(
        source_path=source_path,
        target_path=target_path,
        repo_id=repo_id,
        max_episodes=3
    )


if __name__ == "__main__":
    main()
