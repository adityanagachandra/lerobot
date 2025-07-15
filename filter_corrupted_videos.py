#!/usr/bin/env python3
"""
Script to filter corrupted videos from a LeRobot dataset and create a clean dataset for fine-tuning.

This script will:
1. Load your existing LeRobot dataset
2. Check for corrupted video files
3. Create a filtered dataset with only good episodes
4. Rename parquet files to maintain sequential numbering
5. Update metadata to reflect the filtered dataset
6. Push the clean dataset to the hub for fine-tuning
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import cv2
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_video_corruption(video_path: Path) -> bool:
    """
    Check if a video file is corrupted by attempting to read it.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if video is corrupted, False if it's good
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return True
        
        # Try to read a few frames
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
        
        cap.release()
        return False
    except Exception:
        return True


def find_corrupted_episodes(dataset_root: Path) -> List[int]:
    """
    Find episodes with corrupted video files.
    
    Args:
        dataset_root: Root path of the dataset
        
    Returns:
        List of episode indices that have corrupted videos
    """
    corrupted_episodes = []
    videos_dir = dataset_root / "videos"
    
    if not videos_dir.exists():
        print("No videos directory found. Skipping video corruption check.")
        return corrupted_episodes
    
    # Find all video files
    video_files = list(videos_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files to check...")
    
    for video_file in tqdm(video_files, desc="Checking video corruption"):
        if check_video_corruption(video_file):
            # Extract episode number from filename
            episode_num = int(video_file.stem.split("_")[-1])
            if episode_num not in corrupted_episodes:
                corrupted_episodes.append(episode_num)
    
    return sorted(corrupted_episodes)


def get_good_episodes(dataset_root: Path, total_episodes: int) -> List[int]:
    """
    Get list of episodes that are not corrupted.
    
    Args:
        dataset_root: Root path of the dataset
        total_episodes: Total number of episodes in the dataset
        
    Returns:
        List of good episode indices
    """
    corrupted_episodes = find_corrupted_episodes(dataset_root)
    print(f"Found {len(corrupted_episodes)} corrupted episodes: {corrupted_episodes}")
    
    all_episodes = set(range(total_episodes))
    good_episodes = list(all_episodes - set(corrupted_episodes))
    
    print(f"Found {len(good_episodes)} good episodes")
    return sorted(good_episodes)


def copy_and_rename_parquet_files(
    source_root: Path, 
    target_root: Path, 
    good_episodes: List[int]
) -> Dict[int, int]:
    """
    Copy parquet files for good episodes and rename them sequentially.
    
    Args:
        source_root: Source dataset root
        target_root: Target dataset root
        good_episodes: List of good episode indices
        
    Returns:
        Mapping from old episode indices to new ones
    """
    source_data_dir = source_root / "data"
    target_data_dir = target_root / "data"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all chunk directories
    chunk_dirs = [d for d in source_data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    episode_mapping = {}  # old_episode -> new_episode
    new_episode_idx = 0
    
    for chunk_dir in chunk_dirs:
        target_chunk_dir = target_data_dir / chunk_dir.name
        target_chunk_dir.mkdir(exist_ok=True)
        
        for episode_idx in good_episodes:
            # Find the parquet file for this episode
            parquet_file = chunk_dir / f"episode_{episode_idx:06d}.parquet"
            if parquet_file.exists():
                # Copy to new location with sequential naming
                new_parquet_file = target_chunk_dir / f"episode_{new_episode_idx:06d}.parquet"
                shutil.copy2(parquet_file, new_parquet_file)
                episode_mapping[episode_idx] = new_episode_idx
                new_episode_idx += 1
    
    return episode_mapping


def copy_video_files(
    source_root: Path, 
    target_root: Path, 
    good_episodes: List[int],
    episode_mapping: Dict[int, int]
) -> None:
    """
    Copy video files for good episodes and rename them sequentially.
    
    Args:
        source_root: Source dataset root
        target_root: Target dataset root
        good_episodes: List of good episode indices
        episode_mapping: Mapping from old episode indices to new ones
    """
    source_videos_dir = source_root / "videos"
    target_videos_dir = target_root / "videos"
    
    if not source_videos_dir.exists():
        print("No videos directory found. Skipping video copy.")
        return
    
    target_videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video directories
    video_dirs = [d for d in source_videos_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    for video_dir in video_dirs:
        target_video_dir = target_videos_dir / video_dir.name
        target_video_dir.mkdir(exist_ok=True)
        
        # Find camera directories
        camera_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
        
        for camera_dir in camera_dirs:
            target_camera_dir = target_video_dir / camera_dir.name
            target_camera_dir.mkdir(exist_ok=True)
            
            for episode_idx in good_episodes:
                video_file = camera_dir / f"episode_{episode_idx:06d}.mp4"
                if video_file.exists():
                    new_video_file = target_camera_dir / f"episode_{episode_mapping[episode_idx]:06d}.mp4"
                    shutil.copy2(video_file, new_video_file)


def update_metadata(
    source_root: Path, 
    target_root: Path, 
    good_episodes: List[int],
    episode_mapping: Dict[int, int],
    new_repo_id: str
) -> None:
    """
    Update metadata files for the filtered dataset.
    
    Args:
        source_root: Source dataset root
        target_root: Target dataset root
        good_episodes: List of good episode indices
        episode_mapping: Mapping from old episode indices to new ones
        new_repo_id: New repository ID for the filtered dataset
    """
    source_meta_dir = source_root / "meta"
    target_meta_dir = target_root / "meta"
    target_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and update info.json
    info_file = source_meta_dir / "info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # Update episode count
        info['total_episodes'] = len(good_episodes)
        # Note: total_frames calculation needs to be done properly by reading parquet files
        # For now, we'll keep the original value and let the dataset handle it
        # info['total_frames'] = sum(len(good_episodes))  # This will need to be calculated properly
        
        with open(target_meta_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    # Copy other metadata files
    for meta_file in source_meta_dir.glob("*.json*"):
        if meta_file.name != "info.json":
            shutil.copy2(meta_file, target_meta_dir / meta_file.name)


def create_filtered_dataset(
    original_repo_id: str,
    original_root: Path,
    new_repo_id: str,
    new_root: Path
) -> LeRobotDataset:
    """
    Create a filtered dataset with only good episodes.
    
    Args:
        original_repo_id: Original repository ID
        original_root: Original dataset root
        new_repo_id: New repository ID
        new_root: New dataset root
        
    Returns:
        LeRobotDataset object for the filtered dataset
    """
    # Load original dataset to get metadata
    original_dataset = LeRobotDataset(original_repo_id, root=original_root)
    total_episodes = original_dataset.meta.total_episodes
    
    print(f"Original dataset has {total_episodes} episodes")
    
    # Find good episodes
    good_episodes = get_good_episodes(original_root, total_episodes)
    
    if not good_episodes:
        raise ValueError("No good episodes found!")
    
    # Create new dataset structure
    new_root.mkdir(parents=True, exist_ok=True)
    
    # Copy and rename parquet files
    episode_mapping = copy_and_rename_parquet_files(original_root, new_root, good_episodes)
    
    # Copy video files
    copy_video_files(original_root, new_root, good_episodes, episode_mapping)
    
    # Update metadata
    update_metadata(original_root, new_root, good_episodes, episode_mapping, new_repo_id)
    
    # Create new dataset object
    filtered_dataset = LeRobotDataset(new_repo_id, root=new_root, episodes=list(range(len(good_episodes))))
    
    return filtered_dataset


def main():
    """Main function to filter corrupted videos and create clean dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter corrupted videos from LeRobot dataset")
    parser.add_argument("--original-repo-id", required=True, help="Original repository ID")
    parser.add_argument("--original-root", required=True, help="Original dataset root path")
    parser.add_argument("--new-repo-id", required=True, help="New repository ID for filtered dataset")
    parser.add_argument("--new-root", required=True, help="New dataset root path")
    parser.add_argument("--push-to-hub", action="store_true", help="Push filtered dataset to hub")
    parser.add_argument("--private", action="store_true", help="Make the new dataset private")
    
    args = parser.parse_args()
    
    # Create filtered dataset
    print(f"Creating filtered dataset from {args.original_repo_id}")
    filtered_dataset = create_filtered_dataset(
        args.original_repo_id,
        Path(args.original_root),
        args.new_repo_id,
        Path(args.new_root)
    )
    
    print(f"Filtered dataset created successfully!")
    print(f"New dataset has {filtered_dataset.num_episodes} episodes")
    print(f"New dataset has {filtered_dataset.num_frames} frames")
    
    # Push to hub if requested
    if args.push_to_hub:
        print("Pushing filtered dataset to hub...")
        filtered_dataset.push_to_hub(
            repo_id=args.new_repo_id,
            tags=["filtered", "clean", "fine-tuning"],
            private=args.private
        )
        print(f"Dataset pushed to hub: {args.new_repo_id}")
    
    print("Done!")


if __name__ == "__main__":
    main() 