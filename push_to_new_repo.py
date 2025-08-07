#!/usr/bin/env python3

"""
Script to push PP-v1_local dataset to a new repository on Hugging Face Hub.
Based on the successful patterns seen in terminal output.
"""

import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

def main():
    # Configuration
    local_dataset_path = "/Users/aditya/lerobot/PP-v1_local"
    target_repo_id = "adungus/PP-NEW"  # Change this to your desired repo name
    private = False
    
    # Check if local dataset exists
    if not Path(local_dataset_path).exists():
        logging.error(f"❌ Local dataset path does not exist: {local_dataset_path}")
        sys.exit(1)
    
    # Count episodes
    data_dir = Path(local_dataset_path) / "data"
    if data_dir.exists():
        episode_files = list(data_dir.glob("*/episode_*.parquet"))
        num_episodes = len(episode_files)
        if num_episodes > 0:
            episodes_range = f"{min(int(f.stem.split('_')[-1]) for f in episode_files):06d} to {max(int(f.stem.split('_')[-1]) for f in episode_files):06d}"
        else:
            episodes_range = "No episodes found"
    else:
        num_episodes = 0
        episodes_range = "No data directory found"
    
    print(f"🔍 Found {num_episodes} episode files in {local_dataset_path}")
    print(f"📁 Episodes range from {episodes_range}")
    print(f"🚀 Starting upload to BRAND NEW repository...")
    print(f"  - Local path: {local_dataset_path}")
    print(f"  - Target repo: {target_repo_id} (BRAND NEW REPO)")
    print(f"  - Private: {private}")
    print(f"  - Episodes to upload: {num_episodes}")
    
    try:
        # Load the dataset
        print(f"Loading dataset from {local_dataset_path}...")
        dataset = LeRobotDataset(
            repo_id=target_repo_id,
            root=local_dataset_path
        )
        
        print("Dataset loaded successfully!")
        print(f"  - Total episodes: {dataset.num_episodes}")
        print(f"  - Total frames: {len(dataset)}")
        
        # Get task info if available
        try:
            # Try to get task from dataset metadata
            if hasattr(dataset, 'task') and dataset.task:
                task_info = dataset.task
            else:
                task_info = "Put the black box in the bowl"
            print(f"  - Task: {task_info}")
        except:
            print(f"  - Task: Put the black box in the bowl")
        
        # Push to hub
        print(f"Pushing ALL {dataset.num_episodes} episodes to FRESH repository {target_repo_id}...")
        
        dataset.push_to_hub(
            private=private,
            push_videos=True,
            upload_large_folder=True
        )
        
        print(f"✅ Dataset successfully pushed to https://huggingface.co/datasets/{target_repo_id}")
        print(f"✅ ALL {dataset.num_episodes} episodes uploaded successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error during dataset upload: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()