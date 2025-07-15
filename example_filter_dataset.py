#!/usr/bin/env python3
"""
Example script showing how to filter corrupted videos and create a clean dataset for SmolVLA fine-tuning.

This script demonstrates:
1. How to use the filter_corrupted_videos.py script
2. How to load the filtered dataset for SmolVLA training
3. How to verify the dataset is ready for fine-tuning
"""

import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    """Example workflow for filtering dataset and preparing for SmolVLA fine-tuning."""
    
    # Configuration - adjust these paths and IDs for your setup
    original_repo_id = "your_username/your_original_dataset"
    original_root = Path("/path/to/your/original/dataset")
    new_repo_id = "your_username/your_filtered_dataset"
    new_root = Path("/path/to/your/filtered/dataset")
    
    print("=== LeRobot Dataset Filtering for SmolVLA Fine-tuning ===")
    
    # Step 1: Check if original dataset exists
    if not original_root.exists():
        print(f"❌ Original dataset not found at {original_root}")
        print("Please update the paths in this script and try again.")
        return
    
    print(f"✅ Found original dataset at {original_root}")
    
    # Step 2: Run the filtering script
    print("\n=== Step 1: Filtering corrupted videos ===")
    
    # You can run this command manually:
    cmd = f"""python filter_corrupted_videos.py \\
        --original-repo-id "{original_repo_id}" \\
        --original-root "{original_root}" \\
        --new-repo-id "{new_repo_id}" \\
        --new-root "{new_root}" \\
        --push-to-hub \\
        --private"""
    
    print("Run this command to filter your dataset:")
    print(cmd)
    print("\nOr run the filtering programmatically:")
    
    # Programmatic filtering (uncomment to use)
    """
    from filter_corrupted_videos import create_filtered_dataset
    
    filtered_dataset = create_filtered_dataset(
        original_repo_id=original_repo_id,
        original_root=original_root,
        new_repo_id=new_repo_id,
        new_root=new_root
    )
    
    print(f"✅ Created filtered dataset with {filtered_dataset.num_episodes} episodes")
    """
    
    # Step 3: Verify the filtered dataset
    print("\n=== Step 2: Verifying filtered dataset ===")
    
    try:
        filtered_dataset = LeRobotDataset(new_repo_id, root=new_root)
        print(f"✅ Successfully loaded filtered dataset")
        print(f"   - Episodes: {filtered_dataset.num_episodes}")
        print(f"   - Frames: {filtered_dataset.num_frames}")
        print(f"   - Features: {list(filtered_dataset.features.keys())}")
        
        # Check for video keys (important for SmolVLA)
        video_keys = filtered_dataset.meta.video_keys
        if video_keys:
            print(f"   - Video keys: {video_keys}")
        else:
            print("   ⚠️  No video keys found - SmolVLA needs video data")
            
    except Exception as e:
        print(f"❌ Failed to load filtered dataset: {e}")
        return
    
    # Step 4: Test with SmolVLA
    print("\n=== Step 3: Testing with SmolVLA ===")
    
    try:
        # Load SmolVLA model
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print("✅ Successfully loaded SmolVLA model")
        
        # Test with a sample from the dataset
        sample = filtered_dataset[0]
        print(f"✅ Successfully loaded sample from dataset")
        print(f"   - Sample keys: {list(sample.keys())}")
        
        # Check if sample has the expected format for SmolVLA
        if "observation.images" in sample:
            print("✅ Sample contains image observations (good for SmolVLA)")
        else:
            print("⚠️  Sample doesn't contain image observations")
            
    except Exception as e:
        print(f"❌ Failed to test with SmolVLA: {e}")
        return
    
    # Step 5: Prepare for fine-tuning
    print("\n=== Step 4: Dataset ready for fine-tuning ===")
    print("✅ Your filtered dataset is ready for SmolVLA fine-tuning!")
    print(f"📁 Dataset location: {new_root}")
    print(f"🔗 Hub repository: {new_repo_id}")
    
    print("\nNext steps:")
    print("1. Use this dataset in your SmolVLA training script")
    print("2. Update your training configuration to use the new dataset")
    print("3. Start fine-tuning!")
    
    # Example training command (you'll need to adapt this to your setup)
    print("\nExample training command:")
    print(f"""python -m lerobot.scripts.train \\
        --config_path=your_smolvla_config.yaml \\
        --dataset.repo_id="{new_repo_id}" \\
        --dataset.root="{new_root}" \\
        --output_dir="./outputs/smolvla_finetuned" """)


if __name__ == "__main__":
    main() 