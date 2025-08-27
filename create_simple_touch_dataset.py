#!/usr/bin/env python

"""Create a simple touch dataset by manually copying and modifying the PP-v1 dataset."""

import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm


def create_touch_dataset_manually():
    """Create a touch dataset by manually copying the PP-v1 structure and adding touch data."""
    
    # Paths
    source_path = Path("/Users/aditya/.cache/huggingface/lerobot/adungus/PP-v1")
    target_path = Path("/Users/aditya/.cache/huggingface/lerobot/adungus/touch")
    
    print(f"Creating touch dataset at: {target_path}")
    
    # Remove if exists
    if target_path.exists():
        shutil.rmtree(target_path)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the directory structure
    shutil.copytree(source_path / "meta", target_path / "meta")
    shutil.copytree(source_path / "videos", target_path / "videos")
    
    # Create data directory
    (target_path / "data").mkdir(exist_ok=True)
    (target_path / "data" / "chunk-000").mkdir(exist_ok=True)
    
    print("Copied base structure...")
    
    # Update info.json to include touch feature
    info_path = target_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Add touch feature
    info["features"]["observation.touch"] = {
        "dtype": "float32",
        "shape": [20, 20],
        "names": ["height", "width"]
    }
    
    # Update totals for only 3 episodes
    info["total_episodes"] = 3
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print("Updated info.json with touch feature...")
    
    # Update episodes.jsonl for only 3 episodes
    episodes_path = target_path / "meta" / "episodes.jsonl"
    episodes_data = []
    
    with open(source_path / "meta" / "episodes.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i < 3:  # Only first 3 episodes
                episodes_data.append(json.loads(line.strip()))
    
    with open(episodes_path, 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')
    
    print("Updated episodes.jsonl for 3 episodes...")
    
    # Process each episode parquet file and add touch data
    for episode_idx in range(3):
        print(f"Processing episode {episode_idx}...")
        
        source_parquet = source_path / f"data/chunk-000/episode_{episode_idx:06d}.parquet"
        target_parquet = target_path / f"data/chunk-000/episode_{episode_idx:06d}.parquet"
        
        if not source_parquet.exists():
            print(f"Warning: Source parquet not found: {source_parquet}")
            continue
        
        # Load parquet data
        import pandas as pd
        df = pd.read_parquet(source_parquet)
        
        print(f"Episode {episode_idx} has {len(df)} frames")
        
        # Add touch data column
        touch_data = []
        for frame_idx in range(len(df)):
            # Create synthetic touch data
            touch = np.random.rand(20, 20).astype(np.float32) * 0.3
            
            # Add a central hotspot based on gripper action
            try:
                gripper_pos = df.iloc[frame_idx]['action.gripper.pos']
                touch_intensity = max(0.1, 1.0 - abs(gripper_pos))
            except:
                touch_intensity = 0.5
                
            center_y, center_x = 10, 10
            for y in range(8, 13):
                for x in range(8, 13):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist <= 3:
                        touch[y, x] = touch_intensity * (1 - dist/3)
            
            # Flatten for storage (parquet format)
            touch_data.append(touch.flatten())
        
        # Add touch columns to dataframe
        touch_array = np.array(touch_data)
        for i in range(400):  # 20x20 = 400
            df[f'observation.touch.{i:03d}'] = touch_array[:, i]
        
        # Save updated parquet
        df.to_parquet(target_parquet, index=False)
        
        print(f"Saved episode {episode_idx} with touch data")
    
    print("Dataset creation complete!")
    print(f"Location: {target_path}")
    
    return target_path


if __name__ == "__main__":
    dataset_path = create_touch_dataset_manually()
    print(f"\nTouch dataset created successfully at: {dataset_path}")
    print("You can now use this dataset with the NextAction policy!")
