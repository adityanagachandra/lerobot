#!/usr/bin/env python

"""Convert LeRobot dataset to NextAction format with synthetic touch data.

This script converts datasets from LeRobot's format to the NextAction training format
with seq_xxx/ directories containing frames/, actions.npy, touch.npy, and meta.json.

LeRobot format:
    dataset_root/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   └── stats.json
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── videos/
        └── chunk-000/
            └── observation.images.*/
                └── episode_000000.mp4

NextAction format:
    output_dir/
    ├── seq_000/
    │   ├── frames/
    │   │   ├── 000000.jpg
    │   │   └── ...
    │   ├── actions.npy  # [T, action_dim + state_dim]
    │   ├── touch.npy    # [T, 20, 20]
    │   └── meta.json
    └── seq_001/
        └── ...

Usage:
    python examples/convert_lerobot_to_nextaction.py \
        --input-dir path/to/lerobot/dataset \
        --output-dir path/to/nextaction/dataset \
        --touch-start-frame 10 \
        --camera-key observation.images.top
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Try to import video decoding utilities
try:
    from lerobot.datasets.video_utils import decode_video_frames_torchvision
except ImportError:
    print("Warning: Could not import video utilities. Using fallback method.")
    decode_video_frames_torchvision = None


def generate_synthetic_touch(
    action: np.ndarray,
    state: np.ndarray,
    frame_idx: int,
    episode_idx: int,
    touch_start_frame: int,
    touch_height: int = 20,
    touch_width: int = 20
) -> np.ndarray:
    """Generate synthetic touch data based on gripper state and action.
    
    Args:
        action: Current action vector
        state: Current state vector
        frame_idx: Current frame index
        episode_idx: Episode index
        touch_start_frame: Frame index when touch sensing starts
        touch_height: Height of touch map
        touch_width: Width of touch map
        
    Returns:
        Touch map array of shape [touch_height, touch_width]
    """
    # Create base touch map
    touch = np.zeros((touch_height, touch_width), dtype=np.float32)
    
    # No touch before the start frame
    if frame_idx < touch_start_frame:
        return touch
    
    # Use gripper position from state (assuming last dimension is gripper)
    gripper_pos = state[-1] if len(state) > 0 else 0.5
    
    # Intensity based on gripper closure (0 = open, 1 = closed)
    base_intensity = max(0.1, 1.0 - abs(gripper_pos))
    
    # Add temporal variation
    time_factor = np.sin((frame_idx - touch_start_frame) * 0.1) * 0.2
    intensity = base_intensity * (1 + time_factor)
    
    # Create touch patterns
    center_y, center_x = touch_height // 2, touch_width // 2
    
    # Primary contact point
    for y in range(touch_height):
        for x in range(touch_width):
            # Distance from center
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            
            # Gaussian-like falloff
            if dist <= 5:
                value = intensity * np.exp(-dist**2 / 8)
                # Add noise
                noise = np.random.normal(0, 0.05)
                touch[y, x] = np.clip(value + noise, 0, 1)
    
    # Add secondary contact points based on action
    if len(action) >= 2:
        # Use first two action dimensions to offset secondary contacts
        offset_y = int(action[0] * 3)
        offset_x = int(action[1] * 3)
        
        secondary_y = np.clip(center_y + offset_y, 2, touch_height - 3)
        secondary_x = np.clip(center_x + offset_x, 2, touch_width - 3)
        
        for y in range(max(0, secondary_y - 2), min(touch_height, secondary_y + 3)):
            for x in range(max(0, secondary_x - 2), min(touch_width, secondary_x + 3)):
                dist = np.sqrt((y - secondary_y)**2 + (x - secondary_x)**2)
                if dist <= 2:
                    value = intensity * 0.5 * np.exp(-dist**2 / 2)
                    touch[y, x] = np.clip(touch[y, x] + value, 0, 1)
    
    return touch


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    timestamps: Optional[np.ndarray] = None
) -> List[Path]:
    """Extract frames from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        timestamps: Optional timestamps for frame extraction
        
    Returns:
        List of paths to extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if decode_video_frames_torchvision is not None and timestamps is not None:
        # Use LeRobot's video decoding
        frames = decode_video_frames_torchvision(
            video_path,
            timestamps=timestamps,
            tolerance_s=1e-4
        )
        
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{i:06d}.jpg"
            # Convert tensor to PIL Image
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            img = Image.fromarray((frame * 255).astype(np.uint8))
            img.save(frame_path, quality=95)
            frame_paths.append(frame_path)
            
        return frame_paths
    else:
        # Fallback: Use ffmpeg to extract frames
        import subprocess
        
        # First, get video info
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            num_frames = int(result.stdout.strip())
        except:
            print(f"Warning: Could not get frame count for {video_path}")
            num_frames = 100  # Default assumption
        
        # Extract frames
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-q:v", "2",  # High quality
            str(output_dir / "%06d.jpg")
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Return list of extracted frames
        frame_paths = sorted(output_dir.glob("*.jpg"))
        return frame_paths


def convert_episode(
    episode_idx: int,
    episode_data: pd.DataFrame,
    video_paths: Dict[str, Path],
    output_dir: Path,
    camera_key: str,
    touch_start_frame: int,
    include_state: bool = True
) -> Dict:
    """Convert a single episode to NextAction format.
    
    Args:
        episode_idx: Episode index
        episode_data: DataFrame with episode data
        video_paths: Dictionary mapping camera keys to video file paths
        output_dir: Output directory for this sequence
        camera_key: Which camera to use for frames
        touch_start_frame: Frame when touch sensing starts
        include_state: Whether to concatenate state with actions
        
    Returns:
        Dictionary with episode metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract action and state data
    # PP-v1 format stores actions and states as numpy arrays in single columns
    if 'action' in episode_data.columns and episode_data['action'].dtype == 'object':
        # Actions are stored as numpy arrays
        actions = np.stack(episode_data['action'].values).astype(np.float32)
    else:
        # Try extracting individual action columns
        action_cols = [col for col in episode_data.columns if col.startswith('action.')]
        if action_cols:
            actions = episode_data[action_cols].values.astype(np.float32)
        else:
            raise ValueError("No action data found in episode")
    
    # Extract state data
    if 'observation.state' in episode_data.columns and episode_data['observation.state'].dtype == 'object':
        # States are stored as numpy arrays
        states = np.stack(episode_data['observation.state'].values).astype(np.float32)
    else:
        # Try extracting individual state columns
        state_cols = [col for col in episode_data.columns if col.startswith('observation.state.')]
        states = episode_data[state_cols].values.astype(np.float32) if state_cols else None
    
    # Combine actions and states if requested
    if include_state and states is not None:
        combined_data = np.concatenate([actions, states], axis=1)
    else:
        combined_data = actions
    
    # Save actions
    np.save(output_dir / "actions.npy", combined_data)
    
    # Extract frames from video
    if camera_key in video_paths and video_paths[camera_key].exists():
        timestamps = episode_data['timestamp'].values if 'timestamp' in episode_data else None
        frame_paths = extract_frames_from_video(
            video_paths[camera_key],
            output_dir / "frames",
            timestamps
        )
        num_frames = len(frame_paths)
    else:
        print(f"Warning: No video found for {camera_key} in episode {episode_idx}")
        num_frames = len(actions)
        # Create dummy frames directory
        (output_dir / "frames").mkdir(exist_ok=True)
    
    # Ensure we have the right number of frames
    num_frames = min(num_frames, len(actions))
    
    # Generate touch data
    touch_data = []
    for frame_idx in range(num_frames):
        action = actions[frame_idx] if frame_idx < len(actions) else actions[-1]
        state = states[frame_idx] if states is not None and frame_idx < len(states) else None
        
        touch = generate_synthetic_touch(
            action=action,
            state=state if state is not None else action,
            frame_idx=frame_idx,
            episode_idx=episode_idx,
            touch_start_frame=touch_start_frame
        )
        touch_data.append(touch)
    
    touch_array = np.stack(touch_data, axis=0)
    np.save(output_dir / "touch.npy", touch_array)
    
    # Create metadata
    meta = {
        "episode_idx": episode_idx,
        "num_frames": num_frames,
        "action_dim": combined_data.shape[1],
        "touch_shape": [20, 20],
        "touch_start_frame": touch_start_frame,
        "camera_key": camera_key,
        "includes_state": include_state and states is not None
    }
    
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    return meta


def convert_dataset(
    input_dir: str,
    output_dir: str,
    touch_start_frame: int = 10,
    camera_key: str = "observation.images.top",
    max_episodes: Optional[int] = None,
    include_state: bool = True
):
    """Convert LeRobot dataset to NextAction format.
    
    Args:
        input_dir: Path to LeRobot dataset
        output_dir: Path to output NextAction dataset
        touch_start_frame: Frame when touch sensing starts
        camera_key: Which camera to use for frames
        max_episodes: Maximum number of episodes to convert (None for all)
        include_state: Whether to concatenate state with actions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load dataset metadata
    with open(input_path / "meta" / "info.json", 'r') as f:
        info = json.load(f)
    
    # Load episodes metadata
    episodes = []
    with open(input_path / "meta" / "episodes.jsonl", 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    
    # Limit episodes if requested
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    
    print(f"Converting {len(episodes)} episodes from LeRobot to NextAction format")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Camera: {camera_key}")
    print(f"Touch starts at frame: {touch_start_frame}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each episode
    converted_episodes = []
    
    for ep_idx, episode_info in enumerate(tqdm(episodes, desc="Converting episodes")):
        # Load episode data
        chunk_idx = episode_info.get("chunk", 0)
        episode_file = input_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        
        if not episode_file.exists():
            print(f"Warning: Episode file not found: {episode_file}")
            continue
        
        # Load episode data
        df = pd.read_parquet(episode_file)
        
        # Find video files
        video_dir = input_path / f"videos/chunk-{chunk_idx:03d}"
        video_paths = {}
        
        for cam_dir in video_dir.glob("observation.images.*"):
            cam_key = cam_dir.name
            video_file = cam_dir / f"episode_{ep_idx:06d}.mp4"
            if video_file.exists():
                video_paths[cam_key] = video_file
        
        # Convert episode
        seq_dir = output_path / f"seq_{ep_idx:03d}"
        meta = convert_episode(
            episode_idx=ep_idx,
            episode_data=df,
            video_paths=video_paths,
            output_dir=seq_dir,
            camera_key=camera_key,
            touch_start_frame=touch_start_frame,
            include_state=include_state
        )
        
        converted_episodes.append(meta)
    
    # Save dataset metadata
    dataset_meta = {
        "num_episodes": len(converted_episodes),
        "source_dataset": str(input_path),
        "camera_key": camera_key,
        "touch_start_frame": touch_start_frame,
        "includes_state": include_state,
        "episodes": converted_episodes
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Converted {len(converted_episodes)} episodes")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to NextAction format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output NextAction dataset directory")
    parser.add_argument("--touch-start-frame", type=int, default=10,
                        help="Frame index when touch sensing starts (default: 10)")
    parser.add_argument("--camera-key", type=str, default="observation.images.top",
                        help="Camera key to use for frames (default: observation.images.top)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of episodes to convert (default: all)")
    parser.add_argument("--include-state", action="store_true", default=True,
                        help="Include state observations concatenated with actions")
    parser.add_argument("--no-include-state", dest="include_state", action="store_false",
                        help="Only include actions, not states")
    
    args = parser.parse_args()
    
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        touch_start_frame=args.touch_start_frame,
        camera_key=args.camera_key,
        max_episodes=args.max_episodes,
        include_state=args.include_state
    )


if __name__ == "__main__":
    main()
