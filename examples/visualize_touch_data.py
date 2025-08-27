#!/usr/bin/env python

"""Visualize touch data from a converted NextAction dataset."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json


def visualize_touch_sequence(dataset_dir: str, sequence: int = 0, start_frame: int = 95, num_frames: int = 10):
    """Visualize touch data evolution over time."""
    
    dataset_path = Path(dataset_dir)
    seq_path = dataset_path / f"seq_{sequence:03d}"
    
    # Load data
    touch_data = np.load(seq_path / "touch.npy")
    actions = np.load(seq_path / "actions.npy")
    
    with open(seq_path / "meta.json", 'r') as f:
        meta = json.load(f)
    
    touch_start = meta['touch_start_frame']
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Touch Data Evolution (Touch starts at frame {touch_start})', fontsize=16)
    
    # Show frames around touch start
    frame_indices = range(start_frame, min(start_frame + num_frames, len(touch_data)))
    
    for idx, (ax, frame_idx) in enumerate(zip(axes.flat, frame_indices)):
        touch_frame = touch_data[frame_idx]
        
        # Display touch map
        im = ax.imshow(touch_frame, cmap='hot', vmin=0, vmax=0.3)
        
        # Add title with frame info
        if frame_idx < touch_start:
            ax.set_title(f'Frame {frame_idx}\n(Before touch)', fontsize=10)
        else:
            ax.set_title(f'Frame {frame_idx}\n(Touch active)', fontsize=10)
        
        # Add gripper state info
        gripper_state = actions[frame_idx][-1] if len(actions[frame_idx]) >= 6 else 0
        ax.text(0.5, -0.15, f'Gripper: {gripper_state:.2f}', 
                transform=ax.transAxes, ha='center', fontsize=8)
        
        ax.axis('off')
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label('Touch Intensity', rotation=270, labelpad=20)
    
    plt.savefig('touch_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: touch_visualization.png")
    
    # Print statistics
    print(f"\nTouch Data Statistics:")
    print(f"  Total frames: {len(touch_data)}")
    print(f"  Touch start frame: {touch_start}")
    print(f"  Max touch value: {touch_data.max():.3f}")
    print(f"  Mean touch (after start): {touch_data[touch_start:].mean():.3f}")
    
    # Show touch activation over time
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    mean_touch_per_frame = touch_data.mean(axis=(1, 2))
    ax2.plot(mean_touch_per_frame, label='Mean touch intensity')
    ax2.axvline(x=touch_start, color='r', linestyle='--', label=f'Touch start (frame {touch_start})')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Mean Touch Intensity')
    ax2.set_title('Touch Intensity Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.savefig('touch_timeline.png', dpi=150, bbox_inches='tight')
    print(f"Saved timeline to: touch_timeline.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize touch data from NextAction dataset")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to NextAction dataset directory")
    parser.add_argument("--sequence", type=int, default=0,
                        help="Sequence index to visualize (default: 0)")
    parser.add_argument("--start-frame", type=int, default=95,
                        help="Starting frame for visualization (default: 95)")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of frames to show (default: 10)")
    
    args = parser.parse_args()
    visualize_touch_sequence(args.dataset_dir, args.sequence, args.start_frame, args.num_frames)


if __name__ == "__main__":
    main()
