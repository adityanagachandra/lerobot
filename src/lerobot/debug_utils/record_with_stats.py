#!/usr/bin/env python3
"""
Modified recording script that uses computed dataset statistics with SmolVLA.

This script demonstrates how to modify your recording command to use
precomputed normalization statistics, avoiding the infinity error.

Usage:
    # First compute the stats
    python compute_dataset_stats.py --dataset_path local_dataset

    # Then use this modified recording approach
    python record_with_stats.py \
      --robot.type=so100_follower \
      --robot.id=so100_follow \
      --robot.port=/dev/tty.usbmodem58FD0171971 \
      --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}, "side": {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}}' \
      --display_data=true \
      --dataset.repo_id=adungus/eval_smolVLA_black \
      --dataset.single_task="Put the black box in the bowl" \
      --policy.path=lerobot/smolvla_base \
      --dataset_stats_path=dataset_stats.pt
"""

import argparse
import torch
from pathlib import Path
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs import parser
from lerobot.configs.parser import RecordPipelineConfig


def load_policy_with_stats(policy_path: str, stats_path: Path, device: str = "auto") -> SmolVLAPolicy:
    """Load SmolVLA policy with precomputed dataset statistics."""
    
    print(f"Loading dataset statistics from: {stats_path}")
    dataset_stats = torch.load(stats_path, map_location="cpu")
    
    print("Dataset statistics loaded:")
    for key, stats in dataset_stats.items():
        print(f"  {key}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}")
    
    print(f"Loading SmolVLA policy from: {policy_path}")
    
    # Load the pretrained policy
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = policy.to(device)
    
    print(f"Updating normalization statistics for device: {device}")
    
    # Update normalization statistics
    for key, stats in dataset_stats.items():
        # Convert to the same device as the policy
        device_stats = {k: v.to(device) for k, v in stats.items()}
        
        # Update normalize_inputs
        buffer_name = f"buffer_{key.replace('.', '_')}"
        if hasattr(policy.normalize_inputs, buffer_name):
            buffer = getattr(policy.normalize_inputs, buffer_name)
            if "mean" in device_stats and hasattr(buffer, "mean"):
                buffer.mean.data = device_stats["mean"]
                print(f"  Updated normalize_inputs.{buffer_name}.mean")
            if "std" in device_stats and hasattr(buffer, "std"):
                buffer.std.data = device_stats["std"] 
                print(f"  Updated normalize_inputs.{buffer_name}.std")
        
        # Update normalize_targets and unnormalize_outputs for action
        if key == "action":
            # Update normalize_targets
            if hasattr(policy.normalize_targets, "buffer_action"):
                buffer = policy.normalize_targets.buffer_action
                if "mean" in device_stats and hasattr(buffer, "mean"):
                    buffer.mean.data = device_stats["mean"]
                    print(f"  Updated normalize_targets.buffer_action.mean")
                if "std" in device_stats and hasattr(buffer, "std"):
                    buffer.std.data = device_stats["std"]
                    print(f"  Updated normalize_targets.buffer_action.std")
            
            # Update unnormalize_outputs
            if hasattr(policy.unnormalize_outputs, "buffer_action"):
                buffer = policy.unnormalize_outputs.buffer_action
                if "mean" in device_stats and hasattr(buffer, "mean"):
                    buffer.mean.data = device_stats["mean"]
                    print(f"  Updated unnormalize_outputs.buffer_action.mean")
                if "std" in device_stats and hasattr(buffer, "std"):
                    buffer.std.data = device_stats["std"]
                    print(f"  Updated unnormalize_outputs.buffer_action.std")
    
    # Verify no infinity values remain
    print("\nVerifying normalization buffers...")
    has_inf = False
    for name, param in policy.named_parameters():
        if ("normalize" in name or "unnormalize" in name) and ("mean" in name or "std" in name):
            if torch.isinf(param).any():
                print(f"  ❌ {name} still contains infinity values!")
                has_inf = True
            else:
                print(f"  ✅ {name} is finite")
    
    if has_inf:
        raise ValueError("Some normalization parameters still contain infinity values!")
    
    print("✅ Policy successfully loaded with finite normalization statistics!")
    return policy


def create_modified_record_script():
    """Create a bash script that uses the computed statistics."""
    script_content = '''#!/bin/bash

# Modified recording script that uses precomputed dataset statistics
# This avoids the "mean is infinity" error in SmolVLA

set -e  # Exit on any error

echo "=== LeRobot Recording with Precomputed Statistics ==="

# Check if dataset statistics exist
STATS_FILE="dataset_stats.pt"
if [ ! -f "$STATS_FILE" ]; then
    echo "❌ Dataset statistics not found: $STATS_FILE"
    echo "First compute statistics with:"
    echo "python compute_dataset_stats.py --dataset_path local_dataset"
    exit 1
fi

echo "✅ Found dataset statistics: $STATS_FILE"

# Use the Python script that loads policy with statistics
python record_with_stats.py \\
  --robot.type=so100_follower \\
  --robot.id=so100_follow \\
  --robot.port=/dev/tty.usbmodem58FD0171971 \\
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}, "side": {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}}' \\
  --display_data=true \\
  --dataset.repo_id=adungus/eval_smolVLA_black \\
  --dataset.single_task="Put the black box in the bowl" \\
  --policy.path=lerobot/smolvla_base \\
  --dataset_stats_path=dataset_stats.pt

echo "✅ Recording completed successfully!"
'''
    
    with open("record_with_precomputed_stats.sh", "w") as f:
        f.write(script_content)
    
    # Make it executable
    import os
    os.chmod("record_with_precomputed_stats.sh", 0o755)
    
    print("Created executable script: record_with_precomputed_stats.sh")


# For demonstration - this would be integrated into the actual record.py
@parser.wrap()
def record_with_stats(cfg: RecordPipelineConfig, dataset_stats_path: str = "dataset_stats.pt"):
    """Modified record function that uses precomputed statistics."""
    
    stats_path = Path(dataset_stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset statistics not found: {stats_path}")
    
    # Load policy with statistics
    policy = load_policy_with_stats(cfg.policy.path, stats_path, cfg.policy.device)
    
    print("✅ Policy loaded with precomputed statistics!")
    print("You can now proceed with recording without normalization errors.")
    
    # Here you would continue with the normal recording logic...
    # For this demo, we'll just show that the policy is properly initialized
    
    # Test with a dummy observation
    try:
        dummy_obs = {
            "observation.state": torch.randn(1, 6).to(policy.device),
            "observation.image": torch.randn(1, 3, 256, 256).to(policy.device),
            "observation.image2": torch.randn(1, 3, 256, 256).to(policy.device),
        }
        
        with torch.no_grad():
            action = policy.select_action(dummy_obs)
        
        print(f"✅ Policy inference test successful! Action shape: {action.shape}")
        
    except Exception as e:
        print(f"❌ Policy inference test failed: {e}")
        raise


def main():
    """Main function for standalone testing."""
    parser = argparse.ArgumentParser(description="Record with precomputed dataset statistics")
    parser.add_argument("--dataset_stats_path", type=str, default="dataset_stats.pt",
                       help="Path to computed dataset statistics")
    parser.add_argument("--policy_path", type=str, default="lerobot/smolvla_base",
                       help="Path to SmolVLA model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--create_script", action="store_true",
                       help="Create the bash recording script")
    
    args = parser.parse_args()
    
    if args.create_script:
        create_modified_record_script()
        return
    
    # Test loading policy with stats
    stats_path = Path(args.dataset_stats_path)
    
    if not stats_path.exists():
        print(f"❌ Statistics file not found: {stats_path}")
        print("First run: python compute_dataset_stats.py")
        return
    
    try:
        policy = load_policy_with_stats(args.policy_path, stats_path, args.device)
        print("\n=== Success! ===")
        print("The policy has been successfully loaded with your dataset statistics.")
        print("You can now integrate this approach into your recording workflow.")
        
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()