#!/usr/bin/env python3
"""
Monkey patch for lerobot.record to use precomputed dataset statistics.

This script patches the policy loading process to inject your computed dataset
statistics, allowing you to use the original recording command without modification.

Usage:
    # Run this before your original recording command
    python patch_record_with_stats.py --stats_path dataset_stats.pt

    # Then run your normal recording command (no changes needed!)
    python -m lerobot.record \
      --robot.type=so100_follower \
      --robot.id=so100_follow \
      --robot.port=/dev/tty.usbmodem58FD0171971 \
      --robot.cameras='...' \
      --display_data=true \
      --dataset.repo_id=adungus/eval_smolVLA_black \
      --dataset.single_task="Put the black box in the bowl" \
      --policy.path=lerobot/smolvla_base
"""

import argparse
import sys
import torch
from pathlib import Path


def patch_smolvla_loading(stats_path: Path):
    """Monkey patch SmolVLA to use precomputed statistics."""
    
    # Load the computed statistics
    print(f"Loading dataset statistics from: {stats_path}")
    dataset_stats = torch.load(stats_path, map_location="cpu")
    
    print("Dataset statistics loaded:")
    for key, stats in dataset_stats.items():
        if key in ["observation.state", "action"]:  # Only show the important ones
            print(f"  {key}: mean={stats['mean'][:3]}..., std={stats['std'][:3]}...")
    
    # Import the original SmolVLA class
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Store the original from_pretrained method
    original_from_pretrained = SmolVLAPolicy.from_pretrained
    
    @classmethod
    def patched_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Patched version that automatically applies dataset statistics."""
        
        print(f"🔧 Patching SmolVLA loading for: {pretrained_name_or_path}")
        
        # Call the original from_pretrained
        policy = original_from_pretrained(pretrained_name_or_path, **kwargs)
        
        # Get the device the policy is on
        device = next(policy.parameters()).device
        
        print(f"📊 Applying precomputed statistics to policy on device: {device}")
        
        # Apply the precomputed statistics
        for key, stats in dataset_stats.items():
            # Convert to the same device as the policy
            device_stats = {k: v.to(device) for k, v in stats.items()}
            
            # Update normalize_inputs
            buffer_name = f"buffer_{key.replace('.', '_')}"
            if hasattr(policy.normalize_inputs, buffer_name):
                buffer = getattr(policy.normalize_inputs, buffer_name)
                if "mean" in device_stats and hasattr(buffer, "mean"):
                    buffer.mean.data = device_stats["mean"]
                    print(f"  ✅ Updated normalize_inputs.{buffer_name}.mean")
                if "std" in device_stats and hasattr(buffer, "std"):
                    buffer.std.data = device_stats["std"]
                    print(f"  ✅ Updated normalize_inputs.{buffer_name}.std")
            
            # Update normalize_targets and unnormalize_outputs for action
            if key == "action":
                # Update normalize_targets
                if hasattr(policy.normalize_targets, "buffer_action"):
                    buffer = policy.normalize_targets.buffer_action
                    if "mean" in device_stats and hasattr(buffer, "mean"):
                        buffer.mean.data = device_stats["mean"]
                        print(f"  ✅ Updated normalize_targets.buffer_action.mean")
                    if "std" in device_stats and hasattr(buffer, "std"):
                        buffer.std.data = device_stats["std"]
                        print(f"  ✅ Updated normalize_targets.buffer_action.std")
                
                # Update unnormalize_outputs
                if hasattr(policy.unnormalize_outputs, "buffer_action"):
                    buffer = policy.unnormalize_outputs.buffer_action
                    if "mean" in device_stats and hasattr(buffer, "mean"):
                        buffer.mean.data = device_stats["mean"]
                        print(f"  ✅ Updated unnormalize_outputs.buffer_action.mean")
                    if "std" in device_stats and hasattr(buffer, "std"):
                        buffer.std.data = device_stats["std"]
                        print(f"  ✅ Updated unnormalize_outputs.buffer_action.std")
        
        # Verify no infinity values remain
        has_inf = False
        for name, param in policy.named_parameters():
            if ("normalize" in name or "unnormalize" in name) and ("mean" in name or "std" in name):
                if torch.isinf(param).any():
                    print(f"  ❌ {name} still contains infinity values!")
                    has_inf = True
        
        if has_inf:
            raise ValueError("Some normalization parameters still contain infinity values!")
        
        print("🎉 SmolVLA successfully loaded with finite normalization statistics!")
        return policy
    
    # Apply the monkey patch
    SmolVLAPolicy.from_pretrained = patched_from_pretrained
    print("🔧 Monkey patch applied to SmolVLAPolicy.from_pretrained")


def main():
    parser = argparse.ArgumentParser(description="Patch lerobot.record to use precomputed statistics")
    parser.add_argument("--stats_path", type=str, default="dataset_stats.pt",
                       help="Path to computed dataset statistics")
    
    args = parser.parse_args()
    
    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        print(f"❌ Statistics file not found: {stats_path}")
        print("First run: python compute_dataset_stats.py")
        return 1
    
    # Apply the monkey patch
    patch_smolvla_loading(stats_path)
    
    print("\n🚀 Patch applied successfully!")
    print("You can now run your normal recording command:")
    print()
    print("python -m lerobot.record \\")
    print("  --robot.type=so100_follower \\")
    print("  --robot.id=so100_follow \\")
    print("  --robot.port=/dev/tty.usbmodem58FD0171971 \\")
    print("  --robot.cameras='{...}' \\")
    print("  --display_data=true \\")
    print("  --dataset.repo_id=adungus/eval_smolVLA_black \\")
    print("  --dataset.single_task=\"Put the black box in the bowl\" \\")
    print("  --policy.path=lerobot/smolvla_base")
    print()
    print("The normalization error should be fixed! 🎯")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())