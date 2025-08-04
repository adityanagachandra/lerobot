#!/usr/bin/env python3
"""
Wrapper for lerobot.record that automatically fixes SmolVLA normalization statistics.

This script applies a monkey patch to fix the "mean is infinity" error and then
runs the original lerobot.record with your exact same command-line arguments.

Usage (replace your original command):
    Instead of: python -m lerobot.record [args...]
    Use:        python record_with_fixed_stats.py [args...]

All the same arguments work exactly as before!
"""

import sys
import torch
from pathlib import Path


def apply_smolvla_stats_fix():
    """Apply the normalization statistics fix before any policies are loaded."""
    
    # Try to load the computed statistics
    stats_path = Path("dataset_stats.pt")
    
    if not stats_path.exists():
        print("❌ Dataset statistics not found: dataset_stats.pt")
        print("Please run: python compute_dataset_stats.py --dataset_path local_dataset")
        sys.exit(1)
    
    print(f"📊 Loading dataset statistics from: {stats_path}")
    dataset_stats = torch.load(stats_path, map_location="cpu")
    
    # Import the SmolVLA class
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Store the original from_pretrained method
    original_from_pretrained = SmolVLAPolicy.from_pretrained
    
    @classmethod
    def patched_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Patched version that automatically applies dataset statistics."""
        
        print(f"🔧 Applying statistics fix to SmolVLA: {pretrained_name_or_path}")
        
        # Call the original from_pretrained
        policy = original_from_pretrained(pretrained_name_or_path, **kwargs)
        
        # Get the device the policy is on
        device = next(policy.parameters()).device
        
        # Apply the precomputed statistics
        for key, stats in dataset_stats.items():
            # Convert to the same device as the policy
            device_stats = {k: v.to(device) for k, v in stats.items()}
            
            # Update normalize_inputs for observation.state
            if key == "observation.state":
                buffer = policy.normalize_inputs.buffer_observation_state
                buffer.mean.data = device_stats["mean"]
                buffer.std.data = device_stats["std"]
                print(f"  ✅ Fixed normalize_inputs for {key}")
            
            # Update action normalization
            elif key == "action":
                # normalize_targets
                if hasattr(policy.normalize_targets, "buffer_action"):
                    buffer = policy.normalize_targets.buffer_action
                    buffer.mean.data = device_stats["mean"]
                    buffer.std.data = device_stats["std"]
                    print(f"  ✅ Fixed normalize_targets for {key}")
                
                # unnormalize_outputs
                if hasattr(policy.unnormalize_outputs, "buffer_action"):
                    buffer = policy.unnormalize_outputs.buffer_action
                    buffer.mean.data = device_stats["mean"]
                    buffer.std.data = device_stats["std"]
                    print(f"  ✅ Fixed unnormalize_outputs for {key}")
        
        print("🎉 SmolVLA normalization statistics fixed!")
        return policy
    
    # Apply the monkey patch
    SmolVLAPolicy.from_pretrained = patched_from_pretrained


def main():
    """Main function that applies the fix and runs the original record command."""
    
    print("🚀 Starting lerobot.record with SmolVLA statistics fix...")
    
    # Apply the statistics fix
    apply_smolvla_stats_fix()
    
    # Now import and run the original record module
    # We modify sys.argv to make it look like "python -m lerobot.record [original_args]"
    original_argv = sys.argv.copy()
    
    # Set up sys.argv as if we called "python -m lerobot.record"
    sys.argv = ["python", "-m", "lerobot.record"] + original_argv[1:]
    
    try:
        # Import and run the original record module
        from lerobot.record import main as record_main
        print("📹 Starting recording with fixed statistics...")
        record_main()
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("✅ Recording completed successfully!")


if __name__ == "__main__":
    main()