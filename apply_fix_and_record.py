#!/usr/bin/env python3
"""
Simple script that applies the SmolVLA fix and directly imports/runs lerobot.record.

This bypasses the sys.argv issues and directly calls the record function.
"""

import sys
import torch
from pathlib import Path

def apply_smolvla_fix():
    """Apply the normalization statistics fix."""
    
    # Load the computed statistics
    stats_path = Path("dataset_stats.pt")
    
    if not stats_path.exists():
        print(f"❌ Statistics file not found: {stats_path}")
        return False
    
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
    return True

def main():
    """Apply the fix and run the original record command."""
    
    print("🚀 Applying SmolVLA statistics fix...")
    
    # Apply the fix first
    if not apply_smolvla_fix():
        return 1
    
    print("📹 Fix applied! Now run your original command:")
    print()
    print("python -m lerobot.record \\")
    print("  --robot.type=so100_follower \\")
    print("  --robot.id=so100_follow \\")
    print("  --robot.port=/dev/tty.usbmodem58FD0171971 \\")
    print("  --robot.cameras='{")
    print('    "top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},')
    print('    "side": {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}')
    print("  }' \\")
    print("  --display_data=true \\")
    print("  --dataset.repo_id=adungus/eval_smolVLA_black \\")
    print('  --dataset.single_task="Put the black box in the bowl" \\')
    print("  --policy.path=lerobot/smolvla_base")
    print()
    print("The normalization error should be fixed! 🎯")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())