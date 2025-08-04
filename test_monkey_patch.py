#!/usr/bin/env python3
"""
Test script to verify the monkey patch approach works.

This tests that we can successfully load SmolVLA with the computed statistics
using the same monkey patch that will be used in the recording script.
"""

import sys
import torch
from pathlib import Path


def apply_smolvla_stats_fix():
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


def test_patched_policy():
    """Test that the patched policy works correctly."""
    
    print("🧪 Testing patched SmolVLA policy...")
    
    # Apply the patch
    if not apply_smolvla_stats_fix():
        return False
    
    try:
        # Import after patching
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        # Load the policy (this should now use our patched version)
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        # Get the device from the policy parameters
        device = next(policy.parameters()).device
        
        # Test with a dummy observation (including language instruction)
        dummy_obs = {
            "observation.state": torch.randn(1, 6).to(device),
            "observation.image": torch.randn(1, 3, 256, 256).to(device),
            "observation.image2": torch.randn(1, 3, 256, 256).to(device),
            "task": "Put the black box in the bowl",
        }
        
        print("🔬 Testing policy inference...")
        with torch.no_grad():
            action = policy.select_action(dummy_obs)
        
        print(f"✅ Policy inference successful! Action shape: {action.shape}")
        print(f"Action values: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    print("🚀 Testing SmolVLA monkey patch fix...")
    
    if test_patched_policy():
        print("\n🎉 SUCCESS! The monkey patch works correctly.")
        print("You can now use the recording scripts:")
        print("  - ./record_fixed.sh")
        print("  - python record_with_fixed_stats.py [args...]")
        return 0
    else:
        print("\n❌ FAILED! The monkey patch did not work.")
        return 1


if __name__ == "__main__":
    sys.exit(main())