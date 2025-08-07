#!/usr/bin/env python3
"""
Test script to demonstrate using computed dataset statistics with SmolVLA.

This script shows how to load the computed statistics and initialize SmolVLA
with proper normalization, avoiding the infinity error.

Usage:
    python test_smolvla_with_stats.py --stats_path dataset_stats.pt --model_path lerobot/smolvla_base
"""

import argparse
import torch
from pathlib import Path
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def load_and_test_policy(stats_path: Path, model_path: str):
    """Load SmolVLA policy with computed dataset statistics."""
    
    print(f"Loading dataset statistics from: {stats_path}")
    dataset_stats = torch.load(stats_path)
    
    print("Dataset statistics loaded:")
    for key, stats in dataset_stats.items():
        print(f"  {key}:")
        for stat_name, tensor in stats.items():
            print(f"    {stat_name}: shape {tensor.shape}, finite: {torch.isfinite(tensor).all()}")
    
    print(f"\nLoading SmolVLA model from: {model_path}")
    
    # Method 1: Initialize policy with statistics from scratch
    print("\n=== Method 1: Initialize new policy with statistics ===")
    try:
        config = SmolVLAConfig()
        policy = SmolVLAPolicy(config, dataset_stats=dataset_stats)
        print("✅ Successfully created SmolVLA policy with dataset statistics!")
        
        # Test that normalization buffers are properly initialized
        print("\nChecking normalization buffers...")
        for name, buffer in policy.normalize_inputs.named_parameters():
            if "mean" in name or "std" in name:
                is_finite = torch.isfinite(buffer).all()
                print(f"  {name}: finite={is_finite}, shape={buffer.shape}")
                if not is_finite:
                    print(f"    WARNING: {name} contains non-finite values!")
        
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        return
    
    # Method 2: Load pretrained model and update its statistics
    print(f"\n=== Method 2: Load pretrained model and update statistics ===")
    try:
        # Load the pretrained model
        policy_pretrained = SmolVLAPolicy.from_pretrained(model_path)
        
        # Update the normalization statistics
        print("Updating normalization statistics...")
        
        # Update normalize_inputs statistics
        for key, stats in dataset_stats.items():
            if hasattr(policy_pretrained.normalize_inputs, f"buffer_{key.replace('.', '_')}"):
                buffer = getattr(policy_pretrained.normalize_inputs, f"buffer_{key.replace('.', '_')}")
                if "mean" in stats and hasattr(buffer, "mean"):
                    buffer.mean.data = stats["mean"]
                if "std" in stats and hasattr(buffer, "std"):
                    buffer.std.data = stats["std"]
                print(f"  Updated statistics for {key}")
        
        # Update normalize_targets and unnormalize_outputs for action statistics
        if "action" in dataset_stats:
            action_stats = dataset_stats["action"]
            
            # Update normalize_targets
            if hasattr(policy_pretrained.normalize_targets, "buffer_action"):
                buffer = policy_pretrained.normalize_targets.buffer_action
                if "mean" in action_stats and hasattr(buffer, "mean"):
                    buffer.mean.data = action_stats["mean"]
                if "std" in action_stats and hasattr(buffer, "std"):
                    buffer.std.data = action_stats["std"]
                print(f"  Updated normalize_targets for action")
            
            # Update unnormalize_outputs  
            if hasattr(policy_pretrained.unnormalize_outputs, "buffer_action"):
                buffer = policy_pretrained.unnormalize_outputs.buffer_action
                if "mean" in action_stats and hasattr(buffer, "mean"):
                    buffer.mean.data = action_stats["mean"]
                if "std" in action_stats and hasattr(buffer, "std"):
                    buffer.std.data = action_stats["std"]
                print(f"  Updated unnormalize_outputs for action")
        
        print("✅ Successfully updated pretrained policy statistics!")
        
        # Verify the updates worked
        print("\nVerifying updated normalization buffers...")
        for name, buffer in policy_pretrained.normalize_inputs.named_parameters():
            if "mean" in name or "std" in name:
                is_finite = torch.isfinite(buffer).all()
                print(f"  {name}: finite={is_finite}")
        
        return policy_pretrained
        
    except Exception as e:
        print(f"❌ Failed to update pretrained policy: {e}")
        return None


def create_test_observation(policy):
    """Create a test observation to verify the policy works."""
    print("\n=== Testing policy with sample observation ===")
    
    try:
        # Create a dummy observation matching your robot setup
        batch = {
            "observation.state": torch.randn(1, 6),  # 6-dim state for so100_follower
            "observation.image": torch.randn(1, 3, 256, 256),  # Top camera
            "observation.image2": torch.randn(1, 3, 256, 256),  # Side camera  
            "language_instruction": "Put the black box in the bowl"
        }
        
        print("Running policy inference...")
        with torch.no_grad():
            action = policy.select_action(batch)
        
        print(f"✅ Policy inference successful! Action shape: {action.shape}")
        print(f"Action values: {action}")
        
    except Exception as e:
        print(f"❌ Policy inference failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLA with computed dataset statistics")
    parser.add_argument("--stats_path", type=str, default="dataset_stats.pt",
                       help="Path to the computed dataset statistics file")
    parser.add_argument("--model_path", type=str, default="lerobot/smolvla_base",
                       help="Path to the SmolVLA model")
    parser.add_argument("--test_inference", action="store_true",
                       help="Test policy inference with dummy observation")
    
    args = parser.parse_args()
    
    stats_path = Path(args.stats_path)
    
    if not stats_path.exists():
        print(f"❌ Statistics file not found: {stats_path}")
        print("First run: python compute_dataset_stats.py")
        return
    
    # Load and test the policy
    policy = load_and_test_policy(stats_path, args.model_path)
    
    if policy and args.test_inference:
        create_test_observation(policy)
    
    print(f"\n=== Summary ===")
    print("The dataset statistics have been successfully computed and applied to SmolVLA!")
    print("You can now use this approach in your recording script:")
    print()
    print("1. Compute stats: python compute_dataset_stats.py")
    print("2. Use in recording with updated policy initialization")


if __name__ == "__main__":
    main()