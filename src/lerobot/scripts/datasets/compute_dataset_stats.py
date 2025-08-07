#!/usr/bin/env python3
"""
Script to compute dataset-level normalization statistics from episode statistics.

This script reads episode statistics from episodes_stats.jsonl and computes
aggregate dataset statistics that can be passed to SmolVLA policy for proper normalization.

Usage:
    python compute_dataset_stats.py --dataset_path local_dataset --output_path dataset_stats.pt

The output will be a PyTorch file containing the aggregated statistics that can be
loaded and passed to the SmolVLA policy.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List


def load_episodes_stats(episodes_stats_path: Path) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """Load episode statistics from jsonl file."""
    episodes_stats = {}
    
    with open(episodes_stats_path, 'r') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episode_index = episode_data["episode_index"]
            stats = episode_data["stats"]
            
            # Convert lists to numpy arrays
            for key, stat_dict in stats.items():
                for stat_name, value in stat_dict.items():
                    stats[key][stat_name] = np.array(value)
            
            episodes_stats[episode_index] = stats
    
    return episodes_stats


def aggregate_feature_stats(stats_ft_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Aggregates stats for a single feature across episodes."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    stds = np.stack([s["std"] for s in stats_ft_list])
    variances = stds ** 2
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_dataset_stats(episodes_stats: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate statistics across all episodes for each feature."""
    # Get all unique keys across all episodes
    all_keys = set()
    for episode_stats in episodes_stats.values():
        all_keys.update(episode_stats.keys())
    
    aggregated_stats = {}
    
    for key in all_keys:
        # Skip non-numeric features (like images for now, as they don't need normalization in SmolVLA)
        if key.startswith("observation.images"):
            continue
            
        # Collect statistics for this key from all episodes that have it
        stats_list = []
        for episode_stats in episodes_stats.values():
            if key in episode_stats:
                stats_list.append(episode_stats[key])
        
        if stats_list:
            aggregated_stats[key] = aggregate_feature_stats(stats_list)
    
    return aggregated_stats


def convert_to_torch_stats(aggregated_stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert numpy arrays to torch tensors for use with policies."""
    torch_stats = {}
    
    for key, stats in aggregated_stats.items():
        torch_stats[key] = {}
        for stat_name, value in stats.items():
            if stat_name in ["mean", "std", "min", "max"]:  # Only convert stats used for normalization
                torch_stats[key][stat_name] = torch.from_numpy(value).float()
    
    return torch_stats


def print_stats_summary(stats: Dict[str, Dict[str, np.ndarray]]):
    """Print a summary of the computed statistics."""
    print("\n=== Dataset Statistics Summary ===")
    for key, stat_dict in stats.items():
        print(f"\n{key}:")
        for stat_name, value in stat_dict.items():
            if stat_name == "count":
                print(f"  {stat_name}: {value}")
            else:
                print(f"  {stat_name}: {value} (shape: {value.shape})")


def create_smolvla_example():
    """Create an example of how to use the computed statistics with SmolVLA."""
    example_code = '''
# Example: How to use the computed statistics with SmolVLA

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

# Load the computed statistics
dataset_stats = torch.load("dataset_stats.pt")

# Create SmolVLA config (or load from existing model)
config = SmolVLAConfig()

# Initialize the policy with dataset statistics
policy = SmolVLAPolicy(config, dataset_stats=dataset_stats)

# Now you can use the policy for inference without normalization errors!
# The policy will have proper mean/std values instead of infinity values
'''
    
    with open("smolvla_usage_example.py", "w") as f:
        f.write(example_code)
    
    print(f"\nExample usage code written to: smolvla_usage_example.py")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics from episode statistics")
    parser.add_argument("--dataset_path", type=str, default="local_dataset",
                       help="Path to the dataset directory containing meta/episodes_stats.jsonl")
    parser.add_argument("--output_path", type=str, default="dataset_stats.pt",
                       help="Path to save the computed statistics")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed statistics summary")
    
    args = parser.parse_args()
    
    # Paths
    dataset_path = Path(args.dataset_path)
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    output_path = Path(args.output_path)
    
    if not episodes_stats_path.exists():
        raise FileNotFoundError(f"Episode statistics file not found: {episodes_stats_path}")
    
    print(f"Loading episode statistics from: {episodes_stats_path}")
    episodes_stats = load_episodes_stats(episodes_stats_path)
    print(f"Loaded statistics for {len(episodes_stats)} episodes")
    
    print("Aggregating dataset statistics...")
    aggregated_stats = aggregate_dataset_stats(episodes_stats)
    
    if args.verbose:
        print_stats_summary(aggregated_stats)
    
    print("Converting to PyTorch tensors...")
    torch_stats = convert_to_torch_stats(aggregated_stats)
    
    print(f"Saving dataset statistics to: {output_path}")
    torch.save(torch_stats, output_path)
    
    print(f"\n=== SUCCESS ===")
    print(f"Dataset statistics computed and saved to: {output_path}")
    print(f"Features with statistics: {list(torch_stats.keys())}")
    
    # Create usage example
    create_smolvla_example()
    
    print(f"\nYou can now use these statistics with SmolVLA to fix the normalization error!")


if __name__ == "__main__":
    main()