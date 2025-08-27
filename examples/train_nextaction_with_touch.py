#!/usr/bin/env python

"""Example script to train NextAction policy with touch data using LeRobot framework."""

import argparse
import os
from pathlib import Path

from lerobot.scripts.train import train


def main():
    parser = argparse.ArgumentParser(description="Train NextAction policy with touch data")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to converted NextAction dataset")
    parser.add_argument("--config", type=str, 
                        default="src/lerobot/configs/train_nextaction_touch.yaml",
                        help="Path to training config file")
    parser.add_argument("--output-dir", type=str, default="outputs/nextaction_touch",
                        help="Output directory for checkpoints and logs")
    
    args = parser.parse_args()
    
    # Prepare arguments for LeRobot training script
    train_args = [
        f"config_path={args.config}",
        f"dataset.root={args.dataset_dir}",
        f"output_dir={args.output_dir}",
        # Override some settings for quick testing
        "steps=100",  # Reduced for testing
        "log_freq=10",
        "save_freq=50",
        "eval_freq=50",
    ]
    
    print(f"Training NextAction with touch data")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Note: This is a simplified example. For full training, you would:
    # 1. Remove the steps=100 override for full training
    # 2. Set up proper evaluation datasets
    # 3. Configure wandb logging if desired
    # 4. Adjust hyperparameters based on your specific task
    
    print("To train with LeRobot framework, run:")
    print(f"python src/lerobot/scripts/train.py {' '.join(train_args)}")
    print()
    print("Or for full training with default settings:")
    print(f"python src/lerobot/scripts/train.py config_path={args.config} dataset.root={args.dataset_dir}")


if __name__ == "__main__":
    main()
