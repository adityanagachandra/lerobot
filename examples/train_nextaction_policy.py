#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script to train a NextAction policy using LeRobot's infrastructure.

This script demonstrates how to train the NextAction model that predicts next actions from:
- RGB image sequences
- Touch map sequences  
- Past action sequences

Usage:
    python examples/train_nextaction_policy.py \
        --config-path path/to/config.yaml \
        --output-dir outputs/nextaction_training

Example config YAML:
    ```yaml
    seed: 42
    batch_size: 64
    steps: 20000
    log_freq: 100
    save_freq: 1000
    eval_freq: 2000
    
    dataset:
      repo_id: "your_dataset"
      root: "data"
      
    policy:
      type: "nextaction"
      device: "cuda"
      n_obs_steps: 5
      vision_backbone_dim: 256
      touch_backbone_dim: 128
      hidden_dim: 512
      gru_num_layers: 2
      img_size: 128
      touch_height: 20
      touch_width: 20
      tanh_output: false
      use_amp: true
      optimizer_lr: 1e-3
      optimizer_weight_decay: 1e-5
      
    optimizer:
      grad_clip_norm: 1.0
    ```
"""

import logging
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.train import train
from lerobot.utils.utils import init_logging


def main():
    """Main training function for NextAction policy."""
    init_logging()
    
    # Set up configuration for NextAction training
    @parser.wrap()
    def train_nextaction(cfg: TrainPipelineConfig):
        """Train NextAction policy with LeRobot infrastructure."""
        
        # Validate that we're using the nextaction policy
        if cfg.policy.type != "nextaction":
            raise ValueError(f"This script is specifically for NextAction policy, got {cfg.policy.type}")
        
        # Log training info specific to NextAction
        logging.info("=" * 60)
        logging.info("NEXTACTION POLICY TRAINING")
        logging.info("=" * 60)
        logging.info(f"Policy: {cfg.policy.type}")
        logging.info(f"Observation steps: {cfg.policy.n_obs_steps}")
        logging.info(f"Vision backbone dim: {cfg.policy.vision_backbone_dim}")
        logging.info(f"Touch backbone dim: {cfg.policy.touch_backbone_dim}")
        logging.info(f"Hidden dim: {cfg.policy.hidden_dim}")
        logging.info(f"Image size: {cfg.policy.img_size}")
        logging.info(f"Touch size: {cfg.policy.touch_height}x{cfg.policy.touch_width}")
        logging.info(f"Use AMP: {cfg.policy.use_amp}")
        logging.info("=" * 60)
        
        # Call the main training function
        train(cfg)
    
    # Run training
    train_nextaction()


if __name__ == "__main__":
    main()
