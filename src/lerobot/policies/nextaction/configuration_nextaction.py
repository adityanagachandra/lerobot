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
"""NextAction Policy Configuration"""

from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.policies.pretrained import PretrainedConfig


@dataclass
class NextActionConfig(PretrainedConfig):
    """Configuration class for NextAction policy.
    
    NextAction predicts the next action from RGB frames, touch maps, and past actions.
    It uses separate CNN encoders for vision and touch modalities, combines them with
    past actions, and feeds through a GRU to predict the next action.
    
    Args:
        n_obs_steps: Number of observation steps (frames) to use as input.
        vision_backbone_dim: Output dimension of the vision CNN encoder.
        touch_backbone_dim: Output dimension of the touch CNN encoder.
        hidden_dim: Hidden dimension of the GRU.
        gru_num_layers: Number of layers in the GRU.
        vision_cnn_channels: Base number of channels for vision CNN.
        touch_cnn_channels: Base number of channels for touch CNN.
        img_size: Size of input images (assumed square).
        touch_height: Height of touch sensor data.
        touch_width: Width of touch sensor data.
        cameras: List of camera names to use (if multiple cameras).
        tanh_output: Whether to apply tanh to the output actions.
        dropout: Dropout rate for regularization.
        use_amp: Whether to use automatic mixed precision during training.
        optimizer_lr: Learning rate for the optimizer.
        optimizer_weight_decay: Weight decay for the optimizer.
        optimizer_grad_clip_norm: Gradient clipping norm (if > 0).
        scheduler_name: Name of the learning rate scheduler.
        scheduler_lr_lambda_fn_str: String representation of lr lambda function.
        scheduler_lr_warmup_steps: Number of warmup steps for scheduler.
    """
    
    policy_type: str = "nextaction"
    
    # Observation configuration
    n_obs_steps: int = 5
    cameras: List[str] = field(default_factory=lambda: ["observation.images.camera"])
    
    # Architecture parameters
    vision_backbone_dim: int = 256
    touch_backbone_dim: int = 128
    hidden_dim: int = 512
    gru_num_layers: int = 2
    
    # CNN architecture
    vision_cnn_channels: int = 32
    touch_cnn_channels: int = 16
    
    # Input dimensions
    img_size: int = 128
    touch_height: int = 20
    touch_width: int = 20
    
    # Output configuration
    tanh_output: bool = False
    
    # Training configuration
    dropout: float = 0.1
    use_amp: bool = False
    
    # Optimizer configuration
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 0.0
    
    # Scheduler configuration
    scheduler_name: str = "cosine"
    scheduler_lr_lambda_fn_str: Optional[str] = None
    scheduler_lr_warmup_steps: int = 100
    
    # Dataset configuration
    dataset_stats: Optional[dict] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        
        # Validate dimensions
        if self.vision_backbone_dim <= 0:
            raise ValueError("vision_backbone_dim must be positive")
        if self.touch_backbone_dim <= 0:
            raise ValueError("touch_backbone_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.gru_num_layers <= 0:
            raise ValueError("gru_num_layers must be positive")
        
        # Validate CNN channels
        if self.vision_cnn_channels <= 0:
            raise ValueError("vision_cnn_channels must be positive")
        if self.touch_cnn_channels <= 0:
            raise ValueError("touch_cnn_channels must be positive")
        
        # Validate input dimensions
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.touch_height <= 0 or self.touch_width <= 0:
            raise ValueError("touch dimensions must be positive")
        
        # Validate training parameters
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.optimizer_lr <= 0:
            raise ValueError("optimizer_lr must be positive")
        if self.optimizer_weight_decay < 0:
            raise ValueError("optimizer_weight_decay must be non-negative")
        if self.optimizer_grad_clip_norm < 0:
            raise ValueError("optimizer_grad_clip_norm must be non-negative")
        if self.scheduler_lr_warmup_steps < 0:
            raise ValueError("scheduler_lr_warmup_steps must be non-negative")
    
    @property
    def input_shapes(self):
        """Return expected input shapes for the policy."""
        shapes = {}
        
        # Vision input shape: [B, T, C, H, W]
        for camera in self.cameras:
            shapes[camera] = [self.n_obs_steps, 3, self.img_size, self.img_size]
        
        # Touch input shape: [B, T, 1, H, W]
        shapes["observation.touch"] = [self.n_obs_steps, 1, self.touch_height, self.touch_width]
        
        # Action input shape: [B, T, action_dim]
        shapes["action"] = [self.n_obs_steps, None]  # action_dim determined at runtime
        
        return shapes
    
    @property
    def output_shapes(self):
        """Return expected output shapes for the policy."""
        # Output is next action: [B, action_dim]
        return {"action": [None]}  # action_dim determined at runtime
