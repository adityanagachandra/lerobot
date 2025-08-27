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
"""NextAction Policy

Predicts next action from: RGB frames + touch maps + past actions.
Uses separate CNN encoders for vision and touch, combined with past actions,
fed through a GRU to predict the next action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.nextaction.configuration_nextaction import NextActionConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


class TinyCNN(nn.Module):
    """Lightweight CNN encoder for RGB images."""
    
    def __init__(self, out_dim: int = 256, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, 2, 2), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c*2, 3, 2, 1), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*4, 3, 2, 1), nn.BatchNorm2d(c*4), nn.ReLU(inplace=True),
            nn.Conv2d(c*4, c*4, 3, 2, 1), nn.BatchNorm2d(c*4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(c*4, out_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, C, H, W] tensor of images
        Returns:
            [B, T, out_dim] tensor of encoded features
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.net(x).view(B*T, -1)
        x = self.proj(x).view(B, T, -1)
        return x


class TouchCNN(nn.Module):
    """Lightweight CNN encoder for touch maps."""
    
    def __init__(self, out_dim: int = 128, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),   # 20->10
            nn.Conv2d(c, c*2, 3, 2, 1), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True), # 10->5
            nn.Conv2d(c*2, c*4, 3, 2, 1), nn.BatchNorm2d(c*4), nn.ReLU(inplace=True),# 5->3
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(c*4, out_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, 1, H, W] tensor of touch maps
        Returns:
            [B, T, out_dim] tensor of encoded features
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.net(x).view(B*T, -1)
        x = self.proj(x).view(B, T, -1)
        return x


class NextActionModel(nn.Module):
    """
    Core NextAction model that combines vision, touch, and action history
    to predict the next action using a GRU.
    """
    
    def __init__(self, config: NextActionConfig):
        super().__init__()
        self.config = config
        
        # Get action dimension from config
        action_features = [f for f in config.output_features if f.name == "action"]
        if not action_features:
            raise ValueError("No action feature found in output_features")
        self.action_dim = action_features[0].shape[0]
        
        # Encoders
        self.vision = TinyCNN(
            out_dim=config.vision_backbone_dim,
            base_channels=config.vision_cnn_channels
        )
        self.touch = TouchCNN(
            out_dim=config.touch_backbone_dim,
            base_channels=config.touch_cnn_channels
        )
        
        # Input projection
        input_dim = config.vision_backbone_dim + config.touch_backbone_dim + self.action_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # GRU
        self.gru = nn.GRU(
            config.hidden_dim, 
            config.hidden_dim, 
            num_layers=config.gru_num_layers, 
            batch_first=True,
            dropout=config.dropout if config.gru_num_layers > 1 else 0.0
        )
        
        # Output head
        head_layers = [
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, self.action_dim)
        ]
        if config.tanh_output:
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)
        
    def forward(self, imgs: Tensor, touch: Tensor, acts: Tensor) -> Tensor:
        """
        Args:
            imgs: [B, T, 3, H, W] RGB images
            touch: [B, T, 1, touch_H, touch_W] touch maps
            acts: [B, T, action_dim] past actions
        Returns:
            [B, action_dim] next action prediction
        """
        # Encode inputs
        v = self.vision(imgs)   # [B, T, vision_dim]
        s = self.touch(touch)   # [B, T, touch_dim]
        
        # Combine all modalities
        x = torch.cat([v, s, acts], dim=-1)  # [B, T, vision_dim+touch_dim+action_dim]
        x = self.input_proj(x)
        
        # Process through GRU
        out, _ = self.gru(x)
        h_last = out[:, -1]  # Take last hidden state
        
        # Predict next action
        return self.head(h_last)  # [B, action_dim]


class NextActionPolicy(PreTrainedPolicy):
    """
    NextAction policy for predicting next actions from visual observations,
    touch sensory data, and action history.
    """
    
    config_class = NextActionConfig
    name = "nextaction"
    
    def __init__(
        self,
        config: NextActionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # Set up normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        
        # Initialize model
        self.model = NextActionModel(config)
        
        # Image preprocessing
        self.image_transforms = T.Compose([
            T.Resize(config.img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.img_size),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.reset()
    
    def reset(self):
        """Called at the beginning of each episode."""
        pass
    
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select an action given environment observations.
        
        Args:
            batch: Dictionary containing:
                - Image observations: "observation.images.{camera_name}" -> [B, T, C, H, W]
                - Touch observations: "observation.touch" -> [B, T, touch_H, touch_W]  
                - Past actions: "action" -> [B, T, action_dim]
        
        Returns:
            Predicted actions [B, action_dim]
        """
        batch = self.normalize_inputs(batch)
        batch = self._format_batch_for_model(batch)
        
        with torch.no_grad():
            # Extract modalities
            imgs = self._extract_images(batch)  # [B, T, 3, H, W]
            touch = self._extract_touch(batch)  # [B, T, H, W]
            if touch.ndim == 3:  # [B, H, W] -> [B, 1, H, W] 
                touch = touch.unsqueeze(1)  # Add time dimension
            touch = touch.unsqueeze(-3)  # [B, T, 1, H, W]
            acts = batch["action"]  # [B, T, action_dim]
            
            # Apply image transforms if needed
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            
            # Predict next action
            action = self.model(imgs, touch, acts)
            
        return self.unnormalize_outputs({"action": action})["action"]
    
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Forward pass for training.
        
        Args:
            batch: Dictionary containing normalized inputs and targets
            
        Returns:
            Dictionary with "action" key containing predicted actions
        """
        batch = self._format_batch_for_model(batch)
        
        # Extract modalities
        imgs = self._extract_images(batch)  # [B, T, 3, H, W]
        touch = self._extract_touch(batch)  # [B, T, H, W]
        if touch.ndim == 3:  # [B, H, W] -> [B, 1, H, W] 
            touch = touch.unsqueeze(1)  # Add time dimension
        touch = touch.unsqueeze(-3)  # [B, T, 1, H, W]
        acts = batch["action"]  # [B, T, action_dim]
        
        # Apply image transforms if needed
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
            
        # Predict next action
        action = self.model(imgs, touch, acts)
        
        return {"action": action}
    
    def _extract_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract and concatenate images from all cameras."""
        image_keys = [key for key in batch.keys() if key.startswith("observation.images.")]
        if not image_keys:
            raise ValueError("No image observations found in batch")
        
        # For now, just use the first camera. 
        # TODO: Support multiple cameras by concatenating features
        imgs = batch[image_keys[0]]  # [B, T, C, H, W]
        return imgs
    
    def _extract_touch(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract and reconstruct touch data from individual scalars."""
        # Check if we have the unified touch feature
        if "observation.touch" in batch:
            return batch["observation.touch"]
        
        # Otherwise, reconstruct from individual scalars
        touch_keys = [key for key in batch.keys() if key.startswith("observation.touch.")]
        if not touch_keys:
            raise ValueError("No touch observations found in batch")
        
        # Sort keys by index to ensure correct ordering
        touch_keys.sort(key=lambda x: int(x.split('.')[-1]))
        
        # Stack the scalars and reshape to [B, T, H, W] or [B, H, W]
        touch_scalars = torch.stack([batch[key] for key in touch_keys], dim=-1)
        
        if touch_scalars.ndim == 2:  # [B, 400]
            B = touch_scalars.shape[0]
            touch_map = touch_scalars.view(B, 20, 20)
        elif touch_scalars.ndim == 3:  # [B, T, 400]
            B, T = touch_scalars.shape[:2]
            touch_map = touch_scalars.view(B, T, 20, 20)
        else:
            raise ValueError(f"Unexpected touch_scalars shape: {touch_scalars.shape}")
        
        return touch_map
    
    def _format_batch_for_model(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Ensure batch has the right format for the model."""
        formatted_batch = {}
        
        for key, value in batch.items():
            if value.ndim == 2:
                # Add time dimension if missing [B, D] -> [B, 1, D]
                formatted_batch[key] = value.unsqueeze(1)
            else:
                formatted_batch[key] = value
                
        return formatted_batch
    
    def get_optim_params(self) -> list[dict]:
        """Get parameters for optimizer."""
        return [{"params": self.parameters()}]
    
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk (required by PreTrainedPolicy interface)."""
        # For NextAction, we predict single actions, so just call select_action
        return self.select_action(batch)
