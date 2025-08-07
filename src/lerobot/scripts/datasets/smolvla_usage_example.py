
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
