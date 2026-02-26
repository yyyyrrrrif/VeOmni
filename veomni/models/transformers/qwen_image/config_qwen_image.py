# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import os
from typing import List, Optional, Tuple

from transformers import PretrainedConfig


class QwenImageConfig(PretrainedConfig):
    model_type = "qwen_image"

    def __init__(
        self,
        # Image generation specific
        image_size: int = 512,
        patch_size: int = 2,  # Changed default to match diffusers
        in_channels: int = 64,  # Changed default to match diffusers
        out_channels: Optional[int] = None,  # Changed to Optional
        
        # Transformer architecture
        hidden_size: int = 1152,
        num_hidden_layers: int = 60,  # Changed default to match diffusers
        num_attention_heads: int = 24,  # Changed default to match diffusers
        intermediate_size: int = 4304,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        
        # Text conditioning
        text_embed_dim: int = 3584,  # Changed default to match diffusers (joint_attention_dim)
        text_max_length: int = 512,
        
        # Time embedding (for diffusion)
        time_embed_dim: Optional[int] = None,
        
        # Attention
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
        attention_head_dim: int = 128,
        
        # Training
        gradient_checkpointing: bool = False,
        
        # Qwen-specific parameters (aligned with diffusers)
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        
        **kwargs,
    ):
        # Image generation parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Transformer parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        
        # Text conditioning
        self.text_embed_dim = text_embed_dim
        self.text_max_length = text_max_length
        
        # Time embedding (for diffusion models)
        self.time_embed_dim = time_embed_dim or hidden_size
        
        # Attention
        self.attention_dropout = attention_dropout
        self.use_flash_attention = use_flash_attention
        self.attention_head_dim = attention_head_dim
        
        # Training
        self.gradient_checkpointing = gradient_checkpointing
        
        # Qwen-specific parameters
        self.axes_dims_rope = axes_dims_rope
        self.zero_cond_t = zero_cond_t
        self.use_additional_t_cond = use_additional_t_cond
        self.use_layer3d_rope = use_layer3d_rope
        
        # Compute derived parameters
        self.num_patches = (image_size // patch_size) ** 2
        self.head_dim = attention_head_dim
        
        super().__init__(**kwargs)

    def save_pretrained(self, path):
        config = {
            "_class_name": "QwenImageModel",
            "model_type": self.model_type,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "text_embed_dim": self.text_embed_dim,
            "text_max_length": self.text_max_length,
            "time_embed_dim": self.time_embed_dim,
            "attention_dropout": self.attention_dropout,
            "use_flash_attention": self.use_flash_attention,
            "attention_head_dim": self.attention_head_dim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "axes_dims_rope": list(self.axes_dims_rope),
            "zero_cond_t": self.zero_cond_t,
            "use_additional_t_cond": self.use_additional_t_cond,
            "use_layer3d_rope": self.use_layer3d_rope,
        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
