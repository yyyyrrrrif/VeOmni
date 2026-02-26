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

"""
QwenImage model wrapper for VeOmni with sequence parallel support.

Wraps diffusers' QwenImageTransformer2DModel, adding VeOmni-specific features
like FSDP, Ulysses SP (attention-level), and Context Parallel support.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.modeling_utils import PreTrainedModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
)
from veomni.models.transformers.qwen_image.config_qwen_image import QwenImageConfig
from veomni.utils import logging
from veomni.utils.import_utils import is_torch_npu_available

logger = logging.get_logger(__name__)

from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)


def _patch_block_forward_for_ulysses(block: "QwenImageTransformerBlock"):
    """
    Patch the block's forward to gather full sequences before attention
    and scatter back after attention for Ulysses SP correctness.
    """
    original_forward = block.forward

    def patched_forward(
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask=None,
        temb=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        parallel_state = get_parallel_state()
        if not parallel_state.ulysses_enabled:
            return original_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = gather_outputs(hidden_states, gather_dim=1)
        encoder_hidden_states = gather_outputs(encoder_hidden_states, gather_dim=1)

        enc_out, img_out = original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        img_out = slice_input_tensor_scale_grad(img_out, dim=1)
        enc_out = slice_input_tensor_scale_grad(enc_out, dim=1)

        return enc_out, img_out

    block.forward = patched_forward


class QwenImageModelOutput(dict):
    """Custom output providing both 'logits' and 'sample' keys."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'") from exc

    def __setattr__(self, key, value):
        self[key] = value
        self.__dict__[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'") from exc
        del self.__dict__[key]


class QwenImageModel(PreTrainedModel, QwenImageTransformer2DModel):
    """
    VeOmni wrapper for QwenImage Transformer2DModel with multi-parallel support.
    """

    config_class = QwenImageConfig
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]

    def __init__(self, config: QwenImageConfig, **kwargs):
        object.__setattr__(self, "_config", config)
        PreTrainedModel.__init__(self, config, **kwargs)
        diffusers_config = self._convert_config(config)
        QwenImageTransformer2DModel.__init__(self, **diffusers_config)

        if get_parallel_state().ulysses_enabled:
            self._apply_ulysses_patches()

    @property
    def config(self) -> QwenImageConfig:
        """VeOmni config (HF-style). Overrides diffusers read-only config for PreTrainedModel compatibility."""
        return self._config

    @config.setter
    def config(self, value: QwenImageConfig) -> None:
        """Allow PreTrainedModel.__init__ to set config."""
        object.__setattr__(self, "_config", value)

    def _apply_ulysses_patches(self):
        """Apply Ulysses SP patches to all transformer blocks."""
        for block in self.transformer_blocks:
            _patch_block_forward_for_ulysses(block)
        logger.info_rank0("Applied Ulysses SP block-level patches to QwenImageModel.")

    def _convert_config(self, config: QwenImageConfig) -> Dict[str, Any]:
        """Convert VeOmni QwenImageConfig to diffusers format."""
        return {
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels or config.in_channels,
            "num_layers": config.num_hidden_layers,
            "attention_head_dim": config.head_dim,
            "num_attention_heads": config.num_attention_heads,
            "joint_attention_dim": config.text_embed_dim,
            "axes_dims_rope": config.axes_dims_rope,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        parallel_state = get_parallel_state()

        if img_shapes is None:
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            side = int(math.sqrt(seq_len))
            img_shapes = [(1, side, side)] * batch_size

        if parallel_state.ulysses_enabled:
            hidden_states = slice_input_tensor_scale_grad(hidden_states, dim=1)
            encoder_hidden_states = slice_input_tensor_scale_grad(encoder_hidden_states, dim=1)
        elif parallel_state.cp_enabled:
            hidden_states = slice_input_tensor(hidden_states, dim=1, padding=True, padding_value=0)
            encoder_hidden_states = slice_input_tensor(
                encoder_hidden_states, dim=1, padding=True, padding_value=0
            )

        output = QwenImageTransformer2DModel.forward(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

        if return_dict:
            sample = output.sample
        else:
            sample = output[0]

        if parallel_state.ulysses_enabled:
            sample = gather_outputs(sample, gather_dim=1)
        elif parallel_state.cp_enabled:
            sample = gather_outputs(sample, gather_dim=1)

        if return_dict:
            return QwenImageModelOutput(logits=sample, sample=sample)
        else:
            return (sample,) if len(output) == 1 else (sample,) + output[1:]


if is_torch_npu_available():
    from .npu_patch import apply_qwen_image_npu_patch

    apply_qwen_image_npu_patch()
