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
NPU-specific optimizations for QwenImage model.

Patches RMSNorm and RoPE to use NPU-accelerated implementations,
following the same pattern as veomni/models/transformers/wan/npu_patch.py.
"""

from veomni.utils import logging


logger = logging.get_logger(__name__)


def apply_qwen_image_npu_patch():
    """Apply NPU-specific patches for QwenImage model."""
    try:
        import torch_npu
    except ImportError:
        logger.warning("torch_npu not available, skipping QwenImage NPU patches.")
        return

    try:
        from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

        _original_rmsnorm_forward = DiffusersRMSNorm.forward

        def _npu_rmsnorm_forward(self, x):
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

        DiffusersRMSNorm.forward = _npu_rmsnorm_forward
        logger.info_rank0("Applied NPU RMSNorm patch for QwenImage model.")
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to apply NPU RMSNorm patch: %s", e)

    try:
        import torch
        from diffusers.models.transformers import transformer_qwenimage

        _original_rope = transformer_qwenimage.apply_rotary_emb_qwen

        def _npu_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
            if use_real and isinstance(freqs_cis, tuple) and len(freqs_cis) == 2:
                cos, sin = freqs_cis
                cos = cos.unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
                sin = sin.unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
                x_float = x.to(torch.float32)
                x_out = torch_npu.npu_rotary_mul(x_float, cos, sin, rotary_mode="interleave")
                return x_out.to(x.dtype)
            return _original_rope(x, freqs_cis, use_real, use_real_unbind_dim)

        transformer_qwenimage.apply_rotary_emb_qwen = _npu_rotary_emb
        logger.info_rank0("Applied NPU RoPE patch for QwenImage model.")
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to apply NPU RoPE patch: %s", e)
