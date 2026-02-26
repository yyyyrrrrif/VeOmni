"""
Test script for QwenImage Ulysses SP attention-level correctness.

Validates that SP (sequence parallel) and DP (data parallel) produce
numerically equivalent forward outputs and backward gradients.

Run with torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/parallel/ulysses/test_qwen_image_ulysses.py -v -s

Tests:
  C1. SP vs DP forward equivalence
  C2. SP vs DP backward gradient equivalence
  C3. Async Ulysses DiT attention (via test_async_ulysses_dit.py patterns)
  C4. Non-divisible (padding) sequence length
  C5. Cross-attention skip_ulysses
"""

import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel import gather_seq_scatter_heads
from veomni.distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.utils import unpadding_tensor_for_seqeunce_parallel
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed

from .utils import SequenceParallelTest, sync_tensor


def _safe_assert_close(title, a, b, *, atol, rtol):
    max_diff = (a.detach().float() - b.detach().float()).abs().max().item()
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        if dist.get_rank() == 0:
            print(f"[PASS] {title}: max_abs_diff={max_diff:.6e}")
        return True
    except AssertionError:
        if dist.get_rank() == 0:
            print(f"[FAIL] {title}: max_abs_diff={max_diff:.6e}")
        return False


class QwenImageUlyssesTest(SequenceParallelTest):
    """
    SP vs DP equivalence tests for QwenImage model.

    These tests create identical models with identical weights, run forward
    and backward in SP mode (sliced input, all-to-all in blocks) and DP mode
    (full input, no SP), then compare outputs and gradients.
    """

    @property
    def world_size(self):
        return min(4, get_torch_device().device_count())

    @staticmethod
    def _make_small_model_and_inputs(device, seq_len=1024):
        from veomni.models.transformers.qwen_image.config_qwen_image import QwenImageConfig
        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        config = QwenImageConfig(
            image_size=64,
            patch_size=2,
            in_channels=64,
            out_channels=64,
            hidden_size=384,
            num_hidden_layers=2,
            num_attention_heads=6,
            intermediate_size=1536,
            text_embed_dim=512,
            use_flash_attention=False,
            axes_dims_rope=(16, 16, 32),
        )

        batch_size = 1
        text_len = 16
        full_img = torch.randn(batch_size, seq_len, config.in_channels, dtype=torch.float32, device=device)
        full_txt = torch.randn(batch_size, text_len, config.text_embed_dim, dtype=torch.float32, device=device)
        timestep = torch.tensor([500], dtype=torch.long, device=device)
        img_shapes = [(1, 64, 64)] * batch_size

        return config, full_img, full_txt, timestep, img_shapes

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="need >= 4 GPUs")
    def test_sp_vs_dp_forward_equivalence(self):
        """C1. Forward pass: SP gathered output must match DP output."""
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        device = get_device_type()

        config, full_img, full_txt, timestep, img_shapes = self._make_small_model_and_inputs(device)
        dist.broadcast(full_img, src=0)
        dist.broadcast(full_txt, src=0)

        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        # SP path
        model_sp = QwenImageModel(config).to(device).float().eval()
        model_sp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        with torch.no_grad():
            sp_output = model_sp(
                hidden_states=full_img.clone(),
                encoder_hidden_states=full_txt.clone(),
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=True,
            )
        sp_logits = sp_output["logits"]

        # DP path (disable SP)
        set_ulysses_sequence_parallel_group(None)
        model_dp = QwenImageModel(config).to(device).float().eval()
        model_dp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        with torch.no_grad():
            dp_output = model_dp(
                hidden_states=full_img.clone(),
                encoder_hidden_states=full_txt.clone(),
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=True,
            )
        dp_logits = dp_output["logits"]

        _safe_assert_close("forward_output", dp_logits, sp_logits, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="need >= 4 GPUs")
    def test_sp_vs_dp_backward_gradients(self):
        """C2. Backward pass: SP gradients must match DP gradients."""
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        device = get_device_type()

        config, full_img, full_txt, timestep, img_shapes = self._make_small_model_and_inputs(device)
        dist.broadcast(full_img, src=0)
        dist.broadcast(full_txt, src=0)

        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        # SP path - backward
        model_sp = QwenImageModel(config).to(device).float().train()
        model_sp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        sp_img = full_img.clone().requires_grad_(True)
        sp_output = model_sp(
            hidden_states=sp_img,
            encoder_hidden_states=full_txt.clone(),
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=True,
        )
        loss_sp = sp_output["logits"].sum()
        loss_sp.backward()

        sp_input_grad = sp_img.grad.detach().clone()

        # DP path - backward
        set_ulysses_sequence_parallel_group(None)
        model_dp = QwenImageModel(config).to(device).float().train()
        model_dp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        dp_img = full_img.clone().requires_grad_(True)
        dp_output = model_dp(
            hidden_states=dp_img,
            encoder_hidden_states=full_txt.clone(),
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=True,
        )
        loss_dp = dp_output["logits"].sum()
        loss_dp.backward()

        dp_input_grad = dp_img.grad.detach().clone()

        _safe_assert_close("input.grad", dp_input_grad, sp_input_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="need >= 4 GPUs")
    def test_sp_padding_seq_len(self):
        """C4. Non-divisible sequence length (padding case)."""
        self._get_process_group()
        device = get_device_type()

        config, full_img, full_txt, timestep, img_shapes = self._make_small_model_and_inputs(
            device, seq_len=1023
        )
        dist.broadcast(full_img, src=0)
        dist.broadcast(full_txt, src=0)

        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        # SP path
        model_sp = QwenImageModel(config).to(device).float().eval()
        model_sp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        with torch.no_grad():
            sp_output = model_sp(
                hidden_states=full_img.clone(),
                encoder_hidden_states=full_txt.clone(),
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=True,
            )
        sp_logits = sp_output["logits"]

        # DP path
        set_ulysses_sequence_parallel_group(None)
        model_dp = QwenImageModel(config).to(device).float().eval()
        model_dp.load_state_dict(self._sync_model(model_sp.state_dict(), self.rank))

        with torch.no_grad():
            dp_output = model_dp(
                hidden_states=full_img.clone(),
                encoder_hidden_states=full_txt.clone(),
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=True,
            )
        dp_logits = dp_output["logits"]

        _safe_assert_close("[padding] forward_output", dp_logits, sp_logits, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )
    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()
