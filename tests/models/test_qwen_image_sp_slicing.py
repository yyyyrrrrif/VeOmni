"""
Unit tests for QwenImage sequence parallel slicing correctness.

Tests B: input slice even/padding, freqs slice, output gather, roundtrip.
Uses monkeypatch to mock distributed environment (no GPU required).

Run with pytest:
    pytest tests/models/test_qwen_image_sp_slicing.py -v
"""

import pytest
import torch

from veomni.distributed.sequence_parallel import data as sp_data


class TestInputSliceEven:
    """B1. Uniform slicing of hidden_states."""

    @pytest.mark.parametrize("rank", [0, 1, 2, 3])
    def test_slice_4_ranks(self, monkeypatch, rank):
        """seq_len=1024, world_size=4 -> each rank gets 256."""
        x = torch.randn(2, 1024, 64)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: rank)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        sliced = sp_data.slice_input_tensor(x, dim=1, padding=False, group=None)
        assert sliced.shape == (2, 256, 64)
        assert torch.equal(sliced, x[:, rank * 256 : (rank + 1) * 256, :])
        assert sliced.is_contiguous()


class TestInputSlicePadding:
    """B2. Padding for non-divisible sequence lengths."""

    def test_padding_last_rank(self, monkeypatch):
        """seq_len=1023, world_size=4, last rank -> 255 real + 1 padding."""
        x = torch.randn(2, 1023, 64)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: 3)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        sliced = sp_data.slice_input_tensor(x, dim=1, padding=True, padding_value=0, group=None)

        chunk_size = -(-1023 // 4)  # ceil(1023/4) = 256
        assert sliced.shape == (2, chunk_size, 64)

        start = 3 * chunk_size
        real_len = min(chunk_size, 1023 - start)
        assert torch.equal(sliced[:, :real_len, :], x[:, start : start + real_len, :])
        if real_len < chunk_size:
            assert (sliced[:, real_len:, :] == 0).all()

    def test_padding_first_rank(self, monkeypatch):
        """First rank should get a full chunk."""
        x = torch.randn(1, 5, 8)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: 0)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        sliced = sp_data.slice_input_tensor(x, dim=1, padding=True, padding_value=9, group=None)
        chunk_size = -(-5 // 4)  # ceil(5/4) = 2
        assert sliced.shape == (1, chunk_size, 8)
        assert torch.equal(sliced[:, :2, :], x[:, :2, :])


class TestFreqsSlice:
    """B3. RoPE freqs slicing correctness."""

    @pytest.mark.parametrize("rank", [0, 1, 2, 3])
    def test_freqs_slice_dim0(self, monkeypatch, rank):
        """freqs [seq_len, 1, rope_dim] sliced along dim=0."""
        freqs = torch.randn(1024, 1, 128)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: rank)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        sliced = sp_data.slice_input_tensor(freqs, dim=0, padding=False, group=None)
        assert sliced.shape == (256, 1, 128)
        assert torch.equal(sliced, freqs[rank * 256 : (rank + 1) * 256, :, :])


class TestNoGroupReturnsUnchanged:
    """B4 helper: when no SP group, input is returned as-is."""

    def test_no_group(self):
        x = torch.randn(2, 8, 4)
        result = sp_data.slice_input_tensor(x, dim=1, padding=False, group=None)
        assert torch.equal(result, x)


class TestSliceConsistency:
    """B5. Slice-then-concatenate recovers the original tensor (single-process simulation)."""

    def test_roundtrip_simulation(self, monkeypatch):
        """Simulate 4-rank slice and manual concatenation."""
        x = torch.randn(2, 1024, 64)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        slices = []
        for rank in range(4):
            monkeypatch.setattr(sp_data.dist, "get_rank", lambda g, r=rank: r)
            sliced = sp_data.slice_input_tensor(x, dim=1, padding=False, group=None)
            slices.append(sliced)

        reconstructed = torch.cat(slices, dim=1)
        assert torch.equal(reconstructed, x)

    def test_roundtrip_with_padding(self, monkeypatch):
        """Simulate 4-rank slice with padding and verify real content is preserved."""
        x = torch.randn(2, 1023, 64)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        slices = []
        for rank in range(4):
            monkeypatch.setattr(sp_data.dist, "get_rank", lambda g, r=rank: r)
            sliced = sp_data.slice_input_tensor(x, dim=1, padding=True, padding_value=0, group=None)
            slices.append(sliced)

        reconstructed = torch.cat(slices, dim=1)
        # First 1023 elements should match
        assert torch.equal(reconstructed[:, :1023, :], x)
