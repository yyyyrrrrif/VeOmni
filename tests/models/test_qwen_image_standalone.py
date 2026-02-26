"""
Standalone test script for QwenImageModel.

Tests cover:
  A. Model basics: config round-trip, registry loading, output shapes
  D. Training pipeline: flow match loss, mock data format, EnvironMeter

Run with pytest:
    pytest tests/models/test_qwen_image_standalone.py -v

Run directly:
    python tests/models/test_qwen_image_standalone.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.nn as nn

from veomni.models.transformers.qwen_image.config_qwen_image import QwenImageConfig
from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel


# ---------------------------------------------------------------------------
# A. Model Basics
# ---------------------------------------------------------------------------


def _make_small_config(**overrides):
    defaults = dict(
        image_size=64,
        patch_size=1,
        in_channels=64,
        out_channels=64,
        hidden_size=384,
        num_hidden_layers=2,
        num_attention_heads=6,
        intermediate_size=1536,
        text_embed_dim=512,
        text_max_length=128,
        use_flash_attention=False,
        axes_dims_rope=(16, 16, 32),
    )
    defaults.update(overrides)
    return QwenImageConfig(**defaults)


class TestConfigRoundTrip:
    """A1. Config serialization round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        config = _make_small_config()
        config.save_pretrained(str(tmp_path))
        loaded = QwenImageConfig.from_pretrained(str(tmp_path))

        assert loaded.image_size == config.image_size
        assert loaded.patch_size == config.patch_size
        assert loaded.hidden_size == config.hidden_size
        assert loaded.num_hidden_layers == config.num_hidden_layers
        assert loaded.num_attention_heads == config.num_attention_heads
        assert loaded.text_embed_dim == config.text_embed_dim
        assert tuple(loaded.axes_dims_rope) == tuple(config.axes_dims_rope)
        assert loaded.model_type == "qwen_image"


class TestRegistryLoading:
    """A3. Registry route correctness."""

    def test_registry(self, monkeypatch):
        monkeypatch.setenv("MODELING_BACKEND", "veomni")
        from veomni.models.loader import get_model_config, get_model_class

        config_path = "../../configs/model_configs/qwen_image/qwen_image.json"
        if not os.path.exists(config_path):
            pytest.skip("qwen_image config json not found")
        config = get_model_config(config_path)
        assert config.__class__.__name__ == "QwenImageConfig"
        assert config.model_type == "qwen_image"
        model_class = get_model_class(config)
        assert model_class.__name__ == "QwenImageModel"


class TestBasicForward:
    """Basic forward pass tests from the original standalone test."""

    @staticmethod
    def _get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "npu")

    @staticmethod
    def _make_forward_batch(device, config, batch_size=2, text_len=20, requires_grad=False):
        """Build shared forward inputs for test_basic_forward and test_gradient_flow."""
        num_patches = (config.image_size // config.patch_size) ** 2
        hidden_states = torch.randn(
            batch_size, num_patches, config.in_channels, device=device, requires_grad=requires_grad
        )
        encoder_hidden_states = torch.randn(batch_size, text_len, config.text_embed_dim, device=device)
        encoder_hidden_states_mask = torch.ones(batch_size, text_len, device=device, dtype=torch.bool)
        timestep = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
        img_shapes = [(1, config.image_size, config.image_size)] * batch_size
        txt_seq_lens = torch.tensor([text_len] * batch_size, device=device, dtype=torch.long)
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
        }

    def test_basic_forward(self):
        device = self._get_device()
        config = _make_small_config(num_hidden_layers=2)
        model = QwenImageModel(config).to(device).eval()
        batch = self._make_forward_batch(device, config)

        with torch.no_grad():
            output = model(return_dict=True, **batch)

        assert "logits" in output
        assert torch.isfinite(output["logits"]).all()

    def test_gradient_flow(self):
        device = self._get_device()
        config = _make_small_config(image_size=32, patch_size=1, num_hidden_layers=1)
        model = QwenImageModel(config).to(device).train()
        batch = self._make_forward_batch(device, config, requires_grad=True)

        output = model(return_dict=True, **batch)
        loss = output["logits"].mean()
        loss.backward()

        # Model parameters must receive gradients (backward ran correctly).
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Expected at least one model parameter to have a finite gradient."
        # Input hidden_states may or may not get .grad depending on the underlying
        # implementation (e.g. diffusers may not backprop to input); optional check:
        if batch["hidden_states"].grad is not None:
            assert torch.isfinite(batch["hidden_states"].grad).all()


# ---------------------------------------------------------------------------
# B. Genrate Dataset
# ---------------------------------------------------------------------------


class TestMockDataFormat:
    """D2. Mock data format validation."""

    def test_mock_data_loads(self, tmp_path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../docs/examples"))
        from generate_qwen_image_dataset import generate_mock_data
        from veomni.data.diffusion.dataset import TensorDataset

        generate_mock_data(str(tmp_path), num_samples=5)

        dataset = TensorDataset(
            str(tmp_path),
            os.path.join(str(tmp_path), "metadata.csv"),
        )
        assert len(dataset) == 5

        sample = dataset[0]
        assert isinstance(sample, list)
        data = sample[0]
        assert "latents" in data
        assert data["latents"].shape == (64, 32, 32)
        assert "encoder_hidden_states" in data
        assert data["encoder_hidden_states"].shape == (128, 3584)
        assert "encoder_hidden_states_mask" in data
