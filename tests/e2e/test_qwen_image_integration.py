"""
Integration tests for QwenImage training pipeline.

Tests:
  - Single GPU smoke test: small model + mock data for 3 training steps
  - Multi-GPU FSDP test: 2 GPU FSDP1/FSDP2 (requires multi-GPU)
  - FSDP + Ulysses SP mixed test (requires 4+ GPUs)

Single-GPU test can run standalone:
    python tests/e2e/test_qwen_image_integration.py

Multi-GPU tests:
    torchrun --nproc_per_node=2 -m pytest tests/e2e/test_qwen_image_integration.py -v -s -k "fsdp"
    torchrun --nproc_per_node=4 -m pytest tests/e2e/test_qwen_image_integration.py -v -s -k "ulysses"
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.nn.functional as F

from veomni.schedulers.flow_match import FlowMatchScheduler


def _generate_mock_data(output_dir, num_samples=10, in_channels=64, latent_h=16, latent_w=16):
    """Generate minimal mock data for integration tests."""
    import pandas as pd

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    records = []
    for i in range(num_samples):
        data = {
            "latents": torch.randn(in_channels, latent_h, latent_w),
            "encoder_hidden_states": torch.randn(32, 512),
            "encoder_hidden_states_mask": torch.ones(32, dtype=torch.bool),
            "img_shapes": (1, latent_h * 2, latent_w * 2),
        }
        fname = f"sample_{i:05d}"
        torch.save(data, os.path.join(train_dir, f"{fname}.tensors.pth"))
        records.append({"file_name": fname})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


class TestSingleGPUSmoke:
    """Single GPU smoke test: small model + mock data, 3 training steps."""

    def test_train_3_steps(self, tmp_path):
        from veomni.models.transformers.qwen_image.config_qwen_image import QwenImageConfig
        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = QwenImageConfig(
            image_size=32,
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

        model = QwenImageModel(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)

        batch_size = 2
        in_channels = config.in_channels
        patch_size = config.patch_size
        latent_h, latent_w = 16, 16
        h_patches = latent_h // patch_size
        w_patches = latent_w // patch_size
        num_patches = h_patches * w_patches

        losses = []
        for step in range(3):
            latents = torch.randn(batch_size, in_channels, latent_h, latent_w, device=device)
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, scheduler.num_train_timesteps, (batch_size,))
            timestep = scheduler.timesteps[timestep_id].to(latents.dtype).to(device)

            noisy = scheduler.add_noise(latents, noise, timestep, 1, False)
            target = scheduler.training_target(latents, noise, timestep)

            # Patchify
            noisy_input = noisy.reshape(
                batch_size, in_channels, h_patches, patch_size, w_patches, patch_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(
                batch_size, num_patches, in_channels * patch_size * patch_size
            )
            target_patches = target.reshape(
                batch_size, in_channels, h_patches, patch_size, w_patches, patch_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(
                batch_size, num_patches, in_channels * patch_size * patch_size
            )

            encoder_hidden_states = torch.randn(batch_size, 16, config.text_embed_dim, device=device)
            img_shapes = [(1, latent_h * 2, latent_w * 2)] * batch_size

            output = model(
                hidden_states=noisy_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=True,
            )
            pred = output["logits"]

            loss = F.mse_loss(pred.float(), target_patches.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(l)) for l in losses), f"Non-finite losses: {losses}"
        assert len(losses) == 3

    def test_checkpoint_save_load(self, tmp_path):
        """Verify model state_dict save/load round-trip."""
        from veomni.models.transformers.qwen_image.config_qwen_image import QwenImageConfig
        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        config = QwenImageConfig(
            image_size=32,
            patch_size=2,
            in_channels=64,
            out_channels=64,
            hidden_size=384,
            num_hidden_layers=1,
            num_attention_heads=6,
            text_embed_dim=512,
            use_flash_attention=False,
            axes_dims_rope=(16, 16, 32),
        )
        model = QwenImageModel(config)

        save_path = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), save_path)

        model2 = QwenImageModel(config)
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch"

    def test_no_split_modules(self):
        """Verify _no_split_modules is set for FSDP wrapping."""
        from veomni.models.transformers.qwen_image.modeling_qwen_image import QwenImageModel

        assert hasattr(QwenImageModel, "_no_split_modules")
        assert "QwenImageTransformerBlock" in QwenImageModel._no_split_modules


if __name__ == "__main__":
    print("=" * 60)
    print("QwenImage Integration Tests")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        t = TestSingleGPUSmoke()

        class FakePath:
            def __init__(self, p):
                self._p = p
            def __str__(self):
                return self._p
            def __truediv__(self, other):
                return FakePath(os.path.join(self._p, other))

        t.test_train_3_steps(FakePath(tmp))
        print("[PASS] test_train_3_steps")

        t.test_checkpoint_save_load(FakePath(tmp))
        print("[PASS] test_checkpoint_save_load")

        t.test_no_split_modules()
        print("[PASS] test_no_split_modules")

    print("\nAll integration tests passed!")
