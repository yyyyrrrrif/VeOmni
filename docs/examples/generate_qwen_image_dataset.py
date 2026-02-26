#!/usr/bin/env python3
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
Generate mock pre-cached .tensors.pth data for qwen_image training.

Format is compatible with TensorDataset / build_tensor_dataset():
  - Each .tensors.pth file is a dict with:
      latents: [C, H, W]
      encoder_hidden_states: [text_seq_len, text_embed_dim]
      encoder_hidden_states_mask: [text_seq_len]
      txt_seq_lens: [1]
  - metadata.csv lists file_name entries (without .tensors.pth suffix)

Usage:
    python docs/examples/generate_qwen_image_dataset.py --output_dir QwenImageMini --num_samples 100
"""

import argparse
import os

import pandas as pd
import torch


def generate_mock_data(
    output_dir: str,
    num_samples: int = 10,
    in_channels: int = 64,
    latent_h: int = 32,
    latent_w: int = 32,
    text_seq_len: int = 128,
    text_embed_dim: int = 3584,
):
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    records = []
    for i in range(num_samples):
        data = {
            "latents": torch.randn(in_channels, latent_h, latent_w),
            "encoder_hidden_states": torch.randn(text_seq_len, text_embed_dim, dtype=torch.bfloat16),
            "encoder_hidden_states_mask": torch.ones(text_seq_len, dtype=torch.bool),
            "txt_seq_lens": torch.tensor(text_seq_len, dtype=torch.long),
        }
        fname = f"sample_{i:05d}"
        torch.save(data, os.path.join(train_dir, f"{fname}.tensors.pth"))
        records.append({"file_name": fname})

    pd.DataFrame(records).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Generated {num_samples} mock samples in {output_dir}")
    print(f"  latents shape: [{in_channels}, {latent_h}, {latent_w}]")
    print(f"  encoder_hidden_states shape: [{text_seq_len}, {text_embed_dim}]")


def main():
    parser = argparse.ArgumentParser(description="Generate mock qwen_image training data")
    parser.add_argument("--output_dir", type=str, default="QwenImageMini")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--in_channels", type=int, default=64)
    parser.add_argument("--latent_h", type=int, default=32)
    parser.add_argument("--latent_w", type=int, default=32)
    parser.add_argument("--text_seq_len", type=int, default=128)
    parser.add_argument("--text_embed_dim", type=int, default=3584)
    args = parser.parse_args()

    generate_mock_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        in_channels=args.in_channels,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        text_seq_len=args.text_seq_len,
        text_embed_dim=args.text_embed_dim,
    )


if __name__ == "__main__":
    main()
