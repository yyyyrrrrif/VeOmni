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
Training script for qwen_image (DiT-style text-to-image diffusion model).

Supports:
  - Pre-cached tensor data (TensorDataset): latents, encoder_hidden_states, etc.
  - Real image+caption data (e.g. LAION-COCO-Aesthetic): use_text_image_data=True with
    build_text_image_dataset, then image and text preprocessing via VAE + text encoder (like train_flux).
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    parse_args,
    save_args,
)
from veomni.checkpoint import build_checkpointer
from veomni.data.diffusion.data_loader import build_dit_dataloader
from veomni.data.diffusion.dataset import build_tensor_dataset, build_text_image_dataset
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import (
    build_foundation_model,
    save_model_assets,
)
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.schedulers.flow_match import FlowMatchScheduler
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.dit_utils import EnvironMeter, save_model_weights
from veomni.utils.lora_utils import add_lora_to_model, freeze_parameters
from veomni.utils.recompute_utils import convert_ops_to_objects
from veomni.utils.save_safetensor_utils import save_hf_safetensor


logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    use_text_image_data: bool = field(
        default=False,
        metadata={"help": "If True, use (image, text) dataset and encode with VAE + text encoder (e.g. LAION-COCO-Aesthetic)."},
    )
    height: int = field(default=512, metadata={"help": "Image height for text-image dataset."})
    width: int = field(default=512, metadata={"help": "Image width for text-image dataset."})
    center_crop: bool = field(default=True, metadata={"help": "Center crop when use_text_image_data."})
    random_flip: bool = field(default=False, metadata={"help": "Random horizontal flip when use_text_image_data."})
    datasets_repeat: int = field(
        default=1,
        metadata={"help": "The number of times to repeat the datasets."},
    )


@dataclass
class MyModelArguments(ModelArguments):
    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to tokenizer when use_text_image_data."})
    pretrained_text_encoder_path: Optional[str] = field(default=None, metadata={"help": "Path to text encoder when use_text_image_data."})
    pretrained_vae_path: Optional[str] = field(default=None, metadata={"help": "Path to VAE encoder when use_text_image_data (64-channel latents)."})
    max_text_length: int = field(default=512, metadata={"help": "Max token length for text when use_text_image_data."})


@dataclass
class MyTrainingArguments(TrainingArguments):
    save_initial_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the initial model."},
    )
    ops_to_save: List[str] = field(
        default_factory=list,
        metadata={"help": "Ops to save."},
    )


@dataclass
class Arguments:
    model: MyModelArguments = field(default_factory=MyModelArguments)
    data: MyDataArguments = field(default_factory=MyDataArguments)
    train: MyTrainingArguments = field(default_factory=MyTrainingArguments)


# QwenImage pipeline-style constants (align with diffusers pipeline_qwenimage)
QWENIMAGE_PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
QWENIMAGE_PROMPT_TEMPLATE_ENCODE_START_IDX = 34
QWENIMAGE_TOKENIZER_MAX_LENGTH = 1024


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    """Extract hidden states for non-padding positions (align with pipeline_qwenimage)._extract_masked_hidden."""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
    return split_result


def _pack_latents_qwenimage(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int) -> torch.Tensor:
    """Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4] (align with pipeline_qwenimage._pack_latents)."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def prepare_qwen_image_batch(micro_batch, device, micro_batch_size):
    """Prepare qwen_image batch for training (pre-cached tensor data)."""
    latents = micro_batch["latents"].to(device)
    encoder_hidden_states = micro_batch["encoder_hidden_states"]
    if micro_batch_size > 1:
        encoder_hidden_states = encoder_hidden_states.squeeze(1).to(device)
    else:
        encoder_hidden_states = encoder_hidden_states[0].to(device)

    encoder_hidden_states_mask = micro_batch.get("encoder_hidden_states_mask")
    if encoder_hidden_states_mask is not None:
        if micro_batch_size > 1:
            encoder_hidden_states_mask = encoder_hidden_states_mask.squeeze(1).to(device)
        else:
            encoder_hidden_states_mask = encoder_hidden_states_mask[0].to(device)
    txt_seq_lens = micro_batch.get("txt_seq_lens")
    return latents, encoder_hidden_states, encoder_hidden_states_mask, txt_seq_lens


@torch.no_grad()
def encode_text_for_qwen_image(
    captions: List[str],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 1024,
    prompt_template: str = QWENIMAGE_PROMPT_TEMPLATE_ENCODE,
    drop_idx: int = QWENIMAGE_PROMPT_TEMPLATE_ENCODE_START_IDX,
    tokenizer_max_length: int = QWENIMAGE_TOKENIZER_MAX_LENGTH,
) -> tuple:
    """Encode captions using QwenImage pipeline-style: template + drop prefix + masked hidden (align with pipeline_qwenimage._get_qwen_prompt_embeds)."""
    captions = [captions] if isinstance(captions, str) else captions
    txt = [prompt_template.format(c) for c in captions]
    txt_tokens = tokenizer(
        txt,
        max_length=tokenizer_max_length + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    encoder_attention_mask = encoder_attention_mask.to(device=device)
    if max_sequence_length is not None and prompt_embeds.size(1) > max_sequence_length:
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        encoder_attention_mask = encoder_attention_mask[:, :max_sequence_length]
    if encoder_attention_mask.all():
        encoder_attention_mask = None
    return prompt_embeds, encoder_attention_mask


def load_text_encoder_and_tokenizer(
    tokenizer_path: str,
    pretrained_text_encoder_path: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load QwenImage-style text encoder and tokenizer (Qwen2.5-VL; align with pipeline_qwenimage)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_text_encoder_path)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    return tokenizer, text_encoder, None


def load_vae_encoder(
    pretrained_vae_path: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load QwenImage VAE encoder (align with pipeline_qwenimage; expects 5D input [B,C,1,H,W])."""
    from diffusers.models import AutoencoderKLQwenImage

    try:
        vae = AutoencoderKLQwenImage.from_pretrained(pretrained_vae_path)
    except Exception:
        vae = AutoencoderKLQwenImage.from_pretrained(pretrained_vae_path, local_files_only=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae = vae.to(device=device, dtype=dtype)
    return vae


def get_qwenimage_image_processor(vae_scale_factor: int = 8):
    """VaeImageProcessor with vae_scale_factor*2 for preprocessing (align with pipeline_qwenimage)."""
    from diffusers.image_processor import VaeImageProcessor
    return VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)


def main():
    args = parse_args(Arguments)
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    torch.utils.checkpoint.set_checkpoint_debug_enabled(args.train.debug_gradient_checkpointing)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )
    logger.info_rank0(
        f"Parallel state: dp:{args.train.data_parallel_mode}, "
        f"tp:{args.train.tensor_parallel_size}, ep:{args.train.expert_parallel_size}, "
        f"pp:{args.train.pipeline_parallel_size}, cp:{args.train.context_parallel_size}, "
        f"ulysses:{args.train.ulysses_parallel_size}"
    )

    if args.data.data_type == "diffusion":
        use_text_image_data = getattr(args.data, "use_text_image_data", False)
        if use_text_image_data:
            train_dataset = build_text_image_dataset(
                base_path=args.data.train_path,
                metadata_path=os.path.join(args.data.train_path, "metadata.csv"),
                height=args.data.height,
                width=args.data.width,
                center_crop=args.data.center_crop,
                random_flip=args.data.random_flip,
                datasets_repeat=args.data.datasets_repeat,
            )
            use_text_image_data = True
        else:
            train_dataset = build_tensor_dataset(
                base_path=args.data.train_path,
                metadata_path=os.path.join(args.data.train_path, "metadata.csv"),
                datasets_repeat=args.data.datasets_repeat,
            )
            use_text_image_data = False

        args.train.compute_train_steps(
            args.data.max_seq_len,
            args.data.train_size,
            len(train_dataset) // args.train.data_parallel_size,
        )

        train_dataloader = build_dit_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            train_steps=args.train.train_steps,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        init_device=args.train.init_device,
        torch_dtype="bfloat16",
        attn_implementation=args.model.attn_implementation,
    )
    model.micro_batch_size = args.train.micro_batch_size

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    text_encoder = None
    tokenizer = None
    text_projection = None
    vae_encoder = None
    image_processor = None
    if use_text_image_data:
        device_str = f"{get_device_type()}:{args.train.local_rank}"
        if args.model.pretrained_text_encoder_path and args.model.tokenizer_path:
            tokenizer, text_encoder, text_projection = load_text_encoder_and_tokenizer(
                args.model.tokenizer_path,
                args.model.pretrained_text_encoder_path,
                device_str,
                torch.bfloat16,
            )
            logger.info_rank0("Loaded text encoder and tokenizer for QwenImage.")
        else:
            raise ValueError(
                "use_text_image_data=True requires model.tokenizer_path and model.pretrained_text_encoder_path"
            )
        if args.model.pretrained_vae_path:
            vae_encoder = load_vae_encoder(
                args.model.pretrained_vae_path,
                device_str,
                torch.bfloat16,
            )
            vae_scale_factor = 2 ** len(vae_encoder.temperal_downsample)
            image_processor = get_qwenimage_image_processor(vae_scale_factor)
            logger.info_rank0("Loaded VAE encoder and image processor for QwenImage.")
        else:
            raise ValueError("use_text_image_data=True requires model.pretrained_vae_path")

    lora_target_modules_support = ["to_q", "to_k", "to_v", "to_out.0", "img_mlp", "txt_mlp"]
    if args.train.train_architecture == "lora":
        logger.info_rank0("train_architecture is lora")
        _use_orig_params = True
        freeze_parameters(model)
        add_lora_to_model(
            model,
            lora_rank=args.model.lora_rank,
            lora_alpha=args.model.lora_alpha,
            lora_target_modules=args.model.lora_target_modules,
            init_lora_weights=args.model.init_lora_weights,
            pretrained_lora_path=args.model.pretrained_lora_path,
            lora_target_modules_support=lora_target_modules_support,
        )
        model.to(torch.bfloat16)
    else:
        logger.info_rank0("train_architecture is full")
        _use_orig_params = False

    logger.info_rank0(f"model: {model}")

    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            state_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
            save_model_weights(args.train.output_dir, model.state_dict(), model_assets=[model_config])
        dist.barrier()
        return

    ops_to_save = convert_ops_to_objects(args.train.ops_to_save)
    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_reshard_after_forward=args.train.enable_reshard_after_forward,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        use_orig_params=_use_orig_params,
        ops_to_save=ops_to_save,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

        model_assets = [model_config]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    flow_scheduler = FlowMatchScheduler(
        shift=5,
        sigma_min=0.0,
        extra_one_step=True,
    )

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )

    helper.empty_cache()
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, "
        f"train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )

    flow_scheduler.set_timesteps(1000, training=True)

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        epoch_start_time = time.time()
        data_iterator = iter(train_dataloader)

        epoch_loss = 0
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            synchronize()
            total_loss = 0
            start_time = time.time()
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            num_micro_steps = len(micro_batches)

            for micro_step, micro_batch in enumerate(micro_batches):
                if (
                    args.train.data_parallel_mode == "fsdp2"
                    and not args.train.enable_reshard_after_backward
                    and num_micro_steps > 1
                ):
                    if micro_step == 0:
                        model.set_reshard_after_backward(False)
                    elif micro_step == num_micro_steps - 1:
                        model.set_reshard_after_backward(True)

                environ_meter.add(micro_batch, model_type="qwen_image")

                if use_text_image_data and "text" in micro_batch and "image" in micro_batch:
                    text_list = micro_batch["text"] if isinstance(micro_batch["text"], list) else [micro_batch["text"]]
                    image = micro_batch["image"].to(dtype=torch.bfloat16, device=model.device)
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    # Align with pipeline_qwenimage: text encoding (template + drop prefix + masked hidden)
                    encoder_hidden_states, encoder_hidden_states_mask = encode_text_for_qwen_image(
                        text_list,
                        tokenizer,
                        text_encoder.to(model.device),
                        model.device,
                        torch.bfloat16,
                        max_sequence_length=args.model.max_text_length,
                    )
                    txt_seq_lens = (
                        encoder_hidden_states_mask.sum(dim=1)
                        if encoder_hidden_states_mask is not None
                        else torch.full((encoder_hidden_states.size(0),), encoder_hidden_states.size(1), device=model.device, dtype=torch.long)
                    )
                    # Align with pipeline_qwenimage: image preprocess (multiple of vae_scale_factor*2), VAE encode, normalize, pack
                    vae_scale_factor = 2 ** len(vae_encoder.temperal_downsample)
                    multiple_of = vae_scale_factor * 2
                    height = (args.data.height // multiple_of) * multiple_of
                    width = (args.data.width // multiple_of) * multiple_of
                    image = image_processor.preprocess(image, height=height, width=width)
                    if image.dim() == 4:
                        image = image.unsqueeze(2)
                    with torch.no_grad():
                        image_latents = vae_encoder.to(model.device).encode(image).latent_dist.mode()
                    latent_channels = getattr(vae_encoder.config, "z_dim", 16)
                    latents_mean = (
                        torch.tensor(vae_encoder.config.latents_mean, device=image_latents.device, dtype=image_latents.dtype)
                        .view(1, latent_channels, 1, 1, 1)
                    )
                    latents_std = (
                        torch.tensor(vae_encoder.config.latents_std, device=image_latents.device, dtype=image_latents.dtype)
                        .view(1, latent_channels, 1, 1, 1)
                    )
                    image_latents = (image_latents - latents_mean) / latents_std
                    image_latents = image_latents.squeeze(2)
                    B, num_channels_latents, latent_h, latent_w = image_latents.shape
                    latents = _pack_latents_qwenimage(image_latents, B, num_channels_latents, latent_h, latent_w)
                    img_shapes = [(1, latent_h // 2, latent_w // 2)] * B
                else:
                    latents, encoder_hidden_states, encoder_hidden_states_mask, txt_seq_lens = (
                        prepare_qwen_image_batch(micro_batch, model.device, args.train.micro_batch_size)
                    )
                    if txt_seq_lens is not None and torch.is_tensor(txt_seq_lens):
                        txt_seq_lens = txt_seq_lens.to(model.device)
                    B = latents.shape[0] if latents.dim() == 4 else 1
                    if latents.dim() == 3:
                        latents = latents.unsqueeze(0)
                    _, C, H, W = latents.shape
                    patch_size = model_config.patch_size
                    h_patches, w_patches = H // patch_size, W // patch_size
                    img_shapes = [(1, h_patches, w_patches)] * B
                    if txt_seq_lens is None:
                        txt_seq_lens = torch.full(
                            (B,), encoder_hidden_states.size(1), device=model.device, dtype=torch.long
                        )

                noise = torch.randn_like(latents)
                timestep_id = torch.randint(0, flow_scheduler.num_train_timesteps, (latents.size(0),))
                timestep = flow_scheduler.timesteps[timestep_id].to(latents.dtype).to(latents.device)

                noisy_latents = flow_scheduler.add_noise(
                    latents,
                    noise,
                    timestep,
                    args.train.micro_batch_size,
                    args.train.enable_mixed_precision,
                )
                training_target = flow_scheduler.training_target(latents, noise, timestep)

                # Branch: packed latents [B, num_patches, 64] (text-image path) vs 4D [B, C, H, W] (pre-cached)
                latents_packed = latents.dim() == 3
                if latents_packed:
                    noisy_input = noisy_latents
                    training_target_patches = training_target
                    B = latents.shape[0]
                else:
                    # qwen_image expects [B, seq_len, in_channels] input (64-dim per patch after mean)
                    if noisy_latents.dim() == 3:
                        noisy_latents = noisy_latents.unsqueeze(0)
                    B, C, H, W = noisy_latents.shape
                    patch_size = model_config.patch_size
                    h_patches = H // patch_size
                    w_patches = W // patch_size
                    noisy_input = noisy_latents.reshape(
                        B, C, h_patches, patch_size, w_patches, patch_size
                    ).mean((3, 5)).permute(0, 2, 3, 1).reshape(B, h_patches * w_patches, C)
                    img_shapes = [(1, h_patches, w_patches)] * B
                    training_target_input = training_target
                    if training_target_input.dim() == 3:
                        training_target_input = training_target_input.unsqueeze(0)
                    training_target_patches = (
                        training_target_input.reshape(
                            B, C, h_patches, patch_size, w_patches, patch_size
                        )
                        .permute(0, 2, 4, 1, 3, 5)
                        .reshape(B, h_patches * w_patches, C * patch_size * patch_size)
                    )

                with model_fwd_context:
                    output = model.forward(
                        hidden_states=noisy_input,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        return_dict=True,
                    )
                    noise_pred = output["logits"]

                    loss = F.mse_loss(noise_pred.float(), training_target_patches.float(), reduction="none")
                    weight = flow_scheduler.training_weight(timestep, args.train.micro_batch_size)
                    loss = (loss.view(B, -1).mean(dim=1) * weight).mean() / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            epoch_loss += total_loss
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, step_time: {delta_time:.2f}s",
                refresh=False,
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                save_hf_safetensor(
                    save_hf_safetensor_path=hf_weights_path,
                    ckpt_manager=args.train.ckpt_manager,
                    model_assets=model_assets,
                    train_architecture=args.train.train_architecture,
                    save_checkpoint_path=save_checkpoint_path,
                    output_dir=args.train.output_dir,
                    is_rank_0=args.train.global_rank == 0,
                    model=model,
                    fqn_to_index_mapping=args.model.fqn_to_index_mapping,
                )

        data_loader_tqdm.close()
        epoch_time = time.time() - epoch_start_time
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.global_rank == 0:
            logger.info_rank0(
                f"Epoch {epoch + 1} completed, epoch_time={epoch_time:.4f}s, "
                f"epoch_loss={epoch_loss / args.train.train_steps:.4f}"
            )
        if args.train.global_rank == 0:
            if args.train.use_wandb:
                wandb.log({"training/loss_per_epoch": epoch_loss / args.train.train_steps}, step=global_step)
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            save_hf_safetensor(
                save_hf_safetensor_path=hf_weights_path,
                ckpt_manager=args.train.ckpt_manager,
                model_assets=model_assets,
                train_architecture=args.train.train_architecture,
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                is_rank_0=args.train.global_rank == 0,
                model=model,
                fqn_to_index_mapping=args.model.fqn_to_index_mapping,
            )

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
