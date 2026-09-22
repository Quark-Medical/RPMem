"""Hypernetwork training script for rpmem.

Trains the Perceiver + HyperLoRAHead (the "hypernetwork") to convert context
into LoRA weights. This is the first stage of the RPMem pipeline — before
CMP Gate training.

Usage:
  # Single GPU
  python -m rpmem.training.train_hypernet --config config.yaml

  # Multi-GPU with accelerate
  accelerate launch -m rpmem.training.train_hypernet --config config.yaml

  # With CLI overrides
  accelerate launch -m rpmem.training.train_hypernet \
      --config config.yaml \
      --base_model_path models/reader \
      --lr 1e-5 --max_steps 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train RPMem hypernetwork")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    # Model
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--ctx_encoder_path", type=str, default=None)
    parser.add_argument(
        "--base_model_device_map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default=None,
        help=(
            "Shard the frozen backbone across the visible GPUs in one process. "
            "This is intended for backbones that cannot be replicated per GPU."
        ),
    )
    parser.add_argument(
        "--model_parallel_data_parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Replicate a locally sharded frozen backbone across distributed "
            "processes and explicitly average only trainable gradients. Each "
            "process must be the sole process on its node."
        ),
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="Initialize hypernetwork weights only from an existing checkpoint.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume model, optimizer, scheduler, step, and per-rank RNG state.",
    )
    parser.add_argument(
        "--perceiver_init_checkpoint",
        type=str,
        default=None,
        help=(
            "Strictly initialize only the Perceiver latent-space module from a "
            "RPMem checkpoint. The target backbone and HyperLoRA head are "
            "constructed from the current configuration."
        ),
    )
    parser.add_argument(
        "--trainable_scope",
        choices=("perceiver_and_head", "head_only"),
        default=None,
        help="Select which hypernetwork components receive gradients.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--base_model_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recompute frozen-backbone activations during backward. This lowers "
            "peak memory for decoder-head transfer while preserving gradients "
            "through generated LoRA weights."
        ),
    )

    # Architecture
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_latent", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--target_modules", type=str, nargs="+", default=None)
    parser.add_argument(
        "--per_layer_processing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--per_rank_gen",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--num_pre_head_layers", type=int, default=None)
    parser.add_argument("--n_latent_queries", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)

    # Data
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        required=True,
        help="Paths to training data (parquet or jsonl)",
    )
    parser.add_argument("--val_data", type=str, nargs="*", default=None)
    parser.add_argument("--max_ctx_len", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--use_packing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use sequence packing (Flash Attention required)",
    )
    parser.add_argument(
        "--teacher_logprobs_dir",
        type=str,
        default=None,
        help="Precomputed teacher top-K logprob dir. If set, "
        "train with offline context distillation instead "
        "of hard-label CE.",
    )
    parser.add_argument(
        "--val_teacher_logprobs_dir",
        type=str,
        default=None,
        help="Teacher top-K store aligned with --val_data.",
    )
    parser.add_argument(
        "--objective",
        choices=("auto", "sft", "offline_fkl", "d2l_topk_ce", "opcd"),
        default=None,
        help=(
            "Phase 1 training objective. offline_fkl is the formal "
            "teacher-top-k-plus-tail baseline; d2l_topk_ce reproduces the "
            "upstream selected-top-k loss."
        ),
    )
    parser.add_argument(
        "--opcd_loss_mode",
        choices=(
            "student_topk_reverse_kl",
            "offline_rkl",
            "online_fkl",
            "k3",
            "k3_plus",
        ),
        default=None,
    )
    parser.add_argument("--opcd_top_k", type=int, default=None)
    parser.add_argument("--opcd_kl_chunk_size", type=int, default=None)
    parser.add_argument("--opcd_rollout_max_new_tokens", type=int, default=None)
    parser.add_argument("--opcd_max_teacher_seq_len", type=int, default=None)
    parser.add_argument("--opcd_queries_per_session", type=int, default=None)
    parser.add_argument(
        "--opcd_context_lift_every",
        type=int,
        default=None,
        help="Measure teacher context lift every N optimizer steps; 0 disables it.",
    )
    parser.add_argument(
        "--opcd_context_lift_eval_batches",
        type=int,
        default=None,
        help="Measure context lift on the first N local validation batches.",
    )
    parser.add_argument(
        "--quality_eval_sessions",
        type=int,
        default=None,
        help=(
            "Generate responses for the first N held-out validation sessions. "
            "Zero disables generation-quality validation."
        ),
    )
    parser.add_argument(
        "--quality_eval_queries_per_session",
        type=int,
        default=None,
        help="Deterministic probe count per quality-evaluation session.",
    )
    parser.add_argument(
        "--quality_eval_max_new_tokens",
        type=int,
        default=None,
        help="Maximum generated response tokens for quality validation.",
    )

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--l1_reg_coef", type=float, default=None)
    parser.add_argument(
        "--use_per_ctx_average_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)

    # Logging / Checkpointing
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help=(
            "TensorBoard log directory. Defaults to TENSORBOARD_LOGGING_DIR, "
            "RPMEM_TENSORBOARD_DIR, or <output_dir>/tb_logs."
        ),
    )
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run validation every N optimizer steps; defaults to save_steps.",
    )
    parser.add_argument(
        "--save_training_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save optimizer/scheduler state for practical long-run resume.",
    )
    parser.add_argument(
        "--save_epoch_boundaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save and validate at every complete data epoch.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Keep only the newest N periodic checkpoints; <=0 keeps all.",
    )
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--preflight_stop_after_step",
        type=int,
        default=None,
        help=(
            "Stop after writing this checkpoint so the comprehensive cluster "
            "preflight can verify exact resume. Do not use for formal training."
        ),
    )
    parser.add_argument(
        "--selected_endpoint_step",
        type=int,
        default=None,
        help=(
            "Stop after writing this reproducible intermediate endpoint while "
            "retaining the full optimization schedule."
        ),
    )

    args = parser.parse_args(argv)

    # Merge YAML config: YAML provides defaults, CLI overrides
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    # Final defaults for anything still unset
    defaults = dict(
        use_flash_attn=True,
        base_model_gradient_checkpointing=False,
        model_parallel_data_parallel=False,
        trainable_scope="perceiver_and_head",
        n_layers=32,
        d_latent=512,
        lora_r=8,
        target_modules=["down_proj"],
        per_layer_processing=True,
        per_rank_gen=True,
        num_pre_head_layers=4,
        n_latent_queries=8,
        num_blocks=9,
        max_ctx_len=768,
        max_seq_len=2048,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        lr=1e-5,
        weight_decay=0.01,
        warmup_steps=50,
        max_steps=2000,
        gradient_accumulation_steps=16,
        max_grad_norm=1.0,
        l1_reg_coef=0.01,
        lora_alpha=32.0,
        lora_dropout=0.0,
        output_dir="train_outputs/hypernet",
        tensorboard_dir=None,
        save_steps=500,
        eval_steps=None,
        save_training_state=False,
        save_epoch_boundaries=True,
        save_total_limit=2,
        logging_steps=100,
        dataloader_num_workers=8,
        seed=42,
        objective="auto",
        opcd_loss_mode="student_topk_reverse_kl",
        opcd_top_k=32,
        opcd_kl_chunk_size=16,
        opcd_rollout_max_new_tokens=128,
        opcd_max_teacher_seq_len=4864,
        opcd_queries_per_session=8,
        opcd_context_lift_every=50,
        opcd_context_lift_eval_batches=32,
        quality_eval_sessions=0,
        quality_eval_queries_per_session=8,
        quality_eval_max_new_tokens=128,
    )
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.eval_steps is None:
        args.eval_steps = args.save_steps

    if args.objective == "auto":
        args.objective = (
            "d2l_topk_ce" if args.teacher_logprobs_dir is not None else "sft"
        )
    if (
        args.objective in {"offline_fkl", "d2l_topk_ce"}
        and not args.teacher_logprobs_dir
    ):
        raise ValueError(
            f"--objective={args.objective} requires --teacher_logprobs_dir"
        )
    if args.objective in {"sft", "opcd"} and args.teacher_logprobs_dir:
        raise ValueError(
            f"--objective={args.objective} cannot be combined with "
            "--teacher_logprobs_dir"
        )
    if args.val_teacher_logprobs_dir and not args.val_data:
        raise ValueError("--val_teacher_logprobs_dir requires --val_data")
    if (
        args.val_data
        and args.objective in {"offline_fkl", "d2l_topk_ce"}
        and not args.val_teacher_logprobs_dir
    ):
        raise ValueError(
            f"--objective={args.objective} requires --val_teacher_logprobs_dir "
            "when --val_data is set"
        )
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    for name in (
        "max_steps",
        "save_steps",
        "eval_steps",
        "logging_steps",
        "gradient_accumulation_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    for name in (
        "opcd_top_k",
        "opcd_kl_chunk_size",
        "opcd_rollout_max_new_tokens",
        "opcd_max_teacher_seq_len",
        "opcd_queries_per_session",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.opcd_context_lift_every < 0:
        raise ValueError("opcd_context_lift_every must be non-negative")
    if args.opcd_context_lift_eval_batches < 0:
        raise ValueError("opcd_context_lift_eval_batches must be non-negative")
    from rpmem.training.generation_quality import validate_quality_eval_contract

    validate_quality_eval_contract(
        quality_eval_sessions=args.quality_eval_sessions,
        quality_eval_queries_per_session=args.quality_eval_queries_per_session,
        quality_eval_max_new_tokens=args.quality_eval_max_new_tokens,
    )
    if args.quality_eval_sessions and not args.val_data:
        raise ValueError("quality_eval_sessions requires --val_data")
    if args.quality_eval_sessions and args.objective == "opcd":
        raise ValueError(
            "Accelerate OPCD does not expose validation rollouts; use the formal "
            "VERL OPCD path for generation-quality validation"
        )
    if (
        args.quality_eval_sessions
        and args.objective != "opcd"
        and args.per_device_eval_batch_size != 1
    ):
        raise ValueError(
            "reference generation currently requires per_device_eval_batch_size=1"
        )
    if args.quality_eval_sessions and args.use_packing:
        raise ValueError("reference generation does not support packed validation")
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise ValueError("per-device batch sizes must be positive")
    if args.from_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "--from_checkpoint and --resume_from_checkpoint are mutually exclusive"
        )
    if args.from_checkpoint and args.perceiver_init_checkpoint:
        raise ValueError(
            "--from_checkpoint and --perceiver_init_checkpoint are mutually exclusive"
        )
    if args.perceiver_init_checkpoint and args.trainable_scope != "head_only":
        raise ValueError(
            "--perceiver_init_checkpoint requires --trainable_scope=head_only"
        )
    if args.trainable_scope == "head_only" and not args.perceiver_init_checkpoint:
        raise ValueError(
            "--trainable_scope=head_only requires --perceiver_init_checkpoint"
        )
    if args.use_packing and args.per_device_train_batch_size != 1:
        raise ValueError(
            "cross-session packed context batches are not supported; use "
            "--no-use_packing for per_device_train_batch_size > 1"
        )
    if args.objective == "opcd":
        if args.use_packing:
            raise ValueError("OPCD rollout/replay does not support sequence packing")
        if args.max_seq_len <= args.opcd_rollout_max_new_tokens:
            raise ValueError("max_seq_len must exceed opcd_rollout_max_new_tokens")
        if args.opcd_max_teacher_seq_len <= args.opcd_rollout_max_new_tokens:
            raise ValueError("opcd_max_teacher_seq_len must exceed rollout length")
    if (
        args.preflight_stop_after_step is not None
        and args.selected_endpoint_step is not None
    ):
        raise ValueError(
            "preflight_stop_after_step and selected_endpoint_step are mutually "
            "exclusive"
        )
    bounded_stop_step = (
        args.selected_endpoint_step or args.preflight_stop_after_step
    )
    if bounded_stop_step is not None:
        if not 0 < bounded_stop_step < args.max_steps:
            raise ValueError(
                "bounded stop step must be between 1 and max_steps - 1"
            )
        if bounded_stop_step % args.save_steps != 0:
            raise ValueError("bounded stop step must coincide with save_steps")
        if not args.save_training_state:
            raise ValueError(
                "bounded stop step requires save_training_state=true"
            )

    return args


def build_config(args):
    """Build RPMemConfig from parsed args."""
    from rpmem.config import (
        GateConfig,
        HeadConfig,
        LoRAConfig,
        RPMemConfig,
        PerceiverConfig,
    )

    base_model_name = args.base_model_path or "mistralai/Mistral-7B-Instruct-v0.2"
    ctx_encoder_name = args.ctx_encoder_path or "answerdotai/ModernBERT-base"

    # Infer feature sizes from target_modules + model config
    from transformers import AutoConfig

    from rpmem.training.model_compat import text_model_config

    model_config = text_model_config(
        AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    )
    hidden_size = model_config.hidden_size
    intermediate_size = getattr(model_config, "intermediate_size", None)

    # Feature size mapping for common target modules
    feature_map_in = {}
    feature_map_out = {}
    for m in args.target_modules:
        if m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            feature_map_in[m] = hidden_size
            feature_map_out[m] = hidden_size
        elif m == "down_proj":
            if intermediate_size is None:
                raise ValueError(
                    "down_proj requires model_config.intermediate_size; use an "
                    "architecture-specific target for this backbone"
                )
            feature_map_in[m] = intermediate_size
            feature_map_out[m] = hidden_size
        elif m == "shared_expert_down_proj":
            shared_intermediate_size = getattr(
                model_config,
                "shared_expert_intermediate_size",
                None,
            )
            if shared_intermediate_size is None:
                raise ValueError(
                    "shared_expert_down_proj requires "
                    "model_config.shared_expert_intermediate_size"
                )
            feature_map_in[m] = shared_intermediate_size
            feature_map_out[m] = hidden_size
        elif m in ("gate_proj", "up_proj"):
            if intermediate_size is None:
                raise ValueError(
                    f"{m} requires model_config.intermediate_size"
                )
            feature_map_in[m] = hidden_size
            feature_map_out[m] = intermediate_size
        else:
            feature_map_in[m] = hidden_size
            feature_map_out[m] = hidden_size

    encoder_config = AutoConfig.from_pretrained(
        ctx_encoder_name, trust_remote_code=True
    )
    encoder_hidden_size = encoder_config.hidden_size

    return RPMemConfig(
        base_model_name=base_model_name,
        ctx_encoder_model_name=ctx_encoder_name,
        n_layers=args.n_layers,
        layer_indices=list(range(args.n_layers)),
        lora=LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        ),
        perceiver=PerceiverConfig(
            input_size=encoder_hidden_size,
            hidden_size=args.d_latent,
            n_latent_queries=args.n_latent_queries,
            num_attention_heads=8,
            num_key_value_heads=8,
            encoder_num_blocks=args.num_blocks,
            encoder_num_self_attn_per_block=0,
        ),
        head=HeadConfig(
            d_latent=args.d_latent,
            n_layers=args.n_layers,
            n_modules=1,
            r=args.lora_r,
            num_pre_head_layers=args.num_pre_head_layers,
            per_layer_processing=args.per_layer_processing,
            use_bias=True,
            target_modules=args.target_modules,
            in_features=feature_map_in,
            out_features=feature_map_out,
        ),
        gate=GateConfig(d_latent=args.d_latent),
    )


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main(argv=None, *, opcd_backend_factory=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    from rpmem.training.sample_store import resolve_sample_paths

    train_paths = resolve_sample_paths(args.train_data)
    val_paths = resolve_sample_paths(args.val_data) if args.val_data else ()
    from rpmem.training.teacher_shards import validate_store_source_files

    if args.teacher_logprobs_dir:
        validate_store_source_files(args.teacher_logprobs_dir, train_paths)
    if args.val_teacher_logprobs_dir:
        validate_store_source_files(args.val_teacher_logprobs_dir, val_paths)

    resume_dir = (
        Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    model_checkpoint = args.from_checkpoint
    if resume_dir is not None:
        model_checkpoint = str(resume_dir / "pytorch_model.bin")
        required_resume_files = [
            Path(model_checkpoint),
            resume_dir / "training_state.pt",
            resume_dir / "checkpoint_complete.json",
        ]
        missing = [str(path) for path in required_resume_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "resume checkpoint is incomplete; missing: " + ", ".join(missing)
            )

    # Try to use accelerate if available
    try:
        from accelerate import Accelerator, DataLoaderConfiguration

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16",
            dataloader_config=DataLoaderConfiguration(
                use_seedable_sampler=True,
                data_seed=args.seed,
            ),
        )
        use_accelerate = True
        device = accelerator.device
        logger.info(
            f"Using accelerate, device: {device}, num_processes: {accelerator.num_processes}"
        )
    except ImportError as exc:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "distributed training requires a compatible Accelerate install"
            ) from exc
        use_accelerate = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Accelerate not available, using device: {device}")

    is_main = (not use_accelerate) or accelerator.is_main_process
    handlers = [logging.StreamHandler()]
    if is_main:
        handlers.append(logging.FileHandler(f"{args.output_dir}/train.log"))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Training hypernetwork, output: {args.output_dir}")
    logger.info(f"Args: {vars(args)}")

    if is_main:
        with open(f"{args.output_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    # Build config
    config = build_config(args)

    # Build model
    from rpmem.training.hypernet_model import HypernetModel

    transfer_report = None
    if model_checkpoint:
        logger.info(f"Loading from checkpoint: {model_checkpoint}")
        model = HypernetModel.from_checkpoint(
            model_checkpoint,
            base_model_path=args.base_model_path,
            ctx_encoder_path=args.ctx_encoder_path,
            use_flash_attn=args.use_flash_attn,
            base_model_device_map=args.base_model_device_map,
            lora_dropout=args.lora_dropout,
            lora_alpha=args.lora_alpha,
            l1_reg_coef=args.l1_reg_coef,
            use_per_ctx_average_loss=args.use_per_ctx_average_loss,
            objective=args.objective,
            opcd_loss_mode=args.opcd_loss_mode,
            opcd_top_k=args.opcd_top_k,
            opcd_kl_chunk_size=args.opcd_kl_chunk_size,
            opcd_rollout_max_new_tokens=args.opcd_rollout_max_new_tokens,
            opcd_queries_per_session=args.opcd_queries_per_session,
        )
    else:
        model = HypernetModel.from_config(
            config,
            base_model_path=args.base_model_path,
            ctx_encoder_path=args.ctx_encoder_path,
            use_flash_attn=args.use_flash_attn,
            base_model_device_map=args.base_model_device_map,
            lora_dropout=args.lora_dropout,
            lora_alpha=args.lora_alpha,
            l1_reg_coef=args.l1_reg_coef,
            use_per_ctx_average_loss=args.use_per_ctx_average_loss,
            objective=args.objective,
            opcd_loss_mode=args.opcd_loss_mode,
            opcd_top_k=args.opcd_top_k,
            opcd_kl_chunk_size=args.opcd_kl_chunk_size,
            opcd_rollout_max_new_tokens=args.opcd_rollout_max_new_tokens,
            opcd_queries_per_session=args.opcd_queries_per_session,
        )

    if args.perceiver_init_checkpoint and resume_dir is None:
        logger.info(
            "Initializing target latent space from Perceiver checkpoint: %s",
            args.perceiver_init_checkpoint,
        )
        transfer_report = model.initialize_perceiver_from_checkpoint(
            args.perceiver_init_checkpoint
        )
    model.configure_trainable_scope(args.trainable_scope)
    if args.base_model_gradient_checkpointing:
        if not hasattr(model.base_model, "gradient_checkpointing_enable"):
            raise TypeError(
                "base model does not support gradient checkpointing: "
                f"{type(model.base_model).__name__}"
            )
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.base_model.config.use_cache = False
        text_config = getattr(model.base_model.config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False
        logger.info(
            "Frozen-backbone activation checkpointing enabled "
            "(use_reentrant=False)"
        )
    trainable_report = model.trainable_parameter_report()
    if is_main and args.perceiver_init_checkpoint:
        contract_path = Path(args.output_dir, "trainable_contract.json")
        contract = {
            "format": "memlora_latent_transfer_training_contract_v1",
            "perceiver_transfer": transfer_report,
            "perceiver_init_checkpoint": args.perceiver_init_checkpoint,
            "resumed_from_checkpoint": str(resume_dir) if resume_dir else None,
            "trainable_parameters": trainable_report,
        }
        if resume_dir is not None:
            if not contract_path.is_file():
                raise FileNotFoundError(
                    "resume requires the original trainable contract: "
                    f"{contract_path}"
                )
            existing_contract = json.loads(contract_path.read_text())
            expected = {
                "format": contract["format"],
                "perceiver_init_checkpoint": args.perceiver_init_checkpoint,
                "scope": args.trainable_scope,
            }
            actual = {
                "format": existing_contract.get("format"),
                "perceiver_init_checkpoint": existing_contract.get(
                    "perceiver_init_checkpoint"
                ),
                "scope": existing_contract.get("trainable_parameters", {}).get(
                    "scope"
                ),
            }
            if actual != expected:
                raise ValueError(
                    "resume trainable contract differs from the original run: "
                    f"expected={expected} actual={actual}"
                )
        else:
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n"
            )

    model_parallel_backbone = args.base_model_device_map is not None
    if model_parallel_backbone:
        if not use_accelerate:
            raise ValueError(
                "base_model_device_map requires Accelerate process management"
            )
        if accelerator.num_processes != 1 and not args.model_parallel_data_parallel:
            raise ValueError(
                "base_model_device_map requires one training process unless "
                "model_parallel_data_parallel is explicitly enabled"
            )
        if args.model_parallel_data_parallel:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            if local_world_size != 1:
                raise ValueError(
                    "model_parallel_data_parallel requires exactly one process "
                    "per node"
                )
        model.ctx_encoder.to(device)
        model.perceiver.to(device)
        model.head.to(device)
        logger.info(
            "Frozen backbone device map: %s (%d visible GPUs)",
            args.base_model_device_map,
            torch.cuda.device_count(),
        )
        if args.model_parallel_data_parallel:
            logger.info(
                "Model-parallel replicas: %d; trainable gradients are averaged "
                "explicitly across nodes",
                accelerator.num_processes,
            )
    else:
        model = model.to(device)
    logger.info(
        "Trainable scope: %s; params: %s",
        args.trainable_scope,
        f"{model.n_trainable_params:,}",
    )

    # Load data
    from rpmem.training.data import (
        HypernetDataset,
        collate_opcd_hypernet,
        collate_hypernet,
        packed_collate_hypernet,
    )
    from rpmem.training.generation_quality import (
        GenerationQualityStats,
        build_reference_generation_batch,
        score_generation_rows,
    )
    from rpmem.training.checkpointing import (
        atomic_torch_save,
        gather_rank_states,
        prune_checkpoints,
    )
    from rpmem.training.sample_store import open_sample_store
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    ctx_tokenizer = AutoTokenizer.from_pretrained(config.ctx_encoder_model_name)
    eos_token_ids = base_tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = getattr(model.base_model.config, "eos_token_id", None)
    if eos_token_ids is None:
        raise ValueError("base tokenizer/model does not define an EOS token")
    pad_token_id = base_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            eos_token_ids[0]
            if isinstance(eos_token_ids, (list, tuple))
            else eos_token_ids
        )
    if args.objective == "opcd":
        model.configure_opcd_token_ids(
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )
        if opcd_backend_factory is not None:
            backend = opcd_backend_factory(
                process_index=accelerator.process_index if use_accelerate else 0,
                layer_indices=config.layer_indices,
                lora_scaling=args.lora_alpha,
                max_new_tokens=args.opcd_rollout_max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_ids=model.opcd_eos_token_ids,
            )
            model.configure_opcd_backend(backend)
            logger.info(
                "OPCD external backend rank=%d backend=%s",
                accelerator.process_index if use_accelerate else 0,
                type(backend).__name__,
            )

    train_samples = open_sample_store(train_paths)
    logger.info(
        "Indexed %d training samples without materializing the rows",
        len(train_samples),
    )

    train_dataset = HypernetDataset(
        train_samples,
        base_tokenizer,
        ctx_tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_seq_len=args.max_seq_len,
        teacher_logprobs_dir=args.teacher_logprobs_dir,
        objective=args.objective,
        opcd_rollout_max_new_tokens=args.opcd_rollout_max_new_tokens,
        opcd_max_teacher_seq_len=args.opcd_max_teacher_seq_len,
    )
    if args.teacher_logprobs_dir:
        logger.info(
            f"Distillation mode: teacher logprobs from {args.teacher_logprobs_dir}"
        )
    logger.info("Phase 1 objective: %s", args.objective)

    if args.objective == "opcd":
        collate_fn = partial(
            collate_opcd_hypernet,
            pad_token_id=model.opcd_pad_token_id,
        )
    else:
        collate_fn = packed_collate_hypernet if args.use_packing else collate_hypernet
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        generator=train_generator,
    )

    # Optimizer (must be created before accelerator.prepare wraps model in DDP)
    trainable_params = model.trainable_parameters
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = args.max_steps
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # Accelerate wrapping
    if use_accelerate:
        if model_parallel_backbone:
            optimizer, train_loader, scheduler = accelerator.prepare(
                optimizer, train_loader, scheduler
            )
        else:
            model, optimizer, train_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, scheduler
            )

    batches_per_epoch = len(train_loader)
    if batches_per_epoch <= 0:
        raise ValueError("training data loader has no batches")
    from rpmem.training.sampling import DataLoaderPosition

    def capture_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device)
            if torch.cuda.is_available()
            else None,
        }

    def restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state(state["cuda"], device=device)

    resume_global_step = 0
    resume_rng_state = None
    resume_data_position = DataLoaderPosition.from_micro_batches(
        0,
        batches_per_epoch=batches_per_epoch,
    )
    if resume_dir is not None:
        training_state = torch.load(
            resume_dir / "training_state.pt",
            weights_only=False,
            map_location="cpu",
        )
        if int(training_state.get("max_steps", -1)) != args.max_steps:
            raise ValueError(
                "resume checkpoint max_steps does not match this run; scheduler "
                "semantics would change"
            )
        current_world_size = accelerator.num_processes if use_accelerate else 1
        if int(training_state.get("world_size", -1)) != current_world_size:
            raise ValueError(
                "resume checkpoint world_size does not match this run; global "
                "batch semantics would change"
            )
        resume_contract = {
            "seed": args.seed,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
        }
        for key, expected in resume_contract.items():
            if key in training_state and int(training_state[key]) != int(expected):
                raise ValueError(
                    f"resume checkpoint {key}={training_state[key]!r} does not "
                    f"match this run ({expected!r})"
                )
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        resume_global_step = int(training_state["global_step"])
        if all(
            key in training_state
            for key in (
                "data_epoch",
                "batches_consumed_in_epoch",
                "batches_per_epoch",
            )
        ):
            resume_data_position = DataLoaderPosition.from_state_dict(
                training_state,
                expected_batches_per_epoch=batches_per_epoch,
            )
        else:
            resume_data_position = DataLoaderPosition.from_micro_batches(
                resume_global_step * args.gradient_accumulation_steps,
                batches_per_epoch=batches_per_epoch,
            )

        process_index = accelerator.process_index if use_accelerate else 0
        rng_path = resume_dir / f"rng_state_rank{process_index:05d}.pt"
        if not rng_path.is_file():
            raise FileNotFoundError(f"rank RNG state not found: {rng_path}")
        resume_rng_state = torch.load(
            rng_path,
            weights_only=False,
            map_location="cpu",
        )
        del training_state
        logger.info("Resumed optimizer/scheduler at global step %d", resume_global_step)
        logger.info(
            "Resumed data loader at epoch %d, batch %d/%d",
            resume_data_position.epoch,
            resume_data_position.batches_consumed_in_epoch,
            resume_data_position.batches_per_epoch,
        )

    # Validation data
    val_loader = None
    if args.val_data:
        val_samples = open_sample_store(val_paths)
        logger.info(
            "Indexed %d validation samples without materializing the rows",
            len(val_samples),
        )
        if args.quality_eval_sessions > len(val_samples):
            raise ValueError(
                "quality_eval_sessions exceeds the held-out validation split: "
                f"{args.quality_eval_sessions} > {len(val_samples)}"
            )
        val_dataset = HypernetDataset(
            val_samples,
            base_tokenizer,
            ctx_tokenizer,
            max_ctx_len=args.max_ctx_len,
            max_seq_len=args.max_seq_len,
            teacher_logprobs_dir=args.val_teacher_logprobs_dir,
            objective=args.objective,
            opcd_rollout_max_new_tokens=args.opcd_rollout_max_new_tokens,
            opcd_max_teacher_seq_len=args.opcd_max_teacher_seq_len,
            quality_eval_sessions=(
                args.quality_eval_sessions
                if args.quality_eval_sessions and args.objective != "opcd"
                else 0
            ),
            quality_eval_max_new_tokens=(
                args.quality_eval_max_new_tokens
                if args.quality_eval_sessions and args.objective != "opcd"
                else 0
            ),
            validation_order_seed=args.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=max(0, args.dataloader_num_workers // 2),
            pin_memory=True,
        )
        if use_accelerate:
            val_loader = accelerator.prepare(val_loader)

    # TensorBoard
    tb_writer = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = (
            args.tensorboard_dir
            or os.environ.get("TENSORBOARD_LOGGING_DIR")
            or os.environ.get("RPMEM_TENSORBOARD_DIR")
            or os.environ.get("MEMLORA_TENSORBOARD_DIR")
            or f"{args.output_dir}/tb_logs"
        )
        tb_writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logging to: {tb_dir}")

    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
        resume_rng_state = None

    # Training loop
    model.train()
    global_step = resume_global_step
    micro_step = resume_data_position.total_micro_batches
    data_epoch = resume_data_position.epoch
    batches_consumed_in_epoch = resume_data_position.batches_consumed_in_epoch
    running_loss = 0.0
    running_loss_count = 0
    log_loss = 0.0
    log_loss_count = 0
    log_metric_sums = {
        "task_loss": 0.0,
        "l1_norm": 0.0,
        "l1_loss": 0.0,
        "total_loss": 0.0,
        "distill_mode": 0.0,
        "teacher_tail_mass": 0.0,
        "teacher_tail_correction": 0.0,
        "teacher_tail_correction_max": 0.0,
        "teacher_support_duplicate_fraction": 0.0,
        "teacher_support_renormalization": 0.0,
        "teacher_support_renormalization_max": 0.0,
        "student_tail_mass": 0.0,
        "opcd_reverse_kl": 0.0,
        "student_entropy": 0.0,
        "teacher_entropy": 0.0,
        "teacher_context_lift": 0.0,
        "rollout_length": 0.0,
        "rollout_eos_rate": 0.0,
        "rollout_tokens": 0.0,
        "context_tokens": 0.0,
        "student_prompt_tokens": 0.0,
        "teacher_scoring_tokens": 0.0,
        "selected_queries": 0.0,
        "probe_rotation_offset": 0.0,
        "teacher_response_projection": 0.0,
        "student_response_projection": 0.0,
        "lora_l2_norm": 0.0,
        "opcd_remote_backend": 0.0,
        "rollout_service_seconds": 0.0,
        "teacher_service_seconds": 0.0,
    }
    log_metric_counts = {key: 0 for key in log_metric_sums}
    last_grad_norm = None
    t0 = time.time()

    logger.info(
        f"Starting training: {total_steps} optimizer steps, "
        f"grad_accum={args.gradient_accumulation_steps}"
    )
    run_until_step = (
        args.selected_endpoint_step
        or args.preflight_stop_after_step
        or total_steps
    )

    from rpmem.training.opcd import deterministic_rollout_seed

    process_index = accelerator.process_index if use_accelerate else 0
    last_saved_step = resume_global_step if args.resume_from_checkpoint else -1
    last_validated_step = -1

    def reduce_metric_window(
        loss_sum: float,
        loss_count: int,
        metric_sums: dict[str, float],
        metric_counts: dict[str, int],
    ) -> tuple[float, dict[str, float]]:
        """Reduce scalar sums/counts across ranks without averaging ranks twice."""

        keys = tuple(metric_sums)
        values = [loss_sum, float(loss_count)]
        values.extend(metric_sums[key] for key in keys)
        values.extend(float(metric_counts[key]) for key in keys)
        reduced = torch.tensor(values, dtype=torch.float64, device=device)
        if use_accelerate:
            reduced = accelerator.reduce(reduced, reduction="sum")
        reduced_values = reduced.cpu().tolist()
        global_loss = reduced_values[0] / max(reduced_values[1], 1.0)
        metric_offset = 2
        count_offset = metric_offset + len(keys)
        averages = {}
        for index, key in enumerate(keys):
            count = reduced_values[count_offset + index]
            if count:
                averages[key] = reduced_values[metric_offset + index] / count
        return global_loss, averages

    def build_model_kwargs(
        batch,
        *,
        corpus_pass: int,
        rollout_seed: int | None = None,
        measure_context_lift: bool = False,
    ):
        common = {
            "ctx_ids": batch["ctx_ids"],
            "ctx_attn_mask": batch.get("ctx_attn_mask"),
            "ctx_position_ids": batch.get("ctx_position_ids"),
            "n_ctx_chunks": batch.get("n_ctx_chunks"),
            "n_queries": batch.get("n_queries"),
        }
        if args.objective == "opcd":
            common.update(
                {
                    "sample_indices": batch["sample_indices"],
                    "opcd_student_prompt_ids": batch["opcd_student_prompt_ids"],
                    "opcd_student_prompt_attention_mask": batch[
                        "opcd_student_prompt_attention_mask"
                    ],
                    "opcd_teacher_prompt_ids": batch["opcd_teacher_prompt_ids"],
                    "opcd_teacher_prompt_attention_mask": batch[
                        "opcd_teacher_prompt_attention_mask"
                    ],
                    "opcd_corpus_pass": corpus_pass,
                    "opcd_rollout_seed": rollout_seed,
                    "opcd_measure_context_lift": measure_context_lift,
                }
            )
        else:
            common.update(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch.get("attention_mask"),
                    "position_ids": batch.get("position_ids"),
                    "labels": batch["labels"],
                    "logprobs_vals": batch.get("logprobs_vals"),
                    "logprobs_indices": batch.get("logprobs_indices"),
                }
            )
        return common

    def save_training_checkpoint(
        step: int,
        completed_micro_batches: int,
        *,
        epoch_boundary: bool = False,
        completed_epochs: int = 0,
    ) -> Path:
        nonlocal last_saved_step
        if use_accelerate:
            accelerator.wait_for_everyone()
        save_path = Path(args.output_dir) / f"checkpoint-{step}"
        rank_rng_states = None
        if args.save_training_state:
            process_index = accelerator.process_index if use_accelerate else 0
            world_size = accelerator.num_processes if use_accelerate else 1
            rank_rng_states = gather_rank_states(
                capture_rng_state(),
                rank=process_index,
                world_size=world_size,
            )
        if is_main:
            save_path.mkdir(parents=True, exist_ok=True)
            if args.save_training_state:
                if rank_rng_states is None:
                    raise RuntimeError("rank zero did not receive distributed RNG states")
                for rank, rng_state in enumerate(rank_rng_states):
                    atomic_torch_save(
                        rng_state,
                        save_path / f"rng_state_rank{rank:05d}.pt",
                    )
            unwrapped = accelerator.unwrap_model(model) if use_accelerate else model
            unwrapped.save_checkpoint(save_path / "pytorch_model.bin")
            if args.save_training_state:
                atomic_torch_save(
                    {
                        "format_version": 1,
                        "global_step": step,
                        "max_steps": args.max_steps,
                        "world_size": accelerator.num_processes
                        if use_accelerate
                        else 1,
                        "seed": args.seed,
                        "gradient_accumulation_steps": (
                            args.gradient_accumulation_steps
                        ),
                        "per_device_train_batch_size": (
                            args.per_device_train_batch_size
                        ),
                        "micro_step": completed_micro_batches,
                        "epoch_boundary": bool(epoch_boundary),
                        "completed_epochs": int(completed_epochs),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        **DataLoaderPosition.from_micro_batches(
                            completed_micro_batches,
                            batches_per_epoch=batches_per_epoch,
                        ).state_dict(),
                    },
                    save_path / "training_state.pt",
                )
            (save_path / "checkpoint_complete.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "global_step": int(step),
                        "epoch_boundary": bool(epoch_boundary),
                        "completed_epochs": int(completed_epochs),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            logger.info("Saved checkpoint: %s", save_path)
            removed = prune_checkpoints(
                args.output_dir,
                save_total_limit=args.save_total_limit,
                preserve_epoch_boundaries=args.save_epoch_boundaries,
            )
            for removed_path in removed:
                logger.info("Pruned old checkpoint: %s", removed_path)
        if use_accelerate:
            accelerator.wait_for_everyone()
        last_saved_step = step
        return save_path

    def run_validation(step: int, *, label: str) -> None:
        nonlocal last_validated_step
        if val_loader is None:
            return

        training_rng_state = capture_rng_state()
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_metric_sums = {key: 0.0 for key in log_metric_sums}
        val_metric_counts = {key: 0 for key in log_metric_sums}
        quality_stats = GenerationQualityStats()
        quality_session_count = 0
        quality_sample_indices: set[int] = set()
        try:
            with torch.no_grad():
                for val_batch_index, val_batch in enumerate(val_loader):
                    if not use_accelerate:
                        val_batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()
                        }
                    val_seed = None
                    if args.objective == "opcd":
                        val_seed = deterministic_rollout_seed(
                            args.seed,
                            process_index,
                            val_batch_index,
                            validation=True,
                        )
                    val_kwargs = build_model_kwargs(
                        val_batch,
                        corpus_pass=0,
                        rollout_seed=val_seed,
                        measure_context_lift=(
                            args.objective == "opcd"
                            and val_batch_index < args.opcd_context_lift_eval_batches
                        ),
                    )
                    v_loss, _, generated_loras = model(**val_kwargs)
                    val_loss_sum += v_loss.item()
                    val_steps += 1
                    metrics_model = model.module if hasattr(model, "module") else model
                    step_metrics = getattr(metrics_model, "last_loss_metrics", {})
                    if step_metrics:
                        for key in val_metric_sums:
                            value = step_metrics.get(key)
                            if value is not None:
                                val_metric_sums[key] += float(
                                    value.detach().float().item()
                                )
                                val_metric_counts[key] += 1
                    if hasattr(model, "_reset_lora_bindings"):
                        model._reset_lora_bindings()
                    elif hasattr(model, "module") and hasattr(
                        model.module, "_reset_lora_bindings"
                    ):
                        model.module._reset_lora_bindings()
                    if args.quality_eval_sessions and args.objective != "opcd":
                        sample_indices = [
                            int(value)
                            for value in val_batch["sample_indices"]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        quality_selected = [
                            bool(value)
                            for value in val_batch["quality_selected"]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        logical_indices = [
                            int(value)
                            for value in val_batch["logical_indices"]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        quality_world_size = (
                            accelerator.num_processes if use_accelerate else 1
                        )
                        eligible = [
                            value
                            for value, logical_index, selected in zip(
                                sample_indices,
                                logical_indices,
                                quality_selected,
                                strict=True,
                            )
                            if selected
                            and logical_index % quality_world_size == process_index
                            and value not in quality_sample_indices
                        ]
                        if eligible:
                            if len(eligible) != len(sample_indices):
                                raise RuntimeError(
                                    "quality generation requires one eligible held-out "
                                    "session per validation batch"
                                )
                            prompt_ids, prompt_mask, references, quality_n_queries = (
                                build_reference_generation_batch(
                                    prompt_rows=val_batch["quality_prompt_ids"],
                                    reference_rows=val_batch["quality_reference_ids"],
                                    n_queries=val_batch["n_queries"],
                                    sample_indices=val_batch["sample_indices"],
                                    queries_per_session=(
                                        args.quality_eval_queries_per_session
                                    ),
                                    pad_token_id=pad_token_id,
                                )
                            )
                            unwrapped = (
                                accelerator.unwrap_model(model)
                                if use_accelerate
                                else model
                            )
                            generated_rows = unwrapped.generate_with_loras(
                                generated_loras=generated_loras,
                                n_ctx_chunks=val_batch["n_ctx_chunks"],
                                n_queries=quality_n_queries,
                                prompt_ids=prompt_ids,
                                prompt_attention_mask=prompt_mask,
                                max_new_tokens=args.quality_eval_max_new_tokens,
                                pad_token_id=pad_token_id,
                                eos_token_ids=eos_token_ids,
                            )
                            quality_stats.merge(
                                score_generation_rows(
                                    list(generated_rows),
                                    references,
                                    tokenizer=base_tokenizer,
                                    eos_token_ids=eos_token_ids,
                                )
                            )
                            quality_sample_indices.update(eligible)
                            quality_session_count += len(eligible)
            avg_val_loss, val_metric_averages = reduce_metric_window(
                val_loss_sum,
                val_steps,
                val_metric_sums,
                val_metric_counts,
            )
            quality_values = torch.tensor(
                quality_stats.raw_values() + [float(quality_session_count)],
                dtype=torch.float64,
                device=device,
            )
            if use_accelerate:
                quality_values = accelerator.reduce(quality_values, reduction="sum")
            quality_values = quality_values.cpu().tolist()
            quality_stats = GenerationQualityStats.from_raw_values(quality_values[:-1])
            quality_metrics = quality_stats.metrics()
            if quality_metrics:
                quality_metrics["quality/session_count"] = quality_values[-1]
                if int(round(quality_values[-1])) != args.quality_eval_sessions:
                    raise RuntimeError(
                        "quality validation session count mismatch after distributed "
                        f"deduplication: {quality_values[-1]} != "
                        f"{args.quality_eval_sessions}"
                    )
            if is_main:
                logger.info("%s | val_loss=%.4f", label, avg_val_loss)
                if quality_metrics:
                    logger.info(
                        "%s | quality_em=%.4f quality_f1=%.4f "
                        "quality_rouge_l=%.4f queries=%d",
                        label,
                        quality_metrics["quality/normalized_exact_match"],
                        quality_metrics["quality/token_f1"],
                        quality_metrics["quality/rouge_l"],
                        int(quality_metrics["quality/query_count"]),
                    )
                validation_record = {
                    "step": int(step),
                    "label": label,
                    "val_loss": float(avg_val_loss),
                    "metrics": {
                        key: float(value) for key, value in val_metric_averages.items()
                    },
                    "quality_metrics": {
                        key: float(value) for key, value in quality_metrics.items()
                    },
                }
                with (Path(args.output_dir) / "validation_history.jsonl").open(
                    "a"
                ) as handle:
                    handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
            if tb_writer:
                tb_writer.add_scalar("val/loss", avg_val_loss, step)
                if args.objective == "sft":
                    tb_writer.add_scalar(
                        "val/perplexity",
                        math.exp(min(avg_val_loss, 20)),
                        step,
                    )
                for key, value in val_metric_averages.items():
                    tb_writer.add_scalar(f"val/{key}", value, step)
                for key, value in quality_metrics.items():
                    tb_writer.add_scalar(f"val/{key}", value, step)
                tb_writer.flush()
            last_validated_step = step
        finally:
            model.train()
            restore_rng_state(training_rng_state)

    while global_step < run_until_step:
        epoch_loader = train_loader
        if batches_consumed_in_epoch:
            if not use_accelerate:
                raise RuntimeError(
                    "exact mid-epoch resume requires Accelerate's "
                    "skip_first_batches support"
                )
            epoch_loader = accelerator.skip_first_batches(
                train_loader,
                num_batches=batches_consumed_in_epoch,
            )
        if hasattr(epoch_loader, "set_epoch"):
            epoch_loader.set_epoch(data_epoch)
        completed_epoch = True
        for batch in epoch_loader:
            if global_step >= run_until_step:
                completed_epoch = False
                break

            if not use_accelerate:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            # Forward
            rollout_seed = None
            measure_context_lift = False
            if args.objective == "opcd":
                rollout_seed = deterministic_rollout_seed(
                    args.seed,
                    process_index,
                    micro_step,
                )
                interval = (
                    args.opcd_context_lift_every * args.gradient_accumulation_steps
                )
                measure_context_lift = bool(
                    interval and (micro_step + 1) % interval == 0
                )
            ctx_kwargs = build_model_kwargs(
                batch,
                corpus_pass=data_epoch,
                rollout_seed=rollout_seed,
                measure_context_lift=measure_context_lift,
            )

            did_optimizer_step = False
            if use_accelerate:
                with accelerator.accumulate(model):
                    loss, logits, _ = model(**ctx_kwargs)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if args.model_parallel_data_parallel:
                            from rpmem.training.distributed import (
                                average_parameter_gradients,
                            )

                            sync_stats = average_parameter_gradients(trainable_params)
                            if global_step == resume_global_step and is_main:
                                logger.info(
                                    "Synchronized trainable gradients: %s",
                                    sync_stats,
                                )
                        grad_norm = accelerator.clip_grad_norm_(
                            trainable_params, args.max_grad_norm
                        )
                        if grad_norm is not None:
                            last_grad_norm = float(grad_norm.detach().float().item())
                        did_optimizer_step = True
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                micro_step += 1
            else:
                loss, logits, _ = model(**ctx_kwargs)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                micro_step += 1

                if micro_step % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )
                    last_grad_norm = float(grad_norm.detach().float().item())
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    did_optimizer_step = True

            batches_consumed_in_epoch += 1

            running_loss += loss.item()
            running_loss_count += 1
            log_loss += loss.item()
            log_loss_count += 1
            metrics_model = model.module if hasattr(model, "module") else model
            step_metrics = getattr(metrics_model, "last_loss_metrics", {})
            if step_metrics:
                for key in log_metric_sums:
                    value = step_metrics.get(key)
                    if value is not None:
                        log_metric_sums[key] += float(value.detach().float().item())
                        log_metric_counts[key] += 1
            if did_optimizer_step:
                global_step += 1

            # Reset LoRA bindings after each step
            if hasattr(model, "_reset_lora_bindings"):
                model._reset_lora_bindings()
            elif hasattr(model, "module") and hasattr(
                model.module, "_reset_lora_bindings"
            ):
                model.module._reset_lora_bindings()

            if not did_optimizer_step:
                continue

            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss, metric_averages = reduce_metric_window(
                    log_loss,
                    log_loss_count,
                    log_metric_sums,
                    log_metric_counts,
                )
                if is_main:
                    elapsed = time.time() - t0
                    steps_per_sec = (global_step - resume_global_step) / elapsed
                    lr_now = (
                        scheduler.get_last_lr()[0]
                        if hasattr(scheduler, "get_last_lr")
                        else args.lr
                    )
                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                        f"speed={steps_per_sec:.1f} steps/s"
                    )
                    if tb_writer:
                        tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        tb_writer.add_scalar("train/lr", lr_now, global_step)
                        tb_writer.add_scalar(
                            "train/speed_steps_per_sec", steps_per_sec, global_step
                        )
                        if args.objective == "sft":
                            tb_writer.add_scalar(
                                "train/perplexity",
                                math.exp(min(avg_loss, 20)),
                                global_step,
                            )
                        for key, value in metric_averages.items():
                            tb_writer.add_scalar(f"train/{key}", value, global_step)
                        if args.objective == "opcd":
                            exposure_scale = args.gradient_accumulation_steps * (
                                accelerator.num_processes if use_accelerate else 1
                            )
                            for key in (
                                "context_tokens",
                                "rollout_tokens",
                                "student_prompt_tokens",
                                "teacher_scoring_tokens",
                            ):
                                if key in metric_averages:
                                    tb_writer.add_scalar(
                                        f"train/{key}_per_optimizer_update",
                                        metric_averages[key] * exposure_scale,
                                        global_step,
                                    )
                        if last_grad_norm is not None:
                            tb_writer.add_scalar(
                                "train/grad_norm", last_grad_norm, global_step
                            )
                log_loss = 0.0
                log_loss_count = 0
                for key in log_metric_sums:
                    log_metric_sums[key] = 0.0
                    log_metric_counts[key] = 0

            epoch_boundary = batches_consumed_in_epoch == batches_per_epoch
            completed_epochs = data_epoch + int(epoch_boundary)
            periodic_checkpoint = (
                args.save_steps > 0 and global_step % args.save_steps == 0
            )
            if epoch_boundary and tb_writer:
                tb_writer.add_scalar(
                    "progress/completed_epochs",
                    completed_epochs,
                    global_step,
                )
                tb_writer.flush()

            epoch_checkpoint = epoch_boundary and args.save_epoch_boundaries
            if periodic_checkpoint or epoch_checkpoint:
                save_training_checkpoint(
                    global_step,
                    micro_step,
                    epoch_boundary=epoch_boundary,
                    completed_epochs=completed_epochs,
                )
            periodic_validation = (
                args.eval_steps > 0 and global_step % args.eval_steps == 0
            )
            if periodic_validation or epoch_boundary:
                run_validation(
                    global_step,
                    label=(
                        f"Epoch {completed_epochs} complete (step {global_step})"
                        if epoch_boundary
                        else f"Step {global_step}"
                    ),
                )

        if completed_epoch:
            data_epoch += 1
            batches_consumed_in_epoch = 0

    if last_validated_step != global_step:
        run_validation(global_step, label="Final")

    if args.save_training_state and last_saved_step != global_step:
        save_training_checkpoint(
            global_step,
            micro_step,
            completed_epochs=data_epoch,
        )

    # Save final
    avg_running_loss, _ = reduce_metric_window(
        running_loss,
        running_loss_count,
        {},
        {},
    )
    if use_accelerate:
        accelerator.wait_for_everyone()
    if is_main:
        final_path = f"{args.output_dir}/pytorch_model.bin"
        unwrapped = accelerator.unwrap_model(model) if use_accelerate else model
        unwrapped.save_checkpoint(final_path)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
            memory_profile = {
                "format": "memlora_training_cuda_memory_v1",
                "base_model_device_map": args.base_model_device_map,
                "devices": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
            Path(args.output_dir, "cuda_memory.json").write_text(
                json.dumps(memory_profile, indent=2, sort_keys=True) + "\n"
            )
        logger.info(f"Training complete. Final checkpoint: {final_path}")
        logger.info(
            f"Total time: {time.time() - t0:.1f}s, avg_loss: {avg_running_loss:.4f}"
        )
        if args.selected_endpoint_step is not None:
            logger.info(
                "Selected endpoint reached at step %d; resumable checkpoint-%d "
                "is complete",
                global_step,
                global_step,
            )
        elif args.preflight_stop_after_step is not None:
            logger.info(
                "Preflight stopped intentionally at step %d; resume from checkpoint-%d",
                global_step,
                global_step,
            )

    if tb_writer:
        tb_writer.close()
    if use_accelerate:
        accelerator.end_training()


if __name__ == "__main__":
    main()
