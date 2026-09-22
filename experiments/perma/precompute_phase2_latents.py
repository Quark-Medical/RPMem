"""Precompute bounded PERMA memory-segment latents for Phase 2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_adapter import ALL_USER_IDS, PERMA_VARIANTS, load_tasks
from rpmem.training.hypernet_model import HypernetModel
from perma_segments import segmentation_contract, segment_task_memory
from phase2_fusion_utils import select_memory_segments


LATENT_FORMAT = "memlora_perma_segment_latents_v2"


def encode_token_ids(
    model: HypernetModel,
    token_ids: list[int],
    max_ctx_tokens: int,
):
    if len(token_ids) > max_ctx_tokens:
        raise ValueError(
            f"memory segment has {len(token_ids)} tokens, limit={max_ctx_tokens}"
        )
    ctx_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    ctx_attn_mask = torch.ones_like(ctx_ids)
    with torch.inference_mode():
        return model.encode_context(ctx_ids, ctx_attn_mask).detach().cpu()


def encode_segment(model: HypernetModel, tokenizer, segment, max_ctx_tokens: int):
    if hasattr(segment, "token_ids"):
        token_ids = list(segment.token_ids)
    else:
        token_ids = tokenizer.encode(
            segment.text,
            add_special_tokens=True,
            truncation=False,
        )
    return encode_token_ids(model, token_ids, max_ctx_tokens)


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cached_latent_is_valid(path: Path) -> bool:
    try:
        latent = torch.load(
            path,
            weights_only=True,
            map_location="cpu",
        )
    except Exception as error:
        print(
            f"[precompute] removing invalid latent cache: "
            f"{path} ({type(error).__name__}: {error})",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return False
    if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
        print(
            f"[precompute] removing invalid latent cache: "
            f"{path} (expected non-empty tensor)",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return False
    return True


def save_latent_atomic(latent: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(latent, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument("--ctx_encoder_path", default=None)
    parser.add_argument("--checkpoint_sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="Reuse latents from an interrupted run with the same inputs.")
    parser.add_argument("--phase1_method", default="")
    parser.add_argument("--variant", default="clean_sd", choices=sorted(PERMA_VARIANTS))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_ctx_tokens", type=int, default=4096)
    parser.add_argument("--event_overlap", type=int, default=1)
    parser.add_argument(
        "--memory_selection",
        choices=('all',),
        default="all",
    )
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument(
        "--user_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit PERMA user ids to precompute. Overrides --max_users.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--base_model_device_map",
        default="",
        help="Optional Transformers device_map for model-parallel checkpoint loading.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    model = HypernetModel.from_checkpoint(
        args.checkpoint,
        base_model_path=args.base_model_path,
        ctx_encoder_path=args.ctx_encoder_path,
        use_flash_attn=args.use_flash_attn,
        train=False,
        base_model_device_map=args.base_model_device_map or None,
    )
    model.eval()
    if args.base_model_device_map:
        hypernet_device = torch.device("cuda:0")
        model.ctx_encoder.to(hypernet_device)
        model.perceiver.to(hypernet_device)
        model.head.to(hypernet_device)
    else:
        model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model.config.ctx_encoder_model_name)
    tokenizer.model_max_length = max(
        int(tokenizer.model_max_length),
        1_000_000_000,
    )
    if args.user_ids:
        unknown = sorted(set(args.user_ids) - set(ALL_USER_IDS))
        if unknown:
            raise ValueError(f"unknown PERMA user ids: {unknown}")
        user_ids = args.user_ids
    else:
        user_ids = (
            ALL_USER_IDS
            if args.max_users <= 0
            else ALL_USER_IDS[: args.max_users]
        )
    all_tasks = load_tasks(user_ids=user_ids, variant=args.variant)
    tasks = [
        task
        for idx, task in enumerate(all_tasks)
        if idx % args.num_shards == args.shard_id
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = segmentation_contract(
        max_context_tokens=args.max_ctx_tokens,
        event_overlap=args.event_overlap,
    )

    saved_segments = 0
    encoded_segments = 0
    reused_segments = 0
    total_episodes = 0
    split_episodes = 0
    split_messages = 0
    message_parts = 0
    truncated_events = 0
    dropped_empty_messages = 0
    empty_episodes = 0
    original_content_tokens = 0
    retained_content_tokens = 0
    encode_seconds = 0.0
    durable_encode_seconds = 0.0
    run_started = time.perf_counter()
    for task in tasks:
        task_started = time.perf_counter()
        task_dir = (
            out_dir
            / f"user{task.user_id}"
            / f"{task.task_id}_type{task.task_type}"
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        meta_path = task_dir / "meta.json"
        existing = {}
        if args.resume and meta_path.is_file():
            existing = json.loads(meta_path.read_text())
            if (
                existing.get("latent_format") != LATENT_FORMAT
                or {k: v for k, v in existing.get("segmentation", {}).items() if k != "sha256"}
                != {k: v for k, v in contract.items() if k != "sha256"}
                or existing.get("memory_selection", "all")
                != args.memory_selection
            ):
                raise ValueError(
                    "latent cache policy mismatch; choose a new --output_dir: "
                    f"{task_dir}"
                )

        segmentation_started = time.perf_counter()
        source_segments, segment_stats = segment_task_memory(
            task,
            tokenizer,
            max_context_tokens=args.max_ctx_tokens,
            event_overlap=args.event_overlap,
        )
        segmentation_seconds = time.perf_counter() - segmentation_started
        segments = (
            select_memory_segments(source_segments, args.memory_selection)
        )
        total_episodes += segment_stats["episodes"]
        split_episodes += segment_stats["split_episodes"]
        split_messages += segment_stats["split_messages"]
        message_parts += segment_stats["message_parts"]
        truncated_events += segment_stats["truncated_events"]
        dropped_empty_messages += segment_stats["dropped_empty_messages"]
        empty_episodes += segment_stats["empty_episodes"]
        original_content_tokens += segment_stats["original_content_tokens"]
        retained_content_tokens += segment_stats["retained_content_tokens"]

        segment_metadata = []
        previous_segments = existing.get("memory_segments", [])
        task_encode_seconds = 0.0
        task_encoded_segments = 0
        task_reused_segments = 0
        for flat_index, segment in enumerate(segments):
            path = task_dir / f"memory_segment_{flat_index:05d}.pt"
            metadata = segment.metadata()
            if args.resume and path.exists() and cached_latent_is_valid(path):
                saved_segments += 1
                reused_segments += 1
                task_reused_segments += 1
                previous_segment = (
                    previous_segments[flat_index]
                    if flat_index < len(previous_segments)
                    else {}
                )
                previous_performance = (
                    previous_segment.get("performance", {})
                    if isinstance(previous_segment, dict)
                    else {}
                )
                metadata["performance"] = {
                    **previous_performance,
                    "cache_reused_this_run": True,
                }
                segment_metadata.append(metadata)
                continue
            encode_baseline = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                encode_baseline = int(torch.cuda.memory_allocated())
            synchronize_device()
            started = time.perf_counter()
            emb = encode_segment(
                model,
                tokenizer,
                segment,
                args.max_ctx_tokens,
            )
            synchronize_device()
            segment_encode_seconds = time.perf_counter() - started
            segment_peak_delta_bytes = (
                max(0, int(torch.cuda.max_memory_allocated()) - encode_baseline)
                if torch.cuda.is_available()
                else 0
            )
            save_latent_atomic(emb, path)
            saved_segments += 1
            encoded_segments += 1
            task_encoded_segments += 1
            encode_seconds += segment_encode_seconds
            task_encode_seconds += segment_encode_seconds
            metadata["performance"] = {
                "context_encode_seconds": segment_encode_seconds,
                "context_encode_peak_delta_bytes": segment_peak_delta_bytes,
                "latent_bytes": emb.numel() * emb.element_size(),
                "cache_reused_this_run": False,
            }
            segment_metadata.append(metadata)

        durable_task_encode_seconds = sum(
            float(segment.get("performance", {}).get("context_encode_seconds", 0.0))
            for segment in segment_metadata
        )
        durable_encode_seconds += durable_task_encode_seconds
        task_peak_delta_bytes = max(
            (
                int(segment.get("performance", {}).get(
                    "context_encode_peak_delta_bytes", 0
                ))
                for segment in segment_metadata
            ),
            default=0,
        )

        meta = {
            "latent_format": LATENT_FORMAT,
            "segmentation": contract,
            "user_id": task.user_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "variant": args.variant,
            "question": task.question,
            "options": task.options,
            "gold_label": task.gold_label,
            "num_sessions": len(task.sessions),
            "num_episodes": len(task.sessions),
            "num_memory_segments": len(segments),
            "num_source_memory_segments": len(source_segments),
            "memory_selection": args.memory_selection,
            "memory_segments": segment_metadata,
            "segmentation_stats": segment_stats,
            "phase1_method": args.phase1_method,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": args.checkpoint_sha256,
            "performance": {
                "segmentation_seconds": segmentation_seconds,
                "context_encode_seconds": durable_task_encode_seconds,
                "context_encode_seconds_this_run": task_encode_seconds,
                "task_wall_seconds": time.perf_counter() - task_started,
                "encoded_segments": task_encoded_segments,
                "reused_segments": task_reused_segments,
                "context_encode_peak_delta_bytes": task_peak_delta_bytes,
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    summary = {
        "latent_format": LATENT_FORMAT,
        "segmentation": contract,
        "memory_selection": args.memory_selection,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase1_method": args.phase1_method,
        "variant": args.variant,
        "output_dir": str(out_dir),
        "tasks_total": len(all_tasks),
        "tasks_this_shard": len(tasks),
        "users": sorted({task.user_id for task in tasks}),
        "episodes": total_episodes,
        "saved_segments": saved_segments,
        "encoded_segments": encoded_segments,
        "reused_segments": reused_segments,
        "split_episodes": split_episodes,
        "split_messages": split_messages,
        "message_parts": message_parts,
        "truncated_events": truncated_events,
        "dropped_empty_messages": dropped_empty_messages,
        "empty_episodes": empty_episodes,
        "original_content_tokens": original_content_tokens,
        "retained_content_tokens": retained_content_tokens,
        "content_token_coverage": (
            retained_content_tokens / original_content_tokens
            if original_content_tokens
            else 1.0
        ),
        "max_ctx_tokens": args.max_ctx_tokens,
        "event_overlap": args.event_overlap,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "performance": {
            "context_encode_seconds": durable_encode_seconds,
            "context_encode_seconds_this_run": encode_seconds,
            "wall_seconds": time.perf_counter() - run_started,
        },
    }
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_id:02d}"
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
