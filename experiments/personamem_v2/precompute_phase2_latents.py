"""Precompute PersonaMem-v2 history latents with the frozen Phase-1 compiler."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from evaluation_utils import history_cache_key
from formal_contract import LATENT_FORMAT, PHASE1_METHOD, segmentation_contract
from formal_data import (
    load_history,
    sha256_file,
    validate_dataset_root,
    write_json_atomic,
)
from rpmem.training.hypernet_model import HypernetModel
from personamem_segments import segment_history_memory


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cached_latent_is_valid(path: Path) -> bool:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        path.unlink(missing_ok=True)
        return False
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        path.unlink(missing_ok=True)
        return False
    return True


def save_latent_atomic(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_segment(model, tokenizer, segment, max_context_tokens: int) -> torch.Tensor:
    token_ids = (
        list(segment.token_ids)
        if hasattr(segment, "token_ids")
        else list(
            tokenizer.encode(
                segment.text,
                add_special_tokens=True,
                truncation=False,
            )
        )
    )
    if len(token_ids) > max_context_tokens:
        raise ValueError(
            f"history segment has {len(token_ids)} tokens, limit={max_context_tokens}"
        )
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        return model.encode_context(ids, torch.ones_like(ids)).detach().cpu()


def unique_histories(split_rows: dict[str, list[dict]]) -> dict[str, str]:
    histories: dict[str, str] = {}
    for rows in split_rows.values():
        for row in rows:
            relative = str(row["chat_history_32k_file"])
            persona = str(row["persona_id"])
            previous = histories.setdefault(relative, persona)
            if previous != persona:
                raise ValueError(
                    "history maps to multiple personas: "
                    f"{relative}: {previous}, {persona}"
                )
    return histories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--ctx-encoder-path", default=None)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--data-root", default="data/personamem_v2/formal_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--event-overlap", type=int, default=1)
    parser.add_argument("--memory-selection", choices=('all',), default="all")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid latent shard")
    if args.phase1_method != PHASE1_METHOD:
        raise ValueError("PersonaMem-v2 requires the formal Offline-FKL compiler")
    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Phase-1 checkpoint SHA mismatch")

    data_root = Path(args.data_root).resolve()
    freeze, split_rows = validate_dataset_root(
        data_root,
        verify_source_files=False,
    )
    histories = unique_histories(split_rows)
    selected = [
        (relative, histories[relative])
        for index, relative in enumerate(sorted(histories))
        if index % args.num_shards == args.shard_id
    ]
    model = HypernetModel.from_checkpoint(
        str(checkpoint),
        base_model_path=args.base_model_path,
        ctx_encoder_path=args.ctx_encoder_path,
        use_flash_attn=args.use_flash_attn,
        train=False,
    )
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model.config.ctx_encoder_model_name)
    tokenizer.model_max_length = max(int(tokenizer.model_max_length), 1_000_000_000)
    contract = (
        segmentation_contract(
            max_context_tokens=args.max_context_tokens,
            event_overlap=args.event_overlap,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_segments = 0
    reused_segments = 0
    started_run = time.perf_counter()
    for completed, (relative, persona) in enumerate(selected, start=1):
        history_dir = output_dir / history_cache_key(relative)
        history_dir.mkdir(parents=True, exist_ok=True)
        meta_path = history_dir / "meta.json"
        existing = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        if existing and any(
            (
                existing.get("latent_format") != LATENT_FORMAT,
                existing.get("dataset_sha256") != freeze["dataset_sha256"],
                existing.get("checkpoint_sha256") != args.checkpoint_sha256,
                existing.get("memory_selection") != args.memory_selection,
                existing.get("segmentation", {}).get("sha256")
                != contract["sha256"],
            )
        ):
            raise ValueError(f"latent provenance mismatch: {history_dir}")
        history = load_history(data_root / relative)
        if str(history["metadata"]["persona_id"]) != persona:
            raise ValueError(f"history/persona mismatch: {relative}")
        segments, stats = segment_history_memory(
            history,
            tokenizer,
            relative_path=relative,
            max_context_tokens=args.max_context_tokens,
            event_overlap=args.event_overlap,
        )
        previous = existing.get("memory_segments", [])
        metadata = []
        for index, segment in enumerate(segments):
            path = history_dir / f"memory_segment_{index:05d}.pt"
            item = segment.metadata()
            if path.is_file() and cached_latent_is_valid(path):
                prior = previous[index] if index < len(previous) else {}
                item["performance"] = {
                    **prior.get("performance", {}),
                    "cache_reused_this_run": True,
                }
                reused_segments += 1
            else:
                baseline = int(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                synchronize_device()
                started = time.perf_counter()
                latent = encode_segment(
                    model, tokenizer, segment, args.max_context_tokens
                )
                synchronize_device()
                elapsed = time.perf_counter() - started
                save_latent_atomic(latent, path)
                item["performance"] = {
                    "context_encode_seconds": elapsed,
                    "context_encode_peak_delta_bytes": max(
                        0, int(torch.cuda.max_memory_allocated()) - baseline
                    ),
                    "latent_bytes": latent.numel() * latent.element_size(),
                    "cache_reused_this_run": False,
                }
                encoded_segments += 1
            metadata.append(item)
        write_json_atomic(
            meta_path,
            {
                "latent_format": LATENT_FORMAT,
                "dataset": "personamem_v2_32k_text",
                "dataset_sha256": freeze["dataset_sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": args.checkpoint_sha256,
                "phase1_method": args.phase1_method,
                "history_file": relative,
                "persona_id": persona,
                "memory_selection": args.memory_selection,
                "segmentation": contract,
                "num_memory_segments": len(segments),
                "memory_segments": metadata,
                "segmentation_stats": stats,
            },
        )
        print(
            f"[precompute {completed}/{len(selected)}] shard={args.shard_id} "
            f"persona={persona} segments={len(segments)}",
            flush=True,
        )
    write_json_atomic(
        output_dir / f"summary.shard{args.shard_id:02d}.json",
        {
            "format": "memlora_personamem_v2_latent_shard_summary_v1",
            "latent_format": LATENT_FORMAT,
            "dataset_sha256": freeze["dataset_sha256"],
            "checkpoint_sha256": args.checkpoint_sha256,
            "phase1_method": args.phase1_method,
            "memory_selection": args.memory_selection,
            "segmentation": contract,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "histories": len(selected),
            "encoded_segments": encoded_segments,
            "reused_segments": reused_segments,
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )


if __name__ == "__main__":
    main()
