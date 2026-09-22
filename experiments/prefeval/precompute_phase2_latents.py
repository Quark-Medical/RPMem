"""Precompute shared-session and all-history PrefEval latents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from experiments.prefeval.formal_contract import LATENT_FORMAT, PHASE1_METHOD, segmentation_contract
from experiments.prefeval.formal_data import sha256_file, validate_dataset_root, write_json_atomic
from experiments.prefeval.phase2_utils import asset_cache_key, noise_asset_id, preference_asset_id, save_tensor_atomic, segment_asset
from rpmem.training.hypernet_model import HypernetModel


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cached_tensor_valid(path: Path) -> bool:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        path.unlink(missing_ok=True)
        return False
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        path.unlink(missing_ok=True)
        return False
    return True


def encode_segment(model, tokenizer, segment, limit: int) -> tuple[torch.Tensor, float]:
    token_ids = (
        list(segment.token_ids)
        if hasattr(segment, "token_ids")
        else list(
            tokenizer.encode(
                segment.text, add_special_tokens=True, truncation=False
            )
        )
    )
    if len(token_ids) > limit:
        raise ValueError(f"PrefEval segment exceeds compiler context: {len(token_ids)}")
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    synchronize_device()
    started = time.perf_counter()
    with torch.inference_mode():
        latent = model.encode_context(ids, torch.ones_like(ids)).detach().cpu()
    synchronize_device()
    return latent, time.perf_counter() - started


def preference_assets(examples: list[dict]) -> dict[str, dict]:
    result = {}
    for example in examples:
        asset_id = preference_asset_id(example)
        session = example["memory_sessions"][0]
        result[asset_id] = {
            "asset_id": asset_id,
            "messages": list(session["messages"]),
            "topic": str(example["topic"]),
            "kind": "preference",
        }
    return result


def noise_assets(noise: list[dict]) -> dict[str, dict]:
    return {
        noise_asset_id(session): {
            "asset_id": noise_asset_id(session),
            "messages": list(session["messages"]),
            "topic": "unrelated_noise",
            "kind": "noise",
        }
        for session in noise
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--ctx-encoder-path", default=None)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--data-root", default="data/prefeval/formal_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--memory-selection",
        choices=('sessions',),
        default="sessions",
    )
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--event-overlap", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid PrefEval latent shard")
    if args.phase1_method != PHASE1_METHOD:
        raise ValueError("PrefEval requires the formal Offline-FKL compiler")
    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Phase-1 checkpoint SHA mismatch")

    freeze, examples, noise = validate_dataset_root(args.data_root)
    model = HypernetModel.from_checkpoint(
        str(checkpoint), use_flash_attn=args.use_flash_attn, train=False,
        base_model_path=args.base_model_path, ctx_encoder_path=args.ctx_encoder_path,
    )
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model.config.ctx_encoder_model_name)
    tokenizer.model_max_length = max(int(tokenizer.model_max_length), 1_000_000_000)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = 0
    reused = 0
    started_run = time.perf_counter()

    policy = segmentation_contract(
        max_context_tokens=args.max_context_tokens,
        event_overlap=args.event_overlap,
    )
    assets = {**preference_assets(examples), **noise_assets(noise)}
    selected = [
        assets[key]
        for index, key in enumerate(sorted(assets))
        if index % args.num_shards == args.shard_id
    ]
    for completed, asset in enumerate(selected, start=1):
        directory = output_root / "assets" / asset_cache_key(asset["asset_id"])
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "meta.json"
        existing = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file()
            else {}
        )
        segments, stats = segment_asset(
            asset_id=asset["asset_id"],
            messages=asset["messages"],
            topic=asset["topic"],
            tokenizer=tokenizer,
            max_context_tokens=args.max_context_tokens,
            event_overlap=args.event_overlap,
        )
        metadata = []
        for index, segment in enumerate(segments):
            path = directory / f"segment_{index:04d}.pt"
            previous = existing.get("segment_metadata", [])
            can_reuse = (
                existing.get("format") == LATENT_FORMAT
                and existing.get("dataset_sha256") == freeze["dataset_sha256"]
                and existing.get("checkpoint_sha256") == args.checkpoint_sha256
                and existing.get("asset_id") == asset["asset_id"]
                and existing.get("segmentation") == policy
                and index < len(previous)
                and previous[index].get("segment") == segment.metadata()
                and path.is_file()
                and cached_tensor_valid(path)
            )
            if can_reuse:
                performance = dict(previous[index].get("performance", {}))
                reused += 1
            else:
                latent, seconds = encode_segment(
                    model, tokenizer, segment, args.max_context_tokens
                )
                save_tensor_atomic(latent, path)
                performance = {"context_encode_seconds": seconds}
                encoded += 1
            metadata.append(
                {"segment": segment.metadata(), "performance": performance}
            )
        write_json_atomic(
            meta_path,
            {
                "format": LATENT_FORMAT,
                "dataset_sha256": freeze["dataset_sha256"],
                "checkpoint_sha256": args.checkpoint_sha256,
                "asset_id": asset["asset_id"],
                "asset_kind": asset["kind"],
                "segments": len(segments),
                "segment_metadata": metadata,
                "segmentation": policy,
                "statistics": stats,
            },
        )
        if completed % 25 == 0 or completed == len(selected):
            print(
                f"[precompute {completed}/{len(selected)}] "
                f"selection=sessions shard={args.shard_id} "
                f"encoded={encoded} reused={reused}",
                flush=True,
            )
    units = len(selected)

    write_json_atomic(
        output_root / f"summary.shard{args.shard_id:02d}.json",
        {
            "format": "memlora_prefeval_latent_shard_summary_v1",
            "dataset_sha256": freeze["dataset_sha256"],
            "checkpoint_sha256": args.checkpoint_sha256,
            "memory_selection": args.memory_selection,
            "segmentation": policy,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "units": units,
            "encoded_segments": encoded,
            "reused_segments": reused,
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )


if __name__ == "__main__":
    main()
