"""Evaluate RPMem on one PrefEval shard."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from experiments.personamem_v2.evaluation_utils import build_question_prompt, label_token_id, model_context_limit, option_label, render_prompt_ids
from experiments.prefeval.formal_contract import EVALUATION_PROTOCOL, GATE_RESULT_FORMAT, PHASE1_METHOD
from experiments.prefeval.formal_data import (
    materialized_rows,
    shuffled_options,
    validate_dataset_root,
    write_json_atomic,
    write_jsonl_atomic,
)
from experiments.prefeval.phase2_utils import cmp_lora, forward_with_lora, load_asset_latents, noise_asset_id, preference_asset_id, tensor_tree_bytes
from rpmem.checkpoint.loader import load_gate_checkpoint
from rpmem.training.hypernet_model import HypernetModel


RESULT_FORMAT = "memlora_prefeval_eval_row_v1"
SHARD_FORMAT = "memlora_prefeval_eval_shard_v1"


def source_example_user_ids(rows: list[dict]) -> dict[str, int]:
    """Assign stable numeric group IDs to the frozen test examples."""

    source_ids = sorted({str(row["source_example_id"]) for row in rows})
    return {source_id: index for index, source_id in enumerate(source_ids)}


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round(quantile * (len(ordered) - 1))]


def performance_summary(rows: list[dict]) -> dict:
    result = {}
    for key in sorted({key for row in rows for key in row}):
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
        ]
        if values:
            result[key] = {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
    return result


def load_gate(path: Path, model, dataset_sha256: str, checkpoint_sha256: str):
    gate, payload = load_gate_checkpoint(path)
    metadata = payload.get("metadata", {})
    if any(
        (
            payload.get("format") != GATE_RESULT_FORMAT,
            int(payload.get("d_latent", -1)) != int(model.config.gate.d_latent),
        )
    ):
        raise ValueError("incompatible PrefEval Gate format or dimensions")
    gate = gate.to(model.device)
    gate.eval()
    return gate, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=('rpmem', 'memlora_cmp_gate'), required=True)
    parser.add_argument("--data-root", default="data/prefeval/formal_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model-path", default="models/Qwen3-8B")
    parser.add_argument("--ctx-encoder-path", default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--session-latent-root", default="")
    parser.add_argument("--gate", default="")
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument(
        "--overflow-policy",
        choices=("error", "head", "tail", "head_tail"),
        default="head_tail",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid PrefEval evaluation shard")

    freeze, examples, noise = validate_dataset_root(args.data_root)
    materialized = materialized_rows(examples, noise, split="test")
    user_ids = source_example_user_ids(materialized)
    all_rows = materialized
    if args.max_examples:
        all_rows = all_rows[: args.max_examples]
    rows = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_id
    ]
    output = Path(args.output)
    completion = output.with_suffix(output.suffix + ".meta.json")
    external_store = None
    external_manifest_sha256 = external_store.sha256 if external_store else ""
    adapter_artifact = None
    adapter_model_sha256 = ""
    if args.skip_completed and output.is_file() and completion.is_file():
        existing = json.loads(completion.read_text(encoding="utf-8"))
        if all(
            (
                existing.get("format") == SHARD_FORMAT,
                existing.get("method") == args.method,
                existing.get("num_shards") == args.num_shards,
                existing.get("shard_id") == args.shard_id,
                existing.get("questions") == len(rows),
                existing.get("evaluation_protocol") == EVALUATION_PROTOCOL,
                existing.get("external_memory_manifest_sha256", "")
                == external_manifest_sha256,
                existing.get("adapter_model_sha256", "")
                == adapter_model_sha256,
            )
        ):
            print(f"reusing completed PrefEval result: {output}")
            return

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True, local_files_only=True
    )
    gate = None
    gate_metadata = None
    checkpoint = Path(args.checkpoint)
    if args.phase1_method != PHASE1_METHOD or not checkpoint.is_file():
        raise ValueError("PrefEval parametric methods require formal Offline-FKL")
    model = HypernetModel.from_checkpoint(
        str(checkpoint), use_flash_attn=args.use_flash_attn, train=False,
        base_model_path=args.base_model_path, ctx_encoder_path=args.ctx_encoder_path,
    )
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    gate, gate_metadata = load_gate(
        Path(args.gate),
        model,
        freeze["dataset_sha256"],
        args.checkpoint_sha256,
    )
    max_input_tokens = model_context_limit(
        model.base_model,
        tokenizer,
        args.max_input_tokens,
    )
    retriever = None
    label_ids = {
        option_label(index): label_token_id(tokenizer, option_label(index))
        for index in range(4)
    }
    session_cache: dict[str, list[torch.Tensor]] = {}
    required_assets = {
        noise_asset_id(session)
        for row in rows
        for session in row["memory_sessions"][1:]
    }
    required_assets.update(preference_asset_id(row) for row in rows)
    for completed, asset_id in enumerate(sorted(required_assets), start=1):
        session_cache[asset_id] = load_asset_latents(
            Path(args.session_latent_root),
            asset_id,
            dataset_sha256=freeze["dataset_sha256"],
            checkpoint_sha256=args.checkpoint_sha256,
            device=model.device,
        )
        if completed % 250 == 0 or completed == len(required_assets):
            print(
                f"[latent preload] assets={completed}/{len(required_assets)}",
                flush=True,
            )

    result_rows = []
    started_run = time.perf_counter()
    for completed, row in enumerate(rows, start=1):
        options, gold_index = shuffled_options(row, seed=args.seed)
        context_stats: dict = {}
        method_stats: dict = {}
        lora = None
        prompt_started = time.perf_counter()
        synchronize_device()
        load_started = time.perf_counter()
        latents = list(session_cache[preference_asset_id(row)])
        for session in row["memory_sessions"][1:]:
            latents.extend(session_cache[noise_asset_id(session)])
        selected = latents
        latent_meta = {}
        synchronize_device()
        latent_load_seconds = time.perf_counter() - load_started
        compile_started = time.perf_counter()
        with torch.inference_mode():
            lora = (
                cmp_lora(model, gate, latents)
            )
        synchronize_device()
        method_stats = {
            "selected_memory_segments": len(selected),
            "latent_load_seconds": latent_load_seconds,
            "lora_compile_seconds": time.perf_counter() - compile_started,
            "compiled_lora_bytes": tensor_tree_bytes(lora),
            "compiled_lora_rank": int(next(iter(lora.values()))["A"].shape[-2]),
            "segmentation_policy_id": latent_meta.get("segmentation", {}).get(
                "policy_id", "shared_session_latents"
            ),
        }
        ids = render_prompt_ids(
            tokenizer,
            build_question_prompt(str(row["question"]), options),
            model.device,
        )
        context_stats = {"input_tokens": int(ids.shape[-1])}

        prompt_seconds = time.perf_counter() - prompt_started
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        synchronize_device()
        query_started = time.perf_counter()
        with torch.inference_mode():
            logits = (
                forward_with_lora(model, lora, ids)[:, -1, :]
            )
        synchronize_device()
        scores = [
            float(logits[0, label_ids[option_label(index)]])
            for index in range(len(options))
        ]
        prediction = max(range(len(scores)), key=scores.__getitem__)
        result_rows.append(
            {
                "format": RESULT_FORMAT,
                "first_session_rule": gate.first_session_rule,
                "dataset_sha256": freeze["dataset_sha256"],
                "checkpoint_sha256": (
                    args.checkpoint_sha256
                ),
                "instance_id": row["instance_id"],
                "source_example_id": row["source_example_id"],
                "topic": row["topic"],
                "preference_form": row["preference_form"],
                "turn_count": row["turn_count"],
                "method": args.method,
                "prediction_index": prediction,
                "gold_index": gold_index,
                "prediction_label": option_label(prediction),
                "gold_label": option_label(gold_index),
                "correct": prediction == gold_index,
                "prompt_build_seconds": prompt_seconds,
                "query_forward_seconds": time.perf_counter() - query_started,
                "query_peak_delta_bytes": max(
                    0, int(torch.cuda.max_memory_allocated()) - baseline
                ),
                **context_stats,
                **method_stats,
            }
        )
        if completed % 25 == 0 or completed == len(rows):
            elapsed = time.perf_counter() - started_run
            print(
                f"[evaluate {completed}/{len(rows)}] method={args.method} "
                f"shard={args.shard_id} rate={completed / max(elapsed, 1e-9):.2f}/s",
                flush=True,
            )
        del ids, logits
        if lora is not None:
            del latents, selected, lora

    result_rows.sort(key=lambda row: str(row["instance_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, result_rows)
    write_json_atomic(
        completion,
        {
            "format": SHARD_FORMAT,
            "first_session_rule": gate.first_session_rule,
            "method": args.method,
            "dataset_sha256": freeze["dataset_sha256"],
            "checkpoint_sha256": (
                args.checkpoint_sha256
            ),
            "gate_contract": (
                gate_metadata.get("gate_contract") if gate_metadata else None
            ),
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "questions": len(result_rows),
            "correct": sum(row["correct"] for row in result_rows),
            "accuracy": (
                sum(row["correct"] for row in result_rows) / len(result_rows)
                if result_rows
                else 0.0
            ),
            "performance": performance_summary(result_rows),
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "external_memory_manifest_sha256": external_manifest_sha256,
            "adapter_model_sha256": adapter_model_sha256,
            "adapter_training_wall_seconds": (
                float(adapter_artifact.get("training_wall_seconds", 0.0))
                if adapter_artifact
                else 0.0
            ),
            "thinking_enabled": False,
            "seed": args.seed,
            "max_input_tokens": max_input_tokens,
            "overflow_policy": args.overflow_policy,
            "rag_top_k": 0,
            "retriever_model_revision": (
                ""
            ),
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )


if __name__ == "__main__":
    main()
