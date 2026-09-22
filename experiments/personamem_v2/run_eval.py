"""Evaluate RPMem on one PersonaMem-v2 shard."""

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

from evaluation_utils import build_question_prompt, group_rows_by_history, label_token_id, model_context_limit, option_label, render_prompt_ids, select_shard_histories, shuffled_options
from formal_contract import EVALUATION_PROTOCOL, GATE_RESULT_FORMAT, PHASE1_METHOD
from formal_data import (
    load_history,
    sha256_file,
    validate_dataset_root,
    write_json_atomic,
    write_jsonl_atomic,
)
from rpmem.checkpoint.loader import load_gate_checkpoint
from rpmem.training.hypernet_model import HypernetModel
from parametric_utils import cmp_lora, forward_with_lora, load_latent_example, load_latents, tensor_tree_bytes


RESULT_FORMAT = "memlora_personamem_v2_eval_row_v1"
SHARD_FORMAT = "memlora_personamem_v2_eval_shard_v1"


def history_user_ids(rows: list[dict]) -> dict[str, int]:
    """Assign stable integer namespaces to the frozen benchmark histories."""

    histories = sorted({str(row["chat_history_32k_file"]) for row in rows})
    return {history_file: index for index, history_file in enumerate(histories)}


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def performance_summary(rows: list[dict]) -> dict:
    metrics = (
        "prompt_build_seconds",
        "query_forward_seconds",
        "query_peak_delta_bytes",
        "context_tokens",
        "used_context_tokens",
        "context_encode_seconds",
        "latent_load_seconds",
        "lora_compile_seconds",
        "compiled_lora_bytes",
        "compiled_lora_rank",
        "history_render_seconds",
        "retrieval_index_seconds",
        "retrieval_chunks",
        "retrieval_truncated_chunks",
    )
    result = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            result[metric] = {
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
            metadata.get("dataset_sha256") != dataset_sha256,
            metadata.get("checkpoint_sha256") != checkpoint_sha256,
            int(payload.get("d_latent", -1)) != int(model.config.gate.d_latent),
        )
    ):
        raise ValueError(f"incompatible PersonaMem-v2 Gate: {path}")
    gate = gate.to(model.device)
    gate.eval()
    return gate, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=('rpmem', 'memlora_cmp_gate'), required=True)
    parser.add_argument("--data-root", default="data/personamem_v2/formal_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model-path", default="models/Qwen3-8B")
    parser.add_argument("--ctx-encoder-path", default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--canonical-latent-root", default="")
    parser.add_argument("--gate", default="")
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument(
        "--overflow-policy",
        choices=("error", "head", "tail", "head_tail"),
        default="head_tail",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    output = Path(args.output)
    completion = output.with_suffix(output.suffix + ".meta.json")

    freeze, split_rows = validate_dataset_root(
        args.data_root,
        verify_source_files=False,
    )
    rows = split_rows["benchmark_text"]
    all_history_user_ids = history_user_ids(rows)
    if args.max_examples:
        rows = rows[: args.max_examples]
    grouped = select_shard_histories(
        group_rows_by_history(rows),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    expected_questions = sum(len(value) for value in grouped.values())
    external_store = None
    external_manifest_sha256 = external_store.sha256 if external_store else ""
    adapter_artifact = None
    adapter_model_sha256 = ""
    gate_sha256 = sha256_file(Path(args.gate))
    if args.skip_completed and completion.is_file() and output.is_file():
        existing = json.loads(completion.read_text(encoding="utf-8"))
        if all(
            (
                existing.get("format") == SHARD_FORMAT,
                existing.get("method") == args.method,
                existing.get("dataset_sha256") == freeze["dataset_sha256"],
                existing.get("checkpoint_sha256")
                == (
                    args.checkpoint_sha256
                ),
                existing.get("num_shards") == args.num_shards,
                existing.get("shard_id") == args.shard_id,
                existing.get("questions") == expected_questions,
                existing.get("evaluation_protocol") == EVALUATION_PROTOCOL,
                existing.get("gate_sha256") == gate_sha256,
                existing.get("external_memory_manifest_sha256", "")
                == external_manifest_sha256,
                existing.get("adapter_model_sha256", "")
                == adapter_model_sha256,
            )
        ):
            print(f"reusing completed result: {output}")
            return

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    gate = None
    gate_metadata = None
    if args.phase1_method != PHASE1_METHOD:
        raise ValueError("parametric methods require formal Offline-FKL")
    checkpoint = Path(args.checkpoint)
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != args.checkpoint_sha256
    ):
        raise ValueError("Phase-1 checkpoint is missing or mismatched")
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
        for index in range(12)
    }

    data_root = Path(args.data_root)
    result_rows = []
    history_rows = []
    completed_questions = 0
    started_run = time.perf_counter()
    for history_index, history_file in enumerate(sorted(grouped), start=1):
        history = load_history(data_root / history_file)
        history_context = None
        chunks = None
        chunk_embeddings = None
        lora = None
        history_metrics: dict = {}
        memory_selection = (
            "all"
        )
        latent_root = Path(args.canonical_latent_root)
        example = load_latent_example(
            latent_root,
            history_file,
            dataset_sha256=freeze["dataset_sha256"],
            checkpoint_sha256=args.checkpoint_sha256,
            memory_selection=memory_selection,
        )
        synchronize_device()
        load_started = time.perf_counter()
        latents = load_latents(example, model.device)
        selected_latents = latents
        synchronize_device()
        latent_load_seconds = time.perf_counter() - load_started
        compile_started = time.perf_counter()
        with torch.inference_mode():
            lora = (
                cmp_lora(model, gate, latents)
            )
        synchronize_device()
        lora_compile_seconds = time.perf_counter() - compile_started
        selected_segment_meta = (
            example["meta"]["memory_segments"]
        )
        context_seconds = sum(
            float(item.get("performance", {}).get("context_encode_seconds", 0.0))
            for item in selected_segment_meta
        )
        history_metrics = {
            "memory_selection": memory_selection,
            "num_memory_segments": int(example["meta"]["num_memory_segments"]),
            "selected_memory_segments": len(selected_latents),
            "context_encode_seconds": context_seconds,
            "latent_load_seconds": latent_load_seconds,
            "lora_compile_seconds": lora_compile_seconds,
            "compiled_lora_bytes": tensor_tree_bytes(lora),
            "compiled_lora_rank": int(next(iter(lora.values()))["A"].shape[-2]),
            "segmentation_policy_id": example["meta"]["segmentation"]["policy_id"],
        }
        history_rows.append(
            {
                "history_file": history_file,
                "persona_id": str(history["metadata"]["persona_id"]),
                "method": args.method,
                "questions": len(grouped[history_file]),
                **history_metrics,
            }
        )
        for row in grouped[history_file]:
            options, gold_index = shuffled_options(row)
            prompt_started = time.perf_counter()
            context_stats: dict = {}
            retrieval_stats: dict = {}
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
            completed_questions += 1
            result_rows.append(
                {
                    "format": RESULT_FORMAT,
                    "first_session_rule": gate.first_session_rule,
                    "dataset_sha256": freeze["dataset_sha256"],
                    "checkpoint_sha256": (
                        args.checkpoint_sha256
                    ),
                    "adapter_model_sha256": adapter_model_sha256,
                    "instance_id": row["instance_id"],
                    "source_row_index": row["source_row_index"],
                    "persona_id": row["persona_id"],
                    "history_file": history_file,
                    "method": args.method,
                    "prediction_index": prediction,
                    "gold_index": gold_index,
                    "prediction_label": option_label(prediction),
                    "gold_label": option_label(gold_index),
                    "correct": prediction == gold_index,
                    "option_count": len(options),
                    "attributes": row["attributes"],
                    "prompt_build_seconds": prompt_seconds,
                    "query_forward_seconds": time.perf_counter() - query_started,
                    "query_peak_delta_bytes": max(
                        0, int(torch.cuda.max_memory_allocated()) - baseline
                    ),
                    **history_metrics,
                    **context_stats,
                    **retrieval_stats,
                }
            )
            if completed_questions % 25 == 0:
                elapsed = time.perf_counter() - started_run
                rate = completed_questions / max(elapsed, 1e-9)
                print(
                    f"[evaluate {completed_questions}/{expected_questions}] "
                    f"method={args.method} shard={args.shard_id} rate={rate:.2f}/s",
                    flush=True,
                )
            del ids, logits
        if lora is not None:
            del latents, selected_latents, lora
        if chunk_embeddings is not None:
            del chunk_embeddings
        print(
            f"[history {history_index}/{len(grouped)}] method={args.method} "
            f"shard={args.shard_id}",
            flush=True,
        )
    result_rows.sort(key=lambda row: str(row["instance_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, result_rows)
    history_output = output.with_suffix(output.suffix + ".history.jsonl")
    write_jsonl_atomic(history_output, history_rows)
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
            "gate_sha256": (
                gate_sha256
            ),
            "gate_contract": (
                gate_metadata.get("gate_contract") if gate_metadata else None
            ),
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "histories": len(grouped),
            "questions": len(result_rows),
            "correct": sum(row["correct"] for row in result_rows),
            "accuracy": (
                sum(row["correct"] for row in result_rows) / len(result_rows)
                if result_rows
                else 0.0
            ),
            "performance": performance_summary(result_rows),
            "history_performance": performance_summary(history_rows),
            "history_performance_file": str(history_output),
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
            "rag_chunk_messages": (
                0
            ),
            "retriever_model_revision": (
                ""
            ),
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )


if __name__ == "__main__":
    main()
