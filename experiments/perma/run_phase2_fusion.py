"""Train and evaluate RPMem on a strict PERMA user fold."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_phase1_data import build_prompt
from data_adapter import ALL_USER_IDS, PERMA_VARIANTS
from phase2_fusion_utils import selected_context_performance
from phase2_user_splits import resolve_user_split
from rpmem.gate import CMPGate, FIRST_SESSION_RULES, run_cmp_sessions
from rpmem.training.trainer import step_gate_optimizer
from rpmem.checkpoint.loader import load_gate_checkpoint
from rpmem.lora.merger import combine_lora
from rpmem.training.hypernet_model import HypernetModel


EVALUATION_PROTOCOL = "memlora_perma_mcq_no_thinking_v2"
SEGMENT_LATENT_FORMAT = "memlora_perma_segment_latents_v2"
PERFORMANCE_METRICS = (
    "context_encode_seconds",
    "memory_compile_seconds_estimate",
    "latent_load_seconds",
    "lora_compile_seconds",
    "prompt_build_seconds",
    "query_forward_seconds",
    "lora_compile_peak_delta_bytes",
    "context_encode_peak_delta_bytes",
    "memory_compile_peak_delta_bytes",
    "query_peak_delta_bytes",
    "selected_context_tokens",
    "selected_latent_bytes",
    "compiled_lora_bytes",
    "compiled_lora_rank",
)


def label_token_id(tokenizer, label: str) -> int:
    for text in (label, " " + label):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(label, add_special_tokens=False)
    if not ids:
        raise ValueError(f"cannot tokenize label: {label}")
    return ids[-1]


def input_ids_tensor(encoded) -> torch.Tensor:
    if isinstance(encoded, torch.Tensor):
        return encoded
    input_ids = getattr(encoded, "input_ids", None)
    if isinstance(input_ids, torch.Tensor):
        return input_ids
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError):
        input_ids = None
    if isinstance(input_ids, torch.Tensor):
        return input_ids
    raise TypeError(
        f"expected token ids tensor or BatchEncoding, got {type(encoded)!r}"
    )


def prompt_ids(tokenizer, question: str, options: list[str], device) -> torch.Tensor:
    prompt = build_prompt(question, options)
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    ids = input_ids_tensor(ids)
    return ids.to(device)


def load_examples(emb_dir: Path, user_ids: set[int], limit: int = 0) -> list[dict]:
    examples = []
    for user_dir in sorted(emb_dir.glob("user*")):
        if not user_dir.is_dir():
            continue
        uid = int(user_dir.name.replace("user", ""))
        if uid not in user_ids:
            continue
        for task_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
            meta_path = task_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get("latent_format") == SEGMENT_LATENT_FORMAT:
                latent_paths = sorted(task_dir.glob("memory_segment_*.pt"))
                expected = int(meta.get("num_memory_segments", -1))
                if len(latent_paths) != expected:
                    raise ValueError(
                        f"incomplete memory-segment cache in {task_dir}: "
                        f"expected={expected}, found={len(latent_paths)}"
                    )
            else:
                latent_paths = sorted(
                    task_dir.glob("session_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]),
                )
            if not latent_paths:
                continue
            examples.append({"meta": meta, "latent_paths": latent_paths})
            if limit and len(examples) >= limit:
                return examples
    return examples


def load_embs(example: dict, device) -> list[torch.Tensor]:
    return [
        torch.load(path, weights_only=True, map_location=device).to(device)
        for path in example["latent_paths"]
    ]


def synchronize_device() -> None:
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def tensor_tree_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_bytes(item) for item in value)
    return 0


def reset_peak_memory() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def peak_memory_delta(baseline: int) -> int:
    if not torch.cuda.is_available():
        return 0
    return max(0, int(torch.cuda.max_memory_allocated()) - baseline)


def freeze_model(model: HypernetModel) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def base_model_input_device(model: HypernetModel) -> torch.device:
    return model.base_model.get_input_embeddings().weight.device


def decode_combined_lora(model: HypernetModel, latent: torch.Tensor):
    return decode_rank_concatenated_lora(model, [latent])


def decode_rank_concatenated_lora(
    model: HypernetModel,
    latents: list[torch.Tensor],
):
    if not latents:
        raise ValueError("at least one latent is required")
    lora_dict = model.head(torch.cat(latents, dim=0))
    n_chunks = torch.tensor([len(latents)], device=model.device)
    return combine_lora(
        lora_dict,
        n_chunks,
        lora_bias=model.head.get_head_bias() if model.config.head.use_bias else None,
    )


def fused_lora(
    model: HypernetModel,
    embs: list[torch.Tensor],
    method: str,
    gate=None,
    *,
    example: dict | None = None,
):
    if gate is None:
        raise ValueError("cmp_gate requires a gate")
    h = run_cmp_sessions(gate, embs)
    return decode_combined_lora(model, h)


def forward_with_lora(model: HypernetModel, lora_dict: dict, input_ids: torch.Tensor):
    n_queries = torch.tensor([1], device=model.device)
    model._apply_lora_to_layers(lora_dict, n_queries)
    try:
        return model.base_model(input_ids=input_ids).logits
    finally:
        model._reset_lora_bindings()


def evaluate(model, tokenizer, examples, method: str, gate=None) -> list[dict]:
    rows = []
    label_ids = {
        chr(ord("A") + i): label_token_id(tokenizer, chr(ord("A") + i))
        for i in range(12)
    }
    with torch.no_grad():
        for idx, example in enumerate(examples, start=1):
            meta = example["meta"]
            synchronize_device()
            started = time.perf_counter()
            embs = load_embs(example, model.device)
            synchronize_device()
            latent_load_seconds = time.perf_counter() - started

            selected_embs = embs
            selected_indices = list(range(len(embs)))
            latest_episode = None

            compile_baseline = reset_peak_memory()
            synchronize_device()
            started = time.perf_counter()
            lora = fused_lora(model, embs, method, gate, example=example)
            synchronize_device()
            lora_compile_seconds = time.perf_counter() - started
            lora_compile_peak_delta_bytes = peak_memory_delta(compile_baseline)

            started = time.perf_counter()
            ids = prompt_ids(
                tokenizer,
                meta["question"],
                meta["options"],
                base_model_input_device(model),
            )
            synchronize_device()
            prompt_build_seconds = time.perf_counter() - started

            query_baseline = reset_peak_memory()
            synchronize_device()
            started = time.perf_counter()
            logits = forward_with_lora(model, lora, ids)[:, -1, :]
            synchronize_device()
            query_forward_seconds = time.perf_counter() - started
            query_peak_delta_bytes = peak_memory_delta(query_baseline)
            context_performance = selected_context_performance(
                meta.get("memory_segments"),
                selected_indices,
            )
            labels = [chr(ord("A") + i) for i in range(len(meta["options"]))]
            scores = torch.tensor(
                [logits[0, label_ids[label]].item() for label in labels]
            )
            pred = labels[int(scores.argmax().item())]
            gold = meta["gold_label"]
            row = {
                "index": idx,
                "user_id": meta["user_id"],
                "task_id": meta["task_id"],
                "task_type": meta["task_type"],
                "variant": meta["variant"],
                "method": method,
                "pred": pred,
                "gold": gold,
                "correct": pred == gold,
                "num_sessions": meta["num_sessions"],
                "num_memory_segments": meta.get(
                    "num_memory_segments",
                    meta["num_sessions"],
                ),
                "selected_memory_segments": len(selected_embs),
                "latent_load_seconds": latent_load_seconds,
                "lora_compile_seconds": lora_compile_seconds,
                "prompt_build_seconds": prompt_build_seconds,
                "query_forward_seconds": query_forward_seconds,
                "lora_compile_peak_delta_bytes": lora_compile_peak_delta_bytes,
                "query_peak_delta_bytes": query_peak_delta_bytes,
                "selected_latent_bytes": tensor_tree_bytes(selected_embs),
                "compiled_lora_bytes": tensor_tree_bytes(lora),
                "compiled_lora_rank": int(
                    next(iter(lora.values()))["A"].shape[-2]
                ),
            }
            if context_performance:
                row.update(context_performance)
                row["memory_compile_seconds_estimate"] = (
                    context_performance["context_encode_seconds"]
                    + lora_compile_seconds
                )
                row["memory_compile_peak_delta_bytes"] = max(
                    context_performance["context_encode_peak_delta_bytes"],
                    lora_compile_peak_delta_bytes,
                )
            rows.append(row)
            del embs, selected_embs, lora, ids, logits
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(rows: list[dict]) -> dict:
    out = {
        "total": len(rows),
        "correct": sum(row["correct"] for row in rows),
    }
    out["accuracy"] = out["correct"] / max(out["total"], 1)
    for task_type in (1, 2, 3):
        group = [row for row in rows if row["task_type"] == task_type]
        out[f"type{task_type}_total"] = len(group)
        out[f"type{task_type}_correct"] = sum(row["correct"] for row in group)
        out[f"type{task_type}_accuracy"] = out[
            f"type{task_type}_correct"
        ] / max(len(group), 1)
    out["performance"] = {}
    for metric in PERFORMANCE_METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            continue
        out["performance"][metric] = {
            "mean": statistics.fmean(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return out


def train_gate(model, tokenizer, train_examples, args) -> tuple[CMPGate, dict]:
    gate = CMPGate(
        d_latent=model.config.gate.d_latent,
        init_bias=args.init_bias,
        first_session_rule=args.first_session_rule or "direct",
    ).to(model.device)
    optimizer = AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    label_ids = {
        chr(ord("A") + i): label_token_id(tokenizer, chr(ord("A") + i))
        for i in range(12)
    }

    stats = {"optimizer_updates": 0, "single_session_skips": 0}
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        total_loss = 0.0
        for example in train_examples:
            meta = example["meta"]
            embs = load_embs(example, model.device)
            lora = fused_lora(model, embs, "cmp_gate", gate)
            ids = prompt_ids(
                tokenizer,
                meta["question"],
                meta["options"],
                base_model_input_device(model),
            )
            logits = forward_with_lora(model, lora, ids)[:, -1, :]
            target = torch.tensor(
                [label_ids[meta["gold_label"]]],
                device=logits.device,
            )
            loss = F.cross_entropy(logits, target)

            grad_norm = step_gate_optimizer(
                gate, optimizer, loss, num_sessions=len(embs), max_grad_norm=args.max_grad_norm,
            )
            stats["single_session_skips" if grad_norm is None else "optimizer_updates"] += 1
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_examples), 1)
        print(f"epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f}")

    return gate, stats


def warm_up_evaluation(model, tokenizer, example: dict) -> None:
    embs = load_embs(example, model.device)
    lora = decode_rank_concatenated_lora(model, embs[:1])
    meta = example["meta"]
    ids = prompt_ids(
        tokenizer,
        meta["question"],
        meta["options"],
        base_model_input_device(model),
    )
    with torch.no_grad():
        forward_with_lora(model, lora, ids)
    synchronize_device()
    del embs, lora, ids


def load_phase2_model(args) -> HypernetModel:
    model = HypernetModel.from_checkpoint(
        args.checkpoint,
        base_model_path=args.base_model_path,
        ctx_encoder_path=args.ctx_encoder_path,
        use_flash_attn=args.use_flash_attn,
        train=False,
        base_model_device_map=args.base_model_device_map or None,
    )
    if args.base_model_device_map:
        hypernet_device = torch.device("cuda:0")
        model.ctx_encoder.to(hypernet_device)
        model.perceiver.to(hypernet_device)
        model.head.to(hypernet_device)
    else:
        model.cuda()
    freeze_model(model)
    return model


def run_fold(
    model: HypernetModel,
    tokenizer,
    args,
    methods: list[str],
    *,
    test_user: int | None,
    eval_user_ids: list[int] | None,
    output_root: Path,
    completion_path: Path | None,
) -> list[dict]:
    train_users, eval_users = resolve_user_split(
        test_user,
        eval_user_ids,
        args.train_users,
    )
    emb_dir = Path(args.emb_dir)
    gate_checkpoint = getattr(args, "gate_checkpoint", None)
    train_examples = [] if gate_checkpoint else load_examples(emb_dir, train_users, args.max_train_tasks)
    eval_examples = load_examples(emb_dir, eval_users, args.max_eval_tasks)
    if not gate_checkpoint and not train_examples:
        raise ValueError(f"no phase-2 training examples found in {emb_dir}")
    if not eval_examples:
        raise ValueError(f"no phase-2 evaluation examples found in {emb_dir}")
    loaded_gate = None
    requested_rule = getattr(args, "first_session_rule", None)
    if gate_checkpoint:
        loaded_gate, gate_metadata = load_gate_checkpoint(gate_checkpoint)
        if requested_rule is not None and requested_rule != loaded_gate.first_session_rule:
            raise ValueError("first_session_rule does not match the saved Gate")
        if (
            loaded_gate.d_latent != model.config.gate.d_latent
            or gate_metadata.get("test_user") != test_user
            or gate_metadata.get("eval_users", [gate_metadata.get("test_user")]) != sorted(eval_users)
            or gate_metadata.get("train_users") != sorted(train_users)
            or gate_metadata.get("args", {}).get("variant") != args.variant
            or gate_metadata.get("method", "cmp_gate") != "cmp_gate"
            or "control_contract" in gate_metadata
        ):
            raise ValueError("Gate dimensions, variant, or user fold do not match this evaluation")
        loaded_gate = loaded_gate.to(model.device).eval().requires_grad_(False)
    first_session_rule = loaded_gate.first_session_rule if loaded_gate else (requested_rule or "direct")
    warm_up_evaluation(model, tokenizer, eval_examples[0])

    output_root.mkdir(parents=True, exist_ok=True)
    multi_method = len(methods) > 1 or args.methods is not None
    completed: list[dict] = []

    for method in methods:
        out_dir = output_root / method if multi_method else output_root
        summary_path = out_dir / "summary.json"
        if args.skip_completed and summary_path.is_file():
            existing = json.loads(summary_path.read_text())
            if (
                existing.get("method") == method
                and existing.get("variant") == args.variant
                and existing.get("test_user") == test_user
                and existing.get(
                    "eval_users",
                    [existing.get("test_user")],
                )
                == sorted(eval_users)
                and existing.get("checkpoint") == args.checkpoint
                and existing.get("train_users") == sorted(train_users)
                and existing.get("evaluation_protocol") == EVALUATION_PROTOCOL
                and existing.get("first_session_rule", "gate_zero_state") == first_session_rule
            ):
                print(f"reusing completed result: {summary_path}")
                completed.append(existing)
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        gate = loaded_gate
        gate_training_seconds = 0.0
        training_stats = {"optimizer_updates": 0, "single_session_skips": 0}
        if gate is None:
            synchronize_device()
            gate_training_started = time.perf_counter()
            gate, training_stats = train_gate(model, tokenizer, list(train_examples), args)
            synchronize_device()
            gate_training_seconds = time.perf_counter() - gate_training_started
            torch.save(
                {
                    "state_dict": gate.state_dict(),
                    "first_session_rule": gate.first_session_rule,
                    "args": vars(args),
                    "train_users": sorted(train_users),
                    "test_user": test_user,
                    "eval_users": sorted(eval_users),
                    "phase1_method": args.phase1_method,
                    "checkpoint_sha256": args.checkpoint_sha256,
                },
                out_dir / "final_gate.pt",
            )

        synchronize_device()
        evaluation_started = time.perf_counter()
        rows = evaluate(model, tokenizer, eval_examples, method, gate)
        synchronize_device()
        evaluation_seconds = time.perf_counter() - evaluation_started
        summary = summarize(rows)
        summary.update(
            {
                "method": method,
                "phase1_method": args.phase1_method,
                "variant": args.variant,
                "test_user": test_user,
                "eval_users": sorted(eval_users),
                "train_users": sorted(train_users),
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": args.checkpoint_sha256,
                "emb_dir": args.emb_dir,
                "latent_format": eval_examples[0]["meta"].get(
                    "latent_format",
                    "legacy_perma_session_latents_v1",
                ),
                "segmentation": eval_examples[0]["meta"].get("segmentation"),
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "thinking_enabled": False,
                "seed": args.seed,
                "epochs": 0 if gate_checkpoint else args.epochs,
                "run_mode": "evaluate_only" if gate_checkpoint else "train_and_evaluate",
                "gate_checkpoint": str(gate_checkpoint) if gate_checkpoint else None,
                "first_session_rule": gate.first_session_rule,
                "training_examples": (
                    len(train_examples)
                ),
                **training_stats,
                "gate_training_seconds": gate_training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "fusion_contract": (
                    method
                ),
            }
        )
        (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        completed.append(summary)

    if multi_method:
        (output_root / "matrix_summary.json").write_text(
            json.dumps(completed, indent=2)
        )
    if completion_path is not None:
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(
            json.dumps(
                {
                    "format": "memlora_perma_phase2_fold_complete_v2",
                    "run_mode": "evaluate_only" if gate_checkpoint else "train_and_evaluate",
                    "first_session_rule": first_session_rule,
                    "phase1_method": args.phase1_method,
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "variant": args.variant,
                    "test_user": test_user,
                    "eval_users": sorted(eval_users),
                    "train_users": sorted(train_users),
                    "methods": methods,
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                    "latent_format": eval_examples[0]["meta"].get(
                        "latent_format",
                        "legacy_perma_session_latents_v1",
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument("--ctx_encoder_path", default=None)
    parser.add_argument("--gate_checkpoint", type=Path, default=None,
                        help="Evaluate one saved Gate without training; requires --test_user.")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--variant", default="clean_sd", choices=sorted(PERMA_VARIANTS))
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--test_user", type=int)
    eval_group.add_argument("--eval_users", type=int, nargs="+")
    eval_group.add_argument(
        "--fold_test_users",
        type=int,
        nargs="+",
        help="Run independent leave-one-user-out folds while loading the model once.",
    )
    parser.add_argument(
        "--train_users",
        type=int,
        nargs="+",
        help="Explicit Phase-2 training users; defaults to all non-evaluation users.",
    )
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument(
        "--method",
        choices=('cmp_gate',),
    )
    method_group.add_argument(
        "--methods",
        nargs="+",
        choices=('cmp_gate',),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--phase1_method", default="")
    parser.add_argument("--checkpoint_sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--completion_marker", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--init_bias", type=float, default=-2.0)
    parser.add_argument("--first-session-rule", "--first_session_rule", choices=FIRST_SESSION_RULES,
                        help="New training defaults to direct; saved Gates restore their recorded rule.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_tasks", type=int, default=0)
    parser.add_argument("--max_eval_tasks", type=int, default=0)
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
    methods = args.methods if args.methods is not None else [args.method]
    if args.gate_checkpoint:
        if args.test_user is None:
            parser.error("--gate_checkpoint requires a single --test_user fold")

    requested_users = (
        args.fold_test_users
        if args.fold_test_users is not None
        else ([args.test_user] if args.test_user is not None else args.eval_users)
    )
    unknown_users = sorted(set(requested_users) - set(ALL_USER_IDS))
    if unknown_users:
        raise ValueError(f"unknown PERMA users: {unknown_users}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_phase2_model(args)
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_name)
    output_root = Path(args.output_dir)

    if args.fold_test_users is None:
        run_fold(
            model,
            tokenizer,
            args,
            methods,
            test_user=args.test_user,
            eval_user_ids=args.eval_users,
            output_root=output_root,
            completion_path=(
                Path(args.completion_marker) if args.completion_marker else None
            ),
        )
        return

    fold_users = list(dict.fromkeys(args.fold_test_users))
    fusion_key = "-".join(methods)
    fold_summaries = []
    for fold_index, test_user in enumerate(fold_users, start=1):
        fold_root = output_root / f"fold_user{test_user}"
        fold_marker = fold_root / f"complete-{fusion_key}.json"
        completed = run_fold(
            model,
            tokenizer,
            args,
            methods,
            test_user=test_user,
            eval_user_ids=None,
            output_root=fold_root,
            completion_path=fold_marker,
        )
        fold_summaries.extend(completed)
        torch.cuda.empty_cache()
        print(
            f"[fold {fold_index}/{len(fold_users)}] user={test_user} complete",
            flush=True,
        )

    if args.completion_marker:
        completion_path = Path(args.completion_marker)
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(
            json.dumps(
                {
                    "format": "memlora_perma_phase2_grouped_folds_complete_v1",
                    "phase1_method": args.phase1_method,
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "variant": args.variant,
                    "test_users": fold_users,
                    "methods": methods,
                    "folds": len(fold_users),
                    "summaries": len(fold_summaries),
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
