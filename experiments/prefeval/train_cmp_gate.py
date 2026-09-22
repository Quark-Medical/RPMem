"""Train one PrefEval CMP Gate on official train topics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from experiments.personamem_v2.evaluation_utils import (
    build_question_prompt,
    label_token_id,
    option_label,
    render_prompt_ids,
)
from experiments.prefeval.formal_contract import (
    GATE_CONTRACT,
    GATE_RESULT_FORMAT,
    PHASE1_METHOD,
)
from experiments.prefeval.formal_data import (
    materialized_rows,
    shuffled_options,
    validate_dataset_root,
    write_json_atomic,
)
from experiments.prefeval.phase2_utils import (
    cmp_lora,
    load_asset_latents,
    noise_asset_id,
    preference_asset_id,
    forward_with_lora,
)
from rpmem.gate import CMPGate, FIRST_SESSION_RULES, normalize_gate_contract
from rpmem.training.trainer import step_gate_optimizer
from rpmem.training.hypernet_model import HypernetModel


STATE_FORMAT = "memlora_prefeval_cmp_training_state_v1"


def save_torch_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def contract(args, training_examples: int) -> dict:
    return {
        **GATE_CONTRACT,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "init_bias": args.init_bias,
        "first_session_rule": args.first_session_rule,
        "optimizer_updates": ("one_per_multi_session_question" if args.first_session_rule == "direct"
                              else "one_per_training_question"),
        "training_examples": training_examples,
    }


def epoch_rows(rows: list[dict], *, seed: int, epoch: int) -> list[dict]:
    result = list(rows)
    random.Random(seed + epoch * 1_000_003).shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--data-root", default="data/prefeval/formal_v1")
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--base-model-path", default="models/Qwen3-8B")
    parser.add_argument("--ctx-encoder-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=GATE_CONTRACT["epochs"])
    parser.add_argument(
        "--learning-rate", type=float, default=GATE_CONTRACT["learning_rate"]
    )
    parser.add_argument(
        "--weight-decay", type=float, default=GATE_CONTRACT["weight_decay"]
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=GATE_CONTRACT["max_grad_norm"]
    )
    parser.add_argument("--init-bias", type=float, default=GATE_CONTRACT["init_bias"])
    parser.add_argument("--first-session-rule", choices=FIRST_SESSION_RULES, default="direct")
    parser.add_argument("--seed", type=int, default=GATE_CONTRACT["seed"])
    parser.add_argument("--save-updates", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.phase1_method != PHASE1_METHOD:
        raise ValueError("PrefEval requires formal Offline-FKL")
    if args.epochs < 1 or args.save_updates < 1:
        raise ValueError("invalid PrefEval Gate configuration")
    checkpoint = Path(args.checkpoint)

    freeze, examples, noise = validate_dataset_root(args.data_root)
    rows = materialized_rows(examples, noise, split="train")
    if args.max_train_examples:
        rows = rows[: args.max_train_examples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "checkpoint_latest.pt"
    progress_path = output_dir / "checkpoint_latest.json"
    final_path = output_dir / "gate.pt"
    summary_path = output_dir / "summary.json"
    frozen_contract = contract(args, len(rows))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = HypernetModel.from_checkpoint(
        str(checkpoint), use_flash_attn=args.use_flash_attn, train=False,
        base_model_path=args.base_model_path, ctx_encoder_path=args.ctx_encoder_path,
    )
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, local_files_only=True
    )
    gate = CMPGate(
        d_latent=model.config.gate.d_latent, init_bias=args.init_bias,
        first_session_rule=args.first_session_rule,
    ).to(model.device)
    optimizer = AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    latent_root = Path(args.latent_root)
    asset_cache: dict[str, list[torch.Tensor]] = {}
    asset_ids = {preference_asset_id(row) for row in rows}
    asset_ids.update(noise_asset_id(session) for session in noise)
    for completed, asset_id in enumerate(sorted(asset_ids), start=1):
        asset_cache[asset_id] = load_asset_latents(
            latent_root,
            asset_id,
            dataset_sha256=freeze["dataset_sha256"],
            checkpoint_sha256=args.checkpoint_sha256,
            device=model.device,
        )
        if completed % 250 == 0 or completed == len(asset_ids):
            print(f"[gate preload] assets={completed}/{len(asset_ids)}", flush=True)

    start_epoch = 1
    row_cursor = 0
    optimizer_updates = 0
    single_session_skips = 0
    epoch_single_session_skips = 0
    epoch_loss_sum = 0.0
    epoch_updates = 0
    history: list[dict] = []
    if args.resume and state_path.is_file():
        state = torch.load(state_path, weights_only=False, map_location="cpu")
        if any(
            (
                state.get("format") != STATE_FORMAT,
                normalize_gate_contract(state.get("gate_contract", {})) != frozen_contract,
            )
        ):
            raise ValueError("incompatible PrefEval Gate resume state")
        gate.load_state_dict(state["gate_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        row_cursor = int(state["row_cursor"])
        optimizer_updates = int(state["optimizer_updates"])
        single_session_skips = int(state.get("single_session_skips", 0))
        epoch_single_session_skips = int(state.get("epoch_single_session_skips", 0))
        epoch_loss_sum = float(state["epoch_loss_sum"])
        epoch_updates = int(state["epoch_updates"])
        history = list(state["training_history"])
        print(
            f"resumed PrefEval Gate: epoch={start_epoch} row={row_cursor} "
            f"updates={optimizer_updates}",
            flush=True,
        )

    labels = {
        option_label(index): label_token_id(tokenizer, option_label(index))
        for index in range(4)
    }

    def state_payload(epoch: int, cursor: int) -> dict:
        return {
            "format": STATE_FORMAT,
            "dataset_sha256": freeze["dataset_sha256"],
            "checkpoint_sha256": args.checkpoint_sha256,
            "gate_contract": frozen_contract,
            "epoch": epoch,
            "row_cursor": cursor,
            "optimizer_updates": optimizer_updates,
            "single_session_skips": single_session_skips,
            "epoch_single_session_skips": epoch_single_session_skips,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_updates": epoch_updates,
            "training_history": history,
            "gate_state_dict": gate.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        ordered = epoch_rows(rows, seed=args.seed, epoch=epoch)
        epoch_started = time.perf_counter()
        for position in range(row_cursor, len(ordered)):
            row = ordered[position]
            latents = list(asset_cache[preference_asset_id(row)])
            for session in row["memory_sessions"][1:]:
                latents.extend(asset_cache[noise_asset_id(session)])
            options, gold_index = shuffled_options(row, seed=args.seed)
            ids = render_prompt_ids(
                tokenizer,
                build_question_prompt(str(row["question"]), options),
                model.device,
            )
            lora = cmp_lora(model, gate, latents)
            logits = forward_with_lora(model, lora, ids)[:, -1, :]
            target = torch.tensor(
                [labels[option_label(gold_index)]],
                dtype=torch.long,
                device=model.device,
            )
            loss = F.cross_entropy(logits, target)
            grad_norm = step_gate_optimizer(
                gate, optimizer, loss, num_sessions=len(latents), max_grad_norm=args.max_grad_norm,
            )
            value = float(loss.detach())
            if grad_norm is None:
                single_session_skips += 1
                epoch_single_session_skips += 1
            else:
                optimizer_updates += 1
                epoch_updates += 1
            epoch_loss_sum += value
            if grad_norm is not None and optimizer_updates % 50 == 0:
                print(
                    f"[gate] epoch={epoch}/{args.epochs} "
                    f"row={position + 1}/{len(ordered)} updates={optimizer_updates} "
                    f"loss={value:.4f} grad_norm={float(grad_norm):.4f}",
                    flush=True,
                )
            if (optimizer_updates + single_session_skips) % args.save_updates == 0:
                state = state_payload(epoch, position + 1)
                save_torch_atomic(state, state_path)
                write_json_atomic(
                    progress_path,
                    {k: v for k, v in state.items() if not k.endswith("state_dict")},
                )
            del ids, lora, logits, target, loss
        if epoch_updates + epoch_single_session_skips != len(rows):
            raise ValueError(
                f"PrefEval epoch example-count mismatch: {epoch_updates + epoch_single_session_skips} != {len(rows)}"
            )
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss_sum / max(epoch_updates + epoch_single_session_skips, 1),
                "optimizer_updates": epoch_updates,
                "single_session_skips": epoch_single_session_skips,
                "seconds_this_process": time.perf_counter() - epoch_started,
            }
        )
        row_cursor = 0
        epoch_updates = 0
        epoch_single_session_skips = 0
        epoch_loss_sum = 0.0
        state = state_payload(epoch + 1, 0)
        save_torch_atomic(state, state_path)
        write_json_atomic(
            progress_path,
            {k: v for k, v in state.items() if not k.endswith("state_dict")},
        )
        print(
            f"[gate] completed epoch={epoch}/{args.epochs} "
            f"mean_loss={history[-1]['loss']:.6f}",
            flush=True,
        )

    expected_updates = len(rows) * args.epochs - single_session_skips
    if optimizer_updates != expected_updates:
        raise ValueError(
            f"PrefEval total update mismatch: {optimizer_updates} != {expected_updates}"
        )
    result = {
        "format": GATE_RESULT_FORMAT,
        "dataset": "prefeval",
        "dataset_sha256": freeze["dataset_sha256"],
        "phase1_method": args.phase1_method,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "gate_contract": frozen_contract,
        "training_examples": len(rows),
        "optimizer_updates": optimizer_updates,
        "single_session_skips": single_session_skips,
        "training_history": history,
        "elapsed_seconds_this_process": time.perf_counter() - run_started,
    }
    save_torch_atomic(
        {
            "format": GATE_RESULT_FORMAT,
            "gate_state_dict": gate.state_dict(),
            "first_session_rule": gate.first_session_rule,
            "d_latent": model.config.gate.d_latent,
            "metadata": result,
        },
        final_path,
    )
    write_json_atomic(summary_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
