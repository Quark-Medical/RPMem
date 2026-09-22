"""Train the PersonaMem-v2 CMP Gate with the frozen formal compiler/reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from evaluation_utils import (
    build_question_prompt,
    group_rows_by_history,
    label_token_id,
    option_label,
    render_prompt_ids,
    shuffled_options,
)
from formal_contract import GATE_CONTRACT, GATE_RESULT_FORMAT, PHASE1_METHOD
from formal_data import sha256_file, validate_dataset_root, write_json_atomic
from rpmem.gate import CMPGate, FIRST_SESSION_RULES, normalize_gate_contract
from rpmem.training.trainer import step_gate_optimizer
from rpmem.training.hypernet_model import HypernetModel
from parametric_utils import (
    cmp_lora,
    forward_with_lora,
    load_latent_example,
    load_latents,
)


STATE_FORMAT = "memlora_personamem_v2_cmp_training_state_v1"
RESULT_FORMAT = GATE_RESULT_FORMAT


def gate_contract(args) -> dict:
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
    }


def save_torch_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_epoch_rows(
    grouped: dict[str, list[dict]], *, seed: int, epoch: int
) -> list[tuple[str, list[dict]]]:
    histories = sorted(grouped)
    epoch_rng = random.Random(seed + 1_000_003 * epoch)
    epoch_rng.shuffle(histories)
    result = []
    for history in histories:
        rows = list(grouped[history])
        row_seed = hashlib.sha256(f"{seed}:{epoch}:{history}".encode()).digest()[:8]
        random.Random(int.from_bytes(row_seed, "big")).shuffle(rows)
        result.append((history, rows))
    return result


def training_state(
    *,
    gate,
    optimizer,
    dataset_sha256: str,
    checkpoint_sha256: str,
    contract: dict,
    epoch: int,
    history_cursor: int,
    epoch_loss_sum: float,
    epoch_updates: int,
    optimizer_updates: int,
    epoch_single_session_skips: int,
    single_session_skips: int,
    training_history: list[dict],
) -> dict:
    return {
        "format": STATE_FORMAT,
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gate_contract": contract,
        "epoch": epoch,
        "history_cursor": history_cursor,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_updates": epoch_updates,
        "optimizer_updates": optimizer_updates,
        "epoch_single_session_skips": epoch_single_session_skips,
        "single_session_skips": single_session_skips,
        "training_history": training_history,
        "gate_state_dict": gate.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase1-method", default=PHASE1_METHOD)
    parser.add_argument("--data-root", default="data/personamem_v2/formal_v1")
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
    parser.add_argument("--save-histories", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument(
        "--use-flash-attn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.phase1_method != PHASE1_METHOD:
        raise ValueError("PersonaMem-v2 requires formal Offline-FKL")
    if args.epochs < 1 or args.save_histories < 1:
        raise ValueError("invalid Gate training configuration")
    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Phase-1 checkpoint SHA mismatch")

    freeze, split_rows = validate_dataset_root(
        args.data_root,
        verify_source_files=False,
    )
    rows = split_rows["train_text"]
    if args.max_train_examples:
        rows = rows[: args.max_train_examples]
    grouped = group_rows_by_history(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "checkpoint_latest.pt"
    progress_path = output_dir / "checkpoint_latest.json"
    final_path = output_dir / "gate.pt"
    summary_path = output_dir / "summary.json"
    contract = {
        **gate_contract(args),
        "training_examples": len(rows),
        "training_histories": len(grouped),
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, local_files_only=True
    )
    gate = CMPGate(
        d_latent=model.config.gate.d_latent,
        init_bias=args.init_bias,
        first_session_rule=args.first_session_rule,
    ).to(model.device)
    optimizer = AdamW(
        gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    history_cursor = 0
    epoch_loss_sum = 0.0
    epoch_updates = 0
    optimizer_updates = 0
    epoch_single_session_skips = 0
    single_session_skips = 0
    training_history: list[dict] = []
    if args.resume and state_path.is_file():
        state = torch.load(state_path, weights_only=False, map_location="cpu")
        if any(
            (
                state.get("format") != STATE_FORMAT,
                state.get("dataset_sha256") != freeze["dataset_sha256"],
                state.get("checkpoint_sha256") != args.checkpoint_sha256,
                normalize_gate_contract(state.get("gate_contract", {})) != contract,
            )
        ):
            raise ValueError("incompatible PersonaMem-v2 Gate resume state")
        gate.load_state_dict(state["gate_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        history_cursor = int(state["history_cursor"])
        epoch_loss_sum = float(state["epoch_loss_sum"])
        epoch_updates = int(state["epoch_updates"])
        optimizer_updates = int(state["optimizer_updates"])
        epoch_single_session_skips = int(state.get("epoch_single_session_skips", 0))
        single_session_skips = int(state.get("single_session_skips", 0))
        training_history = list(state["training_history"])
        print(
            f"resumed Gate: epoch={start_epoch} history_cursor={history_cursor} "
            f"updates={optimizer_updates}",
            flush=True,
        )

    label_ids = {
        option_label(index): label_token_id(tokenizer, option_label(index))
        for index in range(12)
    }
    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_rows = deterministic_epoch_rows(grouped, seed=args.seed, epoch=epoch)
        epoch_started = time.perf_counter()
        for position in range(history_cursor, len(epoch_rows)):
            history_file, history_rows = epoch_rows[position]
            example = load_latent_example(
                Path(args.latent_root),
                history_file,
                dataset_sha256=freeze["dataset_sha256"],
                checkpoint_sha256=args.checkpoint_sha256,
                memory_selection="all",
            )
            latents = load_latents(example, model.device)
            for row in history_rows:
                options, gold_index = shuffled_options(row)
                ids = render_prompt_ids(
                    tokenizer,
                    build_question_prompt(str(row["question"]), options),
                    model.device,
                )
                lora = cmp_lora(model, gate, latents)
                logits = forward_with_lora(model, lora, ids)[:, -1, :]
                target = torch.tensor(
                    [label_ids[option_label(gold_index)]],
                    dtype=torch.long,
                    device=model.device,
                )
                loss = F.cross_entropy(logits, target)
                grad_norm = step_gate_optimizer(
                    gate, optimizer, loss, num_sessions=len(latents), max_grad_norm=args.max_grad_norm,
                )
                value = float(loss.detach())
                epoch_loss_sum += value
                if grad_norm is None:
                    epoch_single_session_skips += 1
                    single_session_skips += 1
                else:
                    epoch_updates += 1
                    optimizer_updates += 1
                if grad_norm is not None and optimizer_updates % 50 == 0:
                    print(
                        f"[gate] epoch={epoch}/{args.epochs} "
                        f"history={position + 1}/{len(epoch_rows)} "
                        f"updates={optimizer_updates} loss={value:.4f} "
                        f"grad_norm={float(grad_norm):.4f}",
                        flush=True,
                    )
                del ids, lora, logits, target, loss
            del latents
            next_cursor = position + 1
            if next_cursor % args.save_histories == 0:
                state = training_state(
                    gate=gate,
                    optimizer=optimizer,
                    dataset_sha256=freeze["dataset_sha256"],
                    checkpoint_sha256=args.checkpoint_sha256,
                    contract=contract,
                    epoch=epoch,
                    history_cursor=next_cursor,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_updates=epoch_updates,
                    optimizer_updates=optimizer_updates,
                    epoch_single_session_skips=epoch_single_session_skips,
                    single_session_skips=single_session_skips,
                    training_history=training_history,
                )
                save_torch_atomic(state, state_path)
                write_json_atomic(
                    progress_path,
                    {
                        key: value
                        for key, value in state.items()
                        if not key.endswith("state_dict")
                    },
                )
        training_history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss_sum / max(epoch_updates + epoch_single_session_skips, 1),
                "optimizer_updates": epoch_updates,
                "single_session_skips": epoch_single_session_skips,
                "seconds_this_process": time.perf_counter() - epoch_started,
            }
        )
        if epoch_updates + epoch_single_session_skips != len(rows):
            raise ValueError(
                f"epoch example-count mismatch: {epoch_updates + epoch_single_session_skips} != {len(rows)}"
            )
        epoch_loss_sum = 0.0
        epoch_updates = 0
        epoch_single_session_skips = 0
        history_cursor = 0
        state = training_state(
            gate=gate,
            optimizer=optimizer,
            dataset_sha256=freeze["dataset_sha256"],
            checkpoint_sha256=args.checkpoint_sha256,
            contract=contract,
            epoch=epoch + 1,
            history_cursor=0,
            epoch_loss_sum=0.0,
            epoch_updates=0,
            optimizer_updates=optimizer_updates,
            epoch_single_session_skips=0,
            single_session_skips=single_session_skips,
            training_history=training_history,
        )
        save_torch_atomic(state, state_path)
        write_json_atomic(
            progress_path,
            {
                key: value
                for key, value in state.items()
                if not key.endswith("state_dict")
            },
        )
        print(
            f"[gate] completed epoch={epoch}/{args.epochs} "
            f"mean_loss={training_history[-1]['loss']:.6f}",
            flush=True,
        )

    expected_updates = len(rows) * args.epochs - single_session_skips
    if optimizer_updates != expected_updates:
        raise ValueError(
            f"total optimizer-update mismatch: {optimizer_updates} != "
            f"{expected_updates}"
        )
    result = {
        "format": RESULT_FORMAT,
        "dataset": "personamem_v2_32k_text",
        "dataset_sha256": freeze["dataset_sha256"],
        "phase1_method": args.phase1_method,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "gate_contract": contract,
        "training_examples": len(rows),
        "training_histories": len(grouped),
        "optimizer_updates": optimizer_updates,
        "single_session_skips": single_session_skips,
        "training_history": training_history,
        "elapsed_seconds_this_process": time.perf_counter() - run_started,
    }
    save_torch_atomic(
        {
            "format": RESULT_FORMAT,
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
