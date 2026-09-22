"""Shared prompt, option, retrieval, and MCQ scoring helpers."""

from __future__ import annotations

import torch

import hashlib
import random
from collections import defaultdict
from typing import Any, Iterable


def shuffled_options(row: dict[str, Any]) -> tuple[list[str], int]:
    """Return a deterministic method-independent option order and gold index."""

    options = [str(value) for value in row["options"]]
    source_gold = int(row["correct_option_index"])
    if not 0 <= source_gold < len(options):
        raise ValueError(f"invalid source gold index: {row['instance_id']}")
    seed_bytes = hashlib.sha256(str(row["instance_id"]).encode()).digest()[:8]
    indices = list(range(len(options)))
    random.Random(int.from_bytes(seed_bytes, "big")).shuffle(indices)
    return [options[index] for index in indices], indices.index(source_gold)


def option_label(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError(f"unsupported option index: {index}")
    return chr(ord("A") + index)


def build_question_prompt(question: str, options: list[str]) -> str:
    labels = [option_label(index) for index in range(len(options))]
    rendered = "\n".join(
        f"{label}. {text}" for label, text in zip(labels, options, strict=True)
    )
    return (
        "Answer the multiple-choice question using the user memory.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{rendered}\n\n"
        f"Answer with the letter only ({labels[0]}-{labels[-1]})."
    )


def input_ids_tensor(encoded) -> torch.Tensor:
    import torch

    if isinstance(encoded, torch.Tensor):
        return encoded
    input_ids = getattr(encoded, "input_ids", None)
    if isinstance(input_ids, torch.Tensor):
        return input_ids
    if isinstance(encoded, dict) and isinstance(encoded.get("input_ids"), torch.Tensor):
        return encoded["input_ids"]
    raise TypeError(f"cannot extract input ids from {type(encoded)!r}")


def render_prompt_ids(tokenizer, prompt: str, device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    else:
        encoded = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    return input_ids_tensor(encoded).to(device)


def label_token_id(tokenizer, label: str) -> int:
    for value in (label, " " + label):
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"MCQ label is not one token: {label!r}")


def group_rows_by_history(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["chat_history_32k_file"])].append(row)
    return dict(grouped)


def history_cache_key(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("./")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"history_{digest}"


def select_shard_histories(
    grouped: dict[str, list[dict[str, Any]]], *, num_shards: int, shard_id: int
) -> dict[str, list[dict[str, Any]]]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("invalid history shard")
    return {
        history: grouped[history]
        for index, history in enumerate(sorted(grouped))
        if index % num_shards == shard_id
    }


def model_context_limit(model, tokenizer, requested: int) -> int:
    candidates = []
    for value in (
        getattr(model.config, "max_position_embeddings", 0),
        getattr(tokenizer, "model_max_length", 0),
    ):
        value = int(value or 0)
        if 0 < value < 1_000_000:
            candidates.append(value)
    if not candidates:
        raise ValueError("cannot determine reader context length")
    maximum = min(candidates) - 1
    if requested:
        if requested > maximum:
            raise ValueError(
                f"requested input {requested} exceeds physical limit {maximum}"
            )
        return requested
    return maximum
