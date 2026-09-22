"""Generic embedding precomputation utilities."""

import os
from typing import Callable

import torch
from torch import Tensor


def precompute_and_save(
    encode_fn: Callable[[str], Tensor],
    sessions: list[dict],
    output_dir: str,
    task_id: str,
    metadata: dict,
) -> None:
    """Precompute lora_emb for each session and save to disk.

    Args:
        encode_fn: Function that takes text and returns lora_emb tensor
        sessions: List of {"text": str, ...}
        output_dir: Base output directory
        task_id: Unique task identifier (used as subdirectory name)
        metadata: Additional metadata to save in meta.pt
    """
    task_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    for i, session in enumerate(sessions):
        emb = encode_fn(session["text"])
        torch.save(emb.cpu(), os.path.join(task_dir, f"session_{i}.pt"))

    meta = {**metadata, "n_sessions": len(sessions)}
    torch.save(meta, os.path.join(task_dir, "meta.pt"))


def load_cached_embs(task_dir: str) -> list[Tensor]:
    """Load precomputed session embeddings from a task directory."""
    meta = torch.load(os.path.join(task_dir, "meta.pt"), weights_only=False)
    embs = []
    for i in range(meta["n_sessions"]):
        emb = torch.load(
            os.path.join(task_dir, f"session_{i}.pt"),
            weights_only=True,
            map_location="cpu",
        )
        embs.append(emb)
    return embs
