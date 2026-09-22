"""Checkpoint retention helpers for long-running Phase 1 jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    """Write a torch checkpoint without exposing a partially written target."""

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def gather_rank_states(
    local_state: Any,
    *,
    rank: int,
    world_size: int,
) -> list[Any] | None:
    """Gather per-rank recovery state onto rank zero without shared storage."""

    import torch

    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid distributed coordinates: rank={rank} world_size={world_size}"
        )
    if world_size == 1:
        return [local_state]
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("rank-state gathering requires an initialized process group")
    actual_world_size = torch.distributed.get_world_size()
    actual_rank = torch.distributed.get_rank()
    if actual_world_size != world_size or actual_rank != rank:
        raise RuntimeError(
            "rank-state gathering coordinates differ from the process group: "
            f"requested=({rank}, {world_size}) actual=({actual_rank}, "
            f"{actual_world_size})"
        )

    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local_state, gathered, dst=0)
    if rank == 0 and (gathered is None or any(state is None for state in gathered)):
        raise RuntimeError("rank-state gathering returned an incomplete result")
    return gathered


def prune_checkpoints(
    output_dir: str | Path,
    *,
    save_total_limit: int,
    preserve_epoch_boundaries: bool = False,
) -> list[Path]:
    """Delete old recovery checkpoints while optionally retaining epoch boundaries."""

    save_total_limit = int(save_total_limit)
    if save_total_limit <= 0:
        return []

    output_dir = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir() or path.is_symlink():
            continue
        if preserve_epoch_boundaries:
            marker = path / "checkpoint_complete.json"
            try:
                is_epoch_boundary = bool(
                    json.loads(marker.read_text()).get("epoch_boundary", False)
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
                is_epoch_boundary = False
            if is_epoch_boundary:
                continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()

    removed: list[Path] = []
    for _, path in checkpoints[:-save_total_limit]:
        shutil.rmtree(path)
        removed.append(path)
    return removed
