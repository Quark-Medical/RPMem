"""Pure helpers for online policy context distillation (OPCD)."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def deterministic_rollout_seed(
    base_seed: int,
    process_index: int,
    stream_index: int,
    *,
    validation: bool = False,
) -> int:
    """Derive a stable rank- and stream-specific 63-bit rollout seed."""

    value = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    for component in (process_index, stream_index, int(validation)):
        value ^= (int(component) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
    return value & 0x7FFFFFFFFFFFFFFF


def rotating_query_rows(
    n_queries: Tensor,
    sample_indices: Tensor,
    *,
    corpus_pass: int,
    queries_per_session: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select a deterministic circular probe window for each session."""

    if n_queries.ndim != 1 or sample_indices.ndim != 1:
        raise ValueError("n_queries and sample_indices must be one-dimensional")
    if n_queries.numel() != sample_indices.numel():
        raise ValueError("n_queries and sample_indices must have matching lengths")
    if queries_per_session <= 0:
        raise ValueError("queries_per_session must be positive")

    rows: list[int] = []
    selected_counts: list[int] = []
    offsets: list[int] = []
    row_start = 0
    for count, sample_index in zip(n_queries.tolist(), sample_indices.tolist()):
        count = int(count)
        if count <= 0:
            raise ValueError("every OPCD session must contain at least one query")
        selected = min(count, queries_per_session)
        offset = (int(sample_index) + int(corpus_pass) * selected) % count
        rows.extend(row_start + ((offset + index) % count) for index in range(selected))
        selected_counts.append(selected)
        offsets.append(offset)
        row_start += count
    device = n_queries.device
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(selected_counts, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


def response_mask_from_tokens(
    token_ids: Tensor,
    eos_token_ids: int | Iterable[int],
    padded_lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Mask generated tokens through the first EOS, inclusive."""

    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("generated token_ids must have shape [batch, nonzero length]")
    if isinstance(eos_token_ids, int):
        eos_values = [eos_token_ids]
    else:
        eos_values = [int(value) for value in eos_token_ids]
    if not eos_values:
        raise ValueError("at least one EOS token ID is required")

    is_eos = torch.zeros_like(token_ids, dtype=torch.bool)
    for eos_token_id in eos_values:
        is_eos |= token_ids.eq(eos_token_id)
    positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
    sentinel = torch.full_like(positions.expand_as(token_ids), token_ids.shape[1])
    first_eos = torch.where(is_eos, positions, sentinel).amin(dim=1)
    eos_lengths = torch.where(
        first_eos.lt(token_ids.shape[1]), first_eos + 1, token_ids.shape[1]
    )
    lengths = eos_lengths
    if padded_lengths is not None:
        if padded_lengths.shape != (token_ids.shape[0],):
            raise ValueError("rollout lengths must have shape [batch]")
        if bool((padded_lengths <= 0).any()) or bool(
            (padded_lengths > token_ids.shape[1]).any()
        ):
            raise ValueError("rollout lengths must be within the padded token width")
        lengths = torch.minimum(eos_lengths, padded_lengths.to(eos_lengths.device))
    mask = positions < lengths.unsqueeze(1)
    return mask, first_eos.lt(token_ids.shape[1])


def append_rollout(
    prompt_ids: Tensor,
    prompt_attention_mask: Tensor,
    rollout_ids: Tensor,
    rollout_attention_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one aligned rollout block to left-padded prompt batches."""

    if prompt_ids.shape != prompt_attention_mask.shape:
        raise ValueError("prompt IDs and attention mask must have matching shapes")
    if rollout_ids.shape != rollout_attention_mask.shape:
        raise ValueError("rollout IDs and attention mask must have matching shapes")
    if prompt_ids.shape[0] != rollout_ids.shape[0]:
        raise ValueError("prompt and rollout batch sizes must match")
    input_ids = torch.cat((prompt_ids, rollout_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, rollout_attention_mask), dim=1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    return input_ids, attention_mask, position_ids
