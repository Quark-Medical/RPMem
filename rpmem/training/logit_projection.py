"""Helpers for projecting vocabulary logits only at supervised positions."""

from __future__ import annotations

from torch import Tensor


def response_position_logits(
    model,
    input_ids: Tensor,
    label_positions: Tensor,
    *,
    squeeze_batch: bool = True,
    **model_kwargs,
) -> tuple[Tensor, bool]:
    """Return next-token logits at shared response positions for a batch.

    The boolean result reports whether the model projected only the requested
    positions internally. Models without ``logits_to_keep`` fall back to a
    full vocabulary projection for compatibility.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] < 1:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if label_positions.ndim != 1:
        raise ValueError("label_positions must be one-dimensional")
    if label_positions.numel() == 0:
        raise ValueError("at least one response label position is required")
    if bool((label_positions <= 0).any()):
        raise ValueError("response labels must have a preceding prediction position")

    prediction_positions = (
        label_positions.to(device=input_ids.device, dtype=input_ids.dtype) - 1
    )
    try:
        outputs = model(
            input_ids=input_ids,
            logits_to_keep=prediction_positions,
            **model_kwargs,
        )
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        outputs = model(input_ids=input_ids, **model_kwargs)
        selected = outputs.logits.index_select(1, prediction_positions)
        return (
            selected[0] if squeeze_batch and selected.shape[0] == 1 else selected
        ), False

    logits = outputs.logits
    if logits.shape[1] == prediction_positions.numel():
        selected = logits
        return (
            selected[0] if squeeze_batch and selected.shape[0] == 1 else selected
        ), True
    if logits.shape[1] == input_ids.shape[1]:
        selected = logits.index_select(1, prediction_positions)
        return (
            selected[0] if squeeze_batch and selected.shape[0] == 1 else selected
        ), False
    raise ValueError(
        "model returned an unexpected sequence dimension for logits_to_keep: "
        f"{logits.shape[1]}"
    )
