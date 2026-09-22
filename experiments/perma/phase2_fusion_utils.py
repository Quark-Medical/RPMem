"""Pure helpers for formal PERMA Phase-2 fusion policies."""

from __future__ import annotations

from typing import Any


def select_memory_segments(segments: list[Any], selection: str) -> list[Any]:
    if selection != "all":
        raise ValueError(f"unsupported RPMem memory selection: {selection}")
    return list(segments)


def selected_context_performance(
    memory_segments: Any,
    selected_indices: list[int],
) -> dict[str, float | int]:
    if not isinstance(memory_segments, list):
        raise ValueError("context performance requires memory-segment metadata")
    try:
        selected = [memory_segments[index] for index in selected_indices]
    except (IndexError, TypeError) as error:
        raise ValueError("selected memory-segment index is out of range") from error
    if not selected:
        raise ValueError("context performance requires at least one segment")

    performances = [segment.get("performance", {}) for segment in selected]
    required = ("context_encode_seconds", "context_encode_peak_delta_bytes")
    if any(
        not all(key in performance for key in required)
        for performance in performances
    ):
        return {}
    try:
        return {
            "context_encode_seconds": sum(
                float(performance["context_encode_seconds"])
                for performance in performances
            ),
            "context_encode_peak_delta_bytes": max(
                int(performance["context_encode_peak_delta_bytes"])
                for performance in performances
            ),
            "selected_context_tokens": sum(
                int(segment["token_count"]) for segment in selected
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid memory-segment performance metadata") from error
