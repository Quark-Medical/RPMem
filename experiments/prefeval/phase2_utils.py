"""Shared instance, history, segmentation, and latent helpers for PrefEval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.personamem_v2.parametric_utils import cmp_lora, forward_with_lora, tensor_tree_bytes
from rpmem.training.corpus.normalize import segment_session
from rpmem.training.corpus.schema import (
    CanonicalEvent,
    CanonicalSession,
    render_session,
)

from experiments.prefeval.formal_contract import LATENT_FORMAT


@dataclass(frozen=True)
class MemorySegment:
    segment_index: int
    segment_count: int
    session_id: str
    event_ids: tuple[str, ...]
    token_count: int
    text: str

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("text")
        payload["event_ids"] = list(self.event_ids)
        return payload


def preference_asset_id(example: dict[str, Any]) -> str:
    return f"preference:{example.get('base_instance_id', example['instance_id'])}"


def noise_asset_id(session: dict[str, Any]) -> str:
    return f"noise:{session['session_id']}"


def asset_cache_key(asset_id: str) -> str:
    return hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:24]


def canonical_session(
    *, asset_id: str, messages: list[dict[str, Any]], topic: str
) -> CanonicalSession:
    events = []
    for index, message in enumerate(messages):
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported PrefEval role: {role!r}")
        events.append(
            CanonicalEvent(
                id=f"e{index:04d}",
                role=role,
                type="message",
                content=content,
                timestamp="",
                metadata={"prefeval_message_index": index},
            )
        )
    if not events:
        raise ValueError(f"PrefEval session is empty: {asset_id}")
    return CanonicalSession(
        session_id=f"prefeval:{asset_id}",
        source="prefeval",
        domain=topic or "personalization",
        events=tuple(events),
        provenance={"asset_id": asset_id, "topic": topic},
    )


def _token_count(tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(
        tokenizer.encode(
            text, add_special_tokens=add_special_tokens, truncation=False
        )
    )


def _event_session(session: CanonicalSession, event: CanonicalEvent) -> CanonicalSession:
    return CanonicalSession(
        session_id=f"{session.session_id}:event",
        source=session.source,
        domain=session.domain,
        events=(event,),
        provenance=session.provenance,
    )


def _split_oversize_events(
    session: CanonicalSession, tokenizer, *, max_context_tokens: int
) -> CanonicalSession:
    expanded = []
    for event in session.events:
        if (
            _token_count(
                tokenizer,
                render_session(_event_session(session, event)),
                add_special_tokens=True,
            )
            <= max_context_tokens
        ):
            expanded.append(event)
            continue
        source_ids = list(
            tokenizer.encode(
                event.content, add_special_tokens=False, truncation=False
            )
        )
        cursor = 0
        part = 0
        while cursor < len(source_ids):
            low, high = cursor + 1, len(source_ids)
            best = None
            best_end = cursor
            while low <= high:
                end = (low + high) // 2
                candidate = CanonicalEvent(
                    id=f"{event.id}.part{part:03d}",
                    role=event.role,
                    type=event.type,
                    content=tokenizer.decode(
                        source_ids[cursor:end], skip_special_tokens=True
                    ),
                    timestamp=event.timestamp,
                    metadata={
                        **event.metadata,
                        "parent_event_id": event.id,
                        "source_token_start": cursor,
                        "source_token_end": end,
                    },
                )
                if (
                    _token_count(
                        tokenizer,
                        render_session(_event_session(session, candidate)),
                        add_special_tokens=True,
                    )
                    <= max_context_tokens
                ):
                    best, best_end = candidate, end
                    low = end + 1
                else:
                    high = end - 1
            if best is None:
                raise ValueError(f"PrefEval event cannot fit a segment: {event.id}")
            expanded.append(best)
            cursor = best_end
            part += 1
    return CanonicalSession(
        session_id=session.session_id,
        source=session.source,
        domain=session.domain,
        events=tuple(expanded),
        provenance=session.provenance,
    )


def segment_asset(
    *,
    asset_id: str,
    messages: list[dict[str, Any]],
    topic: str,
    tokenizer,
    max_context_tokens: int,
    event_overlap: int,
) -> tuple[list[MemorySegment], dict[str, Any]]:
    source = canonical_session(asset_id=asset_id, messages=messages, topic=topic)
    expanded = _split_oversize_events(
        source, tokenizer, max_context_tokens=max_context_tokens
    )
    canonical_segments = segment_session(
        expanded,
        max_context_tokens=max_context_tokens,
        token_count=lambda value: _token_count(
            tokenizer, value, add_special_tokens=True
        ),
        event_overlap=event_overlap,
    )
    segments = []
    for index, segment in enumerate(canonical_segments):
        text = render_session(segment)
        token_count = _token_count(tokenizer, text, add_special_tokens=True)
        if token_count > max_context_tokens:
            raise ValueError("PrefEval segmentation exceeded compiler context")
        segments.append(
            MemorySegment(
                segment_index=index,
                segment_count=len(canonical_segments),
                session_id=segment.session_id,
                event_ids=tuple(event.id for event in segment.events),
                token_count=token_count,
                text=text,
            )
        )
    return segments, {
        "source_messages": len(source.events),
        "expanded_events": len(expanded.events),
        "memory_segments": len(segments),
        "max_segment_tokens": max(item.token_count for item in segments),
    }


def save_tensor_atomic(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def latent_files(directory: Path, expected: int) -> list[Path]:
    paths = sorted(directory.glob("segment_*.pt"))
    if len(paths) != expected or not paths:
        raise ValueError(f"incomplete PrefEval latent cache: {directory}")
    return paths


def load_asset_latents(
    root: Path,
    asset_id: str,
    *,
    dataset_sha256: str,
    checkpoint_sha256: str,
    device,
) -> list[torch.Tensor]:
    directory = root / "assets" / asset_cache_key(asset_id)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if any(
        (
            meta.get("format") != LATENT_FORMAT,
            meta.get("asset_id") != asset_id,
        )
    ):
        raise ValueError(f"incompatible PrefEval latent format or asset: {directory}")
    return [
        torch.load(path, weights_only=True, map_location=device).to(device)
        for path in latent_files(directory, int(meta["segments"]))
    ]
