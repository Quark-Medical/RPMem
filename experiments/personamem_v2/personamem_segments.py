"""Lossless PersonaMem-v2 history segmentation for the frozen compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rpmem.training.corpus.normalize import segment_session
from rpmem.training.corpus.schema import (
    CanonicalEvent,
    CanonicalSession,
    render_session,
)


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


def _token_count(tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    )


def canonical_history(
    history: dict[str, Any], *, relative_path: str
) -> CanonicalSession:
    persona_id = str(history["metadata"]["persona_id"])
    events = tuple(
        CanonicalEvent(
            id=f"e{index:04d}",
            role=str(message["role"]),
            type="message",
            content=str(message["content"]).strip(),
            timestamp="",
            metadata={"personamem_message_index": index},
        )
        for index, message in enumerate(history["chat_history"])
    )
    if not events:
        raise ValueError(f"PersonaMem-v2 history is empty: {relative_path}")
    return CanonicalSession(
        session_id=f"personamem-v2:persona-{persona_id}:history",
        source="personamem_v2",
        domain="personalization",
        events=events,
        provenance={
            "persona_id": persona_id,
            "history_file": Path(relative_path).as_posix(),
        },
    )


def _event_only_session(
    session: CanonicalSession, event: CanonicalEvent
) -> CanonicalSession:
    return CanonicalSession(
        session_id=f"{session.session_id}:segment-000",
        source=session.source,
        domain=session.domain,
        events=(event,),
        provenance=session.provenance,
    )


def _split_oversize_events(
    session: CanonicalSession,
    tokenizer,
    *,
    max_context_tokens: int,
) -> tuple[CanonicalSession, list[str]]:
    expanded: list[CanonicalEvent] = []
    split_ids: list[str] = []
    for event in session.events:
        source_ids = tokenizer.encode(
            event.content,
            add_special_tokens=False,
            truncation=False,
        )
        whole = CanonicalEvent(
            **{
                **event.to_dict(),
                "metadata": {
                    **event.metadata,
                    "parent_event_id": event.id,
                    "source_token_start": 0,
                    "source_token_end": len(source_ids),
                    "source_token_count": len(source_ids),
                },
            }
        )
        if (
            _token_count(
                tokenizer,
                render_session(_event_only_session(session, whole)),
                add_special_tokens=True,
            )
            <= max_context_tokens
        ):
            expanded.append(whole)
            continue
        split_ids.append(event.id)
        cursor = 0
        part_index = 0
        while cursor < len(source_ids):
            low, high = cursor + 1, len(source_ids)
            best_event = None
            best_end = cursor
            while low <= high:
                end = (low + high) // 2
                candidate = CanonicalEvent(
                    id=f"{event.id}.part{part_index:03d}",
                    role=event.role,
                    type=event.type,
                    content=tokenizer.decode(
                        source_ids[cursor:end], skip_special_tokens=True
                    ),
                    timestamp=event.timestamp,
                    metadata={
                        **event.metadata,
                        "parent_event_id": event.id,
                        "message_part_index": part_index,
                        "source_token_start": cursor,
                        "source_token_end": end,
                        "source_token_count": len(source_ids),
                    },
                )
                length = _token_count(
                    tokenizer,
                    render_session(_event_only_session(session, candidate)),
                    add_special_tokens=True,
                )
                if length <= max_context_tokens:
                    best_event = candidate
                    best_end = end
                    low = end + 1
                else:
                    high = end - 1
            if best_event is None:
                raise ValueError(
                    "message cannot fit a bounded segment: "
                    f"{session.session_id}:{event.id}"
                )
            expanded.append(best_event)
            cursor = best_end
            part_index += 1
    return (
        CanonicalSession(
            session_id=session.session_id,
            source=session.source,
            domain=session.domain,
            events=tuple(expanded),
            provenance=session.provenance,
        ),
        split_ids,
    )


def segment_history_memory(
    history: dict[str, Any],
    tokenizer,
    *,
    relative_path: str,
    max_context_tokens: int = 4096,
    event_overlap: int = 1,
) -> tuple[list[MemorySegment], dict[str, Any]]:
    source = canonical_history(history, relative_path=relative_path)
    original_tokens = sum(
        _token_count(tokenizer, event.content, add_special_tokens=False)
        for event in source.events
    )
    expanded, split_ids = _split_oversize_events(
        source,
        tokenizer,
        max_context_tokens=max_context_tokens,
    )
    canonical_segments = segment_session(
        expanded,
        max_context_tokens=max_context_tokens,
        token_count=lambda text: _token_count(
            tokenizer, text, add_special_tokens=True
        ),
        event_overlap=event_overlap,
    )
    segments = []
    for index, segment in enumerate(canonical_segments):
        text = render_session(segment)
        token_count = _token_count(tokenizer, text, add_special_tokens=True)
        if token_count > max_context_tokens:
            raise ValueError(
                f"segmentation produced {token_count} tokens: {segment.session_id}"
            )
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
        "expanded_message_events": len(expanded.events),
        "memory_segments": len(segments),
        "split_message_ids": split_ids,
        "original_content_tokens": original_tokens,
        "retained_content_tokens": original_tokens,
        "content_token_coverage": 1.0,
        "max_segment_tokens": max(segment.token_count for segment in segments),
    }
