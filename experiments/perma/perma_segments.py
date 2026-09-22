"""Deterministic PERMA episode segmentation for RPMem Phase 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rpmem.training.corpus.normalize import segment_session
from rpmem.training.corpus.schema import (
    CanonicalEvent,
    CanonicalSession,
    render_session,
)


SEGMENTATION_FORMAT = "memlora_perma_memory_segments_v1"
ALL_HISTORY_RECOMPILE_FORMAT = "memlora_perma_all_history_token_chunks_v2"


@dataclass(frozen=True)
class MemorySegment:
    episode_index: int
    segment_index: int
    segment_count: int
    session_id: str
    event_ids: tuple[str, ...]
    token_count: int
    text: str
    truncated_event_ids: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("text")
        payload["event_ids"] = list(self.event_ids)
        payload["truncated_event_ids"] = list(self.truncated_event_ids)
        return payload


def segmentation_contract(
    *,
    max_context_tokens: int,
    event_overlap: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": SEGMENTATION_FORMAT,
        "boundary": "perma_message_then_token_chunk",
        "rendering": "formal_phase1_canonical_session",
        "max_context_tokens": max_context_tokens,
        "event_overlap": event_overlap,
        "oversize_event_policy": "lossless_token_chunks",
        "episode_order": "source_order",
        "segment_order": "source_order",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    payload["policy_id"] = (
        f"canonical_message_v1_ctx{max_context_tokens}_overlap{event_overlap}_"
        f"{payload['sha256'][:8]}"
    )
    return payload


def _normalize_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    aliases = {
        "human": "user",
        "bot": "assistant",
        "ai": "assistant",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"system", "user", "assistant", "tool", "environment"}:
        raise ValueError(f"unsupported PERMA message role: {role!r}")
    return normalized


def _token_count(tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    )


def _canonical_episode(
    task,
    episode_index: int,
) -> tuple[CanonicalSession | None, tuple[int, ...]]:
    conversation = task.raw_conversations[episode_index]
    date = task.sessions[episode_index].get("date", "")
    events = []
    empty_message_indices = []
    for message_index, message in enumerate(conversation):
        content = str(message.get("content") or "").strip()
        if not content:
            empty_message_indices.append(message_index)
            continue
        events.append(
            CanonicalEvent(
                id=f"e{message_index:04d}",
                role=_normalize_role(message.get("role")),
                type="message",
                content=content,
                timestamp=str(date),
                metadata={
                    "perma_episode_index": episode_index,
                    "perma_message_index": message_index,
                },
            )
        )
    events = tuple(events)
    if not events:
        return None, tuple(empty_message_indices)
    domain = ",".join(str(topic) for topic in task.topic) or "unknown"
    return (
        CanonicalSession(
            session_id=(
                f"perma:user{task.user_id}:{task.variant}:{task.task_id}:"
                f"type{task.task_type}:episode-{episode_index:03d}"
            ),
            source="perma",
            domain=domain,
            events=events,
            provenance={
                "user_id": task.user_id,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "variant": task.variant,
                "episode_index": episode_index,
                "date": str(date),
                "empty_message_indices": empty_message_indices,
            },
        ),
        tuple(empty_message_indices),
    )


def _event_only_session(
    episode: CanonicalSession,
    event: CanonicalEvent,
) -> CanonicalSession:
    return CanonicalSession(
        session_id=f"{episode.session_id}:segment-000",
        source=episode.source,
        domain=episode.domain,
        events=(event,),
        provenance=episode.provenance,
    )


def _split_oversize_events(
    episode: CanonicalSession,
    tokenizer,
    *,
    max_context_tokens: int,
) -> tuple[CanonicalSession, dict[str, Any]]:
    expanded: list[CanonicalEvent] = []
    split_message_ids = []
    message_parts = 0

    for event in episode.events:
        source_ids = tokenizer.encode(
            event.content,
            add_special_tokens=False,
            truncation=False,
        )
        metadata = {
            **event.metadata,
            "parent_event_id": event.id,
            "source_token_start": 0,
            "source_token_end": len(source_ids),
            "source_token_count": len(source_ids),
        }
        whole_event = CanonicalEvent(
            **{
                **event.to_dict(),
                "metadata": metadata,
            }
        )
        if (
            _token_count(
                tokenizer,
                render_session(_event_only_session(episode, whole_event)),
                add_special_tokens=True,
            )
            <= max_context_tokens
        ):
            expanded.append(whole_event)
            continue

        split_message_ids.append(event.id)
        cursor = 0
        part_index = 0
        while cursor < len(source_ids):
            low = cursor + 1
            high = len(source_ids)
            best_event = None
            best_end = cursor
            while low <= high:
                end = (low + high) // 2
                content = tokenizer.decode(
                    source_ids[cursor:end],
                    skip_special_tokens=True,
                )
                candidate = CanonicalEvent(
                    id=f"{event.id}.part{part_index:03d}",
                    role=event.role,
                    type=event.type,
                    content=content,
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
                    render_session(_event_only_session(episode, candidate)),
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
                    f"PERMA message cannot fit an empty bounded segment: "
                    f"{episode.session_id}:{event.id}"
                )
            expanded.append(best_event)
            message_parts += 1
            cursor = best_end
            part_index += 1

    return (
        CanonicalSession(
            session_id=episode.session_id,
            source=episode.source,
            domain=episode.domain,
            events=tuple(expanded),
            provenance=episode.provenance,
        ),
        {
            "split_message_ids": split_message_ids,
            "message_parts": message_parts,
        },
    )


def segment_task_memory(
    task,
    tokenizer,
    *,
    max_context_tokens: int = 4096,
    event_overlap: int = 1,
) -> tuple[list[MemorySegment], dict[str, Any]]:
    """Split every PERMA episode at message boundaries without dropping episodes."""

    all_segments: list[MemorySegment] = []
    original_content_tokens = 0
    retained_content_tokens = 0
    split_episodes = 0
    split_messages = 0
    message_parts = 0
    truncated_events = 0
    dropped_empty_messages = 0
    empty_episodes = 0
    episode_stats = []

    for episode_index in range(len(task.raw_conversations)):
        source_episode, empty_message_indices = _canonical_episode(
            task,
            episode_index,
        )
        dropped_empty_messages += len(empty_message_indices)
        if source_episode is None:
            empty_episodes += 1
            episode_stats.append(
                {
                    "episode_index": episode_index,
                    "source_messages": len(task.raw_conversations[episode_index]),
                    "expanded_message_events": 0,
                    "memory_segments": 0,
                    "segment_token_counts": [],
                    "empty_message_indices": list(empty_message_indices),
                    "split_message_ids": [],
                    "truncated_event_ids": [],
                }
            )
            continue
        source_tokens = {
            event.id: _token_count(
                tokenizer,
                event.content,
                add_special_tokens=False,
            )
            for event in source_episode.events
        }
        original_content_tokens += sum(source_tokens.values())
        episode, expansion = _split_oversize_events(
            source_episode,
            tokenizer,
            max_context_tokens=max_context_tokens,
        )
        expanded_events = {event.id: event for event in episode.events}
        split_messages += len(expansion["split_message_ids"])
        message_parts += expansion["message_parts"]

        canonical_segments = segment_session(
            episode,
            max_context_tokens=max_context_tokens,
            token_count=lambda text: _token_count(
                tokenizer,
                text,
                add_special_tokens=True,
            ),
            event_overlap=event_overlap,
        )
        if len(canonical_segments) > 1:
            split_episodes += 1

        truncated_ids: set[str] = set()
        episode_token_counts = []
        for segment_index, segment in enumerate(canonical_segments):
            text = render_session(segment)
            tokens = _token_count(tokenizer, text, add_special_tokens=True)
            if tokens > max_context_tokens:
                raise ValueError(
                    f"segmentation produced {tokens} tokens, "
                    f"limit={max_context_tokens}: {segment.session_id}"
                )
            episode_token_counts.append(tokens)
            segment_truncated = []
            for event in segment.events:
                parent_id = str(event.metadata.get("parent_event_id", event.id))
                if event.content != expanded_events[event.id].content:
                    truncated_ids.add(parent_id)
                    segment_truncated.append(parent_id)
            all_segments.append(
                MemorySegment(
                    episode_index=episode_index,
                    segment_index=segment_index,
                    segment_count=len(canonical_segments),
                    session_id=segment.session_id,
                    event_ids=tuple(event.id for event in segment.events),
                    token_count=tokens,
                    text=text,
                    truncated_event_ids=tuple(segment_truncated),
                )
            )

        if truncated_ids:
            raise ValueError(
                "formal PERMA segmentation unexpectedly truncated message content: "
                f"{episode.session_id} {sorted(truncated_ids)}"
            )
        retained_content_tokens += sum(source_tokens.values())
        truncated_events += len(truncated_ids)
        episode_stats.append(
            {
                "episode_index": episode_index,
                "source_messages": len(source_episode.events),
                "expanded_message_events": len(episode.events),
                "memory_segments": len(canonical_segments),
                "segment_token_counts": episode_token_counts,
                "empty_message_indices": list(empty_message_indices),
                "split_message_ids": expansion["split_message_ids"],
                "truncated_event_ids": sorted(truncated_ids),
            }
        )

    if not all_segments:
        raise ValueError(
            f"PERMA task {task.task_id} contains no non-empty memory messages"
        )
    coverage = (
        retained_content_tokens / original_content_tokens
        if original_content_tokens
        else 1.0
    )
    stats = {
        "episodes": len(task.raw_conversations),
        "memory_segments": len(all_segments),
        "split_episodes": split_episodes,
        "split_messages": split_messages,
        "message_parts": message_parts,
        "truncated_events": truncated_events,
        "dropped_empty_messages": dropped_empty_messages,
        "empty_episodes": empty_episodes,
        "original_content_tokens": original_content_tokens,
        "retained_content_tokens": retained_content_tokens,
        "content_token_coverage": coverage,
        "max_segment_tokens": max(
            (segment.token_count for segment in all_segments),
            default=0,
        ),
        "episode_stats": episode_stats,
    }
    return all_segments, stats
