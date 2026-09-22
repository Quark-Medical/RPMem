"""Typed, deterministic schema for formal Phase 1 agent-memory sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "environment"})
VALID_EVENT_TYPES = frozenset(
    {
        "message",
        "tool_call",
        "tool_result",
        "observation",
        "artifact",
        "state_change",
        "decision",
        "action_summary",
    }
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(_text(item) for item in value if _text(item))


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class CanonicalEvent:
    id: str
    role: str
    type: str
    content: str = ""
    name: str = ""
    arguments: Any = None
    status: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalEvent":
        if not isinstance(payload, dict):
            raise TypeError("event must be an object")
        return cls(
            id=_text(payload.get("id")),
            role=_text(payload.get("role")),
            type=_text(payload.get("type", "message")),
            content=_text(payload.get("content")),
            name=_text(payload.get("name")),
            arguments=payload.get("arguments"),
            status=_text(payload.get("status")),
            timestamp=_text(payload.get("timestamp")),
            metadata=_mapping(payload.get("metadata"), field_name="event.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "type": self.type,
        }
        for key in ("content", "name", "status", "timestamp"):
            value = getattr(self, key)
            if value:
                result[key] = value
        if self.arguments is not None:
            result["arguments"] = self.arguments
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class MemoryAtom:
    id: str
    type: str
    value: str
    evidence_event_ids: tuple[str, ...]
    key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryAtom":
        if not isinstance(payload, dict):
            raise TypeError("memory atom must be an object")
        return cls(
            id=_text(payload.get("id")),
            type=_text(payload.get("type")),
            value=_text(payload.get("value")),
            evidence_event_ids=_string_list(payload.get("evidence_event_ids")),
            key=_text(payload.get("key")),
            metadata=_mapping(
                payload.get("metadata"), field_name="memory_atom.metadata"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "value": self.value,
            "evidence_event_ids": list(self.evidence_event_ids),
        }
        if self.key:
            result["key"] = self.key
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class Probe:
    id: str
    prompt: str
    probe_type: str
    answerable: bool
    evidence_event_ids: tuple[str, ...]
    reference: str = ""
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Probe":
        if not isinstance(payload, dict):
            raise TypeError("probe must be an object")
        answerable = payload.get("answerable", True)
        if not isinstance(answerable, bool):
            raise TypeError("probe.answerable must be bool")
        return cls(
            id=_text(payload.get("id")),
            prompt=_text(payload.get("prompt")),
            probe_type=_text(payload.get("probe_type")),
            answerable=answerable,
            evidence_event_ids=_string_list(payload.get("evidence_event_ids")),
            reference=_text(payload.get("reference")),
            split=_text(payload.get("split")),
            metadata=_mapping(payload.get("metadata"), field_name="probe.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "probe_type": self.probe_type,
            "answerable": self.answerable,
            "evidence_event_ids": list(self.evidence_event_ids),
        }
        if self.reference:
            result["reference"] = self.reference
        if self.split:
            result["split"] = self.split
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class CanonicalSession:
    session_id: str
    source: str
    domain: str
    events: tuple[CanonicalEvent, ...]
    memory_atoms: tuple[MemoryAtom, ...] = ()
    probes: tuple[Probe, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalSession":
        if not isinstance(payload, dict):
            raise TypeError("session must be an object")
        return cls(
            session_id=_text(payload.get("session_id")),
            source=_text(payload.get("source")),
            domain=_text(payload.get("domain")),
            events=tuple(
                CanonicalEvent.from_dict(item) for item in payload.get("events", [])
            ),
            memory_atoms=tuple(
                MemoryAtom.from_dict(item)
                for item in payload.get("memory_atoms", [])
            ),
            probes=tuple(Probe.from_dict(item) for item in payload.get("probes", [])),
            provenance=_mapping(
                payload.get("provenance"), field_name="session.provenance"
            ),
            metadata=_mapping(payload.get("metadata"), field_name="session.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "source": self.source,
            "domain": self.domain,
            "events": [event.to_dict() for event in self.events],
            "memory_atoms": [atom.to_dict() for atom in self.memory_atoms],
            "probes": [probe.to_dict() for probe in self.probes],
            "provenance": self.provenance,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


def _validate_unique_ids(kind: str, values: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if not value:
            raise ValueError(f"{kind} id must not be empty")
        if value in seen:
            raise ValueError(f"duplicate {kind} id: {value}")
        seen.add(value)


def validate_session(
    session: CanonicalSession,
    *,
    require_probes: bool = False,
    require_references: bool = False,
) -> None:
    """Validate identity, typed events, evidence links, and fixed targets."""

    if not session.session_id:
        raise ValueError("session_id must not be empty")
    if not session.source:
        raise ValueError(f"session {session.session_id}: source must not be empty")
    if not session.domain:
        raise ValueError(f"session {session.session_id}: domain must not be empty")
    if not session.events:
        raise ValueError(f"session {session.session_id}: events must not be empty")

    _validate_unique_ids("event", [event.id for event in session.events])
    _validate_unique_ids("memory atom", [atom.id for atom in session.memory_atoms])
    _validate_unique_ids("probe", [probe.id for probe in session.probes])
    event_ids = {event.id for event in session.events}

    for event in session.events:
        if event.role not in VALID_ROLES:
            raise ValueError(
                f"session {session.session_id}: invalid event role {event.role!r}"
            )
        if event.type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"session {session.session_id}: invalid event type {event.type!r}"
            )
        if event.type == "tool_call" and not event.name:
            raise ValueError(
                f"session {session.session_id}: tool_call {event.id} requires name"
            )
        if event.type != "tool_call" and not event.content:
            raise ValueError(
                f"session {session.session_id}: event {event.id} requires content"
            )

    def validate_evidence(kind: str, item_id: str, evidence_ids: tuple[str, ...]):
        if not evidence_ids:
            raise ValueError(
                f"session {session.session_id}: {kind} {item_id} has no evidence events"
            )
        unknown = sorted(set(evidence_ids) - event_ids)
        if unknown:
            raise ValueError(
                f"session {session.session_id}: {kind} {item_id} references "
                f"unknown evidence event(s): {', '.join(unknown)}"
            )

    for atom in session.memory_atoms:
        if not atom.type or not atom.value:
            raise ValueError(
                f"session {session.session_id}: memory atom {atom.id} requires type and value"
            )
        validate_evidence("memory atom", atom.id, atom.evidence_event_ids)

    if require_probes and not session.probes:
        raise ValueError(f"session {session.session_id}: probes must not be empty")
    for probe in session.probes:
        if not probe.prompt or not probe.probe_type:
            raise ValueError(
                f"session {session.session_id}: probe {probe.id} requires prompt and type"
            )
        if probe.answerable:
            validate_evidence("probe", probe.id, probe.evidence_event_ids)
        elif probe.evidence_event_ids:
            raise ValueError(
                f"session {session.session_id}: unanswerable probe {probe.id} "
                "must not cite evidence events"
            )
        if require_references and not probe.reference:
            raise ValueError(
                f"session {session.session_id}: probe {probe.id} requires a reference"
            )


def _canonical_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        stripped = arguments.strip()
        try:
            arguments = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def render_session(session: CanonicalSession) -> str:
    """Render typed events without discarding their machine-action semantics."""

    validate_session(session)
    lines = [
        f"[Session | id={session.session_id} | source={session.source} | "
        f"domain={session.domain}]"
    ]
    message_labels = {
        "system": "System Message",
        "user": "User Message",
        "assistant": "Assistant Message",
        "tool": "Tool Message",
        "environment": "Environment Message",
    }
    type_labels = {
        "observation": "Observation",
        "artifact": "Artifact Observation",
        "state_change": "State Change",
        "decision": "Decision",
        "action_summary": "Action Summary",
    }

    for event in session.events:
        if event.type == "tool_call":
            lines.append(f"[Tool Call | id={event.id} | name={event.name}]")
            lines.append(_canonical_arguments(event.arguments))
            if event.content:
                lines.append(event.content)
        elif event.type == "tool_result":
            status = f" | status={event.status}" if event.status else ""
            name = f" | name={event.name}" if event.name else ""
            lines.append(f"[Tool Result | id={event.id}{name}{status}]")
            lines.append(event.content)
        elif event.type == "message":
            lines.append(f"[{message_labels[event.role]} | id={event.id}]")
            lines.append(event.content)
        else:
            lines.append(f"[{type_labels[event.type]} | id={event.id}]")
            lines.append(event.content)
    return "\n".join(lines).strip()
