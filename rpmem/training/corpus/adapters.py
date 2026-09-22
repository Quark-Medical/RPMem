"""Source-row adapters for canonical Phase 1 sessions."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

from rpmem.training.corpus.schema import CanonicalSession


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _json_list(value: Any, *, field_name: str) -> list[Any]:
    """Decode JSON-string sequence fields used by Parquet exports."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must contain a JSON array") from exc
    values = _as_list(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be an array")
    return values


def _tool_payload(value: Any, *, field_name: str) -> dict[str, Any]:
    """Parse JSON or Python-literal tool payloads without executing code."""

    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be an object or serialized object")
    stripped = value.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{field_name} is not valid JSON or a literal object"
            ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must decode to an object")
    return payload


def _as_records(value: Any) -> list[dict[str, Any]]:
    """Accept both list-of-struct and HF struct-of-lists representations."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    if any(not isinstance(item, (list, tuple)) for item in value.values()):
        return [dict(value)]
    sequence_lengths = [
        len(item) for item in value.values() if isinstance(item, (list, tuple))
    ]
    if not sequence_lengths:
        return [dict(value)]
    length = max(sequence_lengths)
    records = []
    for index in range(length):
        record = {}
        for key, item in value.items():
            if isinstance(item, (list, tuple)):
                record[key] = item[index] if index < len(item) else None
            else:
                record[key] = item
        records.append(record)
    return records


def _stable_record_id(row: dict[str, Any], explicit: str | None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for key in (
        "session_id",
        "conversation_id",
        "dialogue_id",
        "trajectory_id",
        "instance_id",
        "uuid",
        "id",
        "record_id",
    ):
        if row.get(key) is not None and str(row[key]).strip():
            return str(row[key]).strip()
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _source_fields(
    source: dict[str, Any], source_record_id: str
) -> tuple[str, str, str, dict[str, Any]]:
    source_name = str(source.get("name", "")).strip()
    domain = str(source.get("domain", "")).strip()
    if not source_name or not domain:
        raise ValueError("source spec requires non-empty name and domain")
    session_id = f"{source_name}:{source_record_id}"
    provenance = {
        "source_record_id": source_record_id,
        "license": str(source.get("license", "")).strip(),
        "revision": str(source.get("revision", "")).strip(),
        "external_api_allowed": bool(source.get("external_api_allowed", False)),
    }
    for key in ("repository", "commit", "timestamp", "language", "source_url"):
        if source.get(key) not in (None, ""):
            provenance[key] = source[key]
    return session_id, source_name, domain, provenance


def _canonical_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    session_id, source_name, domain, provenance = _source_fields(
        source, source_record_id
    )
    payload = dict(row)
    payload["session_id"] = session_id
    payload["source"] = source_name
    payload["domain"] = domain
    payload["provenance"] = {**row.get("provenance", {}), **provenance}
    return CanonicalSession.from_dict(payload)


def _fixed_trajectory_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    session_id, source_name, domain, provenance = _source_fields(
        source, source_record_id
    )
    context = str(row.get("context", "")).strip()
    prompts = _as_list(row.get("prompts", row.get("prompt")))
    responses = _as_list(row.get("responses", row.get("response")))
    if not context:
        raise ValueError(f"{session_id}: fixed_trajectory row has empty context")
    if not prompts or len(prompts) != len(responses):
        raise ValueError(
            f"{session_id}: fixed_trajectory prompts/responses must be non-empty and aligned"
        )

    events = [
        {
            "id": "e0",
            "role": "environment",
            "type": "observation",
            "content": context,
        }
    ]
    memory_atoms = []
    probes = []
    for index, (prompt, response) in enumerate(zip(prompts, responses)):
        memory_atoms.append(
            {
                "id": f"m{index}",
                "type": "qa_evidence",
                "value": str(response),
                "evidence_event_ids": ["e0"],
            }
        )
        probes.append(
            {
                "id": f"p{index}",
                "prompt": str(prompt),
                "probe_type": "grounded_qa",
                "answerable": True,
                "reference": str(response),
                "evidence_event_ids": ["e0"],
            }
        )
    provenance.update(
        {
            "source_file": row.get("source_file", ""),
            "source_row": row.get("source_row"),
        }
    )
    return CanonicalSession.from_dict(
        {
            "session_id": session_id,
            "source": source_name,
            "domain": domain,
            "events": events,
            "memory_atoms": memory_atoms,
            "probes": probes,
            "provenance": provenance,
        }
    )


def _decode_token_ids(tokenizer: Any, value: Any) -> str:
    token_ids = [int(item) for item in _as_list(value) if item is not None]
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _clean_d2l_prompt(text: str) -> str:
    for marker in ("[INST]", "[/INST]", "<s>", "</s>"):
        text = str(text).replace(marker, " ")
    return re.sub(r"\s+", " ", text).strip()


def _is_nested_sequence(value: Any) -> bool:
    values = _as_list(value)
    return bool(values and isinstance(values[0], (list, tuple)))


def _d2l_tokenized_adapter(
    row: dict[str, Any],
    source: dict[str, Any],
    source_record_id: str,
    *,
    decode_tokenizer: Any,
) -> CanonicalSession:
    """Decode upstream Mistral-tokenized D2L rows into grounded QA sessions."""

    if decode_tokenizer is None:
        raise ValueError("d2l_tokenized adapter requires a decode tokenizer")
    required = {"ctx_ids", "input_ids", "response_start_end"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"d2l_tokenized row is missing fields: {missing}")

    context = _decode_token_ids(decode_tokenizer, row.get("ctx_ids"))
    input_groups = _as_list(row.get("input_ids"))
    response_spans = _as_list(row.get("response_start_end"))
    if input_groups and not _is_nested_sequence(input_groups):
        input_groups = [input_groups]
    if response_spans and not _is_nested_sequence(response_spans):
        response_spans = [response_spans]

    prompts: list[str] = []
    responses: list[str] = []
    for raw_input_ids, raw_span in zip(input_groups, response_spans):
        input_ids = [
            int(token_id)
            for token_id in _as_list(raw_input_ids)
            if token_id is not None
        ]
        bounds = [
            int(bound) for bound in _as_list(raw_span) if bound is not None
        ]
        if len(bounds) < 2 or not input_ids:
            continue
        start = max(bounds[0], 0)
        end = min(bounds[1], len(input_ids))
        if start >= end:
            continue
        prompt = _clean_d2l_prompt(
            decode_tokenizer.decode(
                input_ids[:start],
                skip_special_tokens=True,
            )
        )
        response = decode_tokenizer.decode(
            input_ids[start:end],
            skip_special_tokens=True,
        ).strip()
        if prompt and response:
            prompts.append(prompt)
            responses.append(response)

    decoded = dict(row)
    decoded.update(
        {
            "context": context,
            "prompts": prompts,
            "responses": responses,
        }
    )
    return _fixed_trajectory_adapter(
        decoded,
        source,
        source_record_id,
    )


def _tool_call_events(
    message: dict[str, Any],
    start_index: int,
    *,
    excluded_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = []
    pending_calls = []
    for tool_call in _as_list(message.get("tool_calls")):
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = str(function.get("name", "")).strip()
        raw_call_id = (
            str(tool_call.get("id", "")).strip() if isinstance(tool_call, dict) else ""
        )
        if name.casefold() in excluded_names:
            pending_calls.append(
                {
                    "match_id": raw_call_id,
                    "event_id": "",
                    "name": name,
                    "excluded": True,
                }
            )
            continue
        metadata = {}
        if raw_call_id:
            metadata["tool_call_id"] = raw_call_id
        event_id = f"e{start_index + len(events)}"
        events.append(
            {
                "id": event_id,
                "role": "assistant",
                "type": "tool_call",
                "name": name,
                "arguments": function.get("arguments", {}),
                "metadata": metadata,
            }
        )
        pending_calls.append(
            {
                "match_id": raw_call_id or event_id,
                "event_id": event_id,
                "name": name,
                "excluded": False,
            }
        )
    return events, pending_calls


def _messages_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    session_id, source_name, domain, provenance = _source_fields(
        source, source_record_id
    )
    messages = _json_list(
        row.get("messages", row.get("conversation")),
        field_name=f"{session_id}.messages",
    )
    if not messages:
        raise ValueError(f"{session_id}: messages adapter requires messages")

    events: list[dict[str, Any]] = []
    excluded_tool_names = {
        str(name).strip().casefold()
        for name in source.get("excluded_tool_names", [])
        if str(name).strip()
    }
    pending_tool_calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"{session_id}: every message must be an object")
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content") or "").strip()
        if role == "system" and source.get("drop_system_messages", False):
            continue
        if role == "assistant" and (
            message.get("tool_calls") or message.get("function_call")
        ):
            normalized_message = dict(message)
            if message.get("function_call") and not message.get("tool_calls"):
                normalized_message["tool_calls"] = [
                    {"function": message["function_call"]}
                ]
            if content and not source.get("drop_assistant_reasoning", False):
                events.append(
                    {
                        "id": f"e{len(events)}",
                        "role": "assistant",
                        "type": "message",
                        "content": content,
                    }
                )
            call_events, call_pending = _tool_call_events(
                normalized_message,
                len(events),
                excluded_names=excluded_tool_names,
            )
            events.extend(call_events)
            pending_tool_calls.extend(call_pending)
            continue
        if role in {"tool_call", "function_call"}:
            payload = _tool_payload(
                message.get("function_call", message.get("content")),
                field_name=f"{session_id}.messages[{message_index}].tool_call",
            )
            function = payload.get("function", payload)
            if not isinstance(function, dict):
                raise TypeError(
                    f"{session_id}.messages[{message_index}].tool_call function "
                    "must be an object"
                )
            name = str(function.get("name", message.get("name", ""))).strip()
            if not name:
                raise ValueError(
                    f"{session_id}.messages[{message_index}].tool_call requires a name"
                )
            raw_call_id = str(
                message.get("tool_call_id", message.get("id", ""))
            ).strip()
            if name.casefold() in excluded_tool_names:
                pending_tool_calls.append(
                    {
                        "match_id": raw_call_id,
                        "event_id": "",
                        "name": name,
                        "excluded": True,
                    }
                )
                continue
            event_id = f"e{len(events)}"
            metadata = {}
            if raw_call_id:
                metadata["tool_call_id"] = raw_call_id
            events.append(
                {
                    "id": event_id,
                    "role": "assistant",
                    "type": "tool_call",
                    "name": name,
                    "arguments": function.get("arguments", {}),
                    "metadata": metadata,
                }
            )
            pending_tool_calls.append(
                {
                    "match_id": raw_call_id or event_id,
                    "event_id": event_id,
                    "name": name,
                    "excluded": False,
                }
            )
            continue
        if role in {"tool", "function", "tool_response"}:
            metadata = {}
            raw_call_id = str(message.get("tool_call_id", "")).strip()
            pending_index = 0
            if raw_call_id:
                pending_index = next(
                    (
                        index
                        for index, call in enumerate(pending_tool_calls)
                        if call["match_id"] == raw_call_id
                    ),
                    0,
                )
            pending_call = (
                pending_tool_calls.pop(pending_index)
                if pending_tool_calls
                else {
                    "match_id": "",
                    "event_id": "",
                    "name": "",
                    "excluded": False,
                }
            )
            if pending_call["excluded"]:
                continue
            if raw_call_id:
                metadata["tool_call_id"] = raw_call_id
            elif pending_call["event_id"]:
                metadata["tool_call_id"] = pending_call["event_id"]
            events.append(
                {
                    "id": f"e{len(events)}",
                    "role": "tool",
                    "type": "tool_result",
                    "name": str(message.get("name") or pending_call["name"]).strip(),
                    "status": str(message.get("status", "success")).strip(),
                    "content": content,
                    "metadata": metadata,
                }
            )
            continue
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{session_id}: unsupported message role {role!r}")
        events.append(
            {
                "id": f"e{len(events)}",
                "role": role,
                "type": "message",
                "content": content,
            }
        )

    return CanonicalSession.from_dict(
        {
            "session_id": session_id,
            "source": source_name,
            "domain": domain,
            "events": events,
            "memory_atoms": row.get("memory_atoms", []),
            "probes": row.get("probes", []),
            "provenance": {**row.get("provenance", {}), **provenance},
            "metadata": row.get("metadata", {}),
        }
    )


def _tool_messages_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    """Normalize tool-agent messages and derive exact tool-outcome probes."""

    session = _messages_adapter(row, source, source_record_id)
    memory_atoms = [atom.to_dict() for atom in session.memory_atoms]
    probes = [probe.to_dict() for probe in session.probes]
    max_reference_chars = int(source.get("max_tool_result_reference_chars", 512))
    for event in session.events:
        if event.type != "tool_result" or not event.content:
            continue
        reference = event.content[:max_reference_chars].strip()
        tool_name = event.name or "the tool"
        _append_grounded_item(
            memory_atoms,
            probes,
            item_type="tool_outcome",
            key=str(event.metadata.get("tool_call_id", "")).strip() or event.id,
            value=reference,
            evidence_event_id=event.id,
            prompt=f"What result did {tool_name} return?",
        )
    provenance = dict(session.provenance)
    provenance.update(
        {
            "subset_name": row.get("subset_name", ""),
            "target_tools": row.get("target_tools", ""),
        }
    )
    metadata = dict(session.metadata)
    if row.get("question") not in (None, ""):
        metadata["source_question"] = row["question"]
    return CanonicalSession.from_dict(
        {
            "session_id": session.session_id,
            "source": session.source,
            "domain": session.domain,
            "events": [event.to_dict() for event in session.events],
            "memory_atoms": memory_atoms,
            "probes": probes,
            "provenance": provenance,
            "metadata": metadata,
        }
    )


def _sharegpt_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    conversations = _as_list(row.get("conversations"))
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    messages = []
    for message in conversations:
        if not isinstance(message, dict):
            raise TypeError("sharegpt conversations must contain objects")
        raw_role = str(message.get("from", message.get("role", ""))).lower()
        if raw_role not in role_map:
            raise ValueError(f"unsupported ShareGPT role {raw_role!r}")
        messages.append(
            {
                "role": role_map[raw_role],
                "content": message.get("value", message.get("content", "")),
                "name": message.get("name", ""),
                "status": message.get("status", "success"),
                "tool_call_id": message.get("tool_call_id", ""),
            }
        )
    payload = dict(row)
    payload["messages"] = messages
    return _messages_adapter(payload, source, source_record_id)


def _append_grounded_item(
    memory_atoms: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    *,
    item_type: str,
    key: str,
    value: str,
    evidence_event_id: str,
    prompt: str,
) -> None:
    value = str(value).strip()
    if not value:
        return
    item_index = len(memory_atoms)
    memory_atoms.append(
        {
            "id": f"m{item_index}",
            "type": item_type,
            "key": key,
            "value": value,
            "evidence_event_ids": [evidence_event_id],
        }
    )
    probes.append(
        {
            "id": f"p{len(probes)}",
            "prompt": prompt,
            "probe_type": item_type,
            "answerable": True,
            "reference": value,
            "evidence_event_ids": [evidence_event_id],
        }
    )


def _parallel_values(payload: Any, names_key: str, values_key: str):
    if not isinstance(payload, dict):
        return []
    if names_key not in payload and values_key not in payload:
        return [(str(name), value) for name, value in payload.items()]
    names = _as_list(payload.get(names_key))
    values = _as_list(payload.get(values_key))
    return [(str(name), value) for name, value in zip(names, values)]


def _schema_guided_dialog_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    session_id, source_name, domain, provenance = _source_fields(
        source, source_record_id
    )
    events: list[dict[str, Any]] = []
    memory_atoms: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for turn in _as_records(row.get("turns")):
        raw_speaker = str(turn.get("speaker", "")).strip().upper()
        role = "user" if raw_speaker == "USER" else "assistant"
        utterance = str(turn.get("utterance", "")).strip()
        if utterance:
            events.append(
                {
                    "id": f"e{len(events)}",
                    "role": role,
                    "type": "message",
                    "content": utterance,
                }
            )
        for frame in _as_records(turn.get("frames")):
            service = str(frame.get("service", "service")).strip() or "service"
            state_records = _as_records(frame.get("state"))
            for state in state_records:
                slot_values = state.get("slot_values", {})
                pairs = _parallel_values(slot_values, "slot_name", "slot_value_list")
                if pairs or str(state.get("active_intent", "")).strip():
                    state_event_id = f"e{len(events)}"
                    state_payload = {
                        "service": service,
                        "active_intent": state.get("active_intent", ""),
                        "slot_values": slot_values,
                    }
                    events.append(
                        {
                            "id": state_event_id,
                            "role": "environment",
                            "type": "state_change",
                            "content": json.dumps(
                                state_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        }
                    )
                    for slot, values in pairs:
                        value = ", ".join(map(str, _as_list(values)))
                        _append_grounded_item(
                            memory_atoms,
                            probes,
                            item_type="task_state",
                            key=f"{service}.{slot}",
                            value=value,
                            evidence_event_id=state_event_id,
                            prompt=f"What value is stored for {slot} in {service}?",
                        )

            for service_call in _as_records(frame.get("service_call")):
                method = str(service_call.get("method", "")).strip()
                if not method:
                    continue
                parameters = service_call.get("parameters", {})
                arguments = {
                    name: value
                    for name, value in _parallel_values(
                        parameters,
                        "parameter_slot_name",
                        "parameter_canonical_value",
                    )
                }
                call_event_id = f"e{len(events)}"
                events.append(
                    {
                        "id": call_event_id,
                        "role": "assistant",
                        "type": "tool_call",
                        "name": f"{service}.{method}",
                        "arguments": arguments,
                    }
                )
                _append_grounded_item(
                    memory_atoms,
                    probes,
                    item_type="tool_call",
                    key=f"{service}.method",
                    value=method,
                    evidence_event_id=call_event_id,
                    prompt=f"Which service method was called for {service}?",
                )

            for service_result in _as_records(frame.get("service_results")):
                result_pairs = _parallel_values(
                    service_result,
                    "service_slot_name",
                    "service_canonical_value",
                )
                if not result_pairs and isinstance(
                    service_result.get("service_results_list"), list
                ):
                    for entity in _as_records(service_result["service_results_list"]):
                        result_pairs.extend(
                            (str(key), value) for key, value in entity.items()
                        )
                if not result_pairs:
                    continue
                result_payload = {name: value for name, value in result_pairs}
                result_event_id = f"e{len(events)}"
                events.append(
                    {
                        "id": result_event_id,
                        "role": "tool",
                        "type": "tool_result",
                        "name": service,
                        "status": "success",
                        "content": json.dumps(
                            result_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    }
                )
                for slot, value in result_pairs:
                    _append_grounded_item(
                        memory_atoms,
                        probes,
                        item_type="tool_outcome",
                        key=f"{service}.{slot}",
                        value=str(value),
                        evidence_event_id=result_event_id,
                        prompt=f"What value did {service} return for {slot}?",
                    )

    provenance["services"] = _as_list(row.get("services"))
    return CanonicalSession.from_dict(
        {
            "session_id": session_id,
            "source": source_name,
            "domain": domain,
            "events": events,
            "memory_atoms": memory_atoms,
            "probes": probes,
            "provenance": provenance,
        }
    )


def _taskmaster_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    session_id, source_name, domain, provenance = _source_fields(
        source, source_record_id
    )
    events: list[dict[str, Any]] = []
    memory_atoms: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for utterance in _as_records(row.get("utterances")):
        speaker = str(utterance.get("speaker", "")).upper()
        role = "user" if speaker in {"USER", "HUMAN"} else "assistant"
        content = str(utterance.get("text", utterance.get("utterance", ""))).strip()
        event_id = f"e{len(events)}"
        events.append(
            {"id": event_id, "role": role, "type": "message", "content": content}
        )
        for segment in _as_records(utterance.get("segments")):
            value = str(segment.get("text", "")).strip()
            for annotation in _as_records(segment.get("annotations")):
                name = str(annotation.get("name", annotation.get("slot", ""))).strip()
                if not name:
                    continue
                _append_grounded_item(
                    memory_atoms,
                    probes,
                    item_type="task_state",
                    key=name,
                    value=value,
                    evidence_event_id=event_id,
                    prompt=f"What value was provided for {name}?",
                )
    return CanonicalSession.from_dict(
        {
            "session_id": session_id,
            "source": source_name,
            "domain": domain,
            "events": events,
            "memory_atoms": memory_atoms,
            "probes": probes,
            "provenance": provenance,
        }
    )


def _swe_zero_adapter(
    row: dict[str, Any], source: dict[str, Any], source_record_id: str
) -> CanonicalSession:
    trajectory = row.get("trajectory", row.get("messages_json", []))
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    messages = []
    for message in _as_list(trajectory):
        if not isinstance(message, dict):
            continue
        normalized = dict(message)
        tool_calls_json = normalized.get("tool_calls_json")
        if tool_calls_json and not normalized.get("tool_calls"):
            normalized["tool_calls"] = (
                json.loads(tool_calls_json)
                if isinstance(tool_calls_json, str)
                else tool_calls_json
            )
        messages.append(normalized)
    payload = dict(row)
    payload["messages"] = messages
    session = _messages_adapter(payload, source, source_record_id)
    memory_atoms = [atom.to_dict() for atom in session.memory_atoms]
    probes = [probe.to_dict() for probe in session.probes]
    max_reference_chars = int(source.get("max_tool_result_reference_chars", 512))
    for event in session.events:
        if event.type != "tool_result" or not event.content:
            continue
        reference = event.content[:max_reference_chars].strip()
        call_id = str(event.metadata.get("tool_call_id", "")).strip()
        prompt = "What result did the coding environment return"
        if call_id:
            prompt += f" for tool call {call_id}"
        prompt += "?"
        _append_grounded_item(
            memory_atoms,
            probes,
            item_type="tool_outcome",
            key=call_id or event.id,
            value=reference,
            evidence_event_id=event.id,
            prompt=prompt,
        )
    provenance = dict(session.provenance)
    provenance.update(
        {
            "repository": row.get("repo", ""),
            "instance_id": row.get("instance_id", ""),
            "trajectory_id": row.get("trajectory_id", source_record_id),
            "repository_license": row.get("license", ""),
            "upstream_dataset": row.get("dataset", row.get("source_dataset", "")),
        }
    )
    return CanonicalSession.from_dict(
        {
            "session_id": session.session_id,
            "source": session.source,
            "domain": session.domain,
            "events": [event.to_dict() for event in session.events],
            "memory_atoms": memory_atoms,
            "probes": probes,
            "provenance": provenance,
            "metadata": session.metadata,
        }
    )


ADAPTERS = {
    "canonical": _canonical_adapter,
    "d2l_tokenized": _d2l_tokenized_adapter,
    "fixed_trajectory": _fixed_trajectory_adapter,
    "messages": _messages_adapter,
    "schema_guided_dialog": _schema_guided_dialog_adapter,
    "sharegpt": _sharegpt_adapter,
    "swe_zero": _swe_zero_adapter,
    "taskmaster": _taskmaster_adapter,
    "tool_messages": _tool_messages_adapter,
}


def adapt_record(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    adapter: str,
    source_record_id: str | None = None,
    decode_tokenizer: Any = None,
) -> CanonicalSession:
    """Normalize one source row while retaining stable source provenance."""

    try:
        adapter_fn = ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(
            f"unsupported corpus adapter {adapter!r}; choices={sorted(ADAPTERS)}"
        ) from exc
    record_id = _stable_record_id(row, source_record_id)
    if adapter == "d2l_tokenized":
        return adapter_fn(
            row,
            source,
            record_id,
            decode_tokenizer=decode_tokenizer,
        )
    return adapter_fn(row, source, record_id)
