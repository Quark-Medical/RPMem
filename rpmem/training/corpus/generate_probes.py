"""Generate grounded memory atoms and probes for normalized sessions."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

from rpmem.training.corpus.build import _expand_paths, _iter_rows
from rpmem.training.corpus.schema import (
    CanonicalSession,
    render_session,
    validate_session,
)
from rpmem.training.corpus.shards import write_json_atomic


GENERATION_POLICY = "grounded_compact_v3"


SYSTEM_PROMPT = """Create concise grounded probes for one agent session.
Return one compact JSON object and no prose. Use only facts explicitly present
in the event stream. Copy evidence event IDs exactly from the event headers;
never invent an ID and never use `session` as an ID. An answerable probe needs
one or more valid evidence IDs. An unanswerable probe must use an empty evidence
list. Keep prompts and references concise. Do not emit markdown, memory atoms,
or hidden reasoning. Use this compact schema, where each `p` item is
[prompt, probe_type, answerable_as_1_or_0, reference, evidence_event_ids]:
{"p":[["...","fact_recall",1,"...",["e0"]]]}"""


_TRAILING_JSON_COMMA = re.compile(r",\s*([}\]])")


def _next_generated_id(items: list[dict[str, Any]], prefix: str) -> str:
    existing = {str(item.get("id", "")) for item in items}
    index = 0
    while f"{prefix}{index}" in existing:
        index += 1
    return f"{prefix}{index}"


def _repair_json_text(text: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] != "\\":
            output.append(text[cursor])
            cursor += 1
            continue
        end = cursor
        while end < len(text) and text[end] == "\\":
            end += 1
        run = text[cursor:end]
        next_character = text[end] if end < len(text) else ""
        if len(run) % 2 and next_character not in '"\\/bfnrtu':
            run += "\\"
        output.append(run)
        cursor = end
    repaired = "".join(output)
    return _TRAILING_JSON_COMMA.sub(r"\1", repaired)


def _parse_mapping(text: str) -> dict[str, Any]:
    repaired = _repair_json_text(text.strip())
    try:
        payload, _ = json.JSONDecoder().raw_decode(repaired)
    except json.JSONDecodeError:
        literal_text = text.strip()
        final_brace = literal_text.rfind("}")
        if final_brace >= 0:
            literal_text = literal_text[: final_brace + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            payload = ast.literal_eval(literal_text)
    if not isinstance(payload, dict):
        raise ValueError("generator output must be a JSON object")
    return payload


def _extract_complete_array_values(text: str, key: str) -> list[Any]:
    repaired = _repair_json_text(text)
    match = re.search(rf'["\']{re.escape(key)}["\']\s*:\s*\[', repaired)
    if match is None:
        return []
    decoder = json.JSONDecoder()
    cursor = match.end()
    items: list[Any] = []
    while cursor < len(repaired):
        while cursor < len(repaired) and repaired[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(repaired) or repaired[cursor] == "]":
            break
        try:
            item, cursor = decoder.raw_decode(repaired, cursor)
        except json.JSONDecodeError:
            break
        items.append(item)
    return items


def _expand_compact_probes(payload: dict[str, Any]) -> dict[str, Any]:
    compact = payload.get("p", [])
    if compact and not isinstance(compact, list):
        raise ValueError("generated compact probes must be a list")
    probes = list(payload.get("probes", []))
    for item in compact:
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            continue
        prompt, probe_type, answerable, reference, evidence_event_ids = item
        if isinstance(answerable, int):
            answerable = bool(answerable)
        probes.append(
            {
                "prompt": prompt,
                "probe_type": probe_type,
                "answerable": answerable,
                "reference": reference,
                "evidence_event_ids": evidence_event_ids,
            }
        )
    payload = dict(payload)
    payload.pop("p", None)
    payload["probes"] = probes
    payload.setdefault("memory_atoms", [])
    return payload


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("generator output contains no JSON object")
    serialized = stripped[start:]
    try:
        payload = _parse_mapping(serialized)
    except (SyntaxError, ValueError, json.JSONDecodeError) as original_exc:
        probes = _extract_complete_array_values(serialized, "probes")
        compact_probes = _extract_complete_array_values(serialized, "p")
        memory_atoms = _extract_complete_array_values(serialized, "memory_atoms")
        if not probes and not memory_atoms:
            if not compact_probes:
                raise original_exc
        payload = {
            "probes": probes,
            "p": compact_probes,
            "memory_atoms": memory_atoms,
        }
    payload = _expand_compact_probes(payload)
    if not isinstance(payload.get("memory_atoms", []), list):
        raise ValueError("generated memory_atoms must be a list")
    if not isinstance(payload.get("probes", []), list):
        raise ValueError("generated probes must be a list")
    return payload


def merge_generated_payload(
    session: CanonicalSession,
    generated_text: str,
    *,
    target_probes: int,
    tolerate_invalid_items: bool = False,
) -> CanonicalSession:
    """Normalize generator IDs, deduplicate targets, and validate evidence links."""

    payload = _extract_json_object(generated_text)
    atoms = [atom.to_dict() for atom in session.memory_atoms]
    probes = [probe.to_dict() for probe in session.probes]
    atom_signatures = {
        (
            str(atom.get("type", "")),
            str(atom.get("key", "")),
            str(atom.get("value", "")),
        )
        for atom in atoms
    }
    probe_signatures = {
        (
            str(probe.get("prompt", "")).casefold(),
            str(probe.get("reference", "")).casefold(),
        )
        for probe in probes
    }
    event_ids = {event.id for event in session.events}

    def evidence_ids(value: Any) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = []
        return [str(item).strip() for item in values if str(item).strip() in event_ids]

    for generated_atom in payload.get("memory_atoms", []):
        if not isinstance(generated_atom, dict):
            if tolerate_invalid_items:
                continue
            raise ValueError("generated memory atom must be an object")
        atom = dict(generated_atom)
        if tolerate_invalid_items:
            atom["type"] = str(atom.get("type") or "").strip()
            atom["value"] = str(atom.get("value") or "").strip()
            atom["evidence_event_ids"] = evidence_ids(atom.get("evidence_event_ids"))
            if not atom["type"] or not atom["value"] or not atom["evidence_event_ids"]:
                continue
        signature = (
            str(atom.get("type", "")),
            str(atom.get("key", "")),
            str(atom.get("value", "")),
        )
        if signature in atom_signatures:
            continue
        atom["id"] = _next_generated_id(atoms, "g_m")
        atom_signatures.add(signature)
        atoms.append(atom)

    for generated_probe in payload.get("probes", []):
        if len(probes) >= target_probes:
            break
        if not isinstance(generated_probe, dict):
            if tolerate_invalid_items:
                continue
            raise ValueError("generated probe must be an object")
        probe = dict(generated_probe)
        if tolerate_invalid_items:
            prompt = str(probe.get("prompt") or "").strip()
            reference = str(probe.get("reference") or "").strip()
            probe_type = str(probe.get("probe_type") or "grounded_recall").strip()
            raw_answerable = probe.get("answerable", True)
            if isinstance(raw_answerable, str):
                answerable = raw_answerable.strip().casefold() not in {
                    "false",
                    "0",
                    "no",
                }
            else:
                answerable = bool(raw_answerable)
            probe_evidence = evidence_ids(probe.get("evidence_event_ids"))
            if not prompt or not reference or not probe_type:
                continue
            if answerable and not probe_evidence:
                continue
            probe.update(
                {
                    "prompt": prompt,
                    "probe_type": probe_type,
                    "answerable": answerable,
                    "reference": reference,
                    "evidence_event_ids": probe_evidence if answerable else [],
                }
            )
        signature = (
            str(probe.get("prompt", "")).casefold(),
            str(probe.get("reference", "")).casefold(),
        )
        if signature in probe_signatures:
            continue
        probe["id"] = _next_generated_id(probes, "g_p")
        probe_signatures.add(signature)
        probes.append(probe)

    enriched = CanonicalSession.from_dict(
        {
            **session.to_dict(),
            "memory_atoms": atoms,
            "probes": probes,
        }
    )
    validate_session(enriched, require_probes=True, require_references=True)
    return enriched


def ensure_target_probe_count(session: CanonicalSession, target_probes: int) -> None:
    if len(session.probes) != target_probes:
        raise ValueError(
            f"session {session.session_id} produced {len(session.probes)} probes; "
            f"target_probes={target_probes}"
        )


def select_target_probes(
    session: CanonicalSession, target_probes: int
) -> CanonicalSession:
    """Deterministically cap existing probes while retaining type diversity."""

    if target_probes < 1:
        raise ValueError("target_probes must be positive")
    if len(session.probes) <= target_probes:
        return session

    def rank(probe) -> bytes:
        return hashlib.sha256(
            f"{session.session_id}\0{probe.id}".encode("utf-8")
        ).digest()

    selected_ids: set[str] = set()
    probes_by_type: dict[str, list[Any]] = {}
    for probe in session.probes:
        probes_by_type.setdefault(probe.probe_type, []).append(probe)
    for probe_type in sorted(probes_by_type):
        if len(selected_ids) >= target_probes:
            break
        selected_ids.add(min(probes_by_type[probe_type], key=rank).id)
    for probe in sorted(session.probes, key=rank):
        if len(selected_ids) >= target_probes:
            break
        selected_ids.add(probe.id)

    return CanonicalSession(
        session_id=session.session_id,
        source=session.source,
        domain=session.domain,
        events=session.events,
        memory_atoms=session.memory_atoms,
        probes=tuple(probe for probe in session.probes if probe.id in selected_ids),
        provenance=session.provenance,
        metadata=session.metadata,
    )


def seed_grounded_source_probes(
    session: CanonicalSession,
    target_probes: int,
    *,
    max_new_probes: int = 4,
    max_prompt_chars: int = 512,
    max_reference_chars: int = 512,
) -> CanonicalSession:
    """Reuse explicit trajectory outcomes before asking a model to invent probes."""

    if max_new_probes <= 0 or len(session.probes) >= target_probes:
        return select_target_probes(session, target_probes)
    probes = [probe.to_dict() for probe in session.probes]
    signatures = {
        (str(probe["prompt"]).casefold(), str(probe["reference"]).casefold())
        for probe in probes
    }
    added = 0

    def add_probe(
        *,
        prompt: str,
        reference: str,
        evidence_event_ids: list[str],
        probe_type: str,
    ) -> None:
        nonlocal added
        if len(probes) >= target_probes or added >= max_new_probes:
            return
        prompt = str(prompt).strip()[:max_prompt_chars].strip()
        reference = str(reference).strip()[:max_reference_chars].strip()
        signature = (prompt.casefold(), reference.casefold())
        if not prompt or not reference or signature in signatures:
            return
        probes.append(
            {
                "id": _next_generated_id(probes, "s_p"),
                "prompt": prompt,
                "probe_type": probe_type,
                "answerable": True,
                "reference": reference,
                "evidence_event_ids": evidence_event_ids,
                "metadata": {"generation": "source_derived"},
            }
        )
        signatures.add(signature)
        added += 1

    for index, event in enumerate(session.events[1:], start=1):
        previous = session.events[index - 1]
        if event.role == "assistant" and previous.role == "user":
            add_probe(
                prompt=previous.content,
                reference=event.content,
                evidence_event_ids=[previous.id, event.id],
                probe_type="source_turn_response",
            )
        elif event.type == "tool_result" and previous.type == "tool_call":
            tool_name = event.name or previous.name or "tool"
            add_probe(
                prompt=(
                    f"What result did the {tool_name} tool return at step {index + 1}?"
                ),
                reference=event.content,
                evidence_event_ids=[previous.id, event.id],
                probe_type="source_tool_result",
            )

    enriched = CanonicalSession.from_dict(
        {
            **session.to_dict(),
            "probes": probes,
        }
    )
    if enriched.probes:
        validate_session(enriched, require_probes=True, require_references=True)
    return enriched


def fill_grounded_fallback_probes(
    session: CanonicalSession,
    target_probes: int,
    *,
    max_reference_chars: int = 512,
) -> CanonicalSession:
    """Fill rare residual failures with auditable event-grounded recall probes."""

    if len(session.probes) >= target_probes:
        return select_target_probes(session, target_probes)
    probes = [probe.to_dict() for probe in session.probes]
    signatures = {
        (str(probe["prompt"]).casefold(), str(probe["reference"]).casefold())
        for probe in probes
    }

    def add_probe(
        *,
        prompt: str,
        reference: str,
        evidence_event_ids: list[str],
        probe_type: str,
    ) -> None:
        if len(probes) >= target_probes:
            return
        prompt = str(prompt).strip()
        reference = str(reference).strip()[:max_reference_chars].strip()
        signature = (prompt.casefold(), reference.casefold())
        if not prompt or not reference or signature in signatures:
            return
        probes.append(
            {
                "id": _next_generated_id(probes, "g_p"),
                "prompt": prompt,
                "probe_type": probe_type,
                "answerable": True,
                "reference": reference,
                "evidence_event_ids": evidence_event_ids,
            }
        )
        signatures.add(signature)

    for index, event in enumerate(session.events[1:], start=1):
        previous = session.events[index - 1]
        if event.role != "assistant" or previous.role != "user":
            continue
        add_probe(
            prompt=previous.content[:max_reference_chars],
            reference=event.content,
            evidence_event_ids=[previous.id, event.id],
            probe_type="fallback_turn_response",
        )

    templates = (
        "What information was recorded at session step {step}?",
        "What occurred at session step {step}?",
        "What should be remembered from session step {step}?",
        "Summarize the recorded content at session step {step}.",
        "What detail did session step {step} establish?",
        "What does the record say at session step {step}?",
        "What content was communicated at session step {step}?",
        "Which result or statement appears at session step {step}?",
        "What is the key recorded outcome of session step {step}?",
        "What can be recalled directly from session step {step}?",
    )
    event_references: list[tuple[int, Any, str]] = []
    for index, event in enumerate(session.events, start=1):
        reference = str(event.content or "").strip()
        if not reference and event.arguments is not None:
            reference = json.dumps(
                event.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if not reference and event.type == "tool_call":
            reference = f"Called {event.name}."
        if reference:
            event_references.append((index, event, reference))

    for template in templates:
        for step, event, reference in event_references:
            add_probe(
                prompt=template.format(step=step),
                reference=reference,
                evidence_event_ids=[event.id],
                probe_type="fallback_event_recall",
            )
            if len(probes) >= target_probes:
                break
        if len(probes) >= target_probes:
            break

    enriched = CanonicalSession.from_dict(
        {
            **session.to_dict(),
            "probes": probes,
        }
    )
    validate_session(enriched, require_probes=True, require_references=True)
    return enriched


def require_provider_permission(session: CanonicalSession, provider: str) -> None:
    if provider in {"openai", "openai_compatible"} and not bool(
        session.provenance.get("external_api_allowed", False)
    ):
        raise ValueError(
            f"session {session.session_id} has external_api_allowed=false; "
            "cannot send it to an external provider"
        )


def repair_and_load_completed(path: str | Path) -> set[str]:
    """Remove a partial final JSONL line and return completed session IDs."""

    path = Path(path)
    if not path.exists():
        return set()
    completed: set[str] = set()
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                payload = json.loads(line)
                session_id = str(payload["session_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                break
            completed.add(session_id)
            valid_end = handle.tell()
    if path.stat().st_size != valid_end:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
    return completed


def build_generation_prompt(
    session: CanonicalSession,
    target_probes: int,
    *,
    attempt: int = 0,
) -> str:
    existing = len(session.probes)
    missing = max(target_probes - existing, 0)
    repair = ""
    if attempt:
        repair = (
            f"Repair attempt {attempt}: the previous output was invalid or incomplete. "
            "Use shorter strings and strictly valid compact JSON. "
        )
    existing_block = ""
    if session.probes:
        existing_prompts = [probe.prompt for probe in session.probes]
        existing_block = (
            "\nDo not repeat these existing probe prompts: "
            + json.dumps(existing_prompts, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
    return (
        repair + f"Create exactly {missing} additional grounded probes "
        f"so the session has exactly {target_probes} total."
        + existing_block
        + "\n"
        + render_session(session)
    )


def singleton_generation_budgets(max_new_tokens: int) -> tuple[int, ...]:
    """Preserve the formal budget, then progressively reduce only OOM singletons."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    budgets: list[int] = []
    for candidate in (max_new_tokens, 640, 512, 384, 256):
        candidate = min(candidate, max_new_tokens)
        if candidate not in budgets:
            budgets.append(candidate)
    return tuple(budgets)


class LocalTransformersProvider:
    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        if temperature == 0:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_ooms = 0
        self.singleton_ooms = 0
        self.adaptive_singleton_successes = 0
        self.adaptive_budget_successes: dict[int, int] = {}

    def _prompt(
        self,
        session: CanonicalSession,
        target_probes: int,
        *,
        attempt: int = 0,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generation_prompt(
                    session,
                    target_probes,
                    attempt=attempt,
                ),
            },
        ]
        template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _generate_batch_once(
        self,
        sessions: list[CanonicalSession],
        target_probes: int,
        *,
        attempt: int,
        max_new_tokens: int,
    ) -> list[str]:
        prompts = [
            self._prompt(session, target_probes, attempt=attempt)
            for session in sessions
        ]
        inputs = None
        output = None
        generated = None
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                "cuda"
            )
            with self.torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": self.temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                if self.temperature > 0:
                    generation_kwargs["temperature"] = self.temperature
                output = self.model.generate(**inputs, **generation_kwargs)
            generated = output[:, inputs["input_ids"].shape[1] :]
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        finally:
            del generated
            del output
            del inputs

    def generate_batch(
        self,
        sessions: list[CanonicalSession],
        target_probes: int,
        attempt: int = 0,
    ) -> list[str]:
        budgets = (self.max_new_tokens,)
        if len(sessions) == 1:
            budgets = singleton_generation_budgets(self.max_new_tokens)

        last_oom = ""
        for max_new_tokens in budgets:
            oom = False
            try:
                generated = self._generate_batch_once(
                    sessions,
                    target_probes,
                    attempt=attempt,
                    max_new_tokens=max_new_tokens,
                )
            except self.torch.OutOfMemoryError as exc:
                # Keep only text. Retaining the exception would retain its traceback
                # and all CUDA tensors while the caller recursively splits the batch.
                last_oom = str(exc)
                oom = True

            if not oom:
                if max_new_tokens != self.max_new_tokens:
                    self.adaptive_singleton_successes += 1
                    self.adaptive_budget_successes[max_new_tokens] = (
                        self.adaptive_budget_successes.get(max_new_tokens, 0) + 1
                    )
                return generated

            if len(sessions) == 1:
                self.singleton_ooms += 1
            else:
                self.batch_ooms += 1
            # This runs after the except block, when the failed generation traceback
            # and its CUDA tensors are no longer live.
            self.recover()
            if len(sessions) > 1:
                break

        scope = "singleton" if len(sessions) == 1 else f"batch_size={len(sessions)}"
        raise self.torch.OutOfMemoryError(
            f"CUDA OOM after generation budgets {list(budgets)} ({scope}): {last_oom}"
        )

    def generate(
        self,
        session: CanonicalSession,
        target_probes: int,
        attempt: int = 0,
    ) -> str:
        return self.generate_batch([session], target_probes, attempt=attempt)[0]

    def recover(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        return {
            "batch_ooms": self.batch_ooms,
            "singleton_ooms": self.singleton_ooms,
            "adaptive_singleton_successes": self.adaptive_singleton_successes,
            "adaptive_budget_successes": {
                str(budget): count
                for budget, count in sorted(self.adaptive_budget_successes.items())
            },
        }


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        max_new_tokens: int,
        temperature: float,
        retries: int = 5,
    ):
        if not api_key:
            raise ValueError("external provider API key is empty")
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.retries = retries

    def generate(
        self,
        session: CanonicalSession,
        target_probes: int,
        attempt: int = 0,
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_generation_prompt(
                            session,
                            target_probes,
                            attempt=attempt,
                        ),
                    },
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for retry_attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read())
                return str(payload["choices"][0]["message"]["content"])
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                json.JSONDecodeError,
            ):
                if retry_attempt + 1 >= self.retries:
                    raise
                time.sleep(min(2**retry_attempt, 30))
        raise AssertionError("unreachable")

    def generate_batch(
        self,
        sessions: list[CanonicalSession],
        target_probes: int,
        attempt: int = 0,
    ) -> list[str]:
        return [
            self.generate(session, target_probes, attempt=attempt)
            for session in sessions
        ]

    def recover(self) -> None:
        return None


def _build_provider(args):
    if args.provider == "local":
        return LocalTransformersProvider(
            args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    api_key = os.environ.get(args.api_key_env, "")
    return OpenAICompatibleProvider(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        retries=args.retries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base_url",
        default="",
    )
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--target_probes", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--source_batch_rows", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_probes", type=int, default=0)
    parser.add_argument("--generation_policy", default=GENERATION_POLICY)
    parser.add_argument("--max_generation_rounds", type=int, default=2)
    parser.add_argument(
        "--grounded_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fsync_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.target_probes < 1:
        raise ValueError("target_probes must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard_id/num_shards")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.max_source_probes < 0:
        raise ValueError("max_source_probes must be non-negative")
    if not args.generation_policy.strip():
        raise ValueError("generation_policy must be non-empty")
    if args.max_generation_rounds < 1:
        raise ValueError("max_generation_rounds must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"part-{args.shard_id:05d}.jsonl"
    error_path = output_dir / f"errors-{args.shard_id:05d}.jsonl"
    if args.overwrite:
        output_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)
    completed = repair_and_load_completed(output_path)
    completed_before = len(completed)
    provider = _build_provider(args)
    input_paths = _expand_paths(args.inputs, Path.cwd())
    processed = skipped = failures = fallback_sessions = fallback_probes = 0
    source_seeded_sessions = source_seeded_probes = 0
    global_index = 0
    started = time.monotonic()
    pending: list[tuple[int, CanonicalSession]] = []
    next_log = args.log_every if args.log_every > 0 else 0
    last_fsync_processed = 0
    with output_path.open("a") as output_file, error_path.open("a") as error_file:

        def maybe_log() -> None:
            nonlocal next_log
            selected = processed + skipped + failures
            if next_log <= 0 or selected < next_log:
                return
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[probes shard={args.shard_id}/{args.num_shards}] "
                f"selected={selected} written={processed} resumed={skipped} "
                f"failed={failures} fallback={fallback_sessions} "
                f"rate={selected / elapsed:.2f} "
                f"new_rate={processed / elapsed:.2f} sessions/s",
                flush=True,
            )
            while next_log <= selected:
                next_log += args.log_every

        def write_failure(
            current_index: int,
            session_id: str,
            exc: Exception,
            *,
            generated_text: str = "",
        ) -> None:
            nonlocal failures
            failures += 1
            payload = {
                "global_source_index": current_index,
                "session_id": session_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if generated_text:
                payload["generated_text"] = generated_text[:16_384]
            error_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            error_file.flush()
            maybe_log()

        def write_session(current_index: int, session: CanonicalSession) -> None:
            nonlocal processed, last_fsync_processed
            nonlocal fallback_sessions, fallback_probes
            nonlocal source_seeded_sessions, source_seeded_probes
            ensure_target_probe_count(session, args.target_probes)
            fallback_count = sum(
                probe.probe_type.startswith("fallback_") for probe in session.probes
            )
            if fallback_count:
                fallback_sessions += 1
                fallback_probes += fallback_count
            source_probe_count = sum(
                probe.metadata.get("generation") == "source_derived"
                for probe in session.probes
            )
            if source_probe_count:
                source_seeded_sessions += 1
                source_seeded_probes += source_probe_count
            payload = session.to_dict()
            payload["global_source_index"] = current_index
            payload["probe_generation_policy"] = args.generation_policy
            if fallback_count:
                payload["probe_generation_fallback_count"] = fallback_count
            if source_probe_count:
                payload["probe_generation_source_count"] = source_probe_count
            output_file.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            processed += 1
            if processed - last_fsync_processed >= max(args.fsync_every, 1):
                output_file.flush()
                os.fsync(output_file.fileno())
                last_fsync_processed = processed
            maybe_log()

        def generate_pending(
            entries: list[tuple[int, CanonicalSession]],
            *,
            attempt: int = 0,
        ) -> None:
            if not entries:
                return
            generated_texts: list[str] | None = None
            generation_error: tuple[str, str] | None = None
            try:
                generated_texts = provider.generate_batch(
                    [session for _, session in entries],
                    args.target_probes,
                    attempt=attempt,
                )
                if len(generated_texts) != len(entries):
                    raise ValueError(
                        "provider returned a different number of generations than inputs"
                    )
            except Exception as exc:
                # Store no exception object: its traceback can retain the failed
                # generation's CUDA tensors. Recovery and recursion must happen only
                # after leaving this except block.
                generation_error = (type(exc).__name__, str(exc))

            if generation_error is not None:
                provider.recover()
                if len(entries) > 1:
                    middle = len(entries) // 2
                    generate_pending(entries[:middle], attempt=attempt)
                    generate_pending(entries[middle:], attempt=attempt)
                else:
                    current_index, session = entries[0]
                    error_type, error_message = generation_error
                    if args.grounded_fallback and error_type == "OutOfMemoryError":
                        try:
                            fallback = fill_grounded_fallback_probes(
                                session,
                                args.target_probes,
                            )
                            write_session(current_index, fallback)
                        except Exception as fallback_exc:
                            write_failure(
                                current_index,
                                session.session_id,
                                RuntimeError(
                                    f"provider {error_type}: {error_message}; "
                                    f"grounded fallback failed: {fallback_exc}"
                                ),
                            )
                    else:
                        write_failure(
                            current_index,
                            session.session_id,
                            RuntimeError(f"provider {error_type}: {error_message}"),
                        )
                return
            assert generated_texts is not None
            retry_entries: list[tuple[int, CanonicalSession]] = []
            for (current_index, session), generated_text in zip(
                entries, generated_texts
            ):
                try:
                    enriched = merge_generated_payload(
                        session,
                        generated_text,
                        target_probes=args.target_probes,
                        tolerate_invalid_items=True,
                    )
                    if len(enriched.probes) == args.target_probes:
                        write_session(current_index, enriched)
                    elif attempt + 1 < args.max_generation_rounds:
                        retry_entries.append((current_index, enriched))
                    elif args.grounded_fallback:
                        fallback = fill_grounded_fallback_probes(
                            enriched,
                            args.target_probes,
                        )
                        write_session(current_index, fallback)
                    else:
                        write_failure(
                            current_index,
                            session.session_id,
                            ValueError(
                                f"session {session.session_id} produced "
                                f"{len(enriched.probes)} probes after "
                                f"{args.max_generation_rounds} generation rounds; "
                                f"target_probes={args.target_probes}"
                            ),
                            generated_text=generated_text,
                        )
                except Exception as exc:
                    if attempt + 1 < args.max_generation_rounds:
                        retry_entries.append((current_index, session))
                    elif args.grounded_fallback:
                        try:
                            fallback = fill_grounded_fallback_probes(
                                session,
                                args.target_probes,
                            )
                            write_session(current_index, fallback)
                        except Exception as fallback_exc:
                            write_failure(
                                current_index,
                                session.session_id,
                                fallback_exc,
                                generated_text=generated_text,
                            )
                    else:
                        write_failure(
                            current_index,
                            session.session_id,
                            exc,
                            generated_text=generated_text,
                        )
            if retry_entries:
                generate_pending(retry_entries, attempt=attempt + 1)

        for path in input_paths:
            for row in _iter_rows(path, args.source_batch_rows):
                current_index = global_index
                global_index += 1
                if current_index % args.num_shards != args.shard_id:
                    continue
                try:
                    session = CanonicalSession.from_dict(row)
                    if session.session_id in completed:
                        skipped += 1
                        maybe_log()
                        continue
                    validate_session(session)
                    session = select_target_probes(session, args.target_probes)
                    session = seed_grounded_source_probes(
                        session,
                        args.target_probes,
                        max_new_probes=args.max_source_probes,
                    )
                    if len(session.probes) < args.target_probes:
                        require_provider_permission(session, args.provider)
                        pending.append((current_index, session))
                        if len(pending) >= args.batch_size:
                            generate_pending(pending)
                            pending.clear()
                    else:
                        validate_session(
                            session, require_probes=True, require_references=True
                        )
                        write_session(current_index, session)
                except Exception as exc:
                    write_failure(
                        current_index,
                        str(row.get("session_id", row.get("id", ""))),
                        exc,
                    )
        generate_pending(pending)
        pending.clear()
        output_file.flush()
        os.fsync(output_file.fileno())

    provider_stats_fn = getattr(provider, "stats", None)
    provider_stats = provider_stats_fn() if callable(provider_stats_fn) else {}
    stats = {
        "format": "memlora_probe_generation_shard_v1",
        "generation_policy": args.generation_policy,
        "provider": args.provider,
        "model": args.model,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "target_probes": args.target_probes,
        "batch_size": args.batch_size,
        "max_source_probes": args.max_source_probes,
        "max_generation_rounds": args.max_generation_rounds,
        "grounded_fallback": args.grounded_fallback,
        "fallback_sessions": fallback_sessions,
        "fallback_probes": fallback_probes,
        "source_seeded_sessions": source_seeded_sessions,
        "source_seeded_probes": source_seeded_probes,
        "completed_before": completed_before,
        "processed": processed,
        "resumed_skips": skipped,
        "failures": failures,
        "provider_stats": provider_stats,
        "output": output_path.name,
        "errors": error_path.name,
    }
    write_json_atomic(output_dir / f"stats-{args.shard_id:05d}.json", stats)
    if args.strict and failures:
        raise SystemExit(
            f"probe generation shard {args.shard_id} failed for {failures} sessions; "
            f"see {error_path}"
        )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
