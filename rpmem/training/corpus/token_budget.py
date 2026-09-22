"""Tokenizer-exact context length accounting for formal corpora."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ContextTokenizerSpec:
    name: str
    path: str
    revision: str | None
    add_special_tokens: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "revision": self.revision,
            "add_special_tokens": self.add_special_tokens,
        }


class ContextTokenCounters:
    def __init__(self, entries: list[tuple[ContextTokenizerSpec, Any]]):
        if not entries:
            raise ValueError("at least one context tokenizer is required")
        self._entries = tuple(entries)

    @property
    def specs(self) -> tuple[ContextTokenizerSpec, ...]:
        return tuple(spec for spec, _ in self._entries)

    def counts(self, text: str) -> dict[str, int]:
        return {
            spec.name: len(
                tokenizer.encode(
                    text,
                    add_special_tokens=spec.add_special_tokens,
                )
            )
            for spec, tokenizer in self._entries
        }

    def primary_count(self, text: str) -> int:
        spec, tokenizer = self._entries[0]
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=spec.add_special_tokens,
            )
        )

    def max_count(self, text: str) -> int:
        return max(self.counts(text).values())

    def over_limit(self, text: str, *, max_tokens: int) -> dict[str, int]:
        return {
            name: count
            for name, count in self.counts(text).items()
            if count > max_tokens
        }

    def metadata(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs]


def build_context_token_counters(
    config: Mapping[str, Any],
) -> ContextTokenCounters:
    """Load the accounting tokenizer plus every model-facing validator."""

    tokenizer_path = str(config.get("tokenizer", "")).strip()
    if not tokenizer_path:
        raise ValueError("tokenizer is required")

    primary_name = str(config.get("tokenizer_name", "primary")).strip()
    if not primary_name:
        raise ValueError("tokenizer_name must be non-empty")

    specs = [
        ContextTokenizerSpec(
            name=primary_name,
            path=tokenizer_path,
            revision=str(config.get("tokenizer_revision") or "") or None,
            add_special_tokens=bool(config.get("tokenizer_add_special_tokens", False)),
        )
    ]
    seen_names = {primary_name}
    raw_additional = config.get("context_tokenizers", [])
    if raw_additional is None:
        raw_additional = []
    if not isinstance(raw_additional, list):
        raise TypeError("context_tokenizers must be a list")
    for raw_spec in raw_additional:
        if not isinstance(raw_spec, Mapping):
            raise TypeError("each context_tokenizers entry must be a mapping")
        name = str(raw_spec.get("name", "")).strip()
        path = str(raw_spec.get("path", "")).strip()
        if not name or not path:
            raise ValueError("context tokenizer entries require name and path")
        if name in seen_names:
            raise ValueError(f"duplicate context tokenizer name: {name}")
        seen_names.add(name)
        specs.append(
            ContextTokenizerSpec(
                name=name,
                path=path,
                revision=str(raw_spec.get("revision") or "") or None,
                add_special_tokens=bool(raw_spec.get("add_special_tokens", False)),
            )
        )

    from transformers import AutoTokenizer

    entries = [
        (
            spec,
            AutoTokenizer.from_pretrained(
                spec.path,
                revision=spec.revision,
                trust_remote_code=True,
            ),
        )
        for spec in specs
    ]
    return ContextTokenCounters(entries)
