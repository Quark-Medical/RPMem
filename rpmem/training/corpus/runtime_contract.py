"""Validate that formal runtime limits and model paths match a corpus artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _model_ref(value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else str(value)


def validate_runtime_contract(
    manifest_path: str | Path,
    *,
    base_model_path: str | Path,
    ctx_encoder_path: str | Path,
    max_ctx_len: int,
    max_teacher_ctx_tokens: int,
    max_seq_len: int,
    max_teacher_seq_len: int,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    corpus_limit = int(manifest.get("max_context_tokens", 0))
    if corpus_limit <= 0:
        raise ValueError("formal corpus must declare a positive context token limit")
    if int(max_ctx_len) != corpus_limit:
        raise ValueError(
            f"MAX_CTX_LEN={max_ctx_len} does not match corpus limit {corpus_limit}"
        )
    if int(max_teacher_ctx_tokens) != corpus_limit:
        raise ValueError(
            "MAX_TEACHER_CTX_TOKENS does not match the corpus context limit"
        )

    expected_base = _model_ref(manifest.get("tokenizer", ""))
    actual_base = _model_ref(base_model_path)
    if actual_base != expected_base:
        raise ValueError(
            f"BASE_MODEL={actual_base} does not match corpus tokenizer {expected_base}"
        )

    tokenizer_specs = manifest.get("context_tokenizers", [])
    if not tokenizer_specs:
        raise ValueError("formal corpus is missing context tokenizer metadata")
    primary_path = _model_ref(tokenizer_specs[0].get("path", ""))
    if primary_path != actual_base:
        raise ValueError("corpus primary tokenizer does not match BASE_MODEL")
    actual_ctx_encoder = _model_ref(ctx_encoder_path)
    additional_paths = {
        _model_ref(spec.get("path", "")) for spec in tokenizer_specs[1:]
    }
    if actual_ctx_encoder not in additional_paths:
        raise ValueError(
            f"CTX_ENCODER={actual_ctx_encoder} is absent from corpus tokenizers"
        )

    minimum_teacher_sequence = corpus_limit + int(max_seq_len) + 64
    if int(max_teacher_seq_len) < minimum_teacher_sequence:
        raise ValueError(
            "teacher sequence limit is too small for context, student sequence, "
            f"and template margin: {max_teacher_seq_len}<{minimum_teacher_sequence}"
        )
    return {
        "manifest": str(manifest_path),
        "base_model_path": actual_base,
        "ctx_encoder_path": actual_ctx_encoder,
        "effective_context_tokens": corpus_limit,
        "max_seq_len": int(max_seq_len),
        "max_teacher_seq_len": int(max_teacher_seq_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--ctx_encoder_path", required=True)
    parser.add_argument("--max_ctx_len", type=int, required=True)
    parser.add_argument("--max_teacher_ctx_tokens", type=int, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--max_teacher_seq_len", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_runtime_contract(
                args.manifest,
                base_model_path=args.base_model_path,
                ctx_encoder_path=args.ctx_encoder_path,
                max_ctx_len=args.max_ctx_len,
                max_teacher_ctx_tokens=args.max_teacher_ctx_tokens,
                max_seq_len=args.max_seq_len,
                max_teacher_seq_len=args.max_teacher_seq_len,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
