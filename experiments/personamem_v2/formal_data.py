"""Pinned PersonaMem-v2 32K text dataset model and validation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_FORMAT = "memlora_personamem_v2_dataset_freeze_v1"
EXAMPLE_FORMAT = "memlora_personamem_v2_text_example_v1"
SOURCE_DATASET = "bowen-upenn/PersonaMem-v2"
SOURCE_REVISION = "0622e56d1cc6f1bc990a5100a6ec4022a60e66a6"
SPLIT_FILES = {
    "train_text": "benchmark/text/train.csv",
    "val_text": "benchmark/text/val.csv",
    "benchmark_text": "benchmark/text/benchmark.csv",
}
NORMALIZED_FILES = {
    "train_text": "normalized/train_text.jsonl",
    "val_text": "normalized/val_text.jsonl",
    "benchmark_text": "normalized/benchmark_text.jsonl",
}
EXPECTED_SOURCE_SPLIT_COUNTS = {
    "train_text": 18549,
    "val_text": 2061,
    "benchmark_text": 5000,
}
EXPECTED_SPLIT_COUNTS = {
    "train_text": 18527,
    "val_text": 2059,
    "benchmark_text": 5000,
}
KNOWN_INVALID_ROWS = {
    ("train_text", 949): "missing_incorrect_answers",
    ("train_text", 16287): "correct_answer_in_distractors",
}
REQUIRED_COLUMNS = {
    "persona_id",
    "chat_history_32k_link",
    "user_query",
    "correct_answer",
    "incorrect_answers",
    "topic_query",
    "preference",
    "topic_preference",
    "conversation_scenario",
    "pref_type",
    "who",
    "updated",
    "prev_pref",
    "sensitive_info",
}
ATTRIBUTE_COLUMNS = (
    "topic_query",
    "preference",
    "topic_preference",
    "conversation_scenario",
    "pref_type",
    "who",
    "updated",
    "prev_pref",
    "sensitive_info",
    "related_conversation_snippet",
)
NUMERIC_COLUMNS = (
    "total_tokens_in_chat_history_32k",
    "distance_from_related_snippet_to_query_32k",
    "num_persona_relevant_tokens_32k",
    "num_persona_irrelevant_tokens_32k",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _text(value: Any, *, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"PersonaMem-v2 field is empty: {field}")
    return text


def _optional_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_serialized(value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"PersonaMem-v2 field is not JSON/Python literal: {field}"
            ) from exc


def parse_user_query(value: Any) -> str:
    parsed = parse_serialized(value, field="user_query")
    if isinstance(parsed, dict):
        role = str(parsed.get("role", "user")).strip()
        if role != "user":
            raise ValueError(f"PersonaMem-v2 user_query has role={role!r}")
        return _text(parsed.get("content"), field="user_query.content")
    return _text(parsed, field="user_query")


def parse_incorrect_answers(value: Any) -> list[str]:
    parsed = parse_serialized(value, field="incorrect_answers")
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("PersonaMem-v2 incorrect_answers must be a list")
    answers = [_text(item, field="incorrect_answers[]") for item in parsed]
    if not answers or len(answers) != len(set(answers)):
        raise ValueError("PersonaMem-v2 incorrect answers are empty or duplicated")
    return answers


def _optional_int(value: Any, *, field: str) -> int | None:
    text = _optional_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"PersonaMem-v2 field is not numeric: {field}") from exc


def normalize_history_path(value: Any) -> str:
    raw = _text(value, field="chat_history_32k_link").replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe PersonaMem-v2 history path: {raw}")
    normalized = path.as_posix().lstrip("./")
    if not normalized.startswith("data/chat_history_32k/"):
        raise ValueError(f"unexpected PersonaMem-v2 32K history path: {raw}")
    return normalized


def load_history(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"PersonaMem-v2 history must be an object: {path}")
    metadata = raw.get("metadata")
    messages = raw.get("chat_history")
    if not isinstance(metadata, dict) or not isinstance(messages, list) or not messages:
        raise ValueError(f"PersonaMem-v2 history has invalid structure: {path}")
    allowed_roles = {"system", "user", "assistant"}
    cleaned_messages = []
    dropped_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"PersonaMem-v2 history message is invalid: {path}:{index}"
            )
        if message.get("role") not in allowed_roles:
            raise ValueError(f"PersonaMem-v2 history role is invalid: {path}:{index}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            dropped_messages.append(
                {
                    "message_index": index,
                    "reason": "drop_empty_history_message",
                }
            )
            continue
        cleaned_messages.append(message)
    declared = metadata.get("total_messages")
    if declared is not None and int(declared) != len(messages):
        raise ValueError(f"PersonaMem-v2 history message count mismatch: {path}")
    raw["chat_history"] = cleaned_messages
    raw["_memlora_source_repairs"] = dropped_messages
    return raw


def normalize_split(
    root: Path,
    split: str,
    *,
    benchmark_personas: set[str],
) -> tuple[
    list[dict[str, Any]],
    set[str],
    set[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_path = root / SPLIT_FILES[split]
    if not source_path.is_file():
        raise FileNotFoundError(f"missing PersonaMem-v2 split: {source_path}")
    rows = []
    personas: set[str] = set()
    history_paths: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"PersonaMem-v2 split {split} misses columns: {missing}")
        for source_index, raw in enumerate(reader):
            persona_id = _text(raw.get("persona_id"), field="persona_id")
            if split != "benchmark_text" and persona_id in benchmark_personas:
                exclusions.append(
                    {
                        "split": split,
                        "source_row_index": source_index,
                        "persona_id": persona_id,
                        "reason": "benchmark_persona_overlap",
                    }
                )
                continue
            history_path = normalize_history_path(raw.get("chat_history_32k_link"))
            question = parse_user_query(raw.get("user_query"))
            correct = _text(raw.get("correct_answer"), field="correct_answer")
            invalid_reason = KNOWN_INVALID_ROWS.get((split, source_index))
            if invalid_reason is not None:
                parsed_invalid = parse_serialized(
                    raw.get("incorrect_answers"), field="incorrect_answers"
                )
                expected_invalid = (
                    invalid_reason == "missing_incorrect_answers"
                    and parsed_invalid is None
                ) or (
                    invalid_reason == "correct_answer_in_distractors"
                    and isinstance(parsed_invalid, (list, tuple))
                    and correct
                    in {
                        _optional_text(item)
                        for item in parsed_invalid
                        if _optional_text(item)
                    }
                )
                if not expected_invalid:
                    raise ValueError(
                        "known PersonaMem-v2 invalid row changed at pinned revision: "
                        f"{split}:{source_index}"
                    )
                exclusions.append(
                    {
                        "split": split,
                        "source_row_index": source_index,
                        "persona_id": persona_id,
                        "reason": invalid_reason,
                    }
                )
                continue
            parsed_incorrect = parse_serialized(
                raw.get("incorrect_answers"), field="incorrect_answers"
            )
            if isinstance(parsed_incorrect, (list, tuple)):
                cleaned_incorrect = [
                    item for item in parsed_incorrect if _optional_text(item)
                ]
                removed = len(parsed_incorrect) - len(cleaned_incorrect)
                if removed:
                    repairs.append(
                        {
                            "split": split,
                            "source_row_index": source_index,
                            "persona_id": persona_id,
                            "reason": "drop_null_incorrect_answer",
                            "removed_answers": removed,
                        }
                    )
                parsed_incorrect = cleaned_incorrect
            incorrect = parse_incorrect_answers(parsed_incorrect)
            if correct in incorrect:
                raise ValueError(
                    "PersonaMem-v2 correct answer appears among distractors: "
                    f"{split}:{source_index}"
                )
            attributes = {
                field: _optional_text(raw.get(field)) for field in ATTRIBUTE_COLUMNS
            }
            numeric = {
                field: _optional_int(raw.get(field), field=field)
                for field in NUMERIC_COLUMNS
            }
            rows.append(
                {
                    "format": EXAMPLE_FORMAT,
                    "instance_id": (
                        f"personamem_v2_{split}_{source_index:06d}_persona_{persona_id}"
                    ),
                    "split": split,
                    "source_row_index": source_index,
                    "persona_id": persona_id,
                    "chat_history_32k_file": history_path,
                    "question": question,
                    "correct_answer": correct,
                    "incorrect_answers": incorrect,
                    "options": [correct, *incorrect],
                    "correct_option_index": 0,
                    "attributes": attributes,
                    "source_statistics": numeric,
                }
            )
            personas.add(persona_id)
            history_paths.add(history_path)
            if (source_index + 1) % 1000 == 0:
                print(
                    f"[personamem-v2 normalize] split={split} "
                    f"rows={source_index + 1}",
                    flush=True,
                )
    print(
        f"[personamem-v2 normalize] split={split} rows={len(rows)} "
        f"excluded={len(exclusions)} repaired={len(repairs)} complete",
        flush=True,
    )
    return rows, personas, history_paths, exclusions, repairs


def source_split_personas(root: Path, split: str) -> tuple[int, set[str]]:
    source_path = root / SPLIT_FILES[split]
    count = 0
    personas: set[str] = set()
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "persona_id" not in (reader.fieldnames or []):
            raise ValueError(f"PersonaMem-v2 split misses persona_id: {source_path}")
        for raw in reader:
            count += 1
            personas.add(_text(raw.get("persona_id"), field="persona_id"))
    return count, personas


def build_source_manifest(
    root: Path, history_paths: Iterable[str]
) -> list[dict[str, Any]]:
    relative_paths = list(SPLIT_FILES.values()) + sorted(set(history_paths))
    rows = []
    for index, relative in enumerate(relative_paths, start=1):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing PersonaMem-v2 source file: {path}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if index % 100 == 0 or index == len(relative_paths):
            print(
                "[personamem-v2 source-hash] "
                f"files={index}/{len(relative_paths)}",
                flush=True,
            )
    return rows


def prepare_dataset(
    root: Path,
    *,
    require_expected_counts: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_personas: dict[str, set[str]] = {}
    all_history_paths: set[str] = set()
    all_exclusions: list[dict[str, Any]] = []
    all_repairs: list[dict[str, Any]] = []
    expected_history_personas: dict[str, set[str]] = defaultdict(set)
    history_repairs: list[dict[str, Any]] = []
    source_counts = {}
    source_personas = {}
    for split in SPLIT_FILES:
        source_counts[split], source_personas[split] = source_split_personas(
            root, split
        )
        if (
            require_expected_counts
            and source_counts[split] != EXPECTED_SOURCE_SPLIT_COUNTS[split]
        ):
            raise ValueError(
                f"PersonaMem-v2 source {split} has {source_counts[split]} rows, "
                f"expected {EXPECTED_SOURCE_SPLIT_COUNTS[split]}"
            )
    benchmark_personas = source_personas["benchmark_text"]
    for split in SPLIT_FILES:
        rows, personas, history_paths, exclusions, repairs = normalize_split(
            root,
            split,
            benchmark_personas=benchmark_personas,
        )
        split_rows[split] = rows
        split_personas[split] = personas
        all_history_paths.update(history_paths)
        all_exclusions.extend(exclusions)
        all_repairs.extend(repairs)
        for row in rows:
            expected_history_personas[row["chat_history_32k_file"]].add(
                str(row["persona_id"])
            )
        if require_expected_counts and len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"PersonaMem-v2 {split} has {len(rows)} rows, expected "
                f"{EXPECTED_SPLIT_COUNTS[split]}"
            )

    for split in ("train_text", "val_text"):
        overlap = split_personas[split] & split_personas["benchmark_text"]
        if overlap:
            raise ValueError(
                f"PersonaMem-v2 test persona leakage from {split}: "
                f"{sorted(overlap)[:10]}"
            )

    # Validate each distinct history once before freezing it.
    sorted_history_paths = sorted(all_history_paths)
    for index, relative in enumerate(sorted_history_paths, start=1):
        history = load_history(root / relative)
        history_persona = history["metadata"].get("persona_id")
        if history_persona is None:
            raise ValueError(f"PersonaMem-v2 history has no persona_id: {relative}")
        expected = expected_history_personas[relative]
        if len(expected) != 1 or str(history_persona) not in expected:
            raise ValueError(
                "PersonaMem-v2 history/persona mismatch: "
                f"path={relative} metadata={history_persona} rows={sorted(expected)}"
            )
        for repair in history["_memlora_source_repairs"]:
            history_repairs.append({"path": relative, **repair})
        if index % 100 == 0 or index == len(sorted_history_paths):
            print(
                "[personamem-v2 history-check] "
                f"files={index}/{len(sorted_history_paths)}",
                flush=True,
            )

    normalized_hashes = {}
    for split, rows in split_rows.items():
        output_path = root / NORMALIZED_FILES[split]
        write_jsonl_atomic(output_path, rows)
        normalized_hashes[split] = sha256_file(output_path)

    source_manifest = build_source_manifest(root, all_history_paths)
    source_manifest_path = root / "source_manifest.json"
    write_json_atomic(source_manifest_path, source_manifest)
    contract = {
        "history_variant": "chat_history_32k",
        "task_variant": "text",
        "train_split": "train_text",
        "configuration_split": "val_text",
        "final_test_split": "benchmark_text",
        "split_sanitation": "exclude_invalid_mcq_and_benchmark_persona_overlap_v1",
        "question_visible_to_memory_writer": False,
        "primary_metric": "mcq_accuracy",
        "thinking_enabled": False,
        "option_shuffle": "deterministic_per_instance",
    }
    dataset_sha256 = canonical_sha256(
        {
            "source_revision": SOURCE_REVISION,
            "source_manifest_sha256": canonical_sha256(source_manifest),
            "normalized_sha256": normalized_hashes,
            "contract": contract,
        }
    )
    split_attribute_counts = {}
    for split, rows in split_rows.items():
        split_attribute_counts[split] = {
            field: dict(
                sorted(
                    Counter(
                        row["attributes"][field]
                        for row in rows
                        if row["attributes"][field]
                    ).items()
                )
            )
            for field in ("pref_type", "who", "updated")
        }
    freeze = {
        "format": DATASET_FORMAT,
        "dataset": "personamem_v2_32k_text",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "dataset_sha256": dataset_sha256,
        "source_manifest_file": source_manifest_path.name,
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_files": len(source_manifest),
        "normalized_files": NORMALIZED_FILES,
        "normalized_sha256": normalized_hashes,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "source_split_counts": source_counts,
        "split_persona_counts": {
            split: len(personas) for split, personas in split_personas.items()
        },
        "split_persona_overlap": {
            "train_val": len(
                split_personas["train_text"] & split_personas["val_text"]
            ),
            "train_benchmark": 0,
            "val_benchmark": 0,
        },
        "source_exclusions": all_exclusions,
        "source_exclusion_counts": dict(
            sorted(Counter(row["reason"] for row in all_exclusions).items())
        ),
        "source_repairs": all_repairs,
        "source_repair_counts": dict(
            sorted(Counter(row["reason"] for row in all_repairs).items())
        ),
        "history_repairs": history_repairs,
        "history_repair_counts": dict(
            sorted(Counter(row["reason"] for row in history_repairs).items())
        ),
        "unique_history_files": len(all_history_paths),
        "split_attribute_counts": split_attribute_counts,
        "contract": contract,
    }
    write_json_atomic(root / "memlora_dataset_freeze.json", freeze)
    return freeze


def validate_dataset_root(
    root: str | Path,
    *,
    require_expected_counts: bool = True,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = Path(root)
    freeze_path = root / "memlora_dataset_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(
            "formal PersonaMem-v2 data is incomplete; run "
            f"experiments/personamem_v2/prepare_formal_data.sh: {root}"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("format") != DATASET_FORMAT:
        raise ValueError(f"unsupported PersonaMem-v2 dataset freeze: {freeze_path}")
    if freeze.get("source_revision") != SOURCE_REVISION:
        raise ValueError("PersonaMem-v2 freeze uses an unpinned source revision")
    manifest_path = root / str(freeze["source_manifest_file"])
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != freeze.get("source_manifest_file_sha256")
    ):
        raise ValueError(f"PersonaMem-v2 source manifest mismatch: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_sha256(source_manifest) != freeze.get("source_manifest_sha256"):
        raise ValueError("PersonaMem-v2 source manifest content mismatch")
    if verify_source_files:
        for index, item in enumerate(source_manifest, start=1):
            path = root / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"PersonaMem-v2 source checksum mismatch: {path}")
            if index % 100 == 0 or index == len(source_manifest):
                print(
                    "[personamem-v2 validate-source] "
                    f"files={index}/{len(source_manifest)}",
                    flush=True,
                )

    split_rows = {}
    split_personas = {}
    for split in SPLIT_FILES:
        path = root / str(freeze["normalized_files"][split])
        if (
            not path.is_file()
            or sha256_file(path) != freeze["normalized_sha256"][split]
        ):
            raise ValueError(f"PersonaMem-v2 normalized checksum mismatch: {path}")
        rows = read_jsonl(path)
        if require_expected_counts and len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"PersonaMem-v2 {split} cardinality mismatch")
        ids = [row["instance_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"PersonaMem-v2 {split} instance IDs are not unique")
        split_rows[split] = rows
        split_personas[split] = {str(row["persona_id"]) for row in rows}
    for split in ("train_text", "val_text"):
        if split_personas[split] & split_personas["benchmark_text"]:
            raise ValueError(
                f"PersonaMem-v2 normalized {split} leaks benchmark personas"
            )
    return freeze, split_rows
