"""Integrity and relational validation for canonical corpus artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rpmem.training.corpus.build import FORMAT_NAME, SPLIT_FORMAT_NAME
from rpmem.training.corpus.shards import sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object")
    return payload


def _iter_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        try:
            for batch in parquet.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        finally:
            parquet.close()


def _paths(root: Path, values: list[str]) -> list[Path]:
    return [(root / value).resolve() for value in values]


def _advance_session(
    sessions: Iterator[dict[str, Any]],
    current: dict[str, Any] | None,
    target_index: int,
) -> dict[str, Any]:
    while current is None or int(current["session_index"]) < target_index:
        try:
            current = next(sessions)
        except StopIteration as exc:
            raise ValueError(f"session index is out of range: {target_index}") from exc
    if int(current["session_index"]) != target_index:
        raise ValueError(f"missing session index: {target_index}")
    return current


def _validate_split_indices(
    root: Path,
    payload: dict[str, Any],
    session_paths: list[Path],
) -> tuple[int, int]:
    sessions = iter(_iter_rows(session_paths))
    current_session = None
    previous_session_index = -1
    session_count = 0
    probe_count = 0
    for row in _iter_rows(_paths(root, payload["indices"])):
        session_index = int(row["session_index"])
        if session_index <= previous_session_index:
            raise ValueError(
                f"session indices in {payload['split']} are not strictly increasing"
            )
        current_session = _advance_session(sessions, current_session, session_index)
        if row["session_id"] != current_session["session_id"]:
            raise ValueError(f"session ID mismatch in {payload['split']} index")
        start = int(current_session["probe_start"])
        end = start + int(current_session["probe_count"])
        selected = [int(value) for value in row["probe_indices"]]
        if not selected:
            raise ValueError(f"empty probe selection in {payload['split']} index")
        if selected != sorted(set(selected)):
            raise ValueError(
                f"probe indices in {payload['split']} are not unique and ordered"
            )
        if any(not start <= probe_index < end for probe_index in selected):
            raise ValueError(f"probe outside session range in {payload['split']} index")
        previous_session_index = session_index
        session_count += 1
        probe_count += len(selected)
    if session_count != int(payload["session_count"]):
        raise ValueError(f"session count mismatch in {payload['split']} manifest")
    if probe_count != int(payload["probe_count"]):
        raise ValueError(f"probe count mismatch in {payload['split']} manifest")
    return session_count, probe_count


def _assert_context_splits_disjoint(
    root: Path, left: dict[str, Any], right: dict[str, Any]
) -> None:
    left_rows = iter(_iter_rows(_paths(root, left["indices"])))
    right_rows = iter(_iter_rows(_paths(root, right["indices"])))
    left_row = next(left_rows, None)
    right_row = next(right_rows, None)
    while left_row is not None and right_row is not None:
        left_index = int(left_row["session_index"])
        right_index = int(right_row["session_index"])
        if left_index == right_index:
            raise ValueError(
                f"{left['split']} and {right['split']} context splits overlap"
            )
        if left_index < right_index:
            left_row = next(left_rows, None)
        else:
            right_row = next(right_rows, None)


def _assert_probe_splits_disjoint(
    root: Path, left: dict[str, Any], right: dict[str, Any]
) -> None:
    left_rows = iter(_iter_rows(_paths(root, left["indices"])))
    right_rows = iter(_iter_rows(_paths(root, right["indices"])))
    left_row = next(left_rows, None)
    right_row = next(right_rows, None)
    while left_row is not None and right_row is not None:
        left_index = int(left_row["session_index"])
        right_index = int(right_row["session_index"])
        if left_index == right_index:
            overlap = set(left_row["probe_indices"]) & set(right_row["probe_indices"])
            if overlap:
                raise ValueError(
                    f"{left['split']} and {right['split']} probe splits overlap"
                )
            left_row = next(left_rows, None)
            right_row = next(right_rows, None)
        elif left_index < right_index:
            left_row = next(left_rows, None)
        else:
            right_row = next(right_rows, None)


def _validate_context_token_counts(
    session: dict[str, Any],
    *,
    expected_names: set[str],
    primary_name: str,
    max_context_tokens: int,
) -> dict[str, int]:
    raw_counts = session.get("context_token_counts_json")
    if not raw_counts:
        raise ValueError(
            f"session is missing tokenizer counts: {session['session_id']}"
        )
    counts = {str(name): int(count) for name, count in json.loads(raw_counts).items()}
    if set(counts) != expected_names:
        raise ValueError(
            f"session tokenizer keys mismatch: {session['session_id']} "
            f"expected={sorted(expected_names)} actual={sorted(counts)}"
        )
    if int(session["context_tokens"]) != counts[primary_name]:
        raise ValueError(
            f"session primary tokenizer count mismatch: {session['session_id']}"
        )
    if int(session["context_token_max"]) != max(counts.values()):
        raise ValueError(f"session tokenizer maximum mismatch: {session['session_id']}")
    overflow = {
        name: count
        for name, count in counts.items()
        if max_context_tokens > 0 and count > max_context_tokens
    }
    if overflow:
        raise ValueError(
            f"session exceeds tokenizer limits: {session['session_id']} {overflow}"
        )
    return counts


def validate_corpus(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    root = manifest_path.parent
    manifest = _load_json(manifest_path)
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"unexpected corpus format: {manifest.get('format')!r}")

    max_context_tokens = int(manifest.get("max_context_tokens", 0))
    context_tokenizers = manifest.get("context_tokenizers", [])
    observed_tokenizer_maxima = manifest.get("max_context_tokens_observed", {})
    expected_names = {str(item["name"]) for item in context_tokenizers}
    primary_name = str(context_tokenizers[0]["name"]) if context_tokenizers else ""
    if context_tokenizers:
        if len(expected_names) != len(context_tokenizers):
            raise ValueError("manifest has duplicate context tokenizer names")
        if set(observed_tokenizer_maxima) != expected_names:
            raise ValueError(
                "manifest tokenizer maxima keys do not match context tokenizers"
            )
        for name in sorted(expected_names):
            observed = int(observed_tokenizer_maxima[name])
            if max_context_tokens > 0 and observed > max_context_tokens:
                raise ValueError(
                    f"{name} observed maximum {observed} exceeds context token "
                    f"limit {max_context_tokens}"
                )

    for artifact in manifest.get("artifacts", []):
        path = (root / artifact["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus artifact: {path}")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"checksum mismatch for {path}: expected {artifact['sha256']}, got {actual}"
            )
        if path.stat().st_size != artifact["size"]:
            raise ValueError(f"size mismatch for {path}")

    split_payloads = {}
    for split, relative_path in manifest.get("splits", {}).items():
        payload = _load_json(root / relative_path)
        if payload.get("format") != SPLIT_FORMAT_NAME or payload.get("split") != split:
            raise ValueError(f"invalid split manifest: {relative_path}")
        split_payloads[split] = payload
    if not split_payloads:
        raise ValueError("manifest has no split manifests")

    any_split = next(iter(split_payloads.values()))
    session_paths = _paths(root, any_split["sessions"])
    probe_paths = _paths(root, any_split["probes"])
    expected_session_index = 0
    expected_probe_start = 0
    recomputed_tokenizer_maxima = {name: 0 for name in expected_names}
    for session in _iter_rows(session_paths):
        if session["session_index"] != expected_session_index:
            raise ValueError("session indices are not contiguous and ordered")
        if session["probe_start"] != expected_probe_start:
            raise ValueError("session probe ranges are not contiguous and ordered")
        if max_context_tokens > 0 and session["context_tokens"] > max_context_tokens:
            raise ValueError(f"session exceeds token limit: {session['session_id']}")
        if expected_names:
            tokenizer_counts = _validate_context_token_counts(
                session,
                expected_names=expected_names,
                primary_name=primary_name,
                max_context_tokens=max_context_tokens,
            )
            for name, count in tokenizer_counts.items():
                recomputed_tokenizer_maxima[name] = max(
                    recomputed_tokenizer_maxima[name], count
                )
        event_ids = {item["id"] for item in json.loads(session["events_json"])}
        if not event_ids:
            raise ValueError(f"session has no events: {session['session_id']}")
        probe_end = session["probe_start"] + session["probe_count"]
        expected_probe_start = probe_end
        expected_session_index += 1

    if (
        expected_names
        and {name: int(observed_tokenizer_maxima[name]) for name in expected_names}
        != recomputed_tokenizer_maxima
    ):
        raise ValueError(
            "manifest tokenizer maxima do not match recomputed session maxima"
        )

    expected_probe_index = 0
    sessions = iter(_iter_rows(session_paths))
    current_session = next(sessions, None)
    if current_session is not None:
        current_event_ids = {
            item["id"] for item in json.loads(current_session["events_json"])
        }
    else:
        current_event_ids = set()
    for probe in _iter_rows(probe_paths):
        if probe["probe_index"] != expected_probe_index:
            raise ValueError("probe indices are not contiguous and ordered")
        while current_session is not None and probe["probe_index"] >= (
            current_session["probe_start"] + current_session["probe_count"]
        ):
            current_session = next(sessions, None)
            current_event_ids = (
                {item["id"] for item in json.loads(current_session["events_json"])}
                if current_session is not None
                else set()
            )
        if current_session is None:
            raise ValueError("probe is outside all session ranges")
        start = int(current_session["probe_start"])
        end = start + int(current_session["probe_count"])
        if not start <= probe["probe_index"] < end:
            raise ValueError("probe is outside its session range")
        if (
            probe["session_index"] != current_session["session_index"]
            or probe["session_id"] != current_session["session_id"]
        ):
            raise ValueError("probe/session join keys do not match")
        unknown = set(probe["evidence_event_ids"] or []) - current_event_ids
        if unknown:
            raise ValueError(
                f"probe {probe['probe_id']} references unknown evidence events: {unknown}"
            )
        expected_probe_index += 1

    counts = manifest.get("counts", {})
    if expected_session_index != counts.get("sessions"):
        raise ValueError("session row count does not match manifest")
    if expected_probe_index != counts.get("probes"):
        raise ValueError("probe row count does not match manifest")
    if expected_probe_start != expected_probe_index:
        raise ValueError("session probe ranges do not cover every probe")

    split_counts: dict[str, tuple[int, int]] = {}
    for split, payload in split_payloads.items():
        split_counts[split] = _validate_split_indices(root, payload, session_paths)

    if "train" in split_payloads and "validation" in split_payloads:
        _assert_context_splits_disjoint(
            root, split_payloads["train"], split_payloads["validation"]
        )
    if "train" in split_payloads and "query_validation" in split_payloads:
        _assert_probe_splits_disjoint(
            root, split_payloads["train"], split_payloads["query_validation"]
        )

    return {
        "manifest": str(manifest_path),
        "accepted_sessions": expected_session_index,
        "accepted_probes": expected_probe_index,
        "splits": {
            split: {
                "sessions": split_counts[split][0],
                "probes": split_counts[split][1],
            }
            for split in split_payloads
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    print(json.dumps(validate_corpus(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
