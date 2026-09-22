"""Audit and deterministically repair canonical session ID collisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


COLLISION_POLICY = "ordered_content_hash_v1"
_DERIVED_ENRICHMENT_FIELDS = {
    "global_source_index",
    "probe_generation_fallback_count",
    "probe_generation_policy",
    "probe_generation_source_count",
}


def _jsonl_paths(
    directory: str | Path,
    *,
    require_files: bool = True,
) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    paths = sorted(directory.glob("part-*.jsonl"))
    if require_files and not paths:
        raise FileNotFoundError(f"no normalized shards found: {directory}/part-*.jsonl")
    return paths


def _iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise TypeError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                yield path, line_number, row


def session_content_digest(row: dict[str, Any]) -> str:
    """Hash stable canonical content while excluding the session ID itself."""

    payload = {
        key: value
        for key, value in row.items()
        if key != "session_id" and key not in _DERIVED_ENRICHMENT_FIELDS
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def session_id_counts(paths: Iterable[Path]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    rows = 0
    for path, line_number, row in _iter_jsonl(paths):
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            raise ValueError(f"empty session_id at {path}:{line_number}")
        counts[session_id] += 1
        rows += 1
    return counts, rows


def audit_session_ids(
    directory: str | Path,
    *,
    require_files: bool = True,
) -> dict[str, Any]:
    """Return a bounded summary of canonical session ID uniqueness."""

    paths = _jsonl_paths(directory, require_files=require_files)
    counts, row_count = session_id_counts(paths)
    duplicate_ids = sorted(key for key, count in counts.items() if count > 1)
    return {
        "format": "memlora_session_id_audit_v1",
        "rows": row_count,
        "files": len(paths),
        "unique_ids": len(counts),
        "duplicate_ids": duplicate_ids,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_occurrences": sum(counts[key] for key in duplicate_ids),
    }


def assert_unique_session_ids(directory: str | Path) -> dict[str, Any]:
    report = audit_session_ids(directory)
    if report["duplicate_id_count"]:
        examples = report["duplicate_ids"][:10]
        raise ValueError(
            f"normalized source has {report['duplicate_id_count']} duplicate "
            f"session IDs across {report['duplicate_occurrences']} rows; "
            f"examples={examples}"
        )
    return report


def resolve_duplicate_session_ids(
    directory: str | Path,
    *,
    policy: str = COLLISION_POLICY,
    require_files: bool = True,
) -> dict[str, Any]:
    """Rewrite only colliding IDs, preserving shard order and all other content."""

    paths = _jsonl_paths(directory, require_files=require_files)
    counts, row_count = session_id_counts(paths)
    duplicate_ids = tuple(sorted(key for key, count in counts.items() if count > 1))
    report: dict[str, Any] = {
        "format": "memlora_session_id_collision_report_v1",
        "policy": policy,
        "rows": row_count,
        "duplicate_ids": list(duplicate_ids),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_occurrences": sum(counts[key] for key in duplicate_ids),
        "rewritten_rows": 0,
        "files": len(paths),
    }
    if not duplicate_ids:
        return report

    duplicate_set = set(duplicate_ids)
    occurrence: Counter[str] = Counter()
    final_ids = {key for key, count in counts.items() if count == 1}
    rewritten_rows = 0
    for path in paths:
        temporary = path.with_suffix(path.suffix + ".session-id-repair")
        try:
            with path.open() as source, temporary.open("w") as destination:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at {path}:{line_number}"
                        ) from exc
                    original_id = str(row.get("session_id", "")).strip()
                    if original_id in duplicate_set:
                        index = occurrence[original_id]
                        occurrence[original_id] += 1
                        digest = session_content_digest(row)
                        repaired_id = (
                            f"{original_id}:collision-{index:03d}-{digest[:12]}"
                        )
                        if repaired_id in final_ids:
                            raise ValueError(
                                f"collision repair generated duplicate ID: {repaired_id}"
                            )
                        final_ids.add(repaired_id)
                        provenance = dict(row.get("provenance") or {})
                        provenance["session_id_collision"] = {
                            "policy": policy,
                            "original_session_id": original_id,
                            "occurrence": index,
                            "content_sha256": digest,
                        }
                        row["session_id"] = repaired_id
                        row["provenance"] = provenance
                        rewritten_rows += 1
                    destination.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    if len(final_ids) != row_count:
        raise ValueError(
            f"session ID repair produced {len(final_ids)} unique IDs for {row_count} rows"
        )
    report["rewritten_rows"] = rewritten_rows
    return report


def remove_stale_enrichment_rows(
    directory: str | Path,
    stale_session_ids: Iterable[str],
) -> dict[str, int]:
    """Drop old enriched rows for repaired IDs so they are regenerated cleanly."""

    directory = Path(directory).expanduser().resolve()
    paths = sorted(directory.glob("part-*.jsonl"))
    stale = {str(value) for value in stale_session_ids}
    removed = retained = 0
    for path in paths:
        temporary = path.with_suffix(path.suffix + ".session-id-repair")
        try:
            with path.open() as source, temporary.open("w") as destination:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at {path}:{line_number}"
                        ) from exc
                    if str(row.get("session_id", "")) in stale:
                        removed += 1
                        continue
                    destination.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    retained += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    for pattern in (
        "stats-*.json",
        "errors-*.jsonl",
        "enrichment_validation.json",
    ):
        for path in directory.glob(pattern):
            path.unlink()
    return {"removed_enriched_rows": removed, "retained_enriched_rows": retained}


def repair_existing_source(
    normalized_dir: str | Path,
    enriched_dir: str | Path | None = None,
) -> dict[str, Any]:
    report = resolve_duplicate_session_ids(normalized_dir)
    if enriched_dir is not None and report["duplicate_ids"]:
        report.update(
            remove_stale_enrichment_rows(enriched_dir, report["duplicate_ids"])
        )
    report_path = Path(normalized_dir).resolve() / "session_id_collision_report.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", required=True)
    parser.add_argument("--enriched-dir")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        report = audit_session_ids(args.normalized_dir)
    else:
        report = repair_existing_source(args.normalized_dir, args.enriched_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
