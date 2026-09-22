"""Validate exact, grounded coverage of probe-enriched session shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from rpmem.training.corpus.build import _expand_paths, _iter_rows
from rpmem.training.corpus.generate_probes import (
    GENERATION_POLICY,
    ensure_target_probe_count,
)
from rpmem.training.corpus.schema import CanonicalSession, validate_session
from rpmem.training.corpus.shards import write_json_atomic


def _base_digest(session: CanonicalSession) -> str:
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "source": session.source,
        "domain": session.domain,
        "events": [event.to_dict() for event in session.events],
        "provenance": session.provenance,
        "metadata": session.metadata,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_enrichment(
    inputs: list[str],
    outputs: list[str],
    *,
    target_probes: int,
    generation_policy: str = GENERATION_POLICY,
    relative_to: str | Path = ".",
) -> dict[str, Any]:
    if target_probes < 1:
        raise ValueError("target_probes must be positive")
    generation_policy = str(generation_policy).strip()
    if not generation_policy:
        raise ValueError("generation_policy must be non-empty")
    relative_to = Path(relative_to).expanduser().resolve()
    input_paths = _expand_paths(inputs, relative_to)
    output_paths = _expand_paths(outputs, relative_to)

    with tempfile.TemporaryDirectory(prefix="rpmem-enrichment-") as tmp:
        connection = sqlite3.connect(Path(tmp) / "coverage.sqlite")
        connection.execute(
            "CREATE TABLE inputs (session_id TEXT PRIMARY KEY, base_digest TEXT NOT NULL, "
            "seen INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute("CREATE TABLE outputs (session_id TEXT PRIMARY KEY)")
        input_count = output_count = fallback_sessions = fallback_probes = 0
        source_seeded_sessions = source_seeded_probes = 0
        try:
            for path in input_paths:
                for row in _iter_rows(path, 1024):
                    session = CanonicalSession.from_dict(row)
                    validate_session(session)
                    try:
                        connection.execute(
                            "INSERT INTO inputs(session_id, base_digest) VALUES (?, ?)",
                            (session.session_id, _base_digest(session)),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"duplicate normalized session_id: {session.session_id}"
                        ) from exc
                    input_count += 1
            connection.commit()

            for path in output_paths:
                for row in _iter_rows(path, 1024):
                    actual_generation_policy = str(
                        row.get("probe_generation_policy", "")
                    )
                    if actual_generation_policy != generation_policy:
                        raise ValueError(
                            "probe_generation_policy mismatch for "
                            f"{row.get('session_id', '')}: "
                            f"expected={generation_policy!r}, "
                            f"actual={actual_generation_policy!r}"
                        )
                    declared_fallback_count = int(
                        row.get("probe_generation_fallback_count", 0)
                    )
                    declared_source_count = int(
                        row.get("probe_generation_source_count", 0)
                    )
                    if declared_fallback_count < 0:
                        raise ValueError(
                            "probe_generation_fallback_count must be non-negative"
                        )
                    if declared_source_count < 0:
                        raise ValueError(
                            "probe_generation_source_count must be non-negative"
                        )
                    session = CanonicalSession.from_dict(row)
                    validate_session(
                        session, require_probes=True, require_references=True
                    )
                    ensure_target_probe_count(session, target_probes)
                    actual_fallback_count = sum(
                        probe.probe_type.startswith("fallback_")
                        for probe in session.probes
                    )
                    if declared_fallback_count != actual_fallback_count:
                        raise ValueError(
                            "probe_generation_fallback_count mismatch for "
                            f"{session.session_id}: declared={declared_fallback_count}, "
                            f"actual={actual_fallback_count}"
                        )
                    if declared_fallback_count:
                        fallback_sessions += 1
                        fallback_probes += declared_fallback_count
                    actual_source_count = sum(
                        probe.metadata.get("generation") == "source_derived"
                        for probe in session.probes
                    )
                    if declared_source_count != actual_source_count:
                        raise ValueError(
                            "probe_generation_source_count mismatch for "
                            f"{session.session_id}: declared={declared_source_count}, "
                            f"actual={actual_source_count}"
                        )
                    if declared_source_count:
                        source_seeded_sessions += 1
                        source_seeded_probes += declared_source_count
                    try:
                        connection.execute(
                            "INSERT INTO outputs(session_id) VALUES (?)",
                            (session.session_id,),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"duplicate enriched session_id: {session.session_id}"
                        ) from exc
                    expected = connection.execute(
                        "SELECT base_digest FROM inputs WHERE session_id = ?",
                        (session.session_id,),
                    ).fetchone()
                    if expected is None:
                        raise ValueError(
                            f"enriched output has no normalized input: {session.session_id}"
                        )
                    if expected[0] != _base_digest(session):
                        raise ValueError(
                            f"enrichment changed base session content: {session.session_id}"
                        )
                    connection.execute(
                        "UPDATE inputs SET seen = 1 WHERE session_id = ?",
                        (session.session_id,),
                    )
                    output_count += 1
            connection.commit()

            missing_count = connection.execute(
                "SELECT COUNT(*) FROM inputs WHERE seen = 0"
            ).fetchone()[0]
            if missing_count:
                examples = [
                    row[0]
                    for row in connection.execute(
                        "SELECT session_id FROM inputs WHERE seen = 0 ORDER BY session_id LIMIT 10"
                    )
                ]
                raise ValueError(
                    f"probe enrichment is missing {missing_count} sessions; examples={examples}"
                )
        finally:
            connection.close()

    return {
        "format": "memlora_probe_enrichment_validation_v1",
        "target_probes": target_probes,
        "normalized_sessions": input_count,
        "enriched_sessions": output_count,
        "input_files": len(input_paths),
        "output_files": len(output_paths),
        "fallback_sessions": fallback_sessions,
        "fallback_probes": fallback_probes,
        "generation_policy": generation_policy,
        "source_seeded_sessions": source_seeded_sessions,
        "source_seeded_probes": source_seeded_probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--target_probes", type=int, default=10)
    parser.add_argument("--generation_policy", default=GENERATION_POLICY)
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_enrichment(
        args.inputs,
        args.outputs,
        target_probes=args.target_probes,
        generation_policy=args.generation_policy,
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
