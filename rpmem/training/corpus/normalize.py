"""Normalize raw public sources into bounded canonical session shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from rpmem.training.corpus.adapters import adapt_record
from rpmem.training.corpus.build import (
    _bounded_ordered_map,
    _dedup_text,
    _expand_paths,
    _iter_rows,
    _load_config,
)
from rpmem.training.corpus.dedup import SQLiteDeduplicator, normalize_for_dedup
from rpmem.training.corpus.schema import (
    CanonicalEvent,
    CanonicalSession,
    render_session,
    validate_session,
)
from rpmem.training.corpus.session_ids import resolve_duplicate_session_ids
from rpmem.training.corpus.shards import write_json_atomic
from rpmem.training.corpus.token_budget import build_context_token_counters


FORMAT_NAME = "memlora_source_normalization_v1"


def repository_is_excluded(repository: str, excluded: set[str]) -> bool:
    return str(repository).strip().casefold().strip("/") in excluded


def _has_flagged_moderation(value: Any) -> bool:
    if isinstance(value, dict):
        flagged = value.get("flagged")
        if flagged is True or (
            isinstance(flagged, (list, tuple)) and any(item is True for item in flagged)
        ):
            return True
        return any(_has_flagged_moderation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_flagged_moderation(item) for item in value)
    return False


def source_filter_reason(row: dict[str, Any], source: dict[str, Any]) -> str:
    language_allowlist = {
        str(item).strip().casefold()
        for item in source.get("language_allowlist", [])
        if str(item).strip()
    }
    language = str(row.get("language", "")).strip()
    if language_allowlist and language.casefold() not in language_allowlist:
        return f"language={language or '<missing>'}"
    if source.get("exclude_flagged_moderation", False) and _has_flagged_moderation(
        row.get("openai_moderation")
    ):
        return "flagged_moderation"
    return ""


def _session_with_events(
    session: CanonicalSession,
    events: list[CanonicalEvent],
    *,
    session_id: str,
    segment_index: int,
    segment_count: int,
) -> CanonicalSession:
    event_ids = {event.id for event in events}
    memory_atoms = tuple(
        atom
        for atom in session.memory_atoms
        if set(atom.evidence_event_ids) <= event_ids
    )
    probes = tuple(
        probe
        for probe in session.probes
        if probe.evidence_event_ids and set(probe.evidence_event_ids) <= event_ids
    )
    provenance = {
        **session.provenance,
        "parent_session_id": session.session_id,
        "segment_index": segment_index,
        "segment_count": segment_count,
    }
    metadata = {
        **session.metadata,
        "segment_index": segment_index,
        "segment_count": segment_count,
    }
    return CanonicalSession(
        session_id=session_id,
        source=session.source,
        domain=session.domain,
        events=tuple(events),
        memory_atoms=memory_atoms,
        probes=probes,
        provenance=provenance,
        metadata=metadata,
    )


def _candidate_session(
    session: CanonicalSession,
    events: list[CanonicalEvent],
    *,
    segment_index: int,
) -> CanonicalSession:
    return _session_with_events(
        session,
        events,
        session_id=f"{session.session_id}:segment-{segment_index:03d}",
        segment_index=segment_index,
        segment_count=1,
    )


def _truncate_event_to_fit(
    session: CanonicalSession,
    event: CanonicalEvent,
    max_context_tokens: int,
    token_count: Callable[[str], int],
    *,
    segment_index: int,
) -> CanonicalEvent:
    if event.type == "tool_call" and not event.content:
        compact = replace(
            event,
            arguments={"truncated": True, "original_name": event.name},
        )
        if (
            token_count(
                render_session(
                    _candidate_session(
                        session,
                        [compact],
                        segment_index=segment_index,
                    )
                )
            )
            <= max_context_tokens
        ):
            return compact
    if not event.content:
        raise ValueError(
            f"event {event.id} alone exceeds max_context_tokens={max_context_tokens}"
        )
    low, high = 1, len(event.content)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = replace(
            event, content=event.content[:middle].rstrip() + " [truncated]"
        )
        length = token_count(
            render_session(
                _candidate_session(
                    session,
                    [candidate],
                    segment_index=segment_index,
                )
            )
        )
        if length <= max_context_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError(
            f"event {event.id} cannot fit max_context_tokens={max_context_tokens}"
        )
    return best


def segment_session(
    session: CanonicalSession,
    *,
    max_context_tokens: int,
    token_count: Callable[[str], int],
    event_overlap: int = 1,
) -> list[CanonicalSession]:
    """Split only at event boundaries and retain locally evidenced targets."""

    if max_context_tokens <= 0:
        return [session]
    if event_overlap < 0:
        raise ValueError("event_overlap must be non-negative")
    validate_session(session)
    groups: list[list[CanonicalEvent]] = []
    current: list[CanonicalEvent] = []
    for raw_event in session.events:
        event = raw_event
        candidate = [*current, event]
        segment_index = len(groups)
        if (
            token_count(
                render_session(
                    _candidate_session(
                        session,
                        candidate,
                        segment_index=segment_index,
                    )
                )
            )
            <= max_context_tokens
        ):
            current = candidate
            continue
        if not current:
            event = _truncate_event_to_fit(
                session,
                event,
                max_context_tokens,
                token_count,
                segment_index=segment_index,
            )
            current = [event]
            continue
        groups.append(current)
        overlap = current[-event_overlap:] if event_overlap else []
        candidate = [*overlap, event]
        segment_index = len(groups)
        while overlap and (
            token_count(
                render_session(
                    _candidate_session(
                        session,
                        candidate,
                        segment_index=segment_index,
                    )
                )
            )
            > max_context_tokens
        ):
            overlap = overlap[1:]
            candidate = [*overlap, event]
        if (
            token_count(
                render_session(
                    _candidate_session(
                        session,
                        candidate,
                        segment_index=segment_index,
                    )
                )
            )
            > max_context_tokens
        ):
            event = _truncate_event_to_fit(
                session,
                event,
                max_context_tokens,
                token_count,
                segment_index=segment_index,
            )
            candidate = [event]
        current = candidate
    if current:
        groups.append(current)

    segment_count = len(groups)
    segments = [
        _session_with_events(
            session,
            events,
            session_id=(
                session.session_id
                if segment_count == 1
                else f"{session.session_id}:segment-{index:03d}"
            ),
            segment_index=index,
            segment_count=segment_count,
        )
        for index, events in enumerate(groups)
    ]
    for segment in segments:
        validate_session(segment)
        length = token_count(render_session(segment))
        if length > max_context_tokens:
            raise ValueError(
                f"segmentation produced {length} tokens, limit={max_context_tokens}: "
                f"{segment.session_id}"
            )
    return segments


class AtomicJsonlShardWriter:
    def __init__(self, directory: Path, rows_per_shard: int):
        if rows_per_shard < 1:
            raise ValueError("rows_per_shard must be positive")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.rows: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        self.row_count = 0

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.row_count += 1
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.directory / f"part-{len(self.paths):05d}.jsonl"
        temporary = path.with_suffix(path.suffix + ".incomplete")
        with temporary.open("w") as handle:
            for row in self.rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary.replace(path)
        self.paths.append(path)
        self.rows.clear()

    def close(self) -> None:
        self.flush()


def _load_excluded_repositories(source: dict[str, Any], config_dir: Path) -> set[str]:
    excluded = {
        str(item).strip().casefold().strip("/")
        for item in source.get("excluded_repositories", [])
        if str(item).strip()
    }
    raw_file = str(source.get("excluded_repositories_file", "")).strip()
    if raw_file:
        path = Path(raw_file)
        if not path.is_absolute():
            path = config_dir / path
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                excluded.add(line.casefold().strip("/"))
    if source.get("require_excluded_repositories", False) and not excluded:
        raise ValueError(
            "required coding repository exclusion manifest is empty; "
            "freeze the evaluation repositories before normalization"
        )
    return excluded


def _load_source_decode_tokenizer(source: dict[str, Any]):
    adapter = str(source.get("adapter", "")).strip()
    tokenizer_path = str(source.get("decode_tokenizer", "")).strip()
    if adapter == "d2l_tokenized" and not tokenizer_path:
        raise ValueError(
            f"source {source.get('name', '<unnamed>')} requires decode_tokenizer"
        )
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=str(source.get("decode_tokenizer_revision") or "") or None,
        trust_remote_code=True,
    )


def normalize_sources(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    workers: int | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = _load_config(config_path)
    if config.get("format") != FORMAT_NAME:
        raise ValueError(f"config format must be {FORMAT_NAME}")
    sources = config.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be non-empty")
    target_context_tokens = int(config.get("target_context_tokens", 0))
    target_domain_shares = {
        str(key): float(value)
        for key, value in config.get("target_domain_token_shares", {}).items()
    }
    if target_context_tokens < 0:
        raise ValueError("target_context_tokens must be non-negative")
    for domain, share in target_domain_shares.items():
        if not 0.0 <= share <= 1.0:
            raise ValueError(
                f"target_domain_token_shares[{domain}] must be between 0 and 1"
            )
    if target_context_tokens and abs(sum(target_domain_shares.values()) - 1.0) > 1e-6:
        raise ValueError(
            "target_domain_token_shares must sum to 1 when target_context_tokens is set"
        )
    source_excluded_repositories = [
        _load_excluded_repositories(source, config_path.parent) for source in sources
    ]
    source_decode_tokenizers = [
        _load_source_decode_tokenizer(source) for source in sources
    ]
    output_overridden = output_dir is not None
    raw_output = output_dir if output_overridden else config.get("output_dir")
    if not raw_output:
        raise ValueError("output_dir is required")
    output = Path(raw_output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() if output_overridden else config_path.parent) / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    max_context_tokens = int(config.get("max_context_tokens", 2048))
    token_counters = build_context_token_counters(config)
    primary_tokenizer_name = token_counters.specs[0].name
    segmentation_token_count = token_counters.max_count
    rows_per_shard = int(config.get("rows_per_shard", 4096))
    batch_rows = int(config.get("source_batch_rows", 1024))
    event_overlap = int(config.get("event_overlap", 1))
    log_every = int(config.get("log_every", 10_000))
    workers = int(config.get("workers", 1) if workers is None else workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    domain_token_caps: dict[str, int] = {}
    if target_context_tokens:
        allocated = 0
        domains = sorted(target_domain_shares)
        for index, domain in enumerate(domains):
            if index == len(domains) - 1:
                cap = target_context_tokens - allocated
            else:
                cap = int(target_context_tokens * target_domain_shares[domain])
                allocated += cap
            domain_token_caps[domain] = cap
    context_tokens_by_domain: Counter[str] = Counter()
    context_tokens_by_tokenizer: Counter[str] = Counter()
    max_context_tokens_observed: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "workers": workers,
        "target_context_tokens": target_context_tokens,
        "target_domain_token_shares": target_domain_shares,
        "domain_token_caps": domain_token_caps,
        "context_tokenizers": token_counters.metadata(),
        "sources": {},
    }
    reports_dir = output / "reports"
    reports_dir.mkdir()
    work_dir = output / "_work"
    work_dir.mkdir()

    executor_context = (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    )
    with (
        SQLiteDeduplicator(
            work_dir / "dedup.sqlite",
            near_hamming_threshold=int(config.get("near_duplicate_hamming", 3)),
            minimum_near_tokens=int(config.get("minimum_near_tokens", 20)),
        ) as deduplicator,
        executor_context as executor,
    ):
        for source, excluded_repositories, decode_tokenizer in zip(
            sources,
            source_excluded_repositories,
            source_decode_tokenizers,
            strict=True,
        ):
            source_name = str(source["name"])
            source_domain = str(source["domain"])
            source_domain_cap = domain_token_caps.get(source_domain)
            stop_remaining_tokens = int(
                source.get("stop_when_domain_budget_remaining_below", 0)
            )
            if stop_remaining_tokens < 0:
                raise ValueError(
                    "stop_when_domain_budget_remaining_below must be non-negative"
                )
            if stop_remaining_tokens and source_domain_cap is None:
                raise ValueError(
                    f"source {source_name} cannot stop at an unconfigured domain budget"
                )
            if (
                stop_remaining_tokens
                and source_domain_cap is not None
                and stop_remaining_tokens > source_domain_cap
            ):
                raise ValueError(
                    f"source {source_name} stop threshold exceeds its domain cap"
                )
            writer = AtomicJsonlShardWriter(output / source_name, rows_per_shard)
            scanned = filtered = excluded = rejected = duplicates = budget_skipped = 0
            source_tokens = 0
            stopped_at_domain_budget = False
            source_started = time.monotonic()
            rejection_path = reports_dir / f"{source_name}-rejections.jsonl"
            with rejection_path.open("w") as rejection_file:
                for path in _expand_paths(list(source["paths"]), config_path.parent):

                    def prepare_row(item):
                        row_number, row = item
                        raw_id = next(
                            (
                                str(row[key])
                                for key in (
                                    "session_id",
                                    "conversation_id",
                                    "dialogue_id",
                                    "trajectory_id",
                                    "instance_id",
                                    "uuid",
                                    "id",
                                )
                                if row.get(key) not in (None, "")
                            ),
                            f"{path.name}:{row_number}",
                        )
                        repository = str(row.get("repo", row.get("repository", "")))
                        filter_reason = source_filter_reason(row, source)
                        if filter_reason:
                            return "source_filter", raw_id, filter_reason, []
                        if repository and repository_is_excluded(
                            repository, excluded_repositories
                        ):
                            return "excluded_repository", raw_id, repository, []
                        try:
                            session = adapt_record(
                                row,
                                source=source,
                                adapter=str(source["adapter"]),
                                source_record_id=raw_id,
                                decode_tokenizer=decode_tokenizer,
                            )
                            validate_session(session)
                            segments = segment_session(
                                session,
                                max_context_tokens=max_context_tokens,
                                token_count=segmentation_token_count,
                                event_overlap=event_overlap,
                            )
                            return (
                                "accepted",
                                raw_id,
                                "",
                                [
                                    (
                                        segment,
                                        token_counters.counts(render_session(segment)),
                                    )
                                    for segment in segments
                                ],
                            )
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            return "rejected", raw_id, str(exc), []

                    prepared_rows = _bounded_ordered_map(
                        executor,
                        prepare_row,
                        enumerate(_iter_rows(path, batch_rows)),
                        max_pending=max(workers * 4, 1),
                    )
                    for status, raw_id, detail, segments in prepared_rows:
                        scanned += 1
                        if status == "excluded_repository":
                            excluded += 1
                        elif status == "source_filter":
                            filtered += 1
                        elif status == "rejected":
                            rejected += 1
                        elif status == "accepted":
                            for segment, tokenizer_counts in segments:
                                segment_tokens = tokenizer_counts[
                                    primary_tokenizer_name
                                ]
                                domain_cap = domain_token_caps.get(segment.domain)
                                if domain_token_caps and domain_cap is None:
                                    raise ValueError(
                                        f"session {segment.session_id} has untargeted "
                                        f"domain {segment.domain!r}"
                                    )
                                if domain_cap is not None and (
                                    context_tokens_by_domain[segment.domain]
                                    + segment_tokens
                                    > domain_cap
                                ):
                                    budget_skipped += 1
                                    rejection_file.write(
                                        json.dumps(
                                            {
                                                "reason": "domain_token_budget",
                                                "source_record_id": raw_id,
                                                "session_id": segment.session_id,
                                                "detail": (
                                                    f"domain={segment.domain}, "
                                                    f"cap={domain_cap}"
                                                ),
                                            },
                                            ensure_ascii=False,
                                            separators=(",", ":"),
                                        )
                                        + "\n"
                                    )
                                    continue
                                normalized_context = normalize_for_dedup(
                                    _dedup_text(segment)
                                )
                                duplicate_reason = deduplicator.classify_and_add(
                                    normalized_context, segment.session_id
                                )
                                if duplicate_reason is not None:
                                    duplicates += 1
                                    rejection_file.write(
                                        json.dumps(
                                            {
                                                "reason": duplicate_reason,
                                                "source_record_id": raw_id,
                                                "session_id": segment.session_id,
                                            },
                                            ensure_ascii=False,
                                            separators=(",", ":"),
                                        )
                                        + "\n"
                                    )
                                    continue
                                writer.add(segment.to_dict())
                                context_tokens_by_domain[segment.domain] += (
                                    segment_tokens
                                )
                                for name, count in tokenizer_counts.items():
                                    context_tokens_by_tokenizer[name] += count
                                    max_context_tokens_observed[name] = max(
                                        max_context_tokens_observed[name], count
                                    )
                                source_tokens += segment_tokens
                        else:
                            raise AssertionError(
                                f"unknown normalization status: {status}"
                            )
                        if status != "accepted":
                            rejection_file.write(
                                json.dumps(
                                    {
                                        "reason": status,
                                        "source_record_id": raw_id,
                                        "detail": detail,
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                        if log_every > 0 and scanned % log_every == 0:
                            elapsed = max(time.monotonic() - source_started, 1e-6)
                            print(
                                f"[normalize] source={source_name} scanned={scanned} "
                                f"sessions={writer.row_count} excluded={excluded} "
                                f"rejected={rejected} rate={scanned / elapsed:.1f} rows/s",
                                file=sys.stderr,
                                flush=True,
                            )
                        if stop_remaining_tokens and source_domain_cap is not None:
                            remaining_tokens = (
                                source_domain_cap
                                - context_tokens_by_domain[source_domain]
                            )
                            if remaining_tokens < stop_remaining_tokens:
                                stopped_at_domain_budget = True
                                print(
                                    f"[normalize] source={source_name} stopping at "
                                    f"domain budget: remaining={remaining_tokens} "
                                    f"threshold={stop_remaining_tokens}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                break
                    if stopped_at_domain_budget:
                        close = getattr(prepared_rows, "close", None)
                        if close is not None:
                            close()
                        break
            writer.close()
            collision_report = resolve_duplicate_session_ids(
                output / source_name,
                require_files=False,
            )
            if collision_report["duplicate_id_count"]:
                print(
                    f"[normalize] source={source_name} resolved "
                    f"{collision_report['duplicate_id_count']} duplicate session IDs "
                    f"across {collision_report['rewritten_rows']} rows",
                    file=sys.stderr,
                    flush=True,
                )
            domain_budget_remaining = (
                source_domain_cap - context_tokens_by_domain[source_domain]
                if source_domain_cap is not None
                else None
            )
            stats["sources"][source_name] = {
                "scanned": scanned,
                "sessions": writer.row_count,
                "excluded_repositories": excluded,
                "source_filtered": filtered,
                "rejected": rejected,
                "duplicates": duplicates,
                "session_id_collision_policy": collision_report["policy"],
                "session_id_collision_ids": collision_report["duplicate_id_count"],
                "session_id_collision_rows": collision_report["rewritten_rows"],
                "budget_skipped_sessions": budget_skipped,
                "stopped_at_domain_budget": stopped_at_domain_budget,
                "domain_budget_remaining_tokens": domain_budget_remaining,
                "context_tokens": source_tokens,
                "files": [path.relative_to(output).as_posix() for path in writer.paths],
                "rejections": rejection_path.relative_to(output).as_posix(),
            }
    shutil.rmtree(work_dir, ignore_errors=True)
    stats["context_tokens_by_domain"] = dict(sorted(context_tokens_by_domain.items()))
    stats["context_tokens_by_tokenizer"] = dict(
        sorted(context_tokens_by_tokenizer.items())
    )
    stats["max_context_tokens_observed"] = {
        spec.name: max_context_tokens_observed[spec.name]
        for spec in token_counters.specs
    }
    stats["context_tokens"] = sum(context_tokens_by_domain.values())
    stats["sessions"] = sum(
        source_stats["sessions"] for source_stats in stats["sources"].values()
    )
    total_context_tokens = stats["context_tokens"]
    observed_domain_shares = {
        domain: tokens / total_context_tokens if total_context_tokens else 0.0
        for domain, tokens in sorted(context_tokens_by_domain.items())
    }
    stats["observed_domain_token_shares"] = observed_domain_shares
    write_json_atomic(output / "normalization_stats.json", stats)

    gate_failures = []
    minimum_sessions = int(config.get("minimum_accepted_sessions", 0))
    minimum_tokens = int(config.get("minimum_context_tokens", 0))
    if stats["sessions"] < minimum_sessions:
        gate_failures.append(
            f"minimum_accepted_sessions: accepted={stats['sessions']}, "
            f"required={minimum_sessions}"
        )
    if total_context_tokens < minimum_tokens:
        gate_failures.append(
            f"minimum_context_tokens: accepted={total_context_tokens}, "
            f"required={minimum_tokens}"
        )
    share_tolerance = float(config.get("token_share_tolerance", 1.0))
    for domain, target_share in sorted(target_domain_shares.items()):
        deviation = observed_domain_shares.get(domain, 0.0) - target_share
        if abs(deviation) > share_tolerance:
            gate_failures.append(
                f"target_domain_token_shares[{domain}]: deviation={deviation:.6f}, "
                f"tolerance={share_tolerance:.6f}"
            )
    if bool(config.get("enforce_launch_gates", False)) and gate_failures:
        write_json_atomic(
            output / "normalization_gate_failures.json",
            {
                "format": "memlora_normalization_gate_failures_v1",
                "failures": gate_failures,
                "stats": "normalization_stats.json",
            },
        )
        raise ValueError(
            "source normalization launch gates failed: " + "; ".join(gate_failures)
        )
    write_json_atomic(
        output / "normalization_complete.json",
        {
            "format": "memlora_source_normalization_complete_v1",
            "config_sha256": stats["config_sha256"],
            "stats": "normalization_stats.json",
            "sources": [str(source["name"]) for source in sources],
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = normalize_sources(
        args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(f"normalized sources: {output}")


if __name__ == "__main__":
    main()
