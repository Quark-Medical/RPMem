"""Build deterministic two-table Parquet artifacts for formal Phase 1."""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import shutil
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rpmem.training.corpus.adapters import adapt_record
from rpmem.training.corpus.dedup import (
    SQLiteDeduplicator,
    exact_text_hash,
    normalize_for_dedup,
    simhash64,
)
from rpmem.training.corpus.schema import render_session, validate_session
from rpmem.training.corpus.shards import (
    INDEX_SCHEMA,
    PROBE_SCHEMA,
    SESSION_SCHEMA,
    AtomicParquetShardWriter,
    sha256_file,
    write_json_atomic,
)
from rpmem.training.corpus.token_budget import build_context_token_counters


FORMAT_NAME = "memlora_canonical_corpus_v1"
SPLIT_FORMAT_NAME = "memlora_corpus_split_v1"


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is YAML; install PyYAML or use JSON configuration"
            ) from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise TypeError("corpus build config must contain one object")
    return config


def _stable_fraction(seed: int, *parts: str) -> float:
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dedup_text(session) -> str:
    parts: list[str] = []
    for event in session.events:
        parts.extend((event.role, event.type, event.name, event.content, event.status))
        if event.arguments is not None:
            parts.append(_canonical_json(event.arguments))
    return "\n".join(part for part in parts if part)


def _expand_paths(raw_paths: list[str], relative_to: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = relative_to / candidate
        matches = sorted(
            Path(match).resolve()
            for match in glob.glob(str(candidate), recursive=True)
            if Path(match).is_file()
        )
        if not matches:
            raise FileNotFoundError(f"source path did not match any files: {candidate}")
        paths.extend(matches)
    if len(paths) != len(set(paths)):
        raise ValueError("source path list contains duplicate files")
    return paths


def _iter_rows(path: Path, batch_rows: int) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number}: row must be an object")
                yield row
        return
    if suffix == ".json":
        decoder = json.JSONDecoder()
        chunk_chars = max(65_536, min(max(batch_rows, 1) * 1024, 8 * 1024 * 1024))
        with path.open(encoding="utf-8") as handle:
            buffer = ""
            cursor = 0
            eof = False

            def refill() -> None:
                nonlocal buffer, cursor, eof
                if cursor:
                    buffer = buffer[cursor:]
                    cursor = 0
                chunk = handle.read(chunk_chars)
                if not chunk:
                    eof = True
                buffer += chunk

            def skip_space() -> None:
                nonlocal cursor
                while True:
                    while cursor < len(buffer) and (
                        buffer[cursor].isspace() or buffer[cursor] == "\ufeff"
                    ):
                        cursor += 1
                    if cursor < len(buffer) or eof:
                        return
                    refill()

            refill()
            skip_space()
            if cursor >= len(buffer) or buffer[cursor] != "[":
                raise ValueError(f"{path}: top-level JSON value must be an array")
            cursor += 1
            first = True
            while True:
                skip_space()
                if cursor < len(buffer) and buffer[cursor] == "]":
                    cursor += 1
                    break
                if not first:
                    if cursor >= len(buffer) or buffer[cursor] != ",":
                        raise ValueError(f"{path}: expected ',' between JSON rows")
                    cursor += 1
                    skip_space()

                while True:
                    try:
                        row, end = decoder.raw_decode(buffer, cursor)
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(
                                f"{path}: invalid JSON array: {exc}"
                            ) from exc
                        refill()
                        continue
                    break
                if not isinstance(row, dict):
                    raise TypeError(f"{path}: every JSON array row must be an object")
                yield row
                cursor = end
                first = False
        return
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        try:
            for batch in parquet.iter_batches(batch_size=batch_rows):
                yield from batch.to_pylist()
        finally:
            parquet.close()
        return
    raise ValueError(f"unsupported source format: {path}")


def _build_token_counter(config: dict[str, Any]):
    max_tokens = int(config.get("max_context_tokens", 2048))
    if max_tokens <= 0:
        return lambda text: len(normalize_for_dedup(text).split())
    return build_context_token_counters(config).primary_count


def _bounded_ordered_map(executor, function, iterable, max_pending: int):
    if executor is None:
        yield from map(function, iterable)
        return
    iterator = iter(iterable)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def _write_split_manifest(
    output_dir: Path,
    *,
    split: str,
    session_paths: list[Path],
    probe_paths: list[Path],
    index_paths: list[Path],
    session_count: int,
    probe_count: int,
    require_references: bool,
    content_digest: str,
    longest_context_indices: list[int],
) -> Path:
    path = output_dir / f"{split}.corpus.json"
    relative = lambda item: item.relative_to(output_dir).as_posix()
    write_json_atomic(
        path,
        {
            "format": SPLIT_FORMAT_NAME,
            "version": 1,
            "split": split,
            "sessions": [relative(item) for item in session_paths],
            "probes": [relative(item) for item in probe_paths],
            "indices": [relative(item) for item in index_paths],
            "session_count": session_count,
            "probe_count": probe_count,
            "require_references": require_references,
            "content_digest": content_digest,
            "longest_context_indices": longest_context_indices,
        },
    )
    return path


def _logical_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for operational_key in (
        "output_dir",
        "overwrite",
        "workers",
        "log_every",
        "shard_rows",
        "row_group_rows",
        "source_batch_rows",
    ):
        result.pop(operational_key, None)
    return result


def build_corpus(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
) -> Path:
    """Build a corpus artifact and return its absolute output directory."""

    config_path = Path(config_path).expanduser().resolve()
    config = _load_config(config_path)
    if config.get("format") != "memlora_corpus_build_v1":
        raise ValueError("config format must be memlora_corpus_build_v1")
    if not config.get("sources"):
        raise ValueError("config requires at least one source")

    output_overridden = output_dir is not None
    raw_output = output_dir if output_overridden else config.get("output_dir")
    if not raw_output:
        raise ValueError("output_dir is required")
    output = Path(raw_output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() if output_overridden else config_path.parent) / output
    output = output.resolve()
    allow_overwrite = (
        bool(config.get("overwrite", False)) if overwrite is None else overwrite
    )
    if output.exists() and any(output.iterdir()):
        if not allow_overwrite:
            raise FileExistsError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    validation_fraction = float(config.get("session_validation_fraction", 0.01))
    if not 0 <= validation_fraction < 1:
        raise ValueError("session_validation_fraction must be in [0, 1)")
    query_validation_probes = int(config.get("query_validation_probes", 2))
    if query_validation_probes < 0:
        raise ValueError("query_validation_probes must be non-negative")
    rows_per_shard = int(config.get("shard_rows", 4096))
    row_group_rows = int(config.get("row_group_rows", 128))
    source_batch_rows = int(config.get("source_batch_rows", 1024))
    workers = int(config.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be positive")
    log_every = int(config.get("log_every", 10_000))
    longest_context_candidates = int(config.get("longest_context_candidates", 256))
    if longest_context_candidates < 1:
        raise ValueError("longest_context_candidates must be positive")
    max_context_tokens = int(config.get("max_context_tokens", 2048))
    require_references = bool(config.get("require_references", True))
    if max_context_tokens > 0:
        context_token_counters = build_context_token_counters(config)
        primary_tokenizer_name = context_token_counters.specs[0].name
        context_tokenizer_names = [spec.name for spec in context_token_counters.specs]
        token_count = context_token_counters.primary_count
    else:
        context_token_counters = None
        primary_tokenizer_name = "fallback_words"
        context_tokenizer_names = [primary_tokenizer_name]
        token_count = _build_token_counter(config)
    target_domain_shares = {
        str(key): float(value)
        for key, value in config.get("target_domain_token_shares", {}).items()
    }
    if target_domain_shares and abs(sum(target_domain_shares.values()) - 1.0) > 1e-6:
        raise ValueError("target_domain_token_shares must sum to 1.0")
    if any(value < 0 or value > 1 for value in target_domain_shares.values()):
        raise ValueError("target domain token shares must be in [0, 1]")
    target_context_tokens = int(config.get("target_context_tokens", 0))
    if target_context_tokens < 0:
        raise ValueError("target_context_tokens must be non-negative")
    if target_context_tokens and not target_domain_shares:
        raise ValueError("target_context_tokens requires target_domain_token_shares")
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

    excluded_sources = {
        str(item).strip().casefold() for item in config.get("excluded_sources", [])
    }
    excluded_patterns = [
        normalize_for_dedup(item)
        for item in config.get("excluded_text_patterns", [])
        if normalize_for_dedup(item)
    ]
    for source in config["sources"]:
        source_name = str(source.get("name", "")).strip()
        if source_name.casefold() in excluded_sources:
            raise ValueError(f"configured source is deny-listed: {source_name}")
        for required in ("name", "domain", "adapter", "license", "revision", "paths"):
            if source.get(required) in (None, "", []):
                raise ValueError(
                    f"source {source_name or '<unnamed>'} requires {required}"
                )

    work_dir = output / "_work"
    work_dir.mkdir()
    rejections_path = output / "rejections.jsonl"
    logical_digest = hashlib.sha256(
        _canonical_json(_logical_config(config)).encode("utf-8")
    )
    rejection_counts: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter()
    accepted_by_domain: Counter[str] = Counter()
    tokens_by_source: Counter[str] = Counter()
    tokens_by_domain: Counter[str] = Counter()
    context_tokens_by_tokenizer: Counter[str] = Counter()
    max_context_tokens_observed: Counter[str] = Counter()
    split_sessions: Counter[str] = Counter()
    split_probes: Counter[str] = Counter()
    probe_types: Counter[str] = Counter()
    probe_count_histogram: Counter[int] = Counter()
    total_probes = 0
    longest_context_heaps: dict[str, list[tuple[int, int, int]]] = {
        split: [] for split in ("train", "validation", "query_validation")
    }
    build_started = time.monotonic()

    session_writer = AtomicParquetShardWriter(
        output / "sessions",
        schema=SESSION_SCHEMA,
        rows_per_shard=rows_per_shard,
        row_group_rows=row_group_rows,
    )
    probe_writer = AtomicParquetShardWriter(
        output / "probes",
        schema=PROBE_SCHEMA,
        rows_per_shard=rows_per_shard,
        row_group_rows=row_group_rows,
    )
    index_writers = {
        split: AtomicParquetShardWriter(
            output / "indices" / split,
            schema=INDEX_SCHEMA,
            rows_per_shard=rows_per_shard,
            row_group_rows=row_group_rows,
        )
        for split in ("train", "validation", "query_validation")
    }

    executor_context = (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    )
    with (
        rejections_path.open("w") as rejection_file,
        SQLiteDeduplicator(
            work_dir / "dedup.sqlite",
            near_hamming_threshold=int(config.get("near_duplicate_hamming", 3)),
            minimum_near_tokens=int(config.get("minimum_near_tokens", 20)),
        ) as deduplicator,
        executor_context as executor,
    ):

        def reject(reason: str, source_name: str, record_id: str, detail: str = ""):
            rejection_counts[reason] += 1
            rejection_file.write(
                _canonical_json(
                    {
                        "reason": reason,
                        "source": source_name,
                        "source_record_id": record_id,
                        "detail": detail,
                    }
                )
                + "\n"
            )

        for source in config["sources"]:
            source_name = str(source["name"]).strip()
            adapter = str(source["adapter"]).strip()
            paths = _expand_paths(list(source["paths"]), config_path.parent)
            for path in paths:

                def prepare_row(item):
                    row_number, row = item
                    raw_id = next(
                        (
                            str(row[key]).strip()
                            for key in (
                                "session_id",
                                "conversation_id",
                                "id",
                                "record_id",
                            )
                            if row.get(key) not in (None, "")
                        ),
                        f"{path.name}:{row_number}",
                    )
                    try:
                        session = adapt_record(
                            row,
                            source=source,
                            adapter=adapter,
                            source_record_id=raw_id,
                        )
                        validate_session(
                            session,
                            require_probes=True,
                            require_references=require_references,
                        )
                        context = render_session(session)
                        normalized = normalize_for_dedup(_dedup_text(session))
                        tokenizer_counts = (
                            context_token_counters.counts(context)
                            if context_token_counters is not None
                            else {primary_tokenizer_name: token_count(context)}
                        )
                        return (
                            raw_id,
                            session,
                            context,
                            normalized,
                            tokenizer_counts[primary_tokenizer_name],
                            tokenizer_counts,
                            "",
                        )
                    except (TypeError, ValueError) as exc:
                        return raw_id, None, "", "", 0, {}, str(exc)

                prepared_rows = _bounded_ordered_map(
                    executor,
                    prepare_row,
                    enumerate(_iter_rows(path, source_batch_rows)),
                    max_pending=max(workers * 4, 1),
                )
                for (
                    raw_id,
                    session,
                    context,
                    normalized_context,
                    n_context_tokens,
                    tokenizer_counts,
                    error,
                ) in prepared_rows:
                    source_rows[source_name] += 1
                    scanned = sum(source_rows.values())
                    if log_every > 0 and scanned % log_every == 0:
                        elapsed = max(time.monotonic() - build_started, 1e-6)
                        print(
                            f"[corpus] scanned={scanned} accepted={session_writer.row_count} "
                            f"rejected={sum(rejection_counts.values())} "
                            f"rate={scanned / elapsed:.1f} rows/s source={source_name} "
                            f"file={path.name}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if error:
                        reject("schema_error", source_name, raw_id, error)
                        continue

                    matched_pattern = next(
                        (
                            pattern
                            for pattern in excluded_patterns
                            if pattern in normalized_context
                        ),
                        None,
                    )
                    if matched_pattern is not None:
                        reject("excluded_text", source_name, raw_id, matched_pattern)
                        continue

                    over_limit = {
                        name: count
                        for name, count in tokenizer_counts.items()
                        if max_context_tokens > 0 and count > max_context_tokens
                    }
                    if over_limit:
                        reject(
                            "context_too_long",
                            source_name,
                            raw_id,
                            ", ".join(
                                f"{name}={count}>{max_context_tokens}"
                                for name, count in sorted(over_limit.items())
                            ),
                        )
                        continue

                    if domain_token_caps:
                        domain_cap = domain_token_caps.get(session.domain)
                        if domain_cap is None:
                            reject(
                                "unexpected_domain",
                                source_name,
                                raw_id,
                                session.domain,
                            )
                            continue
                        if (
                            tokens_by_domain[session.domain] + n_context_tokens
                            > domain_cap
                        ):
                            reject(
                                "domain_token_budget",
                                source_name,
                                raw_id,
                                f"domain={session.domain}, cap={domain_cap}",
                            )
                            continue

                    duplicate_reason = deduplicator.classify_and_add(
                        normalized_context, session.session_id
                    )
                    if duplicate_reason is not None:
                        reject(duplicate_reason, source_name, raw_id)
                        continue

                    session_index = session_writer.row_count
                    probe_start = probe_writer.row_count
                    probe_indices: list[int] = []
                    for probe in session.probes:
                        probe_index = probe_writer.row_count
                        probe_indices.append(probe_index)
                        probe_writer.add(
                            {
                                "probe_index": probe_index,
                                "session_index": session_index,
                                "session_id": session.session_id,
                                "probe_id": probe.id,
                                "prompt": probe.prompt,
                                "probe_type": probe.probe_type,
                                "answerable": probe.answerable,
                                "reference": probe.reference,
                                "evidence_event_ids": list(probe.evidence_event_ids),
                                "metadata_json": _canonical_json(probe.metadata),
                            }
                        )
                    total_probes += len(probe_indices)
                    probe_count_histogram[len(probe_indices)] += 1
                    probe_types.update(probe.probe_type for probe in session.probes)

                    dedup_hash = exact_text_hash(normalized_context)
                    signature, _ = simhash64(normalized_context)
                    session_writer.add(
                        {
                            "session_index": session_index,
                            "session_id": session.session_id,
                            "probe_start": probe_start,
                            "probe_count": len(probe_indices),
                            "source": session.source,
                            "domain": session.domain,
                            "context": context,
                            "context_tokens": n_context_tokens,
                            "context_token_max": max(tokenizer_counts.values()),
                            "context_token_counts_json": _canonical_json(
                                tokenizer_counts
                            ),
                            "context_hash": dedup_hash,
                            "simhash": signature,
                            "events_json": _canonical_json(
                                [event.to_dict() for event in session.events]
                            ),
                            "memory_atoms_json": _canonical_json(
                                [atom.to_dict() for atom in session.memory_atoms]
                            ),
                            "provenance_json": _canonical_json(session.provenance),
                            "metadata_json": _canonical_json(session.metadata),
                        }
                    )

                    is_validation = (
                        _stable_fraction(seed, session.session_id) < validation_fraction
                    )
                    if is_validation:
                        split_probe_indices = {"validation": probe_indices}
                    else:
                        heldout_count = min(
                            query_validation_probes, max(len(probe_indices) - 1, 0)
                        )
                        heldout = set(
                            sorted(
                                probe_indices,
                                key=lambda index: _stable_fraction(
                                    seed,
                                    session.session_id,
                                    session.probes[index - probe_start].id,
                                ),
                            )[:heldout_count]
                        )
                        split_probe_indices = {
                            "train": [
                                index for index in probe_indices if index not in heldout
                            ],
                            "query_validation": [
                                index for index in probe_indices if index in heldout
                            ],
                        }
                    for split, selected_indices in split_probe_indices.items():
                        if not selected_indices:
                            continue
                        split_row_index = index_writers[split].row_count
                        index_writers[split].add(
                            {
                                "session_index": session_index,
                                "session_id": session.session_id,
                                "probe_indices": selected_indices,
                            }
                        )
                        split_sessions[split] += 1
                        split_probes[split] += len(selected_indices)
                        candidate = (
                            max(tokenizer_counts.values()),
                            -split_row_index,
                            split_row_index,
                        )
                        heap = longest_context_heaps[split]
                        if len(heap) < longest_context_candidates:
                            heapq.heappush(heap, candidate)
                        elif candidate[:2] > heap[0][:2]:
                            heapq.heapreplace(heap, candidate)

                    canonical_session = session.to_dict()
                    logical_digest.update(
                        _canonical_json(canonical_session).encode("utf-8")
                    )
                    accepted_by_source[session.source] += 1
                    accepted_by_domain[session.domain] += 1
                    tokens_by_source[session.source] += n_context_tokens
                    tokens_by_domain[session.domain] += n_context_tokens
                    for name, count in tokenizer_counts.items():
                        context_tokens_by_tokenizer[name] += count
                        max_context_tokens_observed[name] = max(
                            max_context_tokens_observed[name], count
                        )

    session_writer.close()
    probe_writer.close()
    for writer in index_writers.values():
        writer.close()

    corpus_content_digest = logical_digest.hexdigest()
    longest_context_indices = {
        split: [
            split_row_index
            for _, _, split_row_index in sorted(
                heap,
                key=lambda item: (-item[0], item[2]),
            )
        ]
        for split, heap in longest_context_heaps.items()
    }
    split_manifest_paths = []
    for split, writer in index_writers.items():
        split_manifest_paths.append(
            _write_split_manifest(
                output,
                split=split,
                session_paths=session_writer.paths,
                probe_paths=probe_writer.paths,
                index_paths=writer.paths,
                session_count=split_sessions[split],
                probe_count=split_probes[split],
                require_references=require_references,
                content_digest=corpus_content_digest,
                longest_context_indices=longest_context_indices[split],
            )
        )

    total_tokens = sum(tokens_by_source.values())
    observed_domain_shares = {
        key: value / total_tokens if total_tokens else 0.0
        for key, value in sorted(tokens_by_domain.items())
    }
    domain_share_deviation = {
        domain: observed_domain_shares.get(domain, 0.0) - target
        for domain, target in sorted(target_domain_shares.items())
    }
    mean_probes_per_session = (
        total_probes / session_writer.row_count if session_writer.row_count else 0.0
    )
    stats = {
        "source_rows": dict(sorted(source_rows.items())),
        "accepted_sessions": session_writer.row_count,
        "accepted_probes": total_probes,
        "mean_probes_per_session": mean_probes_per_session,
        "probe_count_histogram": {
            str(key): value for key, value in sorted(probe_count_histogram.items())
        },
        "probe_types": dict(sorted(probe_types.items())),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "accepted_by_domain": dict(sorted(accepted_by_domain.items())),
        "context_tokens": total_tokens,
        "context_tokens_by_source": dict(sorted(tokens_by_source.items())),
        "context_tokens_by_domain": dict(sorted(tokens_by_domain.items())),
        "context_tokenizers": (
            context_token_counters.metadata()
            if context_token_counters is not None
            else []
        ),
        "context_tokens_by_tokenizer": dict(
            sorted(
                (name, context_tokens_by_tokenizer[name])
                for name in context_tokenizer_names
            )
        ),
        "max_context_tokens_observed": {
            name: max_context_tokens_observed[name] for name in context_tokenizer_names
        },
        "observed_source_token_shares": {
            key: value / total_tokens if total_tokens else 0.0
            for key, value in sorted(tokens_by_source.items())
        },
        "observed_domain_token_shares": observed_domain_shares,
        "target_domain_token_shares": target_domain_shares,
        "target_context_tokens": target_context_tokens,
        "domain_token_caps": domain_token_caps,
        "domain_token_share_deviation": domain_share_deviation,
        "split_sessions": dict(sorted(split_sessions.items())),
        "split_probes": dict(sorted(split_probes.items())),
        "rejections": dict(sorted(rejection_counts.items())),
        "workers": workers,
        "longest_context_candidates": longest_context_candidates,
    }
    write_json_atomic(output / "stats.json", stats)
    write_json_atomic(output / "build_config.json", config)
    shutil.rmtree(work_dir, ignore_errors=True)

    launch_gate_failures: list[str] = []
    minimum_sessions = int(config.get("minimum_accepted_sessions", 0))
    minimum_tokens = int(config.get("minimum_context_tokens", 0))
    minimum_mean_probes = float(config.get("minimum_mean_probes_per_session", 0.0))
    if minimum_mean_probes < 0:
        raise ValueError("minimum_mean_probes_per_session must be non-negative")
    if session_writer.row_count < minimum_sessions:
        launch_gate_failures.append(
            "minimum_accepted_sessions: "
            f"accepted={session_writer.row_count}, required={minimum_sessions}"
        )
    if total_tokens < minimum_tokens:
        launch_gate_failures.append(
            f"minimum_context_tokens: accepted={total_tokens}, required={minimum_tokens}"
        )
    if mean_probes_per_session < minimum_mean_probes:
        launch_gate_failures.append(
            "minimum_mean_probes_per_session: "
            f"accepted={mean_probes_per_session:.6f}, required={minimum_mean_probes:.6f}"
        )
    share_tolerance = float(config.get("token_share_tolerance", 1.0))
    for domain, deviation in domain_share_deviation.items():
        if abs(deviation) > share_tolerance:
            launch_gate_failures.append(
                f"target_domain_token_shares[{domain}]: deviation={deviation:.6f}, "
                f"tolerance={share_tolerance:.6f}"
            )
    if bool(config.get("enforce_launch_gates", False)) and launch_gate_failures:
        failure_path = output / "launch_gate_failures.json"
        write_json_atomic(
            failure_path,
            {
                "format": "memlora_corpus_launch_gate_failures_v1",
                "failures": launch_gate_failures,
                "stats": "stats.json",
            },
        )
        raise ValueError(
            "corpus launch gates failed: " + "; ".join(launch_gate_failures)
        )

    artifact_paths = sorted(
        [
            *session_writer.paths,
            *probe_writer.paths,
            *(path for writer in index_writers.values() for path in writer.paths),
            *split_manifest_paths,
            output / "stats.json",
            output / "build_config.json",
            rejections_path,
        ],
        key=lambda path: path.relative_to(output).as_posix(),
    )
    manifest = {
        "format": FORMAT_NAME,
        "version": 1,
        "name": str(config.get("name", output.name)),
        "seed": seed,
        "tokenizer": str(config.get("tokenizer", "")),
        "tokenizer_revision": str(config.get("tokenizer_revision", "")),
        "context_tokenizers": stats["context_tokenizers"],
        "max_context_tokens": max_context_tokens,
        "max_context_tokens_observed": stats["max_context_tokens_observed"],
        "content_digest": corpus_content_digest,
        "counts": {
            "sessions": session_writer.row_count,
            "probes": total_probes,
            "context_tokens": total_tokens,
        },
        "splits": {
            split: f"{split}.corpus.json"
            for split in ("train", "validation", "query_validation")
        },
        "artifacts": [
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json_atomic(output / "manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = build_corpus(
        args.config,
        output_dir=args.output_dir,
        overwrite=True if args.overwrite else None,
    )
    print(f"canonical corpus built: {output}")
    print(f"manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
