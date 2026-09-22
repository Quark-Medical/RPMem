"""Recover selected normalization sources without rewriting completed sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from rpmem.training.corpus.shards import write_json_atomic


RECOVERY_FORMAT = "memlora_normalization_recovery_complete_v1"


def _unique_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values if str(value).strip())
    if not names:
        raise ValueError(f"{field_name} must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} contains duplicate source names")
    return names


def select_existing_normalized_root(
    normalized_root: str | Path,
    legacy_normalized_roots: Sequence[str | Path] = (),
) -> Path:
    """Use a single existing legacy root when the canonical root is absent."""

    primary = Path(normalized_root).expanduser().resolve()
    if primary.exists():
        return primary
    existing_legacy = [
        Path(candidate).expanduser().resolve()
        for candidate in legacy_normalized_roots
        if Path(candidate).expanduser().resolve().is_dir()
    ]
    if len(existing_legacy) > 1:
        raise ValueError(
            "multiple legacy normalized roots exist: "
            + ", ".join(str(path) for path in existing_legacy)
        )
    return existing_legacy[0] if existing_legacy else primary


def _domain_caps(config: dict[str, Any]) -> dict[str, int]:
    target = int(config.get("target_context_tokens", 0))
    shares = {
        str(domain): float(share)
        for domain, share in config.get("target_domain_token_shares", {}).items()
    }
    if target <= 0 or not shares:
        raise ValueError(
            "tail recovery requires target_context_tokens and domain token shares"
        )
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError("target_domain_token_shares must sum to 1")

    caps: dict[str, int] = {}
    allocated = 0
    domains = sorted(shares)
    for index, domain in enumerate(domains):
        if index == len(domains) - 1:
            cap = target - allocated
        else:
            cap = int(target * shares[domain])
            allocated += cap
        caps[domain] = cap
    return caps


def derive_tail_config(
    config: dict[str, Any],
    remaining_sources: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select remaining sources while preserving their original absolute caps."""

    remaining = _unique_names(remaining_sources, field_name="remaining_sources")
    sources = config.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("config sources must be a non-empty list")
    source_names = [str(source.get("name", "")) for source in sources]
    unknown = sorted(set(remaining) - set(source_names))
    if unknown:
        raise ValueError(f"unknown remaining sources: {unknown}")

    selected = [source for source in sources if str(source.get("name")) in remaining]
    selected_domains = tuple(dict.fromkeys(str(source["domain"]) for source in selected))
    caps = _domain_caps(config)
    missing_domains = sorted(set(selected_domains) - set(caps))
    if missing_domains:
        raise ValueError(f"remaining sources use untargeted domains: {missing_domains}")
    selected_target = sum(caps[domain] for domain in selected_domains)
    if selected_target <= 0:
        raise ValueError("remaining sources have no context-token budget")

    derived = copy.deepcopy(config)
    derived["sources"] = selected
    derived["target_context_tokens"] = selected_target
    derived["target_domain_token_shares"] = {
        domain: caps[domain] / selected_target for domain in selected_domains
    }
    derived["enforce_launch_gates"] = False
    derived["minimum_accepted_sessions"] = 0
    derived["minimum_context_tokens"] = 0
    derived["token_share_tolerance"] = 1.0
    derived["output_dir"] = str(Path(output_dir).expanduser().resolve())
    return derived


def validate_completed_artifacts(
    normalized_root: str | Path,
    completed_sources: Sequence[str],
) -> None:
    """Require closed shards and reports for every declared completed source."""

    root = Path(normalized_root).expanduser().resolve()
    names = _unique_names(completed_sources, field_name="completed_sources")
    if not root.is_dir():
        raise ValueError(f"normalized root does not exist: {root}")
    incomplete = sorted(root.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"normalized root contains incomplete files: {incomplete[0]}")

    for name in names:
        source_dir = root / name
        shards = sorted(source_dir.glob("part-*.jsonl"))
        if not source_dir.is_dir() or not shards:
            raise ValueError(f"completed source has no closed shards: {name}")
        if any(not path.is_file() or path.stat().st_size == 0 for path in shards):
            raise ValueError(f"completed source contains an empty shard: {name}")
        report = root / "reports" / f"{name}-rejections.jsonl"
        if not report.is_file():
            raise ValueError(f"completed source has no rejection report: {name}")


def _validate_partition(
    config: dict[str, Any],
    completed_sources: Sequence[str],
    remaining_sources: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    completed = _unique_names(completed_sources, field_name="completed_sources")
    remaining = _unique_names(remaining_sources, field_name="remaining_sources")
    overlap = sorted(set(completed) & set(remaining))
    if overlap:
        raise ValueError(f"completed and remaining sources overlap: {overlap}")

    configured = tuple(str(source.get("name", "")) for source in config.get("sources", []))
    if not configured or any(not name for name in configured):
        raise ValueError("config sources require non-empty names")
    if len(configured) != len(set(configured)):
        raise ValueError("config contains duplicate source names")
    supplied = set(completed) | set(remaining)
    if supplied != set(configured):
        missing = sorted(set(configured) - supplied)
        unknown = sorted(supplied - set(configured))
        raise ValueError(
            "completed and remaining sources must cover configured sources; "
            f"missing={missing}, unknown={unknown}"
        )
    return completed, remaining


def _validate_recovery_marker(
    marker: dict[str, Any],
    *,
    config_sha256: str,
    completed_sources: tuple[str, ...],
    remaining_sources: tuple[str, ...],
) -> None:
    expected = {
        "format": RECOVERY_FORMAT,
        "config_sha256": config_sha256,
        "completed_sources": list(completed_sources),
        "recovered_sources": list(remaining_sources),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(
                f"normalization recovery marker mismatch for {key}: "
                f"expected={value!r}, found={marker.get(key)!r}"
            )


def recover_normalization(
    config_path: str | Path,
    normalized_root: str | Path,
    completed_sources: Sequence[str],
    remaining_sources: Sequence[str],
    *,
    normalize_fn: Callable[..., Path] | None = None,
    legacy_normalized_roots: Sequence[str | Path] = (),
) -> Path:
    """Normalize and merge only remaining sources, preserving completed output."""

    config_path = Path(config_path).expanduser().resolve()
    requested_normalized_root = Path(normalized_root).expanduser().resolve()
    normalized_root = select_existing_normalized_root(
        normalized_root,
        legacy_normalized_roots,
    )
    if normalized_root != requested_normalized_root:
        print(f"using existing legacy normalized root: {normalized_root}", flush=True)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("normalization config must contain an object")
    completed, remaining = _validate_partition(
        config, completed_sources, remaining_sources
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    marker_path = normalized_root / "normalization_recovery_complete.json"

    if marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        _validate_recovery_marker(
            marker,
            config_sha256=config_sha256,
            completed_sources=completed,
            remaining_sources=remaining,
        )
        validate_completed_artifacts(normalized_root, (*completed, *remaining))
        return normalized_root

    validate_completed_artifacts(normalized_root, completed)
    temporary_root = normalized_root.parent / f".{normalized_root.name}-tail-recovery"
    temporary_complete = temporary_root / "normalization_complete.json"
    for name in remaining:
        destination = normalized_root / name
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
    existing_destinations = [
        normalized_root / name for name in remaining if (normalized_root / name).exists()
    ]
    if existing_destinations and not temporary_complete.is_file():
        raise ValueError(
            "remaining-source output already exists without a completed recovery: "
            f"{existing_destinations[0]}"
        )

    temporary_config = config_path.parent / f".{config_path.stem}-tail-recovery.yaml"
    derived = derive_tail_config(config, remaining, temporary_root)
    temporary_config_payload = yaml.safe_dump(
        derived,
        sort_keys=False,
        allow_unicode=True,
    )
    expected_tail_config_sha256 = hashlib.sha256(
        temporary_config_payload.encode()
    ).hexdigest()
    if temporary_complete.is_file():
        try:
            existing_tail_marker = json.loads(temporary_complete.read_text())
        except json.JSONDecodeError:
            existing_tail_marker = {}
        if (
            existing_tail_marker.get("config_sha256")
            != expected_tail_config_sha256
        ):
            print(
                "discarding stale tail normalization after config change: "
                f"{temporary_root}",
                flush=True,
            )
            shutil.rmtree(temporary_root)

    try:
        if not temporary_complete.is_file():
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            temporary_config.write_text(temporary_config_payload)
            if normalize_fn is None:
                from rpmem.training.corpus.normalize import normalize_sources

                normalize_fn = normalize_sources
            normalize_fn(
                temporary_config,
                output_dir=temporary_root,
                overwrite=True,
            )
        if not temporary_complete.is_file():
            raise ValueError(
                "tail normalization did not publish normalization_complete.json"
            )
        validate_completed_artifacts(temporary_root, remaining)

        reports = normalized_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        tail_stats = temporary_root / "normalization_stats.json"
        if not tail_stats.is_file():
            raise ValueError("tail normalization did not publish normalization stats")
        shutil.copy2(
            tail_stats,
            reports / "tail-recovery-normalization-stats.json",
        )

        for name in remaining:
            source = temporary_root / name
            destination = normalized_root / name
            if source.exists() and destination.exists():
                raise ValueError(
                    f"both temporary and destination source directories exist: {name}"
                )
            if source.exists():
                source.replace(destination)
            elif not destination.exists():
                raise ValueError(f"recovered source directory is missing: {name}")

            source_report = temporary_root / "reports" / f"{name}-rejections.jsonl"
            destination_report = reports / f"{name}-rejections.jsonl"
            if source_report.exists() and destination_report.exists():
                raise ValueError(
                    f"both temporary and destination rejection reports exist: {name}"
                )
            if source_report.exists():
                source_report.replace(destination_report)
            elif not destination_report.exists():
                raise ValueError(f"recovered rejection report is missing: {name}")

        validate_completed_artifacts(normalized_root, (*completed, *remaining))
        tail_marker = json.loads(temporary_complete.read_text())
        write_json_atomic(
            marker_path,
            {
                "format": RECOVERY_FORMAT,
                "config_sha256": config_sha256,
                "completed_sources": list(completed),
                "recovered_sources": list(remaining),
                "tail_config_sha256": tail_marker.get("config_sha256", ""),
                "tail_stats": "reports/tail-recovery-normalization-stats.json",
            },
        )
        shutil.rmtree(temporary_root)
        return normalized_root
    finally:
        temporary_config.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--normalized_root", required=True)
    parser.add_argument("--legacy_normalized_root", action="append", default=[])
    parser.add_argument("--completed_sources", nargs="+", required=True)
    parser.add_argument("--remaining_sources", nargs="+", required=True)
    args = parser.parse_args()
    output = recover_normalization(
        args.config,
        args.normalized_root,
        args.completed_sources,
        args.remaining_sources,
        legacy_normalized_roots=args.legacy_normalized_root,
    )
    print(f"normalization recovery complete: {output}")


if __name__ == "__main__":
    main()
