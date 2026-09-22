"""Parquet storage for fixed-trajectory teacher top-k distributions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAT_NAME = "rpmem.teacher_topk.parquet"
FORMAT_VERSION = 1
COMPLETION_MARKER = "store_complete.json"
COMPLETION_MARKER_FORMAT = "rpmem.teacher_topk.complete"
SOURCE_BINDING_FORMAT = "rpmem.teacher_sources.v1"


def source_content_digests(source_paths: Iterable[str | Path]) -> list[str]:
    """Read stable logical identities from canonical corpus split manifests."""

    digests: list[str] = []
    for raw_path in source_paths:
        path = Path(raw_path).resolve()
        digest = ""
        if path.name.endswith(".corpus.json"):
            payload = json.loads(path.read_text())
            digest = str(payload.get("content_digest", ""))
            if not digest:
                raise ValueError(
                    f"canonical corpus manifest has no content_digest: {path}"
                )
        digests.append(digest)
    return digests


@dataclass(frozen=True)
class TeacherTopKRecord:
    """Teacher top-k rows for all labelled tokens in one training sample."""

    sample_idx: int
    values: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float16)
        indices = np.asarray(self.indices, dtype=np.int32)
        if values.ndim != 2 or indices.ndim != 2:
            raise ValueError("teacher values and indices must both be rank-2")
        if values.shape != indices.shape:
            raise ValueError(
                "teacher values and indices must have identical shapes, got "
                f"{values.shape} and {indices.shape}"
            )
        if int(self.sample_idx) < 0:
            raise ValueError("sample_idx must be non-negative")
        object.__setattr__(self, "sample_idx", int(self.sample_idx))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)

    @property
    def top_k(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.values.shape[0])


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def write_store_metadata(root: str | Path, metadata: dict[str, Any]) -> Path:
    """Create a teacher-store manifest without silently changing an existing one."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    normalized = dict(metadata)
    normalized.setdefault("format", FORMAT_NAME)
    normalized.setdefault("format_version", FORMAT_VERSION)
    meta_path = root / "meta.json"

    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != normalized:
            differing = sorted(
                key
                for key in set(existing) | set(normalized)
                if existing.get(key) != normalized.get(key)
            )
            raise ValueError(
                "teacher store metadata already exists with incompatible fields: "
                + ", ".join(differing)
            )
        return meta_path

    _atomic_write_text(
        meta_path, json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    )
    return meta_path


def write_teacher_shard(
    path: str | Path,
    records: Iterable[TeacherTopKRecord],
    *,
    top_k: int,
    rows_per_group: int = 16,
) -> Path:
    """Write one atomic Parquet shard containing ragged per-sample token rows."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows_per_group = int(rows_per_group)
    if rows_per_group <= 0:
        raise ValueError("rows_per_group must be positive")

    materialized = list(records)
    seen: set[int] = set()
    for record in materialized:
        if record.top_k != top_k:
            raise ValueError(
                f"sample {record.sample_idx} has top_k={record.top_k}, expected {top_k}"
            )
        if record.sample_idx in seen:
            raise ValueError(
                f"duplicate teacher sample_idx {record.sample_idx} in shard"
            )
        seen.add(record.sample_idx)

    token_counts = np.asarray(
        [record.num_tokens for record in materialized],
        dtype=np.int32,
    )
    offsets = np.empty(len(materialized) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(token_counts, out=offsets[1:])

    if materialized:
        values = np.concatenate(
            [record.values for record in materialized],
            axis=0,
            dtype=np.float16,
        )
        indices = np.concatenate(
            [record.indices for record in materialized],
            axis=0,
            dtype=np.int32,
        )
    else:
        values = np.empty((0, top_k), dtype=np.float16)
        indices = np.empty((0, top_k), dtype=np.int32)

    values_rows = pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float16()),
        top_k,
    )
    indices_rows = pa.FixedSizeListArray.from_arrays(
        pa.array(indices.reshape(-1), type=pa.int32()),
        top_k,
    )
    values_column = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        values_rows,
    )
    indices_column = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        indices_rows,
    )
    table = pa.Table.from_arrays(
        [
            pa.array([record.sample_idx for record in materialized], type=pa.int64()),
            pa.array(token_counts, type=pa.int32()),
            values_column,
            indices_column,
        ],
        names=["sample_idx", "num_tokens", "values", "indices"],
    )
    table = table.replace_schema_metadata(
        {
            b"format": FORMAT_NAME.encode(),
            b"format_version": str(FORMAT_VERSION).encode(),
            b"top_k": str(top_k).encode(),
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pq.write_table(
        table,
        tmp_path,
        compression="zstd",
        row_group_size=rows_per_group,
    )
    os.replace(tmp_path, path)
    return path


class TeacherShardWriter:
    """Buffered, resumable writer for one deterministic generation shard."""

    def __init__(
        self,
        root: str | Path,
        *,
        shard_id: int,
        top_k: int,
        records_per_file: int = 512,
        rows_per_group: int = 16,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_id = int(shard_id)
        self.top_k = int(top_k)
        self.records_per_file = int(records_per_file)
        self.rows_per_group = int(rows_per_group)
        if self.shard_id < 0:
            raise ValueError("shard_id must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.records_per_file <= 0:
            raise ValueError("records_per_file must be positive")
        if self.rows_per_group <= 0:
            raise ValueError("rows_per_group must be positive")

        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"teacher store metadata not found: {meta_path}")
        metadata = json.loads(meta_path.read_text())
        if metadata.get("top_k") != self.top_k:
            raise ValueError(
                f"teacher store top_k={metadata.get('top_k')!r}, expected {self.top_k}"
            )

        self._buffer: list[TeacherTopKRecord] = []
        self._completed: set[int] = set()
        self._part_index = 0
        pattern = f"part-{self.shard_id:05d}-*.parquet"
        for shard_path in sorted(self.root.glob(pattern)):
            part_text = shard_path.stem.rsplit("-", 1)[-1]
            self._part_index = max(self._part_index, int(part_text) + 1)
            sample_ids = pq.read_table(shard_path, columns=["sample_idx"])[
                "sample_idx"
            ].to_pylist()
            for sample_idx in sample_ids:
                sample_idx = int(sample_idx)
                if sample_idx in self._completed:
                    raise ValueError(
                        f"duplicate teacher sample_idx {sample_idx} in shard "
                        f"{self.shard_id}"
                    )
                self._completed.add(sample_idx)

    @property
    def completed_sample_indices(self) -> frozenset[int]:
        return frozenset(self._completed)

    def add(self, record: TeacherTopKRecord) -> bool:
        """Buffer a record, returning False when it was already completed."""

        if record.sample_idx in self._completed or any(
            pending.sample_idx == record.sample_idx for pending in self._buffer
        ):
            return False
        if record.top_k != self.top_k:
            raise ValueError(
                f"sample {record.sample_idx} has top_k={record.top_k}, "
                f"expected {self.top_k}"
            )
        self._buffer.append(record)
        if len(self._buffer) >= self.records_per_file:
            self.flush()
        return True

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        path = self.root / (f"part-{self.shard_id:05d}-{self._part_index:05d}.parquet")
        write_teacher_shard(
            path,
            self._buffer,
            top_k=self.top_k,
            rows_per_group=self.rows_per_group,
        )
        self._completed.update(record.sample_idx for record in self._buffer)
        self._buffer.clear()
        self._part_index += 1
        return path

    def close(self) -> None:
        self.flush()
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "shard_id": self.shard_id,
            "top_k": self.top_k,
            "completed_samples": len(self._completed),
            "part_files": self._part_index,
            "rows_per_group": self.rows_per_group,
        }
        _atomic_write_text(
            self.root / f"done-{self.shard_id:05d}.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    def __enter__(self) -> "TeacherShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


class TeacherLogprobStore:
    """Random-access view over a directory of teacher Parquet shards."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_top_k: int | None = None,
        expected_max_seq_len: int | None = None,
        cache_size: int = 2,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"teacher store metadata not found: {meta_path}")
        self.metadata = json.loads(meta_path.read_text())
        if self.metadata.get("format") != FORMAT_NAME:
            raise ValueError(
                f"unsupported teacher store format: {self.metadata.get('format')!r}"
            )
        if self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "unsupported teacher store format_version: "
                f"{self.metadata.get('format_version')!r}"
            )
        self._check_expected("top_k", expected_top_k)
        self._check_expected("max_seq_len", expected_max_seq_len)

        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[Path, Any] = OrderedDict()
        self._index: dict[int, tuple[Path, int, int]] = {}

        for shard_path in sorted(self.root.glob("part-*.parquet")):
            parquet_file = pq.ParquetFile(shard_path)
            all_sample_ids = pq.read_table(shard_path, columns=["sample_idx"])[
                "sample_idx"
            ].to_pylist()
            linear_offset = 0
            for row_group in range(parquet_file.num_row_groups):
                num_rows = parquet_file.metadata.row_group(row_group).num_rows
                sample_ids = all_sample_ids[linear_offset : linear_offset + num_rows]
                linear_offset += num_rows
                for row_offset, sample_idx in enumerate(sample_ids):
                    sample_idx = int(sample_idx)
                    if sample_idx in self._index:
                        previous_path, _, _ = self._index[sample_idx]
                        raise ValueError(
                            f"duplicate teacher sample_idx {sample_idx} in "
                            f"{previous_path.name} and {shard_path.name}"
                        )
                    self._index[sample_idx] = (
                        shard_path,
                        row_group,
                        row_offset,
                    )

        self.sample_indices = tuple(sorted(self._index))

    def _check_expected(self, key: str, expected: int | None) -> None:
        if expected is None:
            return
        actual = self.metadata.get(key)
        if actual != expected:
            raise ValueError(f"teacher store {key}={actual!r}, expected {expected!r}")

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, sample_idx: object) -> bool:
        return isinstance(sample_idx, int) and sample_idx in self._index

    def validate_coverage(self, expected_n_samples: int) -> None:
        """Require exactly one teacher record for every global sample index."""

        expected_n_samples = int(expected_n_samples)
        if expected_n_samples < 0:
            raise ValueError("expected_n_samples must be non-negative")
        actual = set(self._index)
        missing_examples: list[int] = []
        missing_count = 0
        for sample_idx in range(expected_n_samples):
            if sample_idx not in actual:
                missing_count += 1
                if len(missing_examples) < 20:
                    missing_examples.append(sample_idx)
        out_of_range = sorted(
            sample_idx
            for sample_idx in actual
            if sample_idx < 0 or sample_idx >= expected_n_samples
        )
        if not missing_count and not out_of_range:
            return

        problems = []
        if missing_count:
            problems.append(
                f"missing sample IDs {missing_examples}"
                + (f" ({missing_count} total)" if missing_count > 20 else "")
            )
        if out_of_range:
            problems.append(
                f"out-of-range sample IDs {out_of_range[:20]}"
                + (f" ({len(out_of_range)} total)" if len(out_of_range) > 20 else "")
            )
        raise ValueError("teacher store coverage invalid: " + "; ".join(problems))

    def __getitem__(self, sample_idx: int) -> TeacherTopKRecord:
        try:
            shard_path, row_group, row_offset = self._index[int(sample_idx)]
        except KeyError as exc:
            raise KeyError(f"teacher sample_idx {sample_idx} not found") from exc

        parquet_file = self._load_parquet_file(shard_path)
        table = parquet_file.read_row_group(row_group)
        num_tokens = int(table["num_tokens"][row_offset].as_py())
        values = np.asarray(table["values"][row_offset].as_py(), dtype=np.float16)
        indices = np.asarray(table["indices"][row_offset].as_py(), dtype=np.int32)
        top_k = int(self.metadata["top_k"])
        values = values.reshape(num_tokens, top_k)
        indices = indices.reshape(num_tokens, top_k)
        return TeacherTopKRecord(int(sample_idx), values, indices)

    def _load_parquet_file(self, path: Path):
        import pyarrow.parquet as pq

        if path in self._cache:
            parquet_file = self._cache.pop(path)
            self._cache[path] = parquet_file
            return parquet_file

        parquet_file = pq.ParquetFile(path)
        if self.cache_size > 0:
            self._cache[path] = parquet_file
            while len(self._cache) > self.cache_size:
                _, evicted = self._cache.popitem(last=False)
                evicted.close()
        return parquet_file

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        cache = getattr(self, "_cache", None)
        if cache is None:
            return
        for parquet_file in cache.values():
            parquet_file.close()
        cache.clear()

    def __del__(self):
        self.close()


def validate_completion_manifests(
    root: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Validate that every deterministic generation shard completed."""

    root = Path(root)
    if metadata is None:
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"teacher store metadata not found: {meta_path}")
        metadata = json.loads(meta_path.read_text())
    try:
        n_samples = int(metadata["n_samples"])
        num_shards = int(metadata["num_shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "teacher metadata requires integer n_samples and num_shards"
        ) from exc
    if n_samples < 0 or num_shards <= 0:
        raise ValueError(
            "teacher n_samples must be non-negative and num_shards positive"
        )

    expected_manifests = {
        root / f"done-{shard_id:05d}.json" for shard_id in range(num_shards)
    }
    missing_manifests = sorted(
        path.name for path in expected_manifests if not path.is_file()
    )
    if missing_manifests:
        raise ValueError(
            "missing completion manifests: " + ", ".join(missing_manifests)
        )

    for shard_id in range(num_shards):
        manifest_path = root / f"done-{shard_id:05d}.json"
        manifest = json.loads(manifest_path.read_text())
        expected_count = len(range(shard_id, n_samples, num_shards))
        if manifest.get("shard_id") != shard_id:
            raise ValueError(
                f"completion manifest {manifest_path.name} has wrong shard_id"
            )
        if manifest.get("completed_samples") != expected_count:
            raise ValueError(
                f"completion manifest {manifest_path.name} reports "
                f"{manifest.get('completed_samples')} samples; expected {expected_count}"
            )

    return {
        "n_samples": n_samples,
        "num_shards": num_shards,
    }


def validate_store_source_files(
    root: str | Path,
    source_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Require a teacher store to match the ordered source files exactly."""

    root = Path(root)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"teacher store metadata not found: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    paths = [Path(path).resolve() for path in source_paths]
    if "source_binding_format" in metadata or "source_bindings" in metadata:
        if metadata.get("source_binding_format") != SOURCE_BINDING_FORMAT:
            raise ValueError("unsupported teacher source binding format")
        if metadata.get("source_bindings") != source_file_bindings(paths):
            raise ValueError("teacher store source content or file order does not match")
        return metadata
    expected_paths = [str(path) for path in paths]
    expected_sizes = [path.stat().st_size for path in paths]
    if metadata.get("train_data") != expected_paths:
        raise ValueError(
            "teacher store train_data does not match the resolved training file "
            "order; recompute teacher top-k shards"
        )
    if metadata.get("train_file_sizes") != expected_sizes:
        raise ValueError(
            "teacher store training file sizes changed; recompute teacher top-k shards"
        )
    expected_digests = source_content_digests(paths)
    if "train_content_digests" in metadata and (
        metadata.get("train_content_digests") != expected_digests
    ):
        raise ValueError(
            "teacher store training content digest changed; recompute teacher top-k shards"
        )
    return metadata


def source_file_bindings(source_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Bind relocatable stores to ordered file contents, including split identity."""

    paths = [Path(path) for path in source_paths]
    return [
        {"sha256": _sha256_file(path), "size": path.stat().st_size,
         "content_digest": digest}
        for path, digest in zip(paths, source_content_digests(paths))
    ]


def validate_store_complete(root: str | Path) -> dict[str, int]:
    """Validate completion manifests and exact global sample coverage."""

    root = Path(root)
    report = validate_completion_manifests(root)
    store = TeacherLogprobStore(root)
    try:
        store.validate_coverage(report["n_samples"])
    finally:
        store.close()
    return {
        **report,
        "part_files": len(list(root.glob("part-*.parquet"))),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_store_inventory(root: str | Path) -> list[dict[str, Any]]:
    """Return a fast, stable inventory for an already validated store."""

    root = Path(root)
    paths = [
        root / "meta.json",
        *sorted(root.glob("done-*.json")),
        *sorted(root.glob("part-*.parquet")),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "teacher store inventory missing: " + ", ".join(missing)
        )
    entries = []
    for path in paths:
        entry: dict[str, Any] = {
            "name": path.name,
            "size": path.stat().st_size,
        }
        if path.name == "meta.json" or path.name.startswith("done-"):
            entry["sha256"] = _sha256_file(path)
        entries.append(entry)
    return entries


def write_store_completion_marker(root: str | Path) -> dict[str, Any]:
    """Fully validate a store, then publish its fast completion marker."""

    root = Path(root)
    report = validate_store_complete(root)
    marker = {
        "format": COMPLETION_MARKER_FORMAT,
        "format_version": 1,
        "report": report,
        "inventory": teacher_store_inventory(root),
    }
    _atomic_write_text(
        root / COMPLETION_MARKER,
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
    )
    return marker


def validate_store_completion_marker(root: str | Path) -> dict[str, Any]:
    """Verify that a previously validated store has not changed."""

    root = Path(root)
    marker_path = root / COMPLETION_MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(f"teacher completion marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("format") != COMPLETION_MARKER_FORMAT:
        raise ValueError("unsupported teacher completion marker format")
    if marker.get("format_version") != 1:
        raise ValueError("unsupported teacher completion marker version")
    current_inventory = teacher_store_inventory(root)
    if marker.get("inventory") != current_inventory:
        raise ValueError("teacher store changed after completion validation")
    return marker
