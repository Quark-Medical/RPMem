"""Atomic Parquet sharding and schemas for canonical corpus artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SESSION_SCHEMA = pa.schema(
    [
        ("session_index", pa.int64()),
        ("session_id", pa.string()),
        ("probe_start", pa.int64()),
        ("probe_count", pa.int32()),
        ("source", pa.string()),
        ("domain", pa.string()),
        ("context", pa.string()),
        ("context_tokens", pa.int32()),
        ("context_token_max", pa.int32()),
        ("context_token_counts_json", pa.string()),
        ("context_hash", pa.string()),
        ("simhash", pa.uint64()),
        ("events_json", pa.string()),
        ("memory_atoms_json", pa.string()),
        ("provenance_json", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

PROBE_SCHEMA = pa.schema(
    [
        ("probe_index", pa.int64()),
        ("session_index", pa.int64()),
        ("session_id", pa.string()),
        ("probe_id", pa.string()),
        ("prompt", pa.string()),
        ("probe_type", pa.string()),
        ("answerable", pa.bool_()),
        ("reference", pa.string()),
        ("evidence_event_ids", pa.list_(pa.string())),
        ("metadata_json", pa.string()),
    ]
)

INDEX_SCHEMA = pa.schema(
    [
        ("session_index", pa.int64()),
        ("session_id", pa.string()),
        ("probe_indices", pa.list_(pa.int64())),
    ]
)


class AtomicParquetShardWriter:
    """Write bounded row buffers and publish only fully closed shard files."""

    def __init__(
        self,
        directory: str | Path,
        *,
        schema: pa.Schema,
        rows_per_shard: int,
        row_group_rows: int,
        prefix: str = "part",
    ):
        if rows_per_shard < 1 or row_group_rows < 1:
            raise ValueError("rows_per_shard and row_group_rows must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.row_group_rows = row_group_rows
        self.prefix = prefix
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
        shard_index = len(self.paths)
        path = self.directory / f"{self.prefix}-{shard_index:05d}.parquet"
        temporary = path.with_suffix(path.suffix + ".incomplete")
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            row_group_size=self.row_group_rows,
            use_dictionary=True,
        )
        temporary.replace(path)
        self.paths.append(path)
        self.rows.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "AtomicParquetShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)
