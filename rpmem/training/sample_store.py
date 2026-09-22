"""Lazy random-access stores for Phase 1 training samples."""

from __future__ import annotations

import bisect
import json
from array import array
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq


def _normalize_index(index: int, length: int) -> int:
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError(index)
    return index


class JsonlSampleStore(Sequence[dict[str, Any]]):
    """Index JSONL rows by byte offset and decode only the requested row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets = array("Q")
        self._handle = None

        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = _normalize_index(index, len(self))
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        self._handle.seek(self._offsets[index])
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


class ParquetSampleStore(Sequence[dict[str, Any]]):
    """Read individual Parquet rows while caching a bounded set of row groups."""

    def __init__(self, path: str | Path, row_group_cache_size: int = 2):
        if row_group_cache_size < 1:
            raise ValueError("row_group_cache_size must be at least 1")

        self.path = Path(path)
        self.row_group_cache_size = row_group_cache_size
        self._parquet_file = None
        self._row_group_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

        parquet_file = pq.ParquetFile(self.path)
        self._row_group_ends: list[int] = []
        total = 0
        for row_group in range(parquet_file.metadata.num_row_groups):
            total += parquet_file.metadata.row_group(row_group).num_rows
            self._row_group_ends.append(total)
        parquet_file.close()
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = _normalize_index(index, len(self))
        row_group = bisect.bisect_right(self._row_group_ends, index)
        row_group_start = 0 if row_group == 0 else self._row_group_ends[row_group - 1]
        rows = self._read_row_group(row_group)
        return rows[index - row_group_start]

    def _read_row_group(self, row_group: int) -> list[dict[str, Any]]:
        rows = self._row_group_cache.pop(row_group, None)
        if rows is None:
            if self._parquet_file is None:
                self._parquet_file = pq.ParquetFile(self.path)
            rows = self._parquet_file.read_row_group(row_group).to_pylist()
        self._row_group_cache[row_group] = rows
        while len(self._row_group_cache) > self.row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return rows

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parquet_file"] = None
        state["_row_group_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        if self._parquet_file is not None:
            self._parquet_file.close()
            self._parquet_file = None
        self._row_group_cache.clear()

    def __del__(self):
        self.close()


class CompositeSampleStore(Sequence[dict[str, Any]]):
    """Expose several stores through one stable global sample index."""

    def __init__(
        self,
        stores: Sequence[Sequence[dict[str, Any]]],
        *,
        paths: Sequence[Path] = (),
    ):
        self.stores = list(stores)
        self.paths = tuple(paths)
        self._store_ends: list[int] = []
        total = 0
        for store in self.stores:
            total += len(store)
            self._store_ends.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = _normalize_index(index, len(self))
        store_idx = bisect.bisect_right(self._store_ends, index)
        store_start = 0 if store_idx == 0 else self._store_ends[store_idx - 1]
        return self.stores[store_idx][index - store_start]

    def close(self) -> None:
        for store in self.stores:
            close = getattr(store, "close", None)
            if close is not None:
                close()

    def __del__(self):
        self.close()


def _resolve_data_paths(values: Sequence[str], manifest_path: Path) -> tuple[Path, ...]:
    paths = tuple((manifest_path.parent / value).resolve() for value in values)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing canonical corpus data file(s): " + ", ".join(map(str, missing))
        )
    duplicates = sorted(
        (path for path, count in Counter(paths).items() if count > 1), key=str
    )
    if duplicates:
        raise ValueError(
            "duplicate canonical corpus paths: " + ", ".join(map(str, duplicates))
        )
    return paths


def _open_data_file(path: Path) -> Sequence[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return JsonlSampleStore(path)
    if suffix == ".parquet":
        return ParquetSampleStore(path)
    raise ValueError(f"Unsupported training data format: {path}")


def _open_data_files(paths: Sequence[Path]) -> CompositeSampleStore:
    stores = [_open_data_file(path) for path in paths]
    return CompositeSampleStore(stores, paths=paths)


class CanonicalCorpusSampleStore(Sequence[dict[str, Any]]):
    """Lazily join canonical session, probe, and split-index tables."""

    FORMAT_NAME = "memlora_corpus_split_v1"

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("format") != self.FORMAT_NAME:
            raise ValueError(
                f"unsupported canonical corpus manifest format: {manifest.get('format')!r}"
            )
        self.split = str(manifest.get("split", ""))
        self.require_references = bool(manifest.get("require_references", True))
        self.content_digest = str(manifest.get("content_digest", ""))
        self.session_paths = _resolve_data_paths(
            manifest.get("sessions", []), self.manifest_path
        )
        self.probe_paths = _resolve_data_paths(
            manifest.get("probes", []), self.manifest_path
        )
        self.index_paths = _resolve_data_paths(
            manifest.get("indices", []), self.manifest_path
        )
        self.sessions = _open_data_files(self.session_paths)
        self.probes = _open_data_files(self.probe_paths)
        self.indices = _open_data_files(self.index_paths)
        expected = int(manifest.get("session_count", len(self.indices)))
        if len(self.indices) != expected:
            raise ValueError(
                f"{self.manifest_path}: expected {expected} split sessions, "
                f"found {len(self.indices)}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = _normalize_index(index, len(self))
        selection = self.indices[index]
        session_index = int(selection["session_index"])
        session = dict(self.sessions[session_index])
        if session.get("session_id") != selection.get("session_id"):
            raise ValueError(
                f"{self.manifest_path}: session ID mismatch at split row {index}"
            )

        probe_start = int(session.get("probe_start", 0))
        probe_count = int(session.get("probe_count", 0))
        probe_end = probe_start + probe_count
        selected_probes = []
        seen_probe_indices: set[int] = set()
        for raw_probe_index in selection.get("probe_indices", []):
            probe_index = int(raw_probe_index)
            if probe_index in seen_probe_indices:
                raise ValueError(
                    f"{self.manifest_path}: duplicate probe index {probe_index}"
                )
            seen_probe_indices.add(probe_index)
            if not probe_start <= probe_index < probe_end:
                raise ValueError(
                    f"{self.manifest_path}: probe {probe_index} is outside "
                    f"session {session_index} range [{probe_start}, {probe_end})"
                )
            probe = self.probes[probe_index]
            if (
                int(probe.get("session_index", -1)) != session_index
                or probe.get("session_id") != session.get("session_id")
            ):
                raise ValueError(
                    f"{self.manifest_path}: probe/session join mismatch at {probe_index}"
                )
            if self.require_references and not str(probe.get("reference", "")).strip():
                raise ValueError(
                    f"{self.manifest_path}: probe {probe.get('probe_id', probe_index)} "
                    "is missing reference required by fixed-trajectory training"
                )
            selected_probes.append(probe)

        if not selected_probes:
            raise ValueError(
                f"{self.manifest_path}: split row {index} selects no probes"
            )
        session.update(
            {
                "prompts": [str(probe["prompt"]) for probe in selected_probes],
                "responses": [str(probe.get("reference", "")) for probe in selected_probes],
                "probe_ids": [str(probe["probe_id"]) for probe in selected_probes],
                "probe_types": [str(probe["probe_type"]) for probe in selected_probes],
                "probe_answerable": [bool(probe["answerable"]) for probe in selected_probes],
                "corpus_split": self.split,
            }
        )
        return session

    def close(self) -> None:
        for attribute in ("sessions", "probes", "indices"):
            store = getattr(self, attribute, None)
            if store is not None:
                store.close()

    def __del__(self):
        self.close()


def resolve_sample_paths(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand ordered path manifests into canonical data file paths."""

    resolved: list[Path] = []

    def add_path(raw_path: str | Path, *, relative_to: Path | None = None) -> None:
        path = Path(raw_path).expanduser()
        if relative_to is not None and not path.is_absolute():
            path = relative_to / path
        path = path.resolve()
        if path.suffix.lower() in {".manifest", ".txt"}:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    add_path(line, relative_to=path.parent)
            return
        if not path.is_file():
            raise FileNotFoundError(f"training data file not found: {path}")
        resolved.append(path)

    for raw_path in inputs:
        add_path(raw_path)

    duplicates = sorted(
        (path for path, count in Counter(resolved).items() if count > 1),
        key=str,
    )
    if duplicates:
        raise ValueError(
            "duplicate training data paths: " + ", ".join(map(str, duplicates))
        )
    return tuple(resolved)


def open_sample_store(paths: Sequence[str | Path]) -> CompositeSampleStore:
    """Open JSONL and Parquet files in the caller-provided order."""

    resolved_paths = resolve_sample_paths(paths)
    stores: list[Sequence[dict[str, Any]]] = []
    for path in resolved_paths:
        if path.name.endswith(".corpus.json"):
            stores.append(CanonicalCorpusSampleStore(path))
        else:
            stores.append(_open_data_file(path))
    return CompositeSampleStore(stores, paths=resolved_paths)
