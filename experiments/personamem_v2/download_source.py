"""Download only the pinned PersonaMem-v2 32K text assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .formal_data import (
        SOURCE_DATASET,
        SOURCE_REVISION,
        SPLIT_FILES,
        normalize_history_path,
    )
except ImportError:
    from formal_data import (
        SOURCE_DATASET,
        SOURCE_REVISION,
        SPLIT_FILES,
        normalize_history_path,
    )


DOCUMENTATION_FILES = ("README.md", "column_descriptions.md")
DEFAULT_ENDPOINT = "https://huggingface.co"
RANGE_CHUNK_BYTES = 8 << 20
CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def source_url(endpoint: str, relative_path: str) -> str:
    repo = urllib.parse.quote(SOURCE_DATASET, safe="/")
    revision = urllib.parse.quote(SOURCE_REVISION, safe="")
    path = urllib.parse.quote(relative_path, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/{path}"


def _existing_file_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix != ".json":
        return True
    try:
        with path.open(encoding="utf-8") as handle:
            return isinstance(json.load(handle), dict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _request_bytes(
    url: str,
    headers: dict[str, str],
    *,
    retries: int,
    timeout_seconds: float,
) -> tuple[bytes, int, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return (
                    response.read(),
                    int(response.status),
                    response.headers.get("Content-Range"),
                )
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(30.0, 2 ** (attempt - 1)) + random.random())
    raise RuntimeError(
        f"request failed after {retries} attempts: {last_error}"
    ) from last_error


def _download_ranged_file(
    url: str,
    headers: dict[str, str],
    temporary: Path,
    *,
    retries: int,
    timeout_seconds: float,
) -> None:
    offset = 0
    total_size: int | None = None
    with temporary.open("wb") as handle:
        while total_size is None or offset < total_size:
            end = offset + RANGE_CHUNK_BYTES - 1
            range_headers = {**headers, "Range": f"bytes={offset}-{end}"}
            data, status, content_range = _request_bytes(
                url,
                range_headers,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
            if status == 200 and offset == 0:
                handle.write(data)
                return
            match = CONTENT_RANGE_PATTERN.fullmatch(content_range or "")
            if status != 206 or match is None:
                raise RuntimeError(
                    f"server ignored byte range at offset {offset}: "
                    f"status={status} content_range={content_range!r}"
                )
            start, returned_end, remote_size = map(int, match.groups())
            expected_bytes = returned_end - start + 1
            if start != offset or len(data) != expected_bytes:
                raise RuntimeError(
                    f"invalid byte range at offset {offset}: "
                    f"returned={start}-{returned_end} bytes={len(data)}"
                )
            handle.write(data)
            offset = returned_end + 1
            total_size = remote_size


def download_file(
    endpoint: str,
    relative_path: str,
    output_root: Path,
    *,
    retries: int,
    timeout_seconds: float,
) -> bool:
    destination = output_root / relative_path
    if _existing_file_is_valid(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    headers = {"User-Agent": "memlora-personamem-v2-downloader/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = source_url(endpoint, relative_path)
    last_error: Exception | None = None
    full_attempts = 2
    for attempt in range(1, full_attempts + 1):
        temporary.unlink(missing_ok=True)
        try:
            if relative_path in SPLIT_FILES.values():
                _download_ranged_file(
                    url,
                    headers,
                    temporary,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
            else:
                data, _, _ = _request_bytes(
                    url,
                    headers,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
                with temporary.open("wb") as handle:
                    handle.write(data)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"empty response for {relative_path}")
            if destination.suffix == ".json":
                with temporary.open(encoding="utf-8") as handle:
                    if not isinstance(json.load(handle), dict):
                        raise ValueError(f"JSON root is not an object: {relative_path}")
            os.replace(temporary, destination)
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < full_attempts:
                time.sleep(min(30.0, 2 ** (attempt - 1)) + random.random())
    raise RuntimeError(
        f"failed to download PersonaMem-v2 file after {full_attempts} full attempts: "
        f"{relative_path}: {last_error}"
    ) from last_error


def referenced_history_paths(output_root: Path) -> list[str]:
    histories: set[str] = set()
    for relative_csv in SPLIT_FILES.values():
        path = output_root / relative_csv
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "chat_history_32k_link" not in (reader.fieldnames or []):
                raise ValueError(
                    f"PersonaMem-v2 split misses chat_history_32k_link: {path}"
                )
            for row in reader:
                histories.add(normalize_history_path(row["chat_history_32k_link"]))
    return sorted(histories)


def download_many(
    endpoint: str,
    relative_paths: list[str],
    output_root: Path,
    *,
    workers: int,
    retries: int,
    timeout_seconds: float,
    label: str,
) -> None:
    downloaded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_file,
                endpoint,
                relative,
                output_root,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): relative
            for relative in relative_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            relative = futures[future]
            try:
                changed = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"PersonaMem-v2 download failed: {relative}"
                ) from exc
            downloaded += int(changed)
            reused += int(not changed)
            if completed % 100 == 0 or completed == len(relative_paths):
                print(
                    f"[personamem-v2 download] stage={label} "
                    f"files={completed}/{len(relative_paths)} "
                    f"downloaded={downloaded} reused={reused}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.workers <= 0 or args.retries <= 0 or args.timeout_seconds <= 0:
        parser.error("workers, retries, and timeout-seconds must be positive")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    download_many(
        args.endpoint,
        [*SPLIT_FILES.values(), *DOCUMENTATION_FILES],
        output_root,
        workers=min(args.workers, 5),
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        label="metadata",
    )
    histories = referenced_history_paths(output_root)
    print(
        f"[personamem-v2 download] referenced_histories={len(histories)}",
        flush=True,
    )
    download_many(
        args.endpoint,
        histories,
        output_root,
        workers=args.workers,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        label="histories",
    )
    print(f"PersonaMem-v2 pinned source ready: {output_root}")


if __name__ == "__main__":
    main()
