"""Validate the formal PersonaMem-v2 32K text dataset freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from formal_data import validate_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/personamem_v2/formal_v1")
    parser.add_argument("--skip-source-checksums", action="store_true")
    args = parser.parse_args()
    freeze, splits = validate_dataset_root(
        Path(args.data_root),
        verify_source_files=not args.skip_source_checksums,
    )
    counts = ",".join(f"{split}={len(rows)}" for split, rows in splits.items())
    print(
        "formal_personamem_v2_data_ready:"
        f" sha256={freeze['dataset_sha256']} {counts}"
    )


if __name__ == "__main__":
    main()
