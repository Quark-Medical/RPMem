"""Normalize and freeze the pinned PrefEval release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="data/prefeval/formal_v1")
    args = parser.parse_args()

    freeze = prepare_dataset(
        Path(args.source_root).expanduser(),
        Path(args.output_root).expanduser(),
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"formal PrefEval data ready: {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
