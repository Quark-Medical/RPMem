"""Validate the formal PrefEval dataset freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from formal_data import validate_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/prefeval/formal_v1")
    args = parser.parse_args()
    freeze, examples, noise = validate_dataset_root(Path(args.data_root), verify_source_files=True)
    print(
        "formal_prefeval_data_ready:"
        f" sha256={freeze['dataset_sha256']} examples={len(examples)}"
        f" noise_sessions={len(noise)}"
    )


if __name__ == "__main__":
    main()
