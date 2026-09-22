"""Normalize and freeze the pinned PersonaMem-v2 32K text release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/personamem_v2/formal_v1")
    args = parser.parse_args()
    freeze = prepare_dataset(Path(args.data_root).expanduser())
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"formal PersonaMem-v2 data ready: {Path(args.data_root).resolve()}")


if __name__ == "__main__":
    main()
