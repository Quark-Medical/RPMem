"""CLI validation for a completed teacher top-k store."""

from __future__ import annotations

import argparse
import json

from rpmem.training.teacher_shards import (
    validate_store_complete,
    validate_store_completion_marker,
    write_store_completion_marker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store_dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-marker", action="store_true")
    mode.add_argument("--marker-only", action="store_true")
    args = parser.parse_args()
    if args.write_marker:
        report = write_store_completion_marker(args.store_dir)
    elif args.marker_only:
        report = validate_store_completion_marker(args.store_dir)
    else:
        report = validate_store_complete(args.store_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
