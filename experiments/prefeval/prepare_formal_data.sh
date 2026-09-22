#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PREFEVAL_SOURCE_REVISION=50795054b5ff5f418d2b768a331d71e480f93331
PREFEVAL_SOURCE_ROOT=${PREFEVAL_SOURCE_ROOT:-$REPO_ROOT/data/prefeval/source/$PREFEVAL_SOURCE_REVISION/benchmark_dataset}
PREFEVAL_DATA_ROOT=${PREFEVAL_DATA_ROOT:-$REPO_ROOT/data/prefeval/formal_v1}

if [[ ! -s "$PREFEVAL_SOURCE_ROOT/filtered_inter_turns.json" ]]; then
  temporary=$(mktemp -d "${TMPDIR:-/tmp}/prefeval-source.XXXXXX")
  trap 'rm -rf "$temporary"' EXIT
  archive="$temporary/prefeval.tar.gz"
  curl --fail --location --retry 5 --retry-all-errors \
    "https://github.com/amazon-science/PrefEval/archive/$PREFEVAL_SOURCE_REVISION.tar.gz" \
    -o "$archive"
  tar -xzf "$archive" -C "$temporary"
  extracted=$(find "$temporary" -mindepth 1 -maxdepth 1 -type d -name 'PrefEval-*' | head -1)
  test -n "$extracted"
  mkdir -p "$PREFEVAL_SOURCE_ROOT"
  cp -R "$extracted/benchmark_dataset/." "$PREFEVAL_SOURCE_ROOT/"
fi

"$PYTHON_BIN" experiments/prefeval/prepare_formal_data.py \
  --source-root "$PREFEVAL_SOURCE_ROOT" \
  --output-root "$PREFEVAL_DATA_ROOT"

"$PYTHON_BIN" experiments/prefeval/validate_formal_data.py \
  --data-root "$PREFEVAL_DATA_ROOT"
