#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PERSONAMEM_DATA_ROOT=${PERSONAMEM_DATA_ROOT:-$REPO_ROOT/data/personamem_v2/formal_v1}
PERSONAMEM_DOWNLOAD_WORKERS=${PERSONAMEM_DOWNLOAD_WORKERS:-16}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HOME=${HF_HOME:-/tmp/memlora_huggingface}

"$PYTHON_BIN" experiments/personamem_v2/download_source.py \
  --output-root "$PERSONAMEM_DATA_ROOT" \
  --workers "$PERSONAMEM_DOWNLOAD_WORKERS"
"$PYTHON_BIN" experiments/personamem_v2/prepare_formal_data.py \
  --data-root "$PERSONAMEM_DATA_ROOT"
"$PYTHON_BIN" experiments/personamem_v2/validate_formal_data.py \
  --data-root "$PERSONAMEM_DATA_ROOT"
