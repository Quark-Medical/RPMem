#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PERMA_DATA_ROOT=${PERMA_DATA_ROOT:-$REPO_ROOT/data/perma}
PERMA_DOWNLOAD_ROOT=${PERMA_DOWNLOAD_ROOT:-$PERMA_DATA_ROOT}
PERMA_REVISION=${PERMA_REVISION:-440e64e4fb8baec6f7ad10c1de135505f93e7cb1}
PERMA_VARIANTS_TO_PREP=${PERMA_VARIANTS_TO_PREP:-clean_sd}
PERMA_DOWNLOAD_WORKERS=${PERMA_DOWNLOAD_WORKERS:-32}
FORMAL_HF_ENDPOINT=${FORMAL_HF_ENDPOINT:-${HF_ENDPOINT:-https://huggingface.co}}
export HF_ENDPOINT=$FORMAL_HF_ENDPOINT
read -r -a PERMA_VARIANT_ARGV <<<"$PERMA_VARIANTS_TO_PREP"

mkdir -p "$PERMA_DOWNLOAD_ROOT"
"$PYTHON_BIN" - \
  "$PERMA_DOWNLOAD_ROOT" \
  "$PERMA_REVISION" \
  "$PERMA_DOWNLOAD_WORKERS" \
  "${PERMA_VARIANT_ARGV[@]}" <<'PY'
import sys

from huggingface_hub import snapshot_download

suffixes = {
    "clean_sd": "_c",
    "noisy_sd": "_n",
    "style_sd": "_s",
    "style_long_sd": "_s_long",
    "clean_md": "_multi_c",
    "noisy_md": "_multi_n",
    "style_md": "_multi_s",
}
variants = sys.argv[4:]
unknown = sorted(set(variants) - set(suffixes))
if unknown:
    raise SystemExit(f"unknown PERMA variants: {unknown}")
allow_patterns = ["evaluation/*/meta/overall/*.json"]
allow_patterns.extend(
    f"tasks/*/input_data{suffixes[variant]}.json" for variant in variants
)
snapshot_download(
    repo_id="ustclsc/PERMA",
    repo_type="dataset",
    revision=sys.argv[2],
    allow_patterns=allow_patterns,
    local_dir=sys.argv[1],
    max_workers=int(sys.argv[3]),
)
PY

PERMA_DATA_ROOT="$PERMA_DOWNLOAD_ROOT" \
  "$PYTHON_BIN" experiments/perma/validate_variants.py \
  --variants "${PERMA_VARIANT_ARGV[@]}"

"$PYTHON_BIN" - \
  "$PERMA_DOWNLOAD_ROOT/memlora_dataset_freeze.json" \
  "$PERMA_REVISION" \
  "${PERMA_VARIANT_ARGV[@]}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "format": "memlora_perma_dataset_freeze_v1",
            "repository": "ustclsc/PERMA",
            "revision": sys.argv[2],
            "prepared_variants": sys.argv[3:],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

PERMA_DATA_ROOT="$PERMA_DATA_ROOT" \
  "$PYTHON_BIN" experiments/perma/validate_variants.py \
  --variants "${PERMA_VARIANT_ARGV[@]}"

echo "formal PERMA data ready: $PERMA_DATA_ROOT"
