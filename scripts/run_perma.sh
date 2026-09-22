#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Run after compiler training. Data and model locations are user-supplied.
: "${CHECKPOINT:?Set CHECKPOINT to the trained compiler pytorch_model.bin}"
PYTHON_BIN=${PYTHON_BIN:-python}
export PERMA_DATA_ROOT=${PERMA_DATA_ROOT:-$PWD/data/perma}
VARIANT=${VARIANT:-clean_sd}
TEST_USER=${TEST_USER:-334}
FIRST_SESSION_RULE=${FIRST_SESSION_RULE:-direct}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/perma/$VARIANT}
MODEL_ARGV=()
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  MODEL_ARGV+=(--base_model_path "$BASE_MODEL_PATH")
fi
if [[ -n "${CTX_ENCODER_PATH:-}" ]]; then
  MODEL_ARGV+=(--ctx_encoder_path "$CTX_ENCODER_PATH")
fi

PERMA_VARIANTS_TO_PREP="$VARIANT" bash experiments/perma/prepare_formal_perma_data.sh
"$PYTHON_BIN" experiments/perma/precompute_phase2_latents.py \
  --checkpoint "$CHECKPOINT" --variant "$VARIANT" \
  --output_dir "$OUTPUT_ROOT/latents" --no-use_flash_attn "${MODEL_ARGV[@]}"
"$PYTHON_BIN" experiments/perma/run_phase2_fusion.py \
  --checkpoint "$CHECKPOINT" --variant "$VARIANT" \
  --emb_dir "$OUTPUT_ROOT/latents" --method cmp_gate \
  --test_user "$TEST_USER" --output_dir "$OUTPUT_ROOT/fold_$TEST_USER" \
  --first-session-rule "$FIRST_SESSION_RULE" \
  --no-use_flash_attn "${MODEL_ARGV[@]}"
