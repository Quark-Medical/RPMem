# Main-method benchmark workflows

Run these commands from the repository root after installing
`.[train,experiments]`. They use local files and one CUDA GPU by default. They do
not need Ray, OSS, the company platform, or the old research repository. The
commands below are execution instructions. For one orchestration entry point covering
all three benchmarks with saved gates, see [result reproduction](result-reproduction.md).

The [prepared-data release](prepared-data.md#benchmark-data) is being staged
separately from Git. When using a complete prepared package, skip the source
download/preparation commands below and point the benchmark's data-root setting
to that package. Compilation, Gate training and evaluation stay the same. No
public data-download URL has been published yet.

## Local reader and encoder

Obtain Qwen3-8B and ModernBERT-base from their respective official repositories
(`Qwen/Qwen3-8B` and `answerdotai/ModernBERT-base`). Keep their weights, tokenizer,
and configuration together. Point to the local snapshots here; do not replace
the compiler's reader with a different architecture just by changing its path.

```bash
export BASE_MODEL_PATH="$PWD/models/Qwen3-8B"
export CTX_ENCODER_PATH="$PWD/models/ModernBERT-base"
export CHECKPOINT="$PWD/outputs/compiler/pytorch_model.bin"
```

Use a compiler from the Fixed-FKL training workflow. The existing checkpoint
file can be loaded directly even if it predates the `rpmem` rename. Do not modify
its contents or its old manifests just to move it to a new directory.

The examples use SDPA (`--no-use-flash-attn`, with underscores in PERMA flags)
so FlashAttention is not required. Use the same choice across stages when
validating results; this guide does not promise numerical parity across attention
backends, dependency versions, or hardware.

## PERMA

Run one held-out fold with:

```bash
VARIANT=clean_sd TEST_USER=334 OUTPUT_ROOT=outputs/perma/clean_sd \
  bash scripts/run_perma.sh
```

The script downloads the selected variant, compiles session latents, trains the
Gate on the other nine users, and evaluates the held-out user. Outputs include
`fold_334/final_gate.pt`, `fold_334/summary.json`, and `fold_334/results.json`.

For the complete matrix, compile each variant only once, then run ten independent
folds while keeping the reader loaded. This is sequential, not a distributed
launcher:

```bash
export PERMA_DATA_ROOT="$PWD/data/perma"
variants=(clean_sd noisy_sd style_sd style_long_sd clean_md noisy_md style_md)
users=(334 354 123 1377 507 914 112 419 108 109)
PERMA_VARIANTS_TO_PREP="${variants[*]}" bash experiments/perma/prepare_formal_perma_data.sh
for variant in "${variants[@]}"; do
  out="outputs/perma/$variant"
  python experiments/perma/precompute_phase2_latents.py \
    --checkpoint "$CHECKPOINT" --variant "$variant" --output_dir "$out/latents" \
    --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
    --no-use_flash_attn
  python experiments/perma/run_phase2_fusion.py \
    --checkpoint "$CHECKPOINT" --variant "$variant" --emb_dir "$out/latents" \
    --method cmp_gate --fold_test_users "${users[@]}" --output_dir "$out/folds" \
    --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
    --no-use_flash_attn --skip_completed
done
```

Run the array/loop block in Bash. Each fold has its own trained Gate. Do not reuse
one user's Gate for the other nine folds. Use the saved per-fold summaries for
the user-macro mean and standard deviation, rather than pooling all questions.

### Evaluate a saved PERMA Gate

This path does not train a new Gate. Choose the Gate for the exact variant and
held-out user, and use a fresh latent/output directory for exported checkpoints:

```bash
export CHECKPOINT="$PWD/checkpoints/qwen3_8b/compiler.bin"
GATE="$PWD/checkpoints/qwen3_8b/gates/perma/clean_sd/fold_user334/gate.pt"
OUT="$PWD/outputs/perma/pretrained_clean_sd_user334"
COMPILER_SHA=$(python - "$CHECKPOINT" <<'PY'
import hashlib
import sys
with open(sys.argv[1], 'rb') as handle:
    h = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)
python experiments/perma/precompute_phase2_latents.py \
  --checkpoint "$CHECKPOINT" --checkpoint_sha256 "$COMPILER_SHA" \
  --variant clean_sd --output_dir "$OUT/latents" \
  --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
  --no-use_flash_attn
python experiments/perma/run_phase2_fusion.py \
  --checkpoint "$CHECKPOINT" --checkpoint_sha256 "$COMPILER_SHA" \
  --gate_checkpoint "$GATE" --variant clean_sd --test_user 334 \
  --method cmp_gate --emb_dir "$OUT/latents" --output_dir "$OUT/evaluation" \
  --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
  --no-use_flash_attn --skip_completed
```

Prepare `clean_sd` with the earlier data command first. The evaluation summary
records `run_mode=evaluate_only`, the Gate file identity, zero training updates,
and evaluation time. Loading another user's Gate, another variant's Gate, or a
Gate for a different compiler is an error. This option accepts one fold per
invocation; it does not overwrite the saved Gate.

## PersonaMem-v2 and PrefEval

Run the following workflow once with `BENCHMARK=personamem_v2` and once with
`BENCHMARK=prefeval`. Both train one Gate on the dataset's training split and
evaluate its held-out test split, not PERMA-style leave-one-user-out folds.
Their splits, segmentation, optimization defaults, and option-scoring protocols
remain benchmark-specific in the existing entry points.

```bash
BENCHMARK=personamem_v2
DATA="$PWD/data/$BENCHMARK/formal_v1"
OUT="$PWD/outputs/$BENCHMARK/rpmem"
if [ "$BENCHMARK" = personamem_v2 ]; then
  LATENT_FLAG=--canonical-latent-root
  PERSONAMEM_DATA_ROOT="$DATA" bash experiments/personamem_v2/prepare_formal_data.sh
elif [ "$BENCHMARK" = prefeval ]; then
  LATENT_FLAG=--session-latent-root
  PREFEVAL_DATA_ROOT="$DATA" bash experiments/prefeval/prepare_formal_data.sh
else
  echo "Choose personamem_v2 or prefeval"
fi

COMPILER_SHA=$(python - "$CHECKPOINT" <<'PY'
import hashlib
import sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)

python "experiments/$BENCHMARK/precompute_phase2_latents.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$COMPILER_SHA" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" --output-dir "$OUT/latents" --no-use-flash-attn

python "experiments/$BENCHMARK/train_cmp_gate.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$COMPILER_SHA" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" --latent-root "$OUT/latents" \
  --output-dir "$OUT/gate" --resume --no-use-flash-attn

python "experiments/$BENCHMARK/run_eval.py" \
  --method rpmem --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$COMPILER_SHA" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" "$LATENT_FLAG" "$OUT/latents" --gate "$OUT/gate/gate.pt" \
  --output "$OUT/results.jsonl" --skip-completed --no-use-flash-attn
```

The digest here connects the local compiler, latent cache, and trained Gate; it is
not a remote code-version check. Do not mix caches generated by different
compilers. Dataset construction keeps the existing frozen split policies.

To evaluate a prepared PersonaMem-v2 or PrefEval Gate, run the compilation and
evaluation commands above but skip `train_cmp_gate.py`. In the evaluation command,
replace `--gate "$OUT/gate/gate.pt"` with
`--gate "$PWD/checkpoints/qwen3_8b/gates/$BENCHMARK/gate.pt"`, and set `CHECKPOINT`
to its paired compiler before computing `COMPILER_SHA`. Use a new `OUT` directory
instead of overwriting historical experiment results.

`results.jsonl` contains question-level predictions. `results.jsonl.meta.json`
contains the question count, accuracy, evaluation protocol, and timing metrics.
PersonaMem-v2 additionally writes `results.jsonl.history.jsonl` with per-history
performance measurements. Keeping one evaluation shard makes the metadata's
accuracy the complete test result, not a partial shard score.

For multiple GPUs, the compilation and evaluation commands accept `--num-shards`
and `--shard-id`; launch distinct shard IDs on distinct GPUs and use distinct
evaluation output files. Run Gate training once only after every required latent
shard finishes. Do not average incomplete shards as a complete benchmark result.

## First-session convention

The commands above train new gates with `h1 = q1` (`direct`). For the historical
paper convention, pass `--first-session-rule gate_zero_state` to
`run_phase2_fusion.py` or `train_cmp_gate.py`. The PERMA shell launcher exposes
the same choice as `FIRST_SESSION_RULE=gate_zero_state`.

Evaluation-only commands restore the saved gate's rule; old checkpoints without
this field use `gate_zero_state`. New checkpoints, result metadata, and resume
contracts record the rule. Resume rejects a different rule, and evaluation
cache reuse is tied to the saved Gate identity (and PERMA's explicit rule).
Use separate output directories for comparisons. Single-session examples bypass
Gate optimization under `direct`, with separate skip/update counts.

No historical scores have been reassigned to the new default. A new trained
direct policy requires downstream Gate training and evaluation, not compiler
retraining. See [initialization compatibility](usage.md#first-session-initialization).
