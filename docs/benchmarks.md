# Main-method benchmark workflows

Run these commands from the repository root after installing
`.[train,experiments]`. They use local files and one CUDA GPU by default.
For one orchestration entry point covering
all three benchmarks with saved gates, see [result reproduction](result-reproduction.md).

Prepare data with the commands below. If you have already prepared a benchmark,
point its data-root setting to that directory and skip the download step.

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

Use a compiler from the Fixed-FKL training workflow in this repository.
See [checkpoint setup](checkpoints.md) for the required format.

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
python experiments/perma/precompute_phase2_latents.py \
  --checkpoint "$CHECKPOINT" \
  --variant clean_sd --output_dir "$OUT/latents" \
  --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
  --no-use_flash_attn
python experiments/perma/run_phase2_fusion.py \
  --checkpoint "$CHECKPOINT" \
  --gate_checkpoint "$GATE" --variant clean_sd --test_user 334 \
  --method cmp_gate --emb_dir "$OUT/latents" --output_dir "$OUT/evaluation" \
  --base_model_path "$BASE_MODEL_PATH" --ctx_encoder_path "$CTX_ENCODER_PATH" \
  --no-use_flash_attn --skip_completed
```

Prepare `clean_sd` with the earlier data command first. The evaluation summary
records `run_mode=evaluate_only`, zero training updates,
and evaluation time. Use the Gate trained for this compiler, variant and user.
This option accepts one fold per
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

python "experiments/$BENCHMARK/precompute_phase2_latents.py" \
  --checkpoint "$CHECKPOINT" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" --output-dir "$OUT/latents" --no-use-flash-attn

python "experiments/$BENCHMARK/train_cmp_gate.py" \
  --checkpoint "$CHECKPOINT" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" --latent-root "$OUT/latents" \
  --output-dir "$OUT/gate" --no-use-flash-attn

python "experiments/$BENCHMARK/run_eval.py" \
  --method rpmem --checkpoint "$CHECKPOINT" \
  --base-model-path "$BASE_MODEL_PATH" --ctx-encoder-path "$CTX_ENCODER_PATH" \
  --data-root "$DATA" "$LATENT_FLAG" "$OUT/latents" --gate "$OUT/gate/gate.pt" \
  --output "$OUT/results.jsonl" --skip-completed --no-use-flash-attn
```

No checkpoint hash or completion certificate is required. Use the same compiler
and dataset throughout the workflow. Compilation recomputes latents by default;
pass `--resume` only to continue an interrupted compilation with unchanged inputs.
Use a new output directory after changing a model, dataset, or segmentation setting.

To evaluate a prepared PersonaMem-v2 or PrefEval Gate, run the compilation and
evaluation commands above but skip `train_cmp_gate.py`. In the evaluation command,
replace `--gate "$OUT/gate/gate.pt"` with
`--gate "$PWD/checkpoints/qwen3_8b/gates/$BENCHMARK/gate.pt"`, and set `CHECKPOINT`
to its paired compiler. Use a new `OUT` directory
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
settings record the rule. Resume rejects a different rule. Reuse results with
`--skip-completed` only when inputs and settings are unchanged.
Use separate output directories for comparisons. Single-session examples bypass
Gate optimization under `direct`, with separate skip/update counts.

No historical scores have been reassigned to the new default. A new trained
direct policy requires downstream Gate training and evaluation, not compiler
retraining. See [initialization compatibility](usage.md#first-session-initialization).
