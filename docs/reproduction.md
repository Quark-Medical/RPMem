# Reproducing the main method

This repository provides the method's data, training, and evaluation entry
points. Data will be distributed separately from Git: the planned release
includes the processed corpus, its main-method Qwen3-8B teacher cache, and
benchmark inputs/splits where redistribution is permitted. It is not yet
published; see [data scope and status](prepared-data.md). Trained weights,
company job launchers and historical baseline sweeps are not bundled. See
[benchmark preparation](../experiments/README.md) for public data sources.

## Session compiler

The main reader is Qwen3-8B, with ModernBERT-base as the frozen context encoder.
`configs/compiler/qwen3_8b.yaml` specifies the Fixed-FKL compiler. The internal
objective identifier remains `offline_fkl`; it is not an additional method.

The paper run uses 664128 training sessions, five passes, global batch 64,
and 51885 optimizer steps. With eight GPUs, batch size 1 per GPU and gradient
accumulation 8 give that batch size. Changing GPU count without adjusting
accumulation changes the training schedule.

Start from the [prepared corpus](prepared-data.md), with `CORPUS_ROOT` pointing
to its extracted directory. Its Hugging Face publication is still pending.
The split manifest is a supported input, not a file you need to convert to JSONL.
The intended standard path uses the matching prepared teacher cache, so users
need not regenerate probabilities. The full portable training and validation
caches have been privately packaged and load-checked; public distribution is
still pending. With a completed portable cache, set `TEACHER_ROOT` to its
root (containing `train/`) and run:

```bash
accelerate launch --config_file configs/accelerate_8gpu.yaml -m rpmem.training.train_hypernet \
  --config configs/compiler/qwen3_8b.yaml --train_data "$CORPUS_ROOT/train.corpus.json" \
  --teacher_logprobs_dir "$TEACHER_ROOT/train" --output_dir outputs/compiler \
  --no-use_flash_attn
```

Optional held-out validation additionally supplies
`--val_data "$CORPUS_ROOT/validation.corpus.json"` and
`--val_teacher_logprobs_dir "$TEACHER_ROOT/validation"`. Do not substitute the
query-validation split for that cache. See [cache export](prepared-data.md#main-method-teacher-cache)
for moving recorded historical assets into the portable layout.

The following **rebuild path** instead generates teacher targets and trains with
the same split; it is useful for custom data/models or when the cache is absent:

```bash
python -m rpmem.training.precompute_teacher \
  --base_model_path Qwen/Qwen3-8B --train_data "$CORPUS_ROOT/train.corpus.json" \
  --output_dir outputs/teacher --top_k 32 --max_seq_len 640 \
  --max_teacher_ctx_tokens 4096 --max_teacher_seq_len 4864 --no-use_flash_attn
python -m rpmem.training.validate_teacher_store outputs/teacher --write-marker
accelerate launch --config_file configs/accelerate_8gpu.yaml -m rpmem.training.train_hypernet \
  --config configs/compiler/qwen3_8b.yaml --train_data "$CORPUS_ROOT/train.corpus.json" \
  --teacher_logprobs_dir outputs/teacher --output_dir outputs/compiler \
  --no-use_flash_attn
```

Teacher preparation and training must use the same sample ordering. The step
count above only represents five passes for the stated corpus size. A small
custom dataset is useful for a functional test, but is not that paper run.
Teacher preparation accepts `--num_shards` and `--shard_id` for independent GPU
workers writing the same output directory. Wait for every shard before running
the teacher-store validator or starting training. The command above uses a
single teacher worker for clarity; it is not the fastest full-corpus setup.
`--no-use_flash_attn` provides a portable SDPA path; it is not a throughput claim
for the paper's hardware or attention implementation.

## Consolidation and held-out evaluation

New Gate training defaults to `h1 = q1`. The paper's historical runs instead used
`h1 = gate(0, q1)`: use `FIRST_SESSION_RULE=gate_zero_state` with the PERMA launcher,
or `--first-session-rule gate_zero_state` with each benchmark training CLI, to
reproduce that convention. Saved gates restore their own rule automatically.
Keep outputs from the two configurations separate; no compiler retraining is
needed solely to change the Gate initialization.

PERMA uses leave-one-user-out training. This command covers one variant and one
held-out user, not the complete seven-variant, ten-user matrix:

```bash
CHECKPOINT=outputs/compiler/pytorch_model.bin \
BASE_MODEL_PATH=models/Qwen3-8B CTX_ENCODER_PATH=models/ModernBERT-base \
VARIANT=clean_sd TEST_USER=334 bash scripts/run_perma.sh
```

Repeat for the intended variants and held-out users. PersonaMem-v2 and PrefEval
instead use their respective train/test splits and train a gate on training
histories/topics only. Their preparation, compilation, training, and evaluation
entry points are listed in [experiments/README.md](../experiments/README.md).
See [complete benchmark commands](benchmarks.md) for local model preparation,
the remaining PERMA folds, and the PersonaMem-v2 / PrefEval pipelines.

Use the provided benchmark protocols for next-token option scoring with thinking
disabled. The free-form response example is not the benchmark scoring protocol.

For an existing compiler and the 72 matching downstream gates, use the separate
[saved-weight reproduction workflow](result-reproduction.md). It recompiles all
memory and measures new predictions without retraining. Reference scores are
displayed alongside the new measurements, never substituted for them.

## Decoder-head transfer

`configs/transfer/` contains configurations for Qwen3-4B, Qwen3.5-9B,
Qwen3.5-35B-A3B, and Ministral3-8B. Set the target model/data paths and source
compiler checkpoint explicitly. The trainable scope is `head_only`: the latent
Perceiver is transferred while the target head is trained. Run the same compiler
training CLI with the target config; downstream gates are trained/evaluated with
that target compiler. Inspect `--help` for transfer arguments and local paths.
