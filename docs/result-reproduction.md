# Reproduce results from saved weights

This workflow tests the inference half of reproduction: saved compiler and
benchmark-specific gates -> fresh session latents -> fresh predictions -> main
table metrics. It neither trains gates nor reuses historical predictions.
It is separate from [training from prepared data](reproduction.md).

**Weights are not publicly released.** These commands work with locally supplied
matching artifacts, including your own trained checkpoints. They generate new
predictions; the bundled reference scores do not replace measured outputs.
See [reproducibility scope](reproduction.md#reproducibility-scope) for the
required inputs.

## Inputs

Install `.[train,experiments]`, prepare the three datasets following
[benchmark workflows](benchmarks.md), and arrange the inputs as follows:

```text
data/
  perma/                           # tasks/, evaluation/, freeze manifest
  personamem_v2/formal_v1/          # Normalized data and freeze manifest
  prefeval/formal_v1/               # Normalized data and freeze manifest
checkpoints/qwen3_8b/
  compiler.bin
  gates/
    perma/<variant>/fold_user<id>/gate.pt
    personamem_v2/gate.pt
    prefeval/gate.pt
```

PERMA has 70 gates (seven variants x ten held-out users); the other benchmarks
have one each. The gate metadata must identify the paired compiler and training
split. See [checkpoint setup](checkpoints.md) for file formats and model paths.
Never pair a gate with another compiler
just because their architectures match.

## Run

Select GPUs reserved for this process; there is no Ray dependency or cluster
reconfiguration. This eight-GPU example runs independent local workers:

```bash
python experiments/reproduce_main_table.py \
  --checkpoint checkpoints/qwen3_8b/compiler.bin \
  --gate-root checkpoints/qwen3_8b/gates --data-root data \
  --base-model models/Qwen3-8B --ctx-encoder models/ModernBERT-base \
  --output-root outputs/main_table_reproduction \
  --gpus 0 1 2 3 4 5 6 7 --first-session-rule gate_zero_state \
  --reference configs/reproduction/qwen3_8b_reference.json
```

The default attention path is SDPA. `--gpus 0` also works, with less parallelism.
`--benchmarks perma` restricts the run to one benchmark. `--plan-only` prints the
planned commands without loading models or starting jobs. The complete
eight-GPU plan has 72 compilation shards and 86 evaluation jobs.

Use a new output directory for each reproduction. After an interruption, rerun
the same command with `--resume` to reuse completed jobs from that directory
only. Keep the shard count, inputs and code unchanged when resuming. A failed
job stops this runner's remaining subprocesses, not unrelated GPU processes.
`--stage compile`, `evaluate`, and `summarize` allow separate stages; subsequent
stages using the same output directory also require `--resume`.

For the paper's historical gates, use `gate_zero_state`, as shown. For newly
trained `h1=q1` gates, use `--first-session-rule direct` and a separate output
directory. Loading the wrong convention fails rather than silently changing the
saved gate. Historical reference numbers do not become direct-policy results.

## Outputs and interpretation

The output includes a job plan, runtime versions, per-job logs and completion
markers, fresh latents and question-level predictions, plus `summary.json` and
`summary.md`. It checks complete, unique question coverage before reporting a
complete result.
PERMA coverage uses `(variant, user_id, task_id, task_type)`, because the same
`task_id` can occur in multiple task types. Those are distinct questions, not
duplicate predictions.

| Benchmark | Aggregation |
| --- | --- |
| PERMA | Ten-user macro mean and population standard deviation for each variant; separate means for the four main-table variants and all seven variants |
| PersonaMem-v2 | Accuracy over 5,000 questions; Self and Current subsets reported separately |
| PrefEval | Accuracy over 1,620 questions; separate 10/70/300-turn accuracies |

The optional reference file contains historical source-result values, not
expected outputs substituted for new predictions. Differences are reported in
percentage points. There is no automatic score-equality pass condition. Never
overwrite measured scores with the reference values.
