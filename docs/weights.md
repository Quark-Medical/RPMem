# Checkpoint layout

The first release does not publish pretrained weights. If checkpoints are
published later, they will be hosted on Hugging Face; no download link is
available yet. This page describes how to organize your own Qwen3-8B checkpoints and the historical
artifact layout, not available downloads or a completed CUDA reproduction.

## Components

| Component | Count | Scope |
| --- | ---: | --- |
| Fixed-FKL session compiler | 1 | Final five-pass Qwen3-8B compiler, step 51885 |
| PERMA consolidation gates | 70 | Seven variants, ten held-out-user folds per variant |
| PersonaMem-v2 consolidation gate | 1 | Training-persona split to held-out test personas |
| PrefEval consolidation gate | 1 | Training topics to held-out test topics |

The compiler contains the Perceiver and LoRA-generating head. It does not include
the frozen Qwen3-8B reader or ModernBERT-base context encoder; obtain those models
separately. The same compiler is paired with the 72 downstream gates. A gate is
benchmark-specific, and PERMA gates are also variant- and fold-specific.

One compatible local layout is:

```text
checkpoints/qwen3_8b/
  compiler.bin
  compiler.bin.json
  gates/
    perma/<variant>/fold_user<id>/gate.pt
    perma/<variant>/fold_user<id>/gate.pt.json
    personamem_v2/gate.pt
    personamem_v2/gate.pt.json
    prefeval/gate.pt
    prefeval/gate.pt.json
```

PERMA variants are `clean_sd`, `noisy_sd`, `style_sd`, `style_long_sd`, `clean_md`,
`noisy_md`, and `style_md`. Held-out users are `334`, `354`, `123`, `1377`, `507`,
`914`, `112`, `419`, `108`, and `109`. The main table's four-variant subset uses
40 of these gates; the additional style evaluations use the other 30.

## Evaluation without retraining

With your own matching weights, compile benchmark session latents with the exact
compiler being evaluated. For PERMA, use `run_phase2_fusion.py --gate_checkpoint`
with the matching `--variant` and `--test_user`. For PersonaMem-v2 and PrefEval,
use `run_eval.py --gate` and omit the Gate-training step. Complete commands are
in the [benchmark guide](benchmarks.md).

Historical gates use `h1 = gate(0, q1)` and retain that behavior when loaded.
New training defaults to `h1 = q1` and records `first_session_rule=direct`.
These are separate Gate configurations; do not relabel historical paper scores
as measurements of direct initialization. See [API behavior](usage.md#first-session-initialization).

If weights are exported, each sidecar connects the sanitized export to its
original file without internal storage URLs. Export is optional and does not
upload anything. See [checkpoint export](checkpoints.md#preparing-public-weights).

Cross-backbone decoder-head transfer has separate compiler/Gate pairs. Code and
configurations for that workflow remain in the repository.
