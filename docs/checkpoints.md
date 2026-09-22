# Using checkpoints

Train a session compiler followed by a benchmark-specific consolidation gate
using the [training guide](reproduction.md). Pretrained weights are not bundled.

## Compiler

Compiler training saves `pytorch_model.bin` with tensor weights and a plain
dictionary configuration. The loader also accepts configuration objects from
older checkpoints, mapping their classes to the current RPMem implementation.
Loading uses Python pickle; only load trusted checkpoints. The reader and
context encoder weights are loaded separately.

```python
from rpmem import RPMemModel

model = RPMemModel.from_checkpoint(
    "checkpoints/compiler.bin",
    gate_ckpt_path="checkpoints/gate.pt",
    base_model_path="models/Qwen3-8B",
    ctx_encoder_path="models/ModernBERT-base",
    use_flash_attn=False,
)
```

The model paths can point to a new location, but must identify the same reader
and context encoder used in training. Changing the reader architecture requires
[decoder-head transfer](reproduction.md#decoder-head-transfer).

## Consolidation gate

PERMA training saves `final_gate.pt` with `state_dict` and `args`.
PersonaMem-v2 and PrefEval save `gate.pt` with `gate_state_dict` and benchmark
metadata. Both formats are accepted by the API. Gate files use PyTorch
serialization; load only files from trusted sources.

Use the gate trained with the corresponding compiler and benchmark split.
Matching tensor dimensions alone is not sufficient. Do not substitute a
training-resume file containing optimizer state for the final gate export.

For the full three-benchmark evaluation, arrange checkpoints as follows:

```text
checkpoints/qwen3_8b/
  compiler.bin
  gates/
    perma/<variant>/fold_user<id>/gate.pt
    personamem_v2/gate.pt
    prefeval/gate.pt
```

PERMA requires 70 gates (seven variants, ten held-out users); each other
benchmark requires one. See [saved-weight evaluation](result-reproduction.md).

## First-session initialization

New gates default to `direct` (`h1 = q1`). To train with the paper's evaluated
convention, select `gate_zero_state` (`h1 = gate(0, q1)`). Evaluation restores the
saved rule; gates without rule metadata use `gate_zero_state`.
Changing this rule is a training choice, not a file-format conversion.
See [API initialization](usage.md#first-session-initialization).
