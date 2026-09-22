# Using RPMem

## Load compiler and gate

`RPMemModel.from_checkpoint` loads trusted local compiler and optional gate
checkpoints, with the model in evaluation mode. Always supply a trained gate to
use the full method. Omitting the gate creates the initialized gate, not a
pretrained consolidation policy.

`base_model_path` and `ctx_encoder_path` override machine-specific paths recorded
in the checkpoint. They must identify compatible reader and encoder weights;
changing a path does not perform decoder-head transfer.

```bash
python examples/infer.py \
  --checkpoint checkpoints/compiler.bin --gate checkpoints/gate.pt \
  --base-model Qwen/Qwen3-8B --ctx-encoder answerdotai/ModernBERT-base \
  --sessions session1.txt session2.txt --query 'What should I recommend?'
```

## Explicit memory state

`encode_session(text)` returns a session embedding. `merge_sessions(embeddings)`
consolidates a nonempty sequence. Alternatively,
`update_memory(text, previous_memory)` processes one session at a time and returns
the updated tensor. The two consolidation paths use the same gate recurrence.

State is owned by the caller, not hidden inside the model. Save it with
`torch.save(memory.cpu(), path)` and reload it with
`torch.load(path, weights_only=True, map_location=model.device)`. Keep it paired
with the compatible compiler/gate configuration that produced it. Raw tensor
loading does not check model identity or automatically migrate across backbones.

`apply_memory(memory)` replaces the currently injected LoRA. `reset()` restores
the original reader forwards. It does not erase tensors held by the caller.
Use separate memory tensors for different users, and reset before base-model
inference. Dynamic injection mutates the reader: a single instance is not safe
for concurrent requests with different memories.

## First-session initialization

New `CMPGate` and `GateConfig` instances default to `first_session_rule="direct"`:
`h1 = q1`, followed by `h_t = gate(h_{t-1}, q_t)` for later sessions. Both the
streaming and batch APIs use this same implementation.

The historical alternative is `first_session_rule="gate_zero_state"`, which
computes `h1 = gate(0, q1)`. Select it when creating a gate:

```python
from rpmem import CMPGate
gate = CMPGate(d_latent=512, first_session_rule="gate_zero_state")
```

Saved gates restore their recorded rule. Historical checkpoints without a rule
are interpreted as `gate_zero_state`, not silently switched to the new default.
`RPMemModel.from_checkpoint(..., first_session_rule=...)` selects the rule for a
fresh gate; when loading a saved gate it instead asserts that the rule matches.
Persist the rule with the weights, using `CMPTrainer.save_checkpoint` or the
benchmark trainers. A bare `state_dict` cannot describe the initialization policy.

The paper's historical results used the zero-state rule. New direct-initialization
training is a distinct configuration, not a rerun of those numbers. To evaluate a
trained direct policy, retrain the downstream gate; the session compiler and its
cached latents can be reused. Existing gate weights and result files are unchanged.

## Context and decoding

Sessions must fit the 4096-token compiler budget under both the reader and
context-encoder tokenizers (including encoder special tokens). Oversized input
raises an error; it is never silently truncated. Benchmark workflows perform
their own deterministic session segmentation before compilation.

`generate` formats one user query with the reader chat template, requests
non-thinking mode, and defaults to greedy decoding. The convenience API supports
one sequence, not beam search or multiple returned sequences. Benchmark MCQ
evaluation uses next-token option logits, not generated prose; use the benchmark
entry points for paper evaluation.

## Gate training

The generic `CMPTrainer` provides a small loss-callback-based training helper.
`examples/core_smoke.py` shows its use. It freezes reader/head parameters and
updates the gate. Benchmark-specific trainers implement the actual splits,
prompts, and task losses from the paper; the CPU example does not replace them.

For direct initialization, a single-session example has no dependence on Gate
parameters. Trainers still compute its loss, but skip backward, weight decay, and
the optimizer/scheduler step. They record `single_session_skips` separately from
actual `optimizer_updates`. Multi-session examples train the recurrent gate as
usual; the legacy mode also trains on the first session.
