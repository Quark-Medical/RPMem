# Checkpoints and the package rename

The distribution, Python package, and public classes are now `rpmem`,
`RPMemModel`, and `RPMemConfig`. New code should not import `memlora`.

Native research checkpoints previously pickled dataclasses under
`memlora.config`. The compiler loader remaps these known classes locally during
unpickling, without installing a legacy package or changing `sys.modules`.
The library and benchmark entry points use this loader.

Only load trusted research checkpoints: legacy loading uses Python pickle.
This compatibility path is not a general converter for Doc-to-LoRA checkpoints.

## Gate exports

`RPMemModel.from_checkpoint(..., gate_ckpt_path=...)` accepts the final Gate
exports from all three included benchmark workflows:

- PERMA: `final_gate.pt`, containing `state_dict` and `args`. Older exports
  omit `d_latent`; the loader recovers it from the Gate weight shape. Original
  single-user exports also omit `eval_users`; export and saved-Gate evaluation
  recover it as `[test_user]`, without changing the recorded held-out user.
- PersonaMem-v2 and PrefEval: `gate.pt`, containing `gate_state_dict`,
  `d_latent`, and benchmark metadata.

This is a format compatibility fix, not a weight conversion or a change to the
Gate recurrence. No retraining is required to load an existing gate under its
recorded rule. New exports record `first_session_rule`; unversioned historical
exports load as `gate_zero_state`. Conflicting rule metadata is rejected. The
new-training default is `direct`, so deliberately changing an old gate's policy
requires separate downstream training/evaluation, not just format conversion.
Use the Gate for the corresponding
compiler and benchmark fold; matching latent dimensions alone does not establish
that two checkpoints belong together. Training resume files also contain optimizer
state and should not be used in place of the final evaluation export.

## Relocating models

Old compiler checkpoints may contain model paths from the training machine. Pass
`base_model_path` and `ctx_encoder_path` to the Python API to point to the same
reader and encoder weights on a new machine. Both overrides are also available
in benchmark CLIs: PERMA uses `--base_model_path` / `--ctx_encoder_path`, while
PersonaMem-v2 and PrefEval use `--base-model-path` / `--ctx-encoder-path`.
These arguments relocate the weights; they do not make a compiler compatible
with a different backbone. Use decoder-head transfer for that experiment.

## Compiler conversion

New compiler checkpoints store configuration as plain dictionaries alongside
tensor weights. They can be inspected with `torch.load(..., weights_only=True)`
without the historical package. Convert a trusted old compiler once with:

```bash
python -m rpmem.checkpoint.convert checkpoints/old.bin checkpoints/compiler.bin
```

Conversion refuses to overwrite an existing destination. It preserves weight
values and metadata; only the configuration representation changes. File hashes
will change, so regenerate any derived cache/manifest that uses a file hash as
its identity. Do not alter old experiment records to pretend they used the
converted file.

The `memlora_*` format identifiers and freeze filenames in benchmark data are
intentionally retained for compatibility with existing dataset/latent artifacts.
They describe on-disk schemas, not the public method name. The old
`memlora_cmp_gate` evaluation argument is accepted as an alias; new commands use
`--method rpmem`. Old and new method labels are not interchangeable for skipping
previously completed result rows.

`RPMEM_PERCEIVER_FLASH_ATTN` and `RPMEM_TENSORBOARD_DIR` are the public environment
variables. Their historical `MEMLORA_*` counterparts are fallback aliases.

## Preparing public weights

Publishing weights is optional and is not a prerequisite for this code release.
The following offline tool is available only if a weight release is later approved.

The conversion command above preserves metadata and is not a privacy filter.
Maintainers should use the separate export command for trusted research artifacts:

```bash
python -m rpmem.checkpoint.export compiler private/compiler.bin release/compiler.bin \
  --base-model Qwen/Qwen3-8B --ctx-encoder answerdotai/ModernBERT-base
python -m rpmem.checkpoint.export gate private/gate.pt release/gate.pt \
  --compiler-manifest release/compiler.bin.json
```

Export retains compiler/Gate tensors exactly, stores configuration as plain data,
and removes internal paths, training logs, optimizer state, and unrelated weights.
It checks the compiler tensor layout and refuses to overwrite existing outputs.
Each output has a JSON sidecar identifying both the original and exported file.
Gate export requires its original compiler identity to match the compiler sidecar;
it then binds the exported Gate to the exported compiler's new file identity.
The recorded initialization rule is preserved, including the legacy fallback.
PERMA control/ablation gates and training-resume files are not main-method exports.

No model download, training, upload, or remote modification is performed. Rebuild
latent caches using the exported compiler and use a new result directory, because
serialization changes file hashes even though tensors are unchanged. Keep the
original experiment artifacts intact. The export is for inference, not training
resume. See [weight layout](weights.md) for which Gate belongs to each benchmark.
