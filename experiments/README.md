# RPMem experiment entry points

Run from the repository root after `pip install -e '.[train,experiments]'`.
Training and pretrained evaluation require CUDA. Supply data and checkpoints
locally; no datasets, weights, or recorded results are bundled.
See [complete commands](../docs/benchmarks.md) to run data preparation,
compilation, Gate training, and evaluation using local paths.

| Benchmark | Data preparation | Session compilation | Gate training | RPMem evaluation |
| --- | --- | --- | --- | --- |
| PERMA | `perma/prepare_formal_perma_data.sh` | `perma/precompute_phase2_latents.py` | `perma/run_phase2_fusion.py` | Same script, `--method cmp_gate` |
| PersonaMem-v2 | `personamem_v2/prepare_formal_data.sh` | `personamem_v2/precompute_phase2_latents.py` | `personamem_v2/train_cmp_gate.py` | `personamem_v2/run_eval.py --method rpmem` |
| PrefEval | `prefeval/prepare_formal_data.sh` | `prefeval/precompute_phase2_latents.py` | `prefeval/train_cmp_gate.py` | `prefeval/run_eval.py --method rpmem` |

Prefix paths with `experiments/`. Run each Python entry with `--help` for its
required paths and arguments. Use the same compiler, dataset, latent
caches, and train/test splits across stages. PERMA uses held-out
users; PersonaMem-v2 and PrefEval use their respective predefined splits.
`scripts/run_perma.sh` chains compilation, Gate training, and held-out evaluation
for one PERMA fold using a locally trained compiler.

## Compiler training

The public source definitions are in `phase1/formal_sources_v1.yaml`; the final
corpus configuration is `phase1/formal_corpus_v1.yaml`. Set local model/data
paths, then use the following modules in order (each accepts `--help`):

1. `rpmem.training.corpus.normalize`
2. `rpmem.training.corpus.generate_probes`
3. `rpmem.training.corpus.validate_enrichment`
4. `rpmem.training.corpus.build`
5. `rpmem.training.corpus.validate`

The [reproduction guide](../docs/reproduction.md) gives teacher preparation and Fixed-FKL training commands.
Compiler and decoder-transfer configurations are in `configs/`.
Optional API-based probe generation requires caller-supplied endpoints and credentials.
Training and evaluation write local artifacts.
