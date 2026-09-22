# Preparing data

Datasets and teacher targets are not bundled. This guide describes the inputs
and preprocessing code needed for compiler training. For downstream data, use
the [benchmark preparation commands](benchmarks.md).

## Compiler corpus

The source locations, revisions, adapters, and local file patterns are specified
in [formal_sources_v1.yaml](../experiments/phase1/formal_sources_v1.yaml).
Obtain the listed sources and set the local data and tokenizer paths before
running normalization. Dataset paths in the configuration are relative to its
directory.

| Source | Content |
| --- | --- |
| HuggingFaceH4/ultrachat_200k | Conversations |
| DeepPavlov/TaskMaster2 | Task-oriented dialogues |
| google-research-datasets/dstc8-schema-guided-dialogue | Task-oriented dialogues |
| Agent-Ark/Toucan-1.5M, SFT configuration | Tool trajectories |
| nvidia/SWE-Zero-openhands-trajectories | Coding trajectories |
| SakanaAI/self_gen_qa_d2l | Synthetic question-answer data |

Source datasets retain their own licenses. The synthetic source also requires
the Mistral tokenizer specified in the configuration to decode its tokenized
records. The coding repository exclusion list is an input to normalization.

### Build the corpus

Run the following stages from the repository root with `.[train,experiments]`
installed. Each module provides `--help` for arguments.

1. Normalize source records into bounded sessions:

   ```bash
   python -m rpmem.training.corpus.normalize \
     --config experiments/phase1/formal_sources_v1.yaml
   ```

2. Generate session-grounded probes with
   `rpmem.training.corpus.generate_probes`. Supply the normalized shards through
   `--inputs`, an explicit generator model through `--model`, and a separate
   `--output_dir` for each source. Use `--provider local` for local generation.
   Validate the resulting probes with `rpmem.training.corpus.validate_enrichment`.

3. Set enriched input paths in
   [formal_corpus_v1.yaml](../experiments/phase1/formal_corpus_v1.yaml), then build
   the training and validation splits:

   ```bash
   python -m rpmem.training.corpus.build \
     --config experiments/phase1/formal_corpus_v1.yaml \
     --output_dir data/compiler-corpus
   python -m rpmem.training.corpus.validate data/compiler-corpus/manifest.json
   ```

The builder performs deduplication, separates held-out sessions and probes, and
writes split manifests alongside session, probe, and index tables. Keep these
files together. Pass `train.corpus.json` directly to compiler training; it does
not need conversion to JSONL. The configured size and domain checks target the
paper-scale corpus; use a separate configuration for smaller custom datasets.
Probe generation depends on the chosen generator and decoding configuration,
so a new generation run need not reproduce identical probes.

## Teacher targets

Fixed-FKL training needs a teacher cache for the same corpus and reader. Generate
it with `rpmem.training.precompute_teacher`. The [training guide](reproduction.md#session-compiler)
provides commands. Teacher targets and compiler inputs must share sample order.
Held-out validation uses its own cache generated from `validation.corpus.json`.

## Custom inputs

Compiler training also accepts JSONL, JSON, and Parquet records containing a
`context` string and corresponding `prompts` and `responses` lists. Generate
teacher targets from exactly the inputs used for training. These custom inputs
exercise the same training workflow but are not the paper's fixed corpus.

## Benchmark inputs

PERMA, PersonaMem-v2, and PrefEval preparation produces normalized tasks and
`memlora_dataset_freeze.json` files describing their source data and splits.
Keep each dataset directory intact across compilation, gate training, and
evaluation. Use separate output directories for different datasets or settings.
The historical filename and on-disk format identifiers are retained for artifact
compatibility; the Python package and method name are RPMem.
