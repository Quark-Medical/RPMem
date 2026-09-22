# Prepared reproduction data

The intended main-method training entry point is the processed, frozen
session/probe corpus **and its matching Qwen3-8B teacher top-k cache**, not a
fresh round of probe or teacher generation. A Hugging Face data release is being
prepared; it is **not published yet** and this guide contains no placeholder
download link. The first release does not include trained weights.

## Release scope

| Component | Intended contents | Current preparation status |
| --- | --- | --- |
| Phase 1 corpus | Processed sessions, probes/references, ordered indices, and all three original splits | Privately packaged and load-checked; publication deferred |
| Phase 1 teacher targets | Qwen3-8B top-32 log probabilities and token IDs for training and held-out validation | Both complete stores privately packaged; coverage and sampled tensor loading checked |
| Phase 2 benchmarks | PERMA, PersonaMem-v2 and PrefEval inputs, original training/evaluation splits and protocol metadata | Privately packaged and load-checked, subject to upstream terms before publication |
| Rebuilding code | Source preprocessing, probe generation, teacher preparation and benchmark preparation | Kept in the code release; a full raw-data rebuild has not been revalidated |

The standard reproduction path will download prepared inputs and train the
compiler and gates. Rebuilding probes/teacher targets is an alternative for
provenance or new data/models, not a mandatory cost for reproducing the main
Qwen3-8B method. Other-model teacher caches, historical checkpoints, optimizer
states, latent caches, baseline runs and internal logs are outside this data
release. Publication of the selected inputs is separate from publishing weights.

The combined private package was completed on September 22, 2026. It contains
5,719 files totaling 17,698,172,624 bytes (17.70 GB), excluding the top-level
inventory manifest. This includes corpus, teacher targets, benchmark inputs and
their metadata, not model weights. The package and completion record are stored
privately; no public data download is available yet.

## Training input

The recovered paper corpus contains 664,128 training sessions and 5,313,024
training probes (eight probes per training session). The original train,
query-validation, and held-out-validation splits are preserved. The complete
corpus payload is about 2.9 GB, without teacher logits or model weights.

| Split | Sessions | Probes | Meaning |
| --- | ---: | ---: | --- |
| `train` | 664,128 | 5,313,024 | Eight training probes per session |
| `query_validation` | 664,128 | 1,328,256 | Two other probes for each training session |
| `validation` | 6,902 | 69,020 | Ten probes per held-out session |

There are 671,030 unique sessions, not the sum of all three session counts.
Query validation tests new questions about seen sessions; session validation
tests unseen sessions. Neither validation split is a downstream benchmark.

```text
compiler-corpus/
  manifest.json
  train.corpus.json
  query_validation.corpus.json
  validation.corpus.json
  sessions/                 # Session text and source provenance, Parquet
  probes/                   # Probe text, references and evidence, Parquet
  indices/                  # Ordered session/probe associations, Parquet
```

Use the paths recorded in the manifests, rather than assuming a shard naming
scheme. Set `CORPUS_ROOT` to the downloaded/extracted directory. Both teacher
preparation and compiler training accept `$CORPUS_ROOT/train.corpus.json`
directly. The sample store joins the tables and restores the frozen ordering;
do not concatenate Parquet tables as if each row were a training example.
See [compiler training](reproduction.md#session-compiler).

A small functional subset can be generated without rebuilding the corpus:

```bash
python -m rpmem.training.corpus.export_subset \
  --inputs "$CORPUS_ROOT/train.corpus.json" \
  --output outputs/smoke/train.jsonl --max_samples 32 --selection first
```

Use that same subset for its teacher cache and training. Small-subset training
checks execution only; it does not reproduce the full-data paper scores.

## Main-method teacher cache

The selected historical Qwen3-8B asset contains the following stores. Sizes are
decimal and come from its recorded completed-file inventory, not an estimate of
uncompressed tensors or all historical experiment storage.

| Store | Session records | Parquet parts | Size |
| --- | ---: | ---: | ---: |
| `train` | 664,128 | 1,320 | 13.21 GB |
| `validation` | 6,902 | 24 | 175.21 MB |

Together with asset metadata, the cache is 13,383,156,047 bytes (13.38 GB).
Corpus plus teacher asset totals 16,275,951,763 bytes (16.28 GB), before final
publication documentation and packaging. Benchmark data and base-model downloads
are additional. There is **no `query_validation` teacher store** in this selected
asset; that text split remains available but is not included in these cache counts.

Targets use fixed reference responses and history-conditioned teacher prefixes,
top-k 32, float16 log probabilities and int32 token IDs. The recorded limits are
640 student tokens, 4096 teacher context tokens and 4864 total teacher tokens.
These are the existing targets, not a new teacher inference run. Metadata checks
confirm the corpus content identity, exact train/validation split files, and
sample counts against the staged corpus.

The portable exporter removes machine-local source paths, adapts top-level format
metadata, and copies every Parquet file byte-for-byte. Source bindings use ordered
split-file contents, not absolute paths. Two splits of the same corpus cannot be
substituted just because they share a corpus content digest. Old, unexported stores
retain their original validation behavior.

For maintainers with the original asset directory and prepared corpus:

```bash
python -m rpmem.training.export_teacher \
  --source "$TEACHER_ASSET" --corpus "$CORPUS_ROOT" \
  --output release-data/qwen3-8b-teacher
```

The source directory contains `asset_manifest.json`, `asset_complete.json`, and
`teacher_store/`. Exporting requires the recorded files for each selected split;
it does not download data or run a teacher model. The output contains `train/`,
`validation/`, a public-facing manifest and a data-card draft. `--splits validation`
can export only the held-out store for a bounded loading check; this is not a
complete training cache. A new output directory is required. Interrupted exports
do not leave a final package falsely marked complete.

Both stores have been exported and load-checked on the cluster. Coverage checks
passed for all 664,128 training and 6,902 validation records; three tensors per
split were loaded to check shape, dtype and finite values. No public-download
test or full-corpus retraining occurred. Packaging reused the recorded teacher
probabilities without recomputing them.

## Preparing a publishable corpus draft

For maintainers with the original frozen corpus:

```bash
python -m rpmem.training.corpus.export_release \
  --source "$CORPUS_ROOT" --output release-data/compiler-corpus
```

This command copies only split-referenced training tables and manifests,
validates their relationships, removes operational build files, and replaces
machine-local tokenizer paths with public model identifiers. Tables, split
membership, ordering, and source provenance remain byte-identical. It writes a
`release_review.json` and an explicitly unpublished dataset-card draft. It does
not contact Hugging Face or upload anything.

The generated data card includes real split counts, Arrow schemas, source links,
recorded provenance labels, and an executable loader example. Separate Hub
viewer configurations expose `sessions`, `probes`, and split-specific `indices`.
The first two are raw `records` views, not independent training examples. Publication
can use any approved HF namespace without changing the corpus paths or loaders.
No namespace or URL has been selected yet.

The corpus combines several sources with different licenses. The code's
Apache-2.0 license does not relicense their data. Before publication, review
redistribution terms and attribution source by source, inspect content and
provenance, and document the tokenizer revisions. A metadata-path scan is not
privacy clearance. If any source cannot be redistributed, provide its upstream
retrieval/preprocessing path and explain the remaining reproduction dependency;
do not silently remove it while calling the corpus identical to the paper run.

The [source notes](compiler-data-sources.md) distinguish upstream datasets from
our preprocessing labels. In particular, `controlled-agent-synthetic-v1` derives
from SakanaAI's `self_gen_qa_d2l`; its historical
`project-generated-research-data` label does not establish redistribution rights.

The lower-level corpus builders remain available for provenance and alternative
data. They are not a requirement for users of the intended prepared release.

## Benchmark data

PERMA, PersonaMem-v2, and PrefEval are separate from compiler training data.
Their public preparation scripts and split policies are in
[benchmark workflows](benchmarks.md). Do not train downstream gates on held-out
users, test histories, or test topics. Publishing the compiler corpus does not
imply permission to republish these benchmarks under the code license.

The prepared benchmark packages contain **1.42 GB** of original payload in total:

| Benchmark | Included training/evaluation data | Payload |
| --- | --- | ---: |
| PERMA | Seven variants, ten users, 3,987 questions; all leave-one-user-out inputs | 868.53 MB |
| PersonaMem-v2 | 18,527 train / 2,059 validation / 5,000 test questions and 999 history files; source tables and repair records retained | 543.19 MB |
| PrefEval | 2,460 train / 540 test form rows and 316 noise sessions; 7,380 / 1,620 questions after expansion at three history lengths | 9.49 MB |

These are Gate-training and evaluation inputs, not baseline SFT data, predictions
or compiled latents. PersonaMem-v2 question tables alone are not sufficient: its
historical conversations are required too. The staged packages pass the actual
benchmark loaders with their original data and split identities.

Maintainers can stage each benchmark from an existing prepared data root:

```bash
python experiments/export_benchmark_data.py \
  --benchmark personamem_v2 --source "$PERSONAMEM_DATA_ROOT" \
  --output release-data/benchmarks/personamem_v2/formal_v1
```

Repeat with `perma` or `prefeval` and their corresponding roots. The exporter
selects only required files, copies them unchanged, checks the relocated inputs,
and writes a file inventory, split policy and unpublished data card. All three
packages are privately staged and remain unpublished: this is not an HF
download-path test or approval to redistribute upstream material.
