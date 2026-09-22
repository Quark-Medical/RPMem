# Compiler corpus source notes

The processed corpus is privately packaged and unpublished. These notes describe the
frozen inputs and attribution work; they do not relicense upstream data.
No Hugging Face organization or dataset ID has been selected.

## Composition

Counts come from the recovered corpus, including held-out sessions. Source
names are preserved internal identifiers. Session counts can differ from raw
upstream counts because normalization, segmentation, enrichment, and filtering
preceded the final freeze.

| Stored source | Sessions | Upstream | License label in frozen provenance |
| --- | ---: | --- | --- |
| `ultrachat-200k-enriched-v1` | 188,152 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT |
| `taskmaster2-enriched-v1` | 17,294 | [DeepPavlov/TaskMaster2](https://huggingface.co/datasets/DeepPavlov/TaskMaster2) | CC-BY-4.0 |
| `schema-guided-dialog-enriched-v1` | 16,811 | [Google Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | CC-BY-SA-4.0 |
| `toucan-sft-enriched-v1` | 94,996 | [Agent-Ark/Toucan-1.5M, SFT subset](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Apache-2.0 |
| `swe-zero-openhands-filtered-enriched-v1` | 31,667 | [NVIDIA/SWE-Zero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) | CC-BY-4.0-plus-source-repository-licenses |
| `controlled-agent-synthetic-v1` | 322,110 | [SakanaAI/self_gen_qa_d2l](https://huggingface.co/datasets/SakanaAI/self_gen_qa_d2l) | project-generated-research-data |

The accepted contexts contain 525,898,190 Qwen3-8B tokens before training-pass
repetition. Observed context-token shares are conversation 36.60%, tool 25.90%,
coding 16.58%, and synthetic 20.92%. These are measured shares, not the source
selection targets or percentages of sessions.

## Recorded revisions and transformations

The source-normalization recipe records the following upstream revisions.
They are distinct from `formal-enriched-v3` and `formal-filtered-enriched-v3`,
which are processing labels in the final corpus, not upstream commits.

| Raw source | Revision recorded by the recipe |
| --- | --- |
| UltraChat | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| TaskMaster2 | `333d425dd1f53e9fe8f0cf6d8d08033fd6a811f9` |
| Schema-Guided Dialogue | `e852981ae34990f4358979625854259302feaa78` |
| Toucan | `0df3cf37f2abefb380370cfb02eabea2a35ae782` |
| SWE-Zero | `7b3cd106d00f60918e722d33a1d74bc67072a7ea` |
| SakanaAI synthetic QA | `SakanaAI-self_gen_qa_d2l-main-required-shards` (a label, not an immutable commit) |

The recipes are [source normalization](../experiments/phase1/formal_sources_v1.yaml)
and [final corpus build](../experiments/phase1/formal_corpus_v1.yaml).
Tool/coding normalization drops source system messages and assistant reasoning,
excludes the `think` tool, and caps tool-result references at 512 characters.
Coding normalization also applies the recorded repository exclusion list.
The final build caps contexts at 4,096 tokens under both context tokenizers,
requires references, deduplicates, and freezes the session/probe split indices.
The prepared release preserves that output; it does not rerun these steps.

The synthetic subset is **not wholly new author-generated source data**. The
`d2l_tokenized` adapter decodes the upstream Mistral-7B-Instruct-v0.2-tokenized
context and QA spans before enrichment. Its upstream URL is still present in
the frozen provenance. Keep that attribution rather than interpreting the
internal source name or license label as an ownership statement.

## Publication decisions still needed

The SakanaAI dataset page has no dataset card or license statement visible as
of 2026-09-22. Confirm its redistribution terms before publishing that portion;
public accessibility alone is not the approval recorded for this release.
Do not replace it with a newly generated source or silently omit it while
describing the release as the identical training corpus.

SWE-Zero's [upstream terms](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories#licenseterms-of-use)
state CC-BY-4.0 and additional source-repository licenses. Frozen records retain
the repository and its license. Source-specific notices and attribution,
including the recorded share-alike terms for Schema-Guided Dialogue, need to
accompany the approved data release; a blanket Apache-2.0 data license would
not describe these inputs correctly.

The original tokenizer revisions are recorded as `local-frozen-snapshot`.
Replacing machine-local paths with model IDs is a portability change, not an
assertion that today's upstream tokenizer is identical. The final release
should document the selected tokenizer artifacts without rewriting the data.

The exporter checks metadata, not all free-form content. Its report preserves
30 metadata matches for review rather than silently removing examples. No
full-text privacy clearance or comprehensive contamination claim is implied.
These remaining publication decisions do not require another GPU validation.
