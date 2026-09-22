"""Stage a portable, unpublished corpus bundle without changing training rows.

Only split-referenced tables are included. Logs, build configs and rejection
files are excluded. Publication still requires source-license/content review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import pyarrow.parquet as pq
import yaml

from rpmem.training.corpus.build import FORMAT_NAME, SPLIT_FORMAT_NAME
from rpmem.training.corpus.shards import sha256_file
from rpmem.training.corpus.validate import validate_corpus

PRIVATE_METADATA = re.compile(r"/workspace/|/Users/|/root/|oss://|gitlab\.[A-Za-z0-9.-]+|LTAI[A-Za-z0-9]{12,}")


def local_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"non-portable corpus path: {relative}")
    return resolved


def viewer_configs(root: Path, splits: dict) -> list[dict]:
    def paths(files):
        ordered = list(dict.fromkeys(files))
        for relative in ordered:
            local_path(root, relative)
        parents = {Path(relative).parent for relative in ordered}
        if len(parents) == 1:
            pattern = (next(iter(parents)) / "*.parquet").as_posix()
            matched = {path.relative_to(root).as_posix() for path in root.glob(pattern)}
            # Compact only when a glob selects exactly the same ordered files.
            if sorted(matched) == ordered:
                return pattern
        return ordered

    configs = []
    for table in ("sessions", "probes"):
        files = list(dict.fromkeys(path for split in splits.values() for path in split[table]))
        configs.append({"config_name": table, "data_files": [{"split": "records", "path": paths(files)}]})
    configs.append({"config_name": "indices", "data_files": [
        {"split": name, "path": paths(split["indices"])} for name, split in splits.items()
    ]})
    return configs


def review_metadata(root: Path, tables: set[str]) -> dict:
    sources = Counter()
    licenses = Counter()
    metadata_hits = Counter()
    provenance = {}
    for relative in sorted(tables):
        with pq.ParquetFile(local_path(root, relative)) as parquet:
            columns = [name for name in ("source", "provenance_json", "metadata_json")
                       if name in parquet.schema_arrow.names]
            if not columns:
                continue
            for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    origin = json.loads(row.get("provenance_json") or "{}")
                    if "source" in row:
                        source = row["source"]
                        sources[source] += 1
                        entry = provenance.setdefault(source, {key: set() for key in
                                                               ("licenses", "source_urls", "revisions")})
                        for key, field in (("licenses", "license"), ("source_urls", "source_url"),
                                           ("revisions", "revision")):
                            entry[key].add(str(origin.get(field) or "unspecified"))
                    if row.get("provenance_json"):
                        licenses[origin.get("license", "unspecified")] += 1
                    for field in ("provenance_json", "metadata_json"):
                        if PRIVATE_METADATA.search(row.get(field) or ""):
                            metadata_hits[field] += 1
    return {
        "source_session_counts": dict(sources), "recorded_license_counts": dict(licenses),
        "metadata_review_hits": dict(metadata_hits),
        "source_provenance": {source: {key: sorted(values) for key, values in fields.items()}
                              for source, fields in provenance.items()},
    }


def render_dataset_card(root: Path, manifest: dict, report: dict) -> str:
    splits = {name: json.loads(local_path(root, path).read_text())
              for name, path in manifest["splits"].items()}
    metadata = {"pretty_name": "RPMem Compiler Corpus", "tags": ["rpmem", "parametric-memory"],
                "configs": viewer_configs(root, splits)}
    lines = ["---", yaml.safe_dump(metadata, sort_keys=False).rstrip(), "---", "",
             "# RPMem Compiler Corpus", "",
             "**Local publication draft. Not uploaded or approved for distribution.**",
             "The Hugging Face namespace has not been selected; no download URL is advertised.", "",
             "Processed sessions and evidence-linked probe/reference pairs for training the",
             "RPMem session compiler. This is not a downstream benchmark or a set of model weights.",
             "The frozen Parquet tables and split manifests are copied byte-for-byte; source",
             "labels and legacy `memlora_*` format identifiers are retained for compatibility.", "",
             "## Contents and splits", "",
             f"Unique sessions: **{manifest['counts']['sessions']:,}**. Total probes: "
             f"**{manifest['counts']['probes']:,}**. Payload: **{report['payload_bytes'] / 1e9:.2f} GB**",
             "(decimal bytes, excluding documentation and review metadata).", "",
             "| Split | Sessions | Probes |", "| --- | ---: | ---: |"]
    for name in ("train", "query_validation", "validation"):
        if name in splits:
            split = splits[name]
            lines.append(f"| `{name}` | {split['session_count']:,} | {split['probe_count']:,} |")
    lines += ["", "`train` selects the probes used for compiler training. `query_validation`,",
              "when present, holds out probes from the training sessions, not new sessions.",
              "`validation` holds out sessions. The original split indices, not Parquet file",
              "names or a new random split, define membership and ordering.", "",
              "The Hub viewer has separate `sessions`, `probes`, and `indices` configurations",
              "because these tables have different schemas. The `records` split in the first two",
              "is a raw-table view, **not** a training split. Use the RPMem loader to select",
              "the proper sessions/probes and avoid mixing held-out data into training.", "",
              "## Load for training", "", "Install the RPMem package with its `train` extras.",
              "Set `CORPUS_ROOT` to this complete directory, not a single Parquet shard.", "",
              "```python", "import os", "from pathlib import Path",
              "from rpmem.training.sample_store import open_sample_store", "",
              'root = Path(os.environ["CORPUS_ROOT"])',
              'store = open_sample_store([root / "train.corpus.json"])', "try:",
              "    sample = store[0]", '    print("sessions:", len(store))',
              '    print("probes in first session:", len(sample["prompts"]))',
              '    context = sample["context"]',
              '    probe_reference_pairs = list(zip(sample["prompts"], sample["responses"]))',
              "finally:", "    store.close()", "```", "",
              "Pass the same `train.corpus.json` to teacher preparation and compiler training.",
              "The planned main-method release provides matching Qwen3-8B teacher targets as",
              "a separate cache; public packaging and publication are still pending.",
              "Generating teacher targets is the rebuild path, not a required step once",
              "the matching prepared cache is available.",
              "This bundle does not contain teacher caches, pretrained compilers, downstream",
              "gates, or benchmark datasets. The release repository documents those stages in",
              "`docs/reproduction.md` and `docs/benchmarks.md`.", "",
              "## Table fields", "",
              "Session rows contain rendered `context`, structured `events_json` and",
              "`memory_atoms_json`, source/domain labels, token counts, and provenance.",
              "`*_json` columns are JSON-encoded strings. `context_tokens` uses the primary",
              "tokenizer; per-tokenizer counts are in `context_token_counts_json`.", "",
              "Probe rows contain `prompt`, `reference`, `probe_type`, `answerable`, and",
              "`evidence_event_ids`, which refer to event IDs within their parent session.",
              "Index rows join a session to its selected ordered `probe_indices`. Indices",
              "refer to global table positions, not row positions within individual shards.", ""]
    for table in ("sessions", "probes", "indices"):
        relative = next(iter(splits.values()))[table][0]
        with pq.ParquetFile(local_path(root, relative)) as parquet:
            lines += [f"### {table}", "", "| Field | Arrow type |", "| --- | --- |"]
            lines += [f"| `{field.name}` | `{field.type}` |" for field in parquet.schema_arrow]
        lines.append("")
    lines += ["## Sources and attribution", "",
              "Counts below are accepted processed sessions, not upstream dataset sizes.",
              "Licenses and processing revisions are historical provenance labels, not a",
              "new license grant. Upstream links identify the source datasets and their authors.", "",
              "| Stored source | Sessions | Upstream | Recorded license label |",
              "| --- | ---: | --- | --- |"]
    for source, count in report["source_session_counts"].items():
        entry = report["source_provenance"][source]
        urls = ", ".join(f"[{url.rstrip('/').rsplit('/', 1)[-1]}]({url})"
                         if url.startswith("https://") else f"`{url}`" for url in entry["source_urls"])
        licenses = ", ".join(f"`{value}`" for value in entry["licenses"])
        lines.append(f"| `{source}` | {count:,} | {urls} | {licenses} |")
    lines += ["", "Normalization, segmentation, probe enrichment, and filtering were applied",
              "before this freeze. This export does not regenerate probes or change examples.",
              "Source recipes are retained in the RPMem repository under",
              "`experiments/phase1/formal_sources_v1.yaml` and `formal_corpus_v1.yaml`.",
              "The enriched revision labels are processing versions, not upstream commit IDs.", "",
              "## Licensing and limitations", "",
              "There is intentionally no blanket license declaration in this draft. RPMem's",
              "Apache-2.0 **code** license does not relicense the source data. Source-specific",
              "terms, attribution, and redistribution decisions must accompany publication."]
    if "controlled-agent-synthetic-v1" in report["source_session_counts"]:
        lines += ["", "Despite its internal name and `project-generated-research-data` label,",
                  "`controlled-agent-synthetic-v1` derives from SakanaAI's `self_gen_qa_d2l`.",
                  "Its label is not evidence of authorship of the upstream data or permission",
                  "to redistribute it. The upstream dataset's terms need confirmation."]
    if "swe-zero-openhands-filtered-enriched-v1" in report["source_session_counts"]:
        lines += ["", "SWE-Zero records also retain `repository` and `repository_license` in",
                  "`provenance_json`; the source-repository notices must not be discarded."]
    lines += ["", "Generated or normalized references are not guaranteed to be factually correct.",
              "Upstream text can contain personal information, unsafe material, or dataset biases.",
              "The metadata scan in `release_review.json` is not a full-text privacy review.",
              "It neither certifies privacy nor establishes downstream benchmark decontamination.", "",
              "## Reproducibility scope", "",
              f"Corpus content digest: `{manifest['content_digest']}`.",
              "`manifest.json` lists the exact payload files; split files control their use.",
              "The portable top-level manifest uses public model identifiers in place of",
              "machine-local paths. Those identifiers do not pin a tokenizer revision.",
              "The original build records use `local-frozen-snapshot`; do not treat the current",
              "upstream model revision as the historical tokenizer by assumption.", "",
              "The package is prepared locally. The passed small-data training check and",
              "saved-weight evaluation are not a completed full-corpus from-scratch reproduction.",
              "A future publication will add its approved dataset identifier, revision, and citation.", ""]
    return "\n".join(lines)


def stage_release(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("release destination must be outside the source corpus")
    if destination.exists():
        raise FileExistsError("use a new release destination")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError("expected a canonical corpus manifest")
    splits = {}
    tables = set()
    for name, relative in manifest["splits"].items():
        path = local_path(source, relative)
        payload = json.loads(path.read_text())
        if payload.get("format") != SPLIT_FORMAT_NAME or payload.get("split") != name:
            raise ValueError("invalid split manifest")
        for key in ("sessions", "probes", "indices"):
            for table in payload[key]:
                local_path(source, table)
                tables.add(table)
        splits[relative] = payload
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    selected = sorted(tables | set(splits))
    for relative in selected:
        path = local_path(source, relative)
        item = artifacts[relative]
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"source artifact mismatch: {relative}")
    destination.mkdir(parents=True)
    for index, relative in enumerate(selected, start=1):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path(source, relative), target)
        if index % 100 == 0:
            print(f"[stage] copied {index}/{len(selected)} artifacts", flush=True)
    public = {key: manifest[key] for key in (
        "format", "version", "seed", "max_context_tokens", "max_context_tokens_observed",
        "content_digest", "counts", "splits",
    ) if key in manifest}
    public["name"] = "rpmem_qwen3_8b_compiler_corpus_v1"
    public["tokenizer"] = "Qwen/Qwen3-8B"
    model_ids = {"qwen3_8b": "Qwen/Qwen3-8B", "modernbert_base": "answerdotai/ModernBERT-base"}
    public["context_tokenizers"] = [
        {"name": item["name"], "path": model_ids[item["name"]], "add_special_tokens": item["add_special_tokens"]}
        for item in manifest.get("context_tokenizers", [])
    ]
    public["artifacts"] = [artifacts[relative] for relative in selected]
    (destination / "manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    print("[stage] validating complete corpus tables and split relationships", flush=True)
    validation = validate_corpus(destination / "manifest.json")
    validation["manifest"] = "manifest.json"
    print("[stage] collecting source licenses and scanning metadata", flush=True)
    metadata_review = review_metadata(destination, tables)
    report = {
        "status": "staged_not_published", "publication_approved": False,
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "content_digest": manifest["content_digest"], "counts": manifest["counts"],
        "splits": {value["split"]: {key: value[key] for key in ("session_count", "probe_count")}
                   for value in splits.values()},
        **metadata_review,
        "content_review": "not_performed; metadata scan is not a privacy or license clearance",
        "training_tables_and_splits": "byte_identical", "validation": validation,
        "payload_bytes": sum(item["size"] for item in public["artifacts"]),
        "excluded": ["build_config.json", "stats.json", "rejections and operational logs"],
    }
    (destination / "release_review.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "README.md").write_text(render_dataset_card(destination, public, report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_release(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
