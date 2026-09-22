"""Export a recorded teacher asset without recomputing or rewriting its tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .teacher_shards import (
    FORMAT_NAME, FORMAT_VERSION, SOURCE_BINDING_FORMAT, _sha256_file,
    source_file_bindings, validate_completion_manifests, validate_store_source_files,
    write_store_completion_marker, write_store_metadata,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def local_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"non-portable asset path: {relative}")
    path.resolve().relative_to(root.resolve())
    return path


def stage_teacher_release(source: Path, corpus: Path, output: Path, *, splits: list[str] | None = None) -> dict:
    """Copy selected stores with portable metadata; leave all source files intact."""

    source, corpus, output = Path(source), Path(corpus), Path(output)
    if output.exists():
        raise FileExistsError(f"release destination already exists: {output}")
    manifest = read_json(source / "asset_manifest.json")
    complete = read_json(source / "asset_complete.json")
    if manifest["asset_id"] != complete["asset_id"]:
        raise ValueError("teacher asset identities differ")
    entries = complete["inventory"]["files"]
    inventory = {entry["path"]: entry for entry in entries}
    if len(inventory) != len(entries):
        raise ValueError("duplicate teacher inventory paths")
    for relative in inventory:
        local_file(source, relative)

    def check_file(relative: str, path: Path | None = None) -> dict:
        entry = inventory[relative]
        path = path if path is not None else local_file(source, relative)
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"teacher source artifact mismatch: {relative}")
        return entry

    check_file("asset_manifest.json")
    stable = manifest["stable_inputs"]
    corpus_manifest = read_json(corpus / "manifest.json")
    if corpus_manifest["content_digest"] != stable["corpus"]["content_digest"]:
        raise ValueError("teacher and corpus content identities differ")
    store_root = manifest["storage_layout"]["teacher_store_relative_root"]
    available_splits = manifest["storage_layout"]["splits"]
    splits = available_splits if splits is None else splits
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(available_splits):
        raise ValueError("teacher asset requires unique nonempty splits")
    protocol = stable["teacher_protocol"]
    model = stable["target_model"]
    reports = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only after both stores and their source bindings pass.
    with tempfile.TemporaryDirectory(prefix=".teacher-release-", dir=output.parent) as work:
        draft = Path(work) / "payload"
        draft.mkdir()
        for split in splits:
            if split not in {"train", "validation", "query_validation"}:
                raise ValueError(f"unsupported teacher split: {split}")
            prefix = f"{store_root}/{split}/"
            original_root = local_file(source, prefix)
            check_file(prefix + "meta.json")
            meta = read_json(original_root / "meta.json")
            if meta.get("format") not in {FORMAT_NAME, "memlora.teacher_topk.parquet"}:
                raise ValueError("unsupported source teacher format")
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported source teacher version")
            split_path = local_file(corpus, corpus_manifest["splits"][split])
            split_info = stable["corpus"]["splits"][split]
            binding = source_file_bindings([split_path])[0]
            if binding != {"sha256": split_info["file_sha256"],
                           "size": split_info["file_size"],
                           "content_digest": split_info["content_digest"]}:
                raise ValueError(f"teacher split identity differs: {split}")
            if meta["n_samples"] != split_info["n_samples"] or meta["n_samples"] != read_json(split_path)["session_count"]:
                raise ValueError(f"teacher sample count differs: {split}")
            if meta.get("train_content_digests") != [binding["content_digest"]] or meta["train_file_sizes"] != [binding["size"]]:
                raise ValueError(f"teacher source binding differs: {split}")
            for field in ("top_k", "max_seq_len", "max_teacher_ctx_tokens", "max_teacher_seq_len", "ctx_prompt_sep"):
                if meta[field] != protocol[field]:
                    raise ValueError(f"teacher protocol differs: {split}/{field}")
            validate_completion_manifests(original_root, meta)
            portable = {key: meta[key] for key in (
                "format_version", "top_k", "max_seq_len", "max_teacher_ctx_tokens",
                "max_teacher_seq_len", "ctx_prompt_sep", "n_samples", "num_shards",
                "records_per_file", "rows_per_group")}
            portable.update(format=FORMAT_NAME, base_model_path=model["repository_id"],
                            source_binding_format=SOURCE_BINDING_FORMAT,
                            source_bindings=[binding])
            target = draft / split
            write_store_metadata(target, portable)
            for shard_id in range(meta["num_shards"]):
                name = f"done-{shard_id:05d}.json"
                check_file(prefix + name)
                done = read_json(original_root / name)
                write_json(target / name, {key: done[key] for key in ("shard_id", "completed_samples")})
            parts = sorted(original_root.glob("part-*.parquet"))
            listed_parts = {Path(relative).name for relative in inventory
                            if relative.startswith(prefix) and relative.endswith(".parquet")}
            if {path.name for path in parts} != listed_parts:
                raise ValueError(f"teacher part files differ from inventory: {split}")
            part_inventory = []
            for count, path in enumerate(parts, 1):
                shutil.copyfile(path, target / path.name)
                entry = check_file(prefix + path.name, target / path.name)
                part_inventory.append({"path": f"{split}/{path.name}",
                                       "size": entry["size"], "sha256": entry["sha256"]})
                if count % 100 == 0:
                    print(f"[teacher release] {split}: {count}/{len(parts)} parts", flush=True)
            validate_store_source_files(target, [split_path])
            completion = write_store_completion_marker(target)
            reports[split] = {**completion["report"], "source_bindings": [binding],
                             "parquet_bytes": sum(entry["size"] for entry in part_inventory)}
            write_json(target / "tensor_inventory.json", {"files": part_inventory})
        report = {
            "format": "rpmem.teacher_release.v1", "status": "staged_not_published",
            "publication_approved": False, "teacher_tensors": "byte_identical",
            "source_asset_id": manifest["asset_id"], "corpus_content_digest": corpus_manifest["content_digest"],
            "model": {key: model[key] for key in ("repository_id", "revision")},
            "model_snapshot": model["snapshot"], "teacher_protocol": protocol,
            "available_source_splits": available_splits,
            "stores": reports,
        }
        write_json(draft / "manifest.json", report)
        (draft / "README.md").write_text(render_card(report))
        draft.rename(output)
    return report


def render_card(report: dict) -> str:
    lines = ["# RPMem Qwen3-8B Teacher Targets", "",
             "**Local publication draft. Not uploaded or approved for distribution.**", "",
             "Fixed-reference, history-conditioned top-32 teacher log probabilities and",
              "token IDs for the matching RPMem compiler corpus. Not model weights.", "",
             "| Store | Sessions | Parquet parts |", "| --- | ---: | ---: |"]
    for split, store in report["stores"].items():
        lines.append(f"| `{split}` | {store['n_samples']:,} | {store['part_files']:,} |")
    lines += ["", "The original Parquet files are byte-identical, including historical embedded",
              "format labels. Only store metadata and completion records are repackaged.",
              "Log probabilities remain float16 and token IDs int32. Sample IDs index the",
              "frozen ordered corpus split; do not shuffle or substitute that split.", "",
              "Set `TEACHER_ROOT` to this directory and `CORPUS_ROOT` to the matching corpus.",
              "When the train store is included, use `--teacher_logprobs_dir \"$TEACHER_ROOT/train\"` with",
              "`--train_data \"$CORPUS_ROOT/train.corpus.json\"` in compiler training.",
              "Optional held-out validation uses `validation`, not `query_validation`.",
              "No query-validation cache is claimed unless it appears in the table above.", "",
              "The source model snapshot identity is recorded in `manifest.json`. Its original",
              "HF revision was not recorded; do not interpret the snapshot label as a pinned",
              "public HF commit. Tokenizer and teacher identity matter when reusing targets.", "",
              "Data publication remains subject to source-data redistribution review. The",
              "code license does not relicense upstream content. See the compiler corpus",
              "data card for source attribution and the reproduction guide for rebuilding.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Recorded teacher asset root")
    parser.add_argument("--corpus", type=Path, required=True, help="Matching prepared compiler corpus")
    parser.add_argument("--output", type=Path, required=True, help="New local publication draft")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "query_validation"],
                        help="Export only named stores; default is every store in the asset")
    args = parser.parse_args()
    report = stage_teacher_release(args.source, args.corpus, args.output, splits=args.splits)
    print(json.dumps({split: data["n_samples"] for split, data in report["stores"].items()}))


if __name__ == "__main__":
    main()
