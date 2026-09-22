"""Stage the exact training/evaluation inputs for a main-method benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.perma import data_adapter as perma
from experiments.personamem_v2 import formal_data as persona
from experiments.prefeval import formal_data as pref

BENCHMARKS = ("perma", "personamem_v2", "prefeval")


def read_json(path):
    return json.loads(path.read_text())


def local_file(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"non-portable benchmark path: {relative}")
    path.resolve().relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark input: {relative}")
    return path


def inspect_inputs(benchmark, root):
    root = Path(root)
    freeze = read_json(local_file(root, "memlora_dataset_freeze.json"))
    selected = {"memlora_dataset_freeze.json"}
    if benchmark == "perma":
        if set(freeze["prepared_variants"]) != set(perma.PERMA_VARIANTS):
            raise ValueError("PERMA release needs all seven variants")
        counts = {}
        for variant, suffix in perma.PERMA_VARIANTS.items():
            users = {}
            for user in perma.ALL_USER_IDS:
                relative = f"tasks/user{user}/input_data{suffix}.json"
                tasks = read_json(local_file(root, relative))["overall"]
                selected.add(relative)
                identities = [(task["task_id"], int(task["type"])) for task in tasks]
                if not tasks or len(set(identities)) != len(tasks):
                    raise ValueError(f"empty/duplicated PERMA questions: {variant}/user{user}")
                for task, kind in identities:
                    meta = f"evaluation/user{user}/meta/overall/{task}_{kind}.json"
                    local_file(root, meta)
                    selected.add(meta)
                users[str(user)] = len(tasks)
            counts[variant] = {"questions": sum(users.values()), "questions_by_user": users}
        policy = {"protocol": "leave_one_user_out", "held_out_users": list(perma.ALL_USER_IDS),
                  "training_users": "all other users in the same variant"}
        source = {"repository": freeze["repository"], "revision": freeze["revision"]}
    elif benchmark == "personamem_v2":
        selected.update(freeze["normalized_files"].values())
        selected.add(freeze["source_manifest_file"])
        sources = read_json(local_file(root, freeze["source_manifest_file"]))
        selected.update(item["path"] for item in sources)
        for relative in selected:
            local_file(root, relative)
        _, splits = persona.validate_dataset_root(root)
        histories = {row["chat_history_32k_file"] for rows in splits.values() for row in rows}
        if not histories <= selected:
            raise ValueError("PersonaMem histories are missing from the source inventory")
        for relative in sorted(histories):
            persona.load_history(local_file(root, relative))
        counts = {split: len(rows) for split, rows in splits.items()}
        counts["history_files"] = len(histories)
        policy = {"train": "train_text", "configuration": "val_text", "test": "benchmark_text",
                  "benchmark_personas_excluded_from_training": True}
        source = {"repository": freeze["source_dataset"], "revision": freeze["source_revision"]}
    elif benchmark == "prefeval":
        selected.update(freeze[key] for key in ("examples_file", "noise_sessions_file", "source_manifest_file"))
        for relative in selected:
            local_file(root, relative)
        _, examples, noise = pref.validate_dataset_root(root)
        if max(pref.PRIMARY_TURN_COUNTS) > len(noise):
            raise ValueError("PrefEval noise history is incomplete")
        split_counts = dict(Counter(row["split"] for row in examples))
        counts = {"form_rows_by_split": dict(Counter(row["split"] for row in examples)),
                  "materialized_questions": {split: split_counts.get(split, 0) * len(pref.PRIMARY_TURN_COUNTS)
                                             for split in ("train", "test")},
                  "noise_sessions": len(noise)}
        policy = {"train_topics": list(pref.TRAIN_TOPICS), "test_topics": list(pref.TEST_TOPICS),
                  "turn_counts": list(pref.PRIMARY_TURN_COUNTS), "preference_forms": list(pref.PREFERENCE_FORMS)}
        source = {"repository": freeze["source_repository"], "revision": freeze["source_revision"]}
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return sorted(selected), {"counts": counts, "split_policy": policy, "source": source}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_benchmark(benchmark, source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"release destination already exists: {output}")
    selected, info = inspect_inputs(benchmark, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-release-", dir=output.parent) as work:
        draft = Path(work) / "payload"
        draft.mkdir()
        inventory = []
        for relative in selected:
            original = local_file(source, relative)
            target = draft / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            digest = sha256(original)
            if sha256(target) != digest:
                raise ValueError(f"benchmark copy mismatch: {relative}")
            inventory.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})
        restored_files, restored = inspect_inputs(benchmark, draft)
        if restored_files != selected or restored != info:
            raise ValueError("relocated benchmark inputs changed")
        report = {"format": "rpmem.benchmark_release.v1", "benchmark": benchmark,
                  "status": "staged_not_published", "publication_approved": False,
                  "input_files": "byte_identical", "payload_bytes": sum(f["bytes"] for f in inventory),
                  "files": inventory, **info}
        (draft / "release_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (draft / "README.md").write_text(render_card(report))
        draft.rename(output)
    return report


def render_card(report):
    name = report["benchmark"]
    sources = report["source"]
    return f"""# RPMem {name} Prepared Inputs

**Local publication draft. Not uploaded or approved for distribution.**

Prepared inputs for compiler-memory generation, downstream Gate training and
held-out evaluation. Original splits, option order and data files are preserved;
no predictions, latent caches, trained gates or baseline results are included.

Source: `{sources['repository']}` at `{sources['revision']}`.
Payload: {report['payload_bytes']:,} bytes in {len(report['files']):,} files.
Upstream attribution and redistribution terms still apply. The code license
does not grant a new license to these data. Public release approval is pending.

## Counts

```json
{json.dumps(report['counts'], indent=2)}
```

## Split policy

```json
{json.dumps(report['split_policy'], indent=2)}
```

For PERMA, point `PERMA_DATA_ROOT` at this directory. For PersonaMem-v2 and
PrefEval, pass this directory as `--data-root`. Keep the directory structure
intact: compilation reads historical conversations, not only question tables.
Use the code repository's `docs/benchmarks.md` for the training/evaluation steps.
PrefEval materializes each form row at 10/70/300 turns; form-row counts are not
the final question counts. PERMA's folds share input files but not trained gates.

`release_manifest.json` lists exact files and counts. Historical freeze files
retain their original names and identifiers for loader compatibility. Public
raw-source preparation code remains available; raw upstream snapshots are not
necessarily bundled in every benchmark package.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = stage_benchmark(args.benchmark, args.source, args.output)
    print(json.dumps({"benchmark": args.benchmark, "counts": report["counts"],
                      "payload_bytes": report["payload_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
