"""Recompile memory and evaluate saved RPMem gates on the three main benchmarks.

Local files and explicitly reserved GPUs only. No training, remote storage, Ray,
or historical prediction files are used. See docs/result-reproduction.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.perma.data_adapter import ALL_USER_IDS, PERMA_VARIANTS

BENCHMARKS = ("perma", "personamem_v2", "prefeval")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_jobs(args):
    jobs = {"compile": [], "evaluate": []}
    size = len(args.gpus)

    def add(stage, name, script, argv):
        jobs[stage].append({"name": name, "command": [sys.executable, str(ROOT / script), *map(str, argv)]})

    gates = []
    if "perma" in args.benchmarks:
        common = ["--checkpoint", args.checkpoint,
                  "--base_model_path", args.base_model, "--ctx_encoder_path", args.ctx_encoder,
                  "--use_flash_attn" if args.use_flash_attn else "--no-use_flash_attn"]
        for variant in PERMA_VARIANTS:
            latent = args.output_root / "perma" / variant / "latents"
            for shard in range(size):
                add("compile", f"perma-{variant}-s{shard:02d}", "experiments/perma/precompute_phase2_latents.py",
                    [*common, "--variant", variant, "--output_dir", latent,
                     "--num_shards", size, "--shard_id", shard])
            for user in ALL_USER_IDS:
                gate = args.gate_root / "perma" / variant / f"fold_user{user}" / "gate.pt"
                gates.append(gate)
                add("evaluate", f"perma-{variant}-user{user}", "experiments/perma/run_phase2_fusion.py",
                    [*common, "--variant", variant, "--test_user", user, "--gate_checkpoint", gate,
                     "--emb_dir", latent, "--method", "cmp_gate", "--first-session-rule", args.first_session_rule,
                     "--output_dir", args.output_root / "perma" / variant / f"fold_user{user}"])
    for benchmark in (name for name in args.benchmarks if name != "perma"):
        gate = args.gate_root / benchmark / "gate.pt"
        gates.append(gate)
        common = ["--checkpoint", args.checkpoint,
                  "--base-model-path", args.base_model, "--ctx-encoder-path", args.ctx_encoder,
                  "--data-root", args.data_root / benchmark / "formal_v1",
                  "--num-shards", size,
                  "--use-flash-attn" if args.use_flash_attn else "--no-use-flash-attn"]
        latent = args.output_root / benchmark / "latents"
        flag = "--canonical-latent-root" if benchmark == "personamem_v2" else "--session-latent-root"
        for shard in range(size):
            add("compile", f"{benchmark}-s{shard:02d}", f"experiments/{benchmark}/precompute_phase2_latents.py",
                [*common, "--shard-id", shard, "--output-dir", latent])
            add("evaluate", f"{benchmark}-s{shard:02d}", f"experiments/{benchmark}/run_eval.py",
                [*common, "--shard-id", shard, "--method", "rpmem", "--gate", gate, flag, latent,
                 "--output", args.output_root / benchmark / "results" / f"shard{shard:02d}.jsonl"])
    return jobs, gates


def run_jobs(jobs, args, stage):
    devices = queue.Queue()
    for gpu in args.gpus:
        devices.put(gpu)
    stop = threading.Event()
    started = time.monotonic()

    def run(job):
        if stop.is_set():
            return
        gpu = devices.get()
        log = args.output_root / "logs" / f"{stage}-{job['name']}.log"
        marker = args.output_root / "completed" / f"{stage}-{job['name']}.json"
        try:
            if args.resume and marker.is_file() and read_json(marker).get("command") == job["command"]:
                return "reused"
            if stop.is_set():
                return
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                       PERMA_DATA_ROOT=str(args.data_root / "perma"))
            env["PYTHONPATH"] = str(ROOT)
            env["RPMEM_PERCEIVER_FLASH_ATTN"] = "1" if args.use_flash_attn else "0"
            with log.open("w") as handle:
                handle.write(json.dumps(job["command"]) + "\n")
                handle.flush()
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if stop.wait(1):
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        except ProcessLookupError:
                            process.wait()
                        return
                if process.returncode:
                    stop.set()
                    raise RuntimeError(f"{job['name']} failed (rc={process.returncode}); log={log}")
            write_json(marker, {"command": job["command"], "status": "complete"})
            return "complete"
        finally:
            devices.put(gpu)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        pending = {pool.submit(run, job): job for job in jobs}
        completed = 0
        try:
            while pending:
                ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = pending.pop(future)
                    status = future.result()
                    completed += 1
                    print(f"[{stage} {completed}/{len(jobs)}] {job['name']} {status}", flush=True)
                if not ready:
                    print(f"[{stage}] completed={completed}/{len(jobs)} elapsed={time.monotonic()-started:.0f}s", flush=True)
        except BaseException:
            stop.set()
            for future in pending:
                future.cancel()
            raise


def checked_rows(rows, expected_ids, key, prediction, gold, *, context="evaluation"):
    identifiers = [key(row) for row in rows]
    expected_ids = list(expected_ids)
    counts = Counter(identifiers)
    expected_counts = Counter(expected_ids)
    duplicates = sum(count - 1 for count in counts.values())
    expected_duplicates = sum(count - 1 for count in expected_counts.values())
    missing = expected_counts.keys() - counts.keys()
    unexpected = counts.keys() - expected_counts.keys()
    if duplicates or expected_duplicates or missing or unexpected:
        raise ValueError(
            f"{context}: incomplete or duplicated evaluation questions; "
            f"rows={len(rows)} expected={len(expected_ids)} duplicates={duplicates} "
            f"expected_duplicates={expected_duplicates} missing={len(missing)} "
            f"unexpected={len(unexpected)}; refusing a complete summary"
        )
    for row in rows:
        if type(row.get("correct")) is not bool or row["correct"] != (row[prediction] == row[gold]):
            raise ValueError("invalid prediction/correctness record")
    return rows


def accuracy(rows):
    if not rows:
        raise ValueError("empty metric group")
    return sum(row["correct"] for row in rows) / len(rows)


def summarize(args):
    metrics = {}
    details = {}
    if "perma" in args.benchmarks:
        from experiments.perma import data_adapter
        data_adapter.PERMA_DATA_ROOT = str(args.data_root / "perma")
        details["perma"] = {}
        for variant in PERMA_VARIANTS:
            fold_metrics = []
            tasks = data_adapter.load_tasks(variant=variant)
            for user in ALL_USER_IDS:
                folder = args.output_root / "perma" / variant / f"fold_user{user}"
                # PERMA repeats task_id across Type 1/2/3; the full identity is composite.
                rows = checked_rows(read_json(folder / "results.json"),
                                    [(variant, user, task.task_id, task.task_type)
                                     for task in tasks if task.user_id == user],
                                    lambda row: (row["variant"], row["user_id"], row["task_id"], row["task_type"]),
                                    "pred", "gold", context=f"perma/{variant}/fold_user{user}")
                saved = read_json(folder / "summary.json")
                if saved.get("run_mode") != "evaluate_only" or saved.get("first_session_rule") != args.first_session_rule:
                    raise ValueError("PERMA results are not the requested saved-gate evaluation")
                fold_metrics.append(accuracy(rows))
            mean = statistics.fmean(fold_metrics)
            details["perma"][variant] = {"accuracy": mean, "std": statistics.pstdev(fold_metrics), "folds": len(fold_metrics)}
            metrics[f"perma/{variant}"] = mean
        metrics["perma/average"] = statistics.fmean(metrics[f"perma/{v}"] for v in PERMA_VARIANTS)
        main_variants = ("clean_sd", "noisy_sd", "clean_md", "noisy_md")
        if all(v in PERMA_VARIANTS for v in main_variants):
            metrics["perma/main_four_average"] = statistics.fmean(metrics[f"perma/{v}"] for v in main_variants)
    for benchmark in (name for name in args.benchmarks if name != "perma"):
        rows = []
        for shard in range(len(args.gpus)):
            path = args.output_root / benchmark / "results" / f"shard{shard:02d}.jsonl"
            metadata = read_json(str(path) + ".meta.json")
            if metadata.get("first_session_rule") != args.first_session_rule:
                raise ValueError(f"{benchmark} Gate rule does not match requested convention")
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        data = args.data_root / benchmark / "formal_v1"
        if benchmark == "personamem_v2":
            from experiments.personamem_v2.formal_data import validate_dataset_root
            _, split = validate_dataset_root(data, verify_source_files=False)
            expected = [row["instance_id"] for row in split["benchmark_text"]]
        else:
            from experiments.prefeval.formal_data import validate_dataset_root, materialized_rows
            _, examples, noise = validate_dataset_root(data)
            expected = [row["instance_id"] for row in materialized_rows(examples, noise, split="test")]
        checked_rows(rows, expected, lambda row: row["instance_id"], "prediction_index", "gold_index",
                     context=benchmark)
        metrics[f"{benchmark}/overall"] = accuracy(rows)
        details[benchmark] = {"questions": len(rows)}
        if benchmark == "personamem_v2":
            metrics[f"{benchmark}/self"] = accuracy([r for r in rows if r["attributes"]["who"] == "self"])
            metrics[f"{benchmark}/current"] = accuracy([r for r in rows if str(r["attributes"]["updated"]).lower() == "false"])
        else:
            for turns in (10, 70, 300):
                metrics[f"{benchmark}/turns{turns}"] = accuracy([r for r in rows if r["turn_count"] == turns])
    result = {"status": "complete", "first_session_rule": args.first_session_rule,
              "metrics": metrics, "details": details}
    reference = read_json(args.reference)["metrics"] if args.reference else {}
    result["difference_percentage_points"] = {key: 100 * (value - reference[key])
                                             for key, value in metrics.items() if key in reference}
    write_json(args.output_root / "summary.json", result)
    lines = ["# RPMem Saved-Weight Reproduction", "", "| Metric | Accuracy (%) | Reference (%) | Delta (pp) |",
             "| --- | ---: | ---: | ---: |"]
    for key, value in metrics.items():
        ref = f"{100*reference[key]:.2f}" if key in reference else "-"
        delta = f"{100*(value-reference[key]):+.2f}" if key in reference else "-"
        lines.append(f"| {key} | {100*value:.2f} | {ref} | {delta} |")
    text = "\n".join(lines) + "\n"
    (args.output_root / "summary.md").write_text(text)
    print(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "gate-root", "data-root", "base-model", "ctx-encoder", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU indices/UUIDs explicitly reserved for this run")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--first-session-rule", choices=("gate_zero_state", "direct"), default="gate_zero_state")
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage", choices=("all", "compile", "evaluate", "summarize"), default="all")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any("," in value for value in args.gpus):
        parser.error("select unique individual GPUs, e.g. --gpus 0 1")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("duplicate benchmarks")
    for name in ("checkpoint", "gate_root", "data_root", "base_model", "ctx_encoder", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    jobs, gates = build_jobs(args)
    plan = {"format": "rpmem_saved_weight_reproduction_v1", "checkpoint": str(args.checkpoint),
            "gates": [str(path) for path in gates], "jobs": jobs,
            "data_root": str(args.data_root), "first_session_rule": args.first_session_rule,
            "benchmarks": args.benchmarks, "num_shards": len(args.gpus)}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    from rpmem.checkpoint.loader import load_gate_checkpoint
    for path in gates:
        gate, metadata = load_gate_checkpoint(path)
        if gate.first_session_rule != args.first_session_rule:
            parser.error(f"Gate first-session convention mismatch: {path}")
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if not args.resume or read_json(plan_path) != plan:
            parser.error("use a fresh output directory, or --resume with the unchanged plan")
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("output directory must be empty for a new run")
    write_json(plan_path, plan)
    if args.stage != "summarize":
        import torch
        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "cuda": torch.version.cuda, "requested_gpus": args.gpus,
                   "attention": "flash_attention_2" if args.use_flash_attn else "sdpa",
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "numpy", "pyarrow")}}
        write_json(args.output_root / "runtime.json", runtime)
    if args.stage in ("all", "compile"):
        run_jobs(jobs["compile"], args, "compile")
    if args.stage in ("all", "evaluate"):
        run_jobs(jobs["evaluate"], args, "evaluate")
    if args.stage in ("all", "evaluate", "summarize"):
        summarize(args)


if __name__ == "__main__":
    main()
