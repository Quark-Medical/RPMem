"""Export a bounded deterministic JSONL subset from a corpus split manifest."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

from rpmem.training.sample_store import open_sample_store, resolve_sample_paths


def _precomputed_longest_indices(inputs: list[str], count: int) -> list[int] | None:
    paths = resolve_sample_paths(inputs)
    if len(paths) != 1 or not paths[0].name.endswith(".corpus.json"):
        return None
    payload = json.loads(paths[0].read_text())
    if payload.get("format") != "memlora_corpus_split_v1":
        return None
    if "longest_context_indices" not in payload:
        raise ValueError(
            f"{paths[0]} has no precomputed longest-context index; rebuild the corpus"
        )
    indices = [int(value) for value in payload["longest_context_indices"]]
    if count > len(indices):
        raise ValueError(
            f"requested {count} longest contexts but corpus stores only "
            f"{len(indices)} candidates"
        )
    return indices[:count]


def export_subset(
    inputs: list[str],
    output: str | Path,
    *,
    max_samples: int,
    selection: str = "first",
) -> Path:
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    samples = open_sample_store(inputs)
    count = min(max_samples, len(samples))
    if selection == "first":
        selected = [samples[index] for index in range(count)]
    elif selection == "longest_context":
        precomputed = _precomputed_longest_indices(inputs, count)
        if precomputed is not None:
            selected = [samples[index] for index in precomputed]
        else:
            heap: list[tuple[int, int, int, dict]] = []
            for index in range(len(samples)):
                sample = samples[index]
                if "context_token_max" in sample:
                    context_tokens = int(sample["context_token_max"])
                elif "context_tokens" in sample:
                    context_tokens = int(sample["context_tokens"])
                else:
                    raise ValueError(
                        "longest_context selection requires context_token_max or "
                        "context_tokens on every sample"
                    )
                candidate = (context_tokens, -index, index, sample)
                if len(heap) < count:
                    heapq.heappush(heap, candidate)
                elif candidate[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, candidate)
            selected = [
                sample
                for _, _, _, sample in sorted(
                    heap,
                    key=lambda item: (-item[0], item[2]),
                )
            ]
    else:
        raise ValueError(f"unsupported subset selection: {selection}")

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    with temporary.open("w") as handle:
        for sample in selected:
            handle.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, required=True)
    parser.add_argument(
        "--selection",
        choices=("first", "longest_context"),
        default="first",
    )
    args = parser.parse_args()
    output = export_subset(
        args.inputs,
        args.output,
        max_samples=args.max_samples,
        selection=args.selection,
    )
    print(f"exported subset: {output}")


if __name__ == "__main__":
    main()
