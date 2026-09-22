"""Precompute sharded teacher top-K logprobs for offline context distillation.

Teacher = the frozen base model with the context in its prompt:
    teacher input  = [context \n\n prompt] + response
    student input  = [prompt] + response          (context goes through the hypernet)

For every response token position we store the teacher's top-K next-token
log-probabilities. The response token ids are produced by the exact same
tokenization as HypernetDataset (shared helper), so positions align 1:1 with
the student's label positions at training time.

Output layout ({output_dir}/):
    meta.json                         - settings validated at training time
    part-{shard_id}-{part}.parquet   - batched ragged top-K records
    done-{shard_id}.json             - completion manifest for one GPU shard

Usage (shard across 8 GPUs):
    for i in 0..7:
      CUDA_VISIBLE_DEVICES=$i python -m rpmem.training.precompute_teacher \
          --base_model_path models/reader \
          --train_data data/train.jsonl \
          --output_dir teacher_logprobs/train \
          --num_shards 8 --shard_id $i &
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from rpmem.training.data import (
    iter_prompt_response_pairs,
    tokenize_teacher_aligned_to_student,
)
from rpmem.training.logit_projection import response_position_logits
from rpmem.training.sample_store import open_sample_store
from rpmem.training.teacher_shards import (
    TeacherShardWriter,
    TeacherTopKRecord,
    source_content_digests,
    write_store_metadata,
)

logger = logging.getLogger(__name__)


AUTO_MODEL_LOADERS = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForMultimodalLM",
)


def load_teacher_model(model_path: str, *, use_flash_attn: bool):
    """Load text logits from causal or conditional multimodal checkpoints."""

    import transformers

    attention_modes = ["flash_attention_2", "sdpa"] if use_flash_attn else ["sdpa"]
    failures = []
    for loader_name in AUTO_MODEL_LOADERS:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        for attention_mode in attention_modes:
            try:
                model = loader.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attention_mode,
                )
                logger.info(
                    "Loaded teacher with %s (%s), model_class=%s",
                    loader_name,
                    attention_mode,
                    type(model).__name__,
                )
                return model
            except (ImportError, ValueError, TypeError) as exc:
                failures.append(
                    f"{loader_name}/{attention_mode}: {type(exc).__name__}: {exc}"
                )
    raise RuntimeError(
        "no Transformers auto-model loader accepted the teacher checkpoint:\n"
        + "\n".join(failures)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute teacher top-K logprobs")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        required=True,
        help="Same paths and order as train_hypernet --train_data",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="MUST match train_hypernet --max_seq_len",
    )
    parser.add_argument(
        "--max_teacher_ctx_tokens",
        type=int,
        default=8192,
        help="Context token budget in the teacher prompt",
    )
    parser.add_argument(
        "--max_teacher_seq_len",
        type=int,
        default=None,
        help="Teacher context+query+response limit; defaults to ctx budget + max_seq_len.",
    )
    parser.add_argument("--ctx_prompt_sep", type=str, default="\n\n")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--records_per_file", type=int, default=512)
    parser.add_argument("--rows_per_group", type=int, default=16)
    parser.add_argument(
        "--use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recompute files owned by this shard",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}"
        )

    if args.overwrite:
        for path in out_dir.glob(f"part-{args.shard_id:05d}-*.parquet"):
            path.unlink()
        done_path = out_dir / f"done-{args.shard_id:05d}.json"
        if done_path.exists():
            done_path.unlink()

    samples = open_sample_store(args.train_data)
    logger.info(
        "Indexed %d samples from %d file(s) without materializing the rows",
        len(samples),
        len(args.train_data),
    )

    max_teacher_seq_len = args.max_teacher_seq_len or (
        args.max_teacher_ctx_tokens + args.max_seq_len
    )
    meta = {
        "base_model_path": args.base_model_path,
        "train_data": [str(path) for path in samples.paths],
        "train_file_sizes": [path.stat().st_size for path in samples.paths],
        "train_content_digests": source_content_digests(samples.paths),
        "top_k": args.top_k,
        "max_seq_len": args.max_seq_len,
        "max_teacher_ctx_tokens": args.max_teacher_ctx_tokens,
        "max_teacher_seq_len": max_teacher_seq_len,
        "ctx_prompt_sep": args.ctx_prompt_sep,
        "n_samples": len(samples),
        "num_shards": args.num_shards,
        "records_per_file": args.records_per_file,
        "rows_per_group": args.rows_per_group,
    }
    write_store_metadata(out_dir, meta)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = load_teacher_model(
        args.base_model_path,
        use_flash_attn=args.use_flash_attn,
    )
    model.eval().cuda()
    model.requires_grad_(False)

    indices = range(args.shard_id, len(samples), args.num_shards)
    logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} samples")

    t0 = time.time()
    n_done, n_skipped = 0, 0
    projection_mode_logged = False
    with TeacherShardWriter(
        out_dir,
        shard_id=args.shard_id,
        top_k=args.top_k,
        records_per_file=args.records_per_file,
        rows_per_group=args.rows_per_group,
    ) as writer:
        completed = writer.completed_sample_indices
        for n, idx in enumerate(indices):
            if idx in completed:
                n_skipped += 1
                continue

            sample = samples[idx]
            ctx_text = str(sample["context"])
            ctx_tokens = tokenizer.encode(ctx_text, add_special_tokens=False)
            if len(ctx_tokens) > args.max_teacher_ctx_tokens:
                ctx_text = tokenizer.decode(
                    ctx_tokens[: args.max_teacher_ctx_tokens], skip_special_tokens=True
                )

            vals_parts = []
            idx_parts = []
            for prompt_text, response_text in iter_prompt_response_pairs(sample):
                input_ids_list, label_positions = tokenize_teacher_aligned_to_student(
                    prompt_text,
                    response_text,
                    tokenizer,
                    args.max_seq_len,
                    max_teacher_seq_len,
                    context_text=ctx_text,
                    ctx_prompt_sep=args.ctx_prompt_sep,
                    system_message=str(sample.get("system_message", "")),
                )
                if not label_positions:
                    continue

                input_ids = torch.tensor(
                    [input_ids_list], dtype=torch.long, device="cuda"
                )
                label_positions_tensor = torch.tensor(
                    label_positions, device="cuda", dtype=torch.long
                )
                with torch.inference_mode():
                    resp_logits, model_side_projection = response_position_logits(
                        model,
                        input_ids,
                        label_positions_tensor,
                        use_cache=False,
                    )
                    resp_logits = resp_logits.float()
                if not projection_mode_logged:
                    logger.info(
                        "Teacher vocabulary projection: %s",
                        "response positions only (model-side)"
                        if model_side_projection
                        else "full logits fallback",
                    )
                    projection_mode_logged = True
                logprobs = torch.log_softmax(resp_logits, dim=-1)
                vals, topk_idx = logprobs.topk(args.top_k, dim=-1)
                vals_parts.append(vals.cpu().to(torch.float16))
                idx_parts.append(topk_idx.cpu().to(torch.int32))

            if vals_parts:
                vals = torch.cat(vals_parts, dim=0).numpy()
                topk_idx = torch.cat(idx_parts, dim=0).numpy()
            else:
                vals = torch.empty(0, args.top_k, dtype=torch.float16).numpy()
                topk_idx = torch.empty(0, args.top_k, dtype=torch.int32).numpy()

            writer.add(
                TeacherTopKRecord(
                    sample_idx=idx,
                    values=vals,
                    indices=topk_idx,
                )
            )
            n_done += 1

            if (n + 1) % 200 == 0:
                elapsed = time.time() - t0
                visited = n + 1
                rate = visited / max(elapsed, 1e-6)
                eta = (len(indices) - visited) / max(rate, 1e-6) / 60
                logger.info(
                    f"[{visited}/{len(indices)}] {rate:.1f} samples/s, "
                    f"computed={n_done}, skipped={n_skipped}, ETA {eta:.0f} min"
                )

    logger.info(
        f"Done. computed={n_done}, skipped={n_skipped}, "
        f"total time={(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
