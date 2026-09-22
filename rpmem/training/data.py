"""Data loading and preprocessing for hypernetwork training.

Handles the full pipeline:
  1. Load parquet/json datasets with (context, prompts, responses) fields
  2. Tokenize with base model tokenizer + ctx encoder tokenizer
  3. Optional sequence packing for efficient batching
  4. Collation into ctx_ids, input_ids, labels, position_ids
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from rpmem.training.opcd import rotating_query_rows
from rpmem.training.teacher_shards import (
    FORMAT_NAME,
    TeacherLogprobStore,
)

logger = logging.getLogger(__name__)


def stable_validation_order(length: int, seed: int) -> tuple[int, ...]:
    """Return a reproducible pseudo-random order for a frozen split."""

    if length < 0:
        raise ValueError("validation length must be non-negative")

    def sort_key(index: int) -> tuple[bytes, int]:
        payload = f"memlora_validation_order_v1:{seed}:{index}".encode()
        return hashlib.sha256(payload).digest(), index

    return tuple(sorted(range(length), key=sort_key))


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def iter_prompt_response_pairs(sample: dict) -> list[tuple[str, str]]:
    """Return all prompt/response pairs for one context sample."""
    if "prompts" in sample:
        prompts = _as_list(sample["prompts"])
    else:
        prompts = _as_list(sample.get("prompt", ""))
    if "responses" in sample:
        responses = _as_list(sample["responses"])
    else:
        responses = _as_list(sample.get("response", ""))

    pairs = []
    for prompt, response in zip(prompts, responses):
        if prompt is None or response is None:
            continue
        pairs.append((str(prompt), str(response)))
    return pairs


def iter_prompts(sample: dict) -> list[str]:
    """Return non-empty prompts without requiring fixed reference responses."""

    values = (
        _as_list(sample["prompts"])
        if "prompts" in sample
        else _as_list(sample.get("prompt", ""))
    )
    return [str(value) for value in values if value is not None and str(value).strip()]


def get_prompt_response_text(sample: dict) -> tuple[str, str]:
    """Extract prompt and response strings from a raw training sample."""
    pairs = iter_prompt_response_pairs(sample)
    if not pairs:
        return "", ""
    return pairs[0]


def _truncate_keep_labels(
    input_ids: list[int],
    labels: list[int],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    if len(input_ids) <= max_seq_len:
        return input_ids, labels

    label_positions = [i for i, label in enumerate(labels) if label != -100]
    if not label_positions:
        return input_ids[:max_seq_len], labels[:max_seq_len]

    first_label = label_positions[0]
    last_label = label_positions[-1]
    label_span = last_label - first_label + 1
    if label_span >= max_seq_len:
        start = max(0, last_label - max_seq_len + 1)
    else:
        prompt_budget = max_seq_len - label_span
        start = max(0, first_label - prompt_budget)
    end = min(len(input_ids), start + max_seq_len)
    start = max(0, end - max_seq_len)
    return input_ids[start:end], labels[start:end]


def tokenize_chat_prompt_response(
    prompt_text: str,
    response_text: str,
    base_tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    context_text: str | None = None,
    ctx_prompt_sep: str = "\n\n",
    system_message: str = "",
) -> tuple[list[int], list[int]]:
    """Tokenize one prompt/response pair and label assistant response tokens.

    This uses the chat-template path when the tokenizer supports assistant
    masks. If a tokenizer has no chat template, it falls back to prompt +
    response tokenization while preserving response labels.
    """
    user_text = prompt_text
    if context_text:
        user_text = str(context_text).strip() + ctx_prompt_sep + prompt_text

    chat_template = getattr(base_tokenizer, "chat_template", None)
    if chat_template and "{% generation" in str(chat_template):
        messages = []
        if system_message.strip():
            messages.append({"role": "system", "content": system_message.strip()})
        messages.extend(
            [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": response_text},
            ]
        )
        try:
            tokens = base_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            input_ids = list(tokens["input_ids"])
            labels = [
                token_id if mask else -100
                for token_id, mask in zip(input_ids, tokens["assistant_masks"])
            ]
            if any(label != -100 for label in labels):
                return _truncate_keep_labels(input_ids, labels, max_seq_len)
            logger.warning(
                "chat template returned no assistant labels; falling back to "
                "prompt-length labeling"
            )
        except Exception as exc:
            logger.warning(
                "chat template assistant mask failed; falling back to prompt-length "
                "labeling: %s",
                exc,
            )

    prompt_ids = base_tokenizer.encode(
        user_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_len,
    )
    response_ids = base_tokenizer.encode(
        response_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(max_seq_len - 1, 1),
    )
    if not response_ids:
        return prompt_ids[:max_seq_len], [-100] * min(len(prompt_ids), max_seq_len)

    prompt_budget = max(max_seq_len - len(response_ids), 1)
    prompt_ids = base_tokenizer.encode(
        user_text,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_budget,
    )
    if not prompt_ids:
        prompt_ids = [base_tokenizer.eos_token_id or 0]
    input_ids = prompt_ids + response_ids[: max_seq_len - len(prompt_ids)]
    labels = [-100] * len(prompt_ids) + response_ids[: max_seq_len - len(prompt_ids)]
    return input_ids, labels


def tokenize_chat_prompt(
    prompt_text: str,
    base_tokenizer: PreTrainedTokenizerBase,
    max_prompt_len: int,
    *,
    context_text: str | None = None,
    ctx_prompt_sep: str = "\n\n",
    system_message: str = "",
) -> list[int]:
    """Tokenize a generation prompt with Qwen thinking explicitly disabled."""

    if max_prompt_len <= 0:
        raise ValueError("max_prompt_len must be positive")
    user_text = prompt_text
    if context_text:
        user_text = str(context_text).strip() + ctx_prompt_sep + prompt_text

    chat_template = getattr(base_tokenizer, "chat_template", None)
    if chat_template:
        messages = []
        if system_message.strip():
            messages.append({"role": "system", "content": system_message.strip()})
        messages.append({"role": "user", "content": user_text.strip()})
        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        try:
            tokens = base_tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            tokens = base_tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
        token_ids = list(tokens)
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("chat template returned an unexpected prompt batch")
            token_ids = token_ids[0]
    else:
        token_ids = base_tokenizer.encode(
            user_text,
            add_special_tokens=True,
            truncation=False,
        )

    token_ids = token_ids[-max_prompt_len:]
    if not token_ids:
        token_ids = [base_tokenizer.eos_token_id or 0]
    return token_ids


def tokenize_prompt_response(
    sample: dict,
    base_tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a sample into prompt and response ids.

    This is shared by student training and teacher-logprob precompute so their
    response-token positions stay aligned.
    """
    prompt_text, response_text = get_prompt_response_text(sample)
    input_ids, labels = tokenize_chat_prompt_response(
        prompt_text,
        response_text,
        base_tokenizer,
        max_seq_len,
        system_message=str(sample.get("system_message", "")),
    )
    response_ids = [
        token_id for token_id, label in zip(input_ids, labels) if label != -100
    ]
    prompt_ids = input_ids[: len(input_ids) - len(response_ids)]
    return prompt_ids, response_ids


def tokenize_teacher_aligned_to_student(
    prompt_text: str,
    response_text: str,
    tokenizer: PreTrainedTokenizerBase,
    student_max_seq_len: int,
    teacher_max_seq_len: int,
    *,
    context_text: str,
    ctx_prompt_sep: str = "\n\n",
    system_message: str = "",
) -> tuple[list[int], list[int]]:
    """Tokenize a long-context teacher while preserving student target tokens."""

    _, student_labels = tokenize_chat_prompt_response(
        prompt_text,
        response_text,
        tokenizer,
        student_max_seq_len,
        system_message=system_message,
    )
    student_response_ids = [label for label in student_labels if label != -100]
    if not student_response_ids:
        return [], []

    teacher_ids, teacher_labels = tokenize_chat_prompt_response(
        prompt_text,
        response_text,
        tokenizer,
        teacher_max_seq_len,
        context_text=context_text,
        ctx_prompt_sep=ctx_prompt_sep,
        system_message=system_message,
    )
    teacher_positions = [
        position for position, label in enumerate(teacher_labels) if label != -100
    ]
    teacher_response_ids = [teacher_ids[position] for position in teacher_positions]

    width = len(student_response_ids)
    match_start = next(
        (
            start
            for start in range(len(teacher_response_ids) - width + 1)
            if teacher_response_ids[start : start + width] == student_response_ids
        ),
        None,
    )
    if match_start is None:
        raise ValueError(
            "teacher response tokens do not contain the student target sequence; "
            "check tokenizer, chat template, and sequence limits"
        )
    return teacher_ids, teacher_positions[match_start : match_start + width]


def _flatten_teacher_logprobs(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return torch.tensor(value)


class HypernetDataset(Dataset):
    """Dataset for hypernetwork training.

    Each sample contains:
      - ctx_ids: tokenized context [ctx_len]
      - input_ids: tokenized prompt + response [seq_len]
      - labels: same as input_ids but with prompt tokens masked as -100
      - optional teacher top-K logprobs at response positions
    """

    def __init__(
        self,
        samples: Sequence[dict],
        base_tokenizer: PreTrainedTokenizerBase,
        ctx_tokenizer: PreTrainedTokenizerBase,
        max_ctx_len: int = 768,
        max_seq_len: int = 2048,
        teacher_logprobs_dir: Optional[str] = None,
        objective: str = "sft",
        opcd_rollout_max_new_tokens: int = 128,
        opcd_max_teacher_seq_len: int = 4864,
        opcd_include_references: bool = False,
        quality_eval_sessions: int = 0,
        quality_eval_max_new_tokens: int = 0,
        validation_order_seed: int | None = None,
    ):
        self.samples = samples
        self.base_tokenizer = base_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_seq_len = max_seq_len
        self.teacher_logprobs_dir = teacher_logprobs_dir
        self.objective = objective
        self.opcd_rollout_max_new_tokens = opcd_rollout_max_new_tokens
        self.opcd_max_teacher_seq_len = opcd_max_teacher_seq_len
        self.opcd_include_references = bool(opcd_include_references)
        self.quality_eval_sessions = quality_eval_sessions
        self.quality_eval_max_new_tokens = quality_eval_max_new_tokens
        self.sample_order = (
            stable_validation_order(len(samples), validation_order_seed)
            if validation_order_seed is not None
            else None
        )
        self.teacher_logprob_store: TeacherLogprobStore | None = None

        if quality_eval_sessions < 0:
            raise ValueError("quality_eval_sessions must be non-negative")
        if quality_eval_sessions and quality_eval_max_new_tokens <= 0:
            raise ValueError(
                "quality_eval_max_new_tokens must be positive when quality eval is enabled"
            )
        if quality_eval_sessions and quality_eval_max_new_tokens >= max_seq_len:
            raise ValueError(
                "quality_eval_max_new_tokens must leave room for a generation prompt"
            )

        if objective == "opcd":
            if teacher_logprobs_dir is not None:
                raise ValueError("OPCD produces teacher distributions online")
            if opcd_rollout_max_new_tokens <= 0:
                raise ValueError("OPCD rollout length must be positive")
            if max_seq_len <= opcd_rollout_max_new_tokens:
                raise ValueError("max_seq_len must leave room for the OPCD query")
            if opcd_max_teacher_seq_len <= opcd_rollout_max_new_tokens:
                raise ValueError(
                    "opcd_max_teacher_seq_len must leave room for teacher input"
                )

        if teacher_logprobs_dir is not None:
            meta_path = Path(teacher_logprobs_dir) / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("max_seq_len") != max_seq_len:
                    raise ValueError(
                        "teacher logprobs were precomputed with max_seq_len="
                        f"{meta.get('max_seq_len')} but dataset uses {max_seq_len}; "
                        "token positions would misalign."
                    )
                if meta.get("format") == FORMAT_NAME:
                    self.teacher_logprob_store = TeacherLogprobStore(
                        teacher_logprobs_dir,
                        expected_max_seq_len=max_seq_len,
                    )
                    self.teacher_logprob_store.validate_coverage(len(samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        source_idx = self.sample_order[idx] if self.sample_order is not None else idx
        sample = self.samples[source_idx]
        ctx_text = sample["context"]
        quality_enabled = idx < self.quality_eval_sessions

        ctx_ids = self.ctx_tokenizer.encode(
            ctx_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_ctx_len,
        )

        if self.objective == "opcd":
            student_prompt_ids = []
            teacher_prompt_ids = []
            include_references = quality_enabled or self.opcd_include_references
            reference_ids = [] if include_references else None
            system_message = str(sample.get("system_message", ""))
            student_prompt_limit = self.max_seq_len - self.opcd_rollout_max_new_tokens
            teacher_prompt_limit = (
                self.opcd_max_teacher_seq_len - self.opcd_rollout_max_new_tokens
            )
            for prompt_text, response_text in iter_prompt_response_pairs(sample):
                student_prompt_ids.append(
                    torch.tensor(
                        tokenize_chat_prompt(
                            prompt_text,
                            self.base_tokenizer,
                            student_prompt_limit,
                            system_message=system_message,
                        ),
                        dtype=torch.long,
                    )
                )
                teacher_prompt_ids.append(
                    torch.tensor(
                        tokenize_chat_prompt(
                            prompt_text,
                            self.base_tokenizer,
                            teacher_prompt_limit,
                            context_text=ctx_text,
                            system_message=system_message,
                        ),
                        dtype=torch.long,
                    )
                )
                if include_references:
                    _, response_labels = tokenize_chat_prompt_response(
                        prompt_text,
                        response_text,
                        self.base_tokenizer,
                        self.max_seq_len,
                        system_message=system_message,
                    )
                    response_tokens = [
                        label for label in response_labels if label != -100
                    ][
                        : (
                            self.quality_eval_max_new_tokens
                            or self.opcd_rollout_max_new_tokens
                        )
                    ]
                    if not response_tokens:
                        raise ValueError(
                            f"sample {source_idx} has an empty OPCD reference"
                        )
                    reference_ids.append(
                        torch.tensor(response_tokens, dtype=torch.long)
                    )
            if not student_prompt_ids:
                raise ValueError(
                    f"sample {source_idx} has no OPCD prompt/reference pairs"
                )
            result = {
                "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "student_prompt_ids": student_prompt_ids,
                "teacher_prompt_ids": teacher_prompt_ids,
                "sample_index": torch.tensor(source_idx, dtype=torch.long),
                "logical_index": torch.tensor(idx, dtype=torch.long),
                "quality_selected": torch.tensor(quality_enabled, dtype=torch.bool),
            }
            if include_references:
                result["reference_ids"] = reference_ids
            return result

        input_ids = []
        labels = []
        quality_prompt_ids = []
        quality_reference_ids = []
        system_message = str(sample.get("system_message", ""))
        for prompt_text, response_text in iter_prompt_response_pairs(sample):
            ids, lab = tokenize_chat_prompt_response(
                prompt_text,
                response_text,
                self.base_tokenizer,
                self.max_seq_len,
                system_message=system_message,
            )
            if any(label != -100 for label in lab):
                input_ids.append(torch.tensor(ids, dtype=torch.long))
                labels.append(torch.tensor(lab, dtype=torch.long))
                if quality_enabled:
                    quality_prompt_ids.append(
                        torch.tensor(
                            tokenize_chat_prompt(
                                prompt_text,
                                self.base_tokenizer,
                                self.max_seq_len - self.quality_eval_max_new_tokens,
                                system_message=system_message,
                            ),
                            dtype=torch.long,
                        )
                    )
                    reference_tokens = [label for label in lab if label != -100][
                        : self.quality_eval_max_new_tokens
                    ]
                    if not reference_tokens:
                        raise ValueError(
                            f"sample {source_idx} has an empty quality reference"
                        )
                    quality_reference_ids.append(
                        torch.tensor(reference_tokens, dtype=torch.long)
                    )

        if not input_ids:
            raise ValueError(
                f"sample {source_idx} has no response labels after tokenization"
            )

        out = {
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "input_ids": input_ids,
            "labels": labels,
            "sample_index": torch.tensor(source_idx, dtype=torch.long),
            "logical_index": torch.tensor(idx, dtype=torch.long),
            "quality_selected": torch.tensor(quality_enabled, dtype=torch.bool),
        }
        if quality_enabled:
            out["quality_prompt_ids"] = quality_prompt_ids
            out["quality_reference_ids"] = quality_reference_ids

        if self.teacher_logprobs_dir is not None:
            if self.teacher_logprob_store is not None:
                teacher = self.teacher_logprob_store[source_idx]
                vals = torch.from_numpy(teacher.values).float()
                indices = torch.from_numpy(teacher.indices).long()
            else:
                teacher_path = (
                    Path(self.teacher_logprobs_dir) / f"{source_idx:08d}.pt"
                )
                teacher = torch.load(
                    teacher_path, weights_only=True, map_location="cpu"
                )
                vals = teacher["vals"]
                indices = teacher["idx"]
            n_labels = sum((label != -100).sum().item() for label in labels)
            if vals.shape[0] != n_labels:
                raise ValueError(
                    f"sample {source_idx}: teacher logprobs have {vals.shape[0]} rows "
                    f"but response has {n_labels} tokens; recompute teacher "
                    "logprobs with matching tokenizer and max_seq_len."
                )
            out["logprobs_vals"] = vals.float()
            out["logprobs_indices"] = indices.long()
        elif "logprobs_vals" in sample and "logprobs_indices" in sample:
            vals = _flatten_teacher_logprobs(sample["logprobs_vals"]).float()
            indices = _flatten_teacher_logprobs(sample["logprobs_indices"]).long()
            if vals.dim() > 2:
                vals = vals.reshape(-1, vals.shape[-1])
                indices = indices.reshape(-1, indices.shape[-1])
            n_labels = sum((label != -100).sum().item() for label in labels)
            if vals.shape[0] != n_labels:
                raise ValueError(
                    f"sample {source_idx}: inline teacher logprobs have "
                    f"{vals.shape[0]} rows "
                    f"but response has {n_labels} tokens; regenerate logprobs with "
                    "matching tokenizer and max_seq_len."
                )
            out["logprobs_vals"] = vals
            out["logprobs_indices"] = indices

        return out


def collate_hypernet(batch: list[dict]) -> dict:
    """Collate function for HypernetDataset.

    Pads to max length within batch.
    Returns:
      ctx_ids: [n_ctx, max_ctx_len]
      ctx_attn_mask: [n_ctx, max_ctx_len]
      input_ids: [bs, max_seq_len]
      attention_mask: [bs, max_seq_len]
      labels: [bs, max_seq_len]
      position_ids: [bs, max_seq_len]
      n_ctx_chunks: [bs] (1 per sample for simple case)
    """
    ctx_ids_list = [s["ctx_ids"] for s in batch]
    query_pairs = [
        (sample_idx, inp, lab)
        for sample_idx, s in enumerate(batch)
        for inp, lab in zip(s["input_ids"], s["labels"])
    ]

    # Pad ctx_ids
    max_ctx_len = max(c.shape[0] for c in ctx_ids_list)
    ctx_ids = torch.zeros(len(batch), max_ctx_len, dtype=torch.long)
    ctx_attn_mask = torch.zeros(len(batch), max_ctx_len, dtype=torch.long)
    for i, c in enumerate(ctx_ids_list):
        ctx_ids[i, : c.shape[0]] = c
        ctx_attn_mask[i, : c.shape[0]] = 1

    # Pad input_ids and labels
    max_seq_len = max(inp.shape[0] for _, inp, _ in query_pairs)
    input_ids = torch.zeros(len(query_pairs), max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(len(query_pairs), max_seq_len, dtype=torch.long)
    labels = torch.full((len(query_pairs), max_seq_len), -100, dtype=torch.long)
    position_ids = torch.zeros(len(query_pairs), max_seq_len, dtype=torch.long)

    for i, (_, inp, lab) in enumerate(query_pairs):
        seq_len = inp.shape[0]
        input_ids[i, :seq_len] = inp
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = lab
        position_ids[i, :seq_len] = torch.arange(seq_len)

    n_ctx_chunks = torch.ones(len(batch), dtype=torch.int32)
    n_queries = torch.tensor([len(s["input_ids"]) for s in batch], dtype=torch.int32)

    out = {
        "ctx_ids": ctx_ids,
        "ctx_attn_mask": ctx_attn_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "n_ctx_chunks": n_ctx_chunks,
        "n_queries": n_queries,
        "sample_indices": torch.stack([sample["sample_index"] for sample in batch]),
        "logical_indices": torch.stack(
            [sample["logical_index"] for sample in batch]
        ),
        "quality_selected": torch.stack(
            [sample["quality_selected"] for sample in batch]
        ),
    }
    if "quality_prompt_ids" in batch[0]:
        out["quality_prompt_ids"] = [
            prompt for sample in batch for prompt in sample["quality_prompt_ids"]
        ]
        out["quality_reference_ids"] = [
            reference
            for sample in batch
            for reference in sample["quality_reference_ids"]
        ]

    if "logprobs_vals" in batch[0]:
        out["logprobs_vals"] = torch.cat([s["logprobs_vals"] for s in batch], dim=0)
        out["logprobs_indices"] = torch.cat(
            [s["logprobs_indices"] for s in batch], dim=0
        )

    return out


def _left_pad_sequences(
    sequences: list[torch.Tensor], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(sequence.shape[0] for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        width = sequence.shape[0]
        input_ids[row, -width:] = sequence
        attention_mask[row, -width:] = 1
    return input_ids, attention_mask


def collate_opcd_hypernet(
    batch: list[dict], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Collate contexts and left-padded student/teacher generation prompts."""

    ctx_ids_list = [sample["ctx_ids"] for sample in batch]
    max_ctx_len = max(context.shape[0] for context in ctx_ids_list)
    ctx_ids = torch.zeros(len(batch), max_ctx_len, dtype=torch.long)
    ctx_attn_mask = torch.zeros(len(batch), max_ctx_len, dtype=torch.long)
    for row, context in enumerate(ctx_ids_list):
        ctx_ids[row, : context.shape[0]] = context
        ctx_attn_mask[row, : context.shape[0]] = 1

    student_prompts = [
        prompt for sample in batch for prompt in sample["student_prompt_ids"]
    ]
    teacher_prompts = [
        prompt for sample in batch for prompt in sample["teacher_prompt_ids"]
    ]
    student_ids, student_mask = _left_pad_sequences(student_prompts, pad_token_id)
    teacher_ids, teacher_mask = _left_pad_sequences(teacher_prompts, pad_token_id)
    return {
        "ctx_ids": ctx_ids,
        "ctx_attn_mask": ctx_attn_mask,
        "n_ctx_chunks": torch.ones(len(batch), dtype=torch.int32),
        "n_queries": torch.tensor(
            [len(sample["student_prompt_ids"]) for sample in batch],
            dtype=torch.int32,
        ),
        "sample_indices": torch.stack([sample["sample_index"] for sample in batch]),
        "opcd_student_prompt_ids": student_ids,
        "opcd_student_prompt_attention_mask": student_mask,
        "opcd_teacher_prompt_ids": teacher_ids,
        "opcd_teacher_prompt_attention_mask": teacher_mask,
    }


def collate_opcd_sft_hypernet(
    batch: list[dict],
    *,
    pad_token_id: int,
    corpus_pass: int,
    queries_per_session: int,
) -> dict[str, torch.Tensor]:
    """Collate grounded references for the teacher-forced OPCD bootstrap."""

    if not batch or any("reference_ids" not in sample for sample in batch):
        raise ValueError("OPCD SFT bootstrap requires references for every session")

    opcd_batch = collate_opcd_hypernet(batch, pad_token_id=pad_token_id)
    selected_rows, selected_n_queries, _ = rotating_query_rows(
        opcd_batch["n_queries"],
        opcd_batch["sample_indices"],
        corpus_pass=corpus_pass,
        queries_per_session=queries_per_session,
    )
    prompts = [
        prompt for sample in batch for prompt in sample["student_prompt_ids"]
    ]
    references = [
        reference for sample in batch for reference in sample["reference_ids"]
    ]
    if len(prompts) != len(references):
        raise ValueError("OPCD prompt/reference counts differ during SFT bootstrap")

    selected_indices = selected_rows.tolist()
    selected_prompts = [prompts[index] for index in selected_indices]
    selected_references = [references[index] for index in selected_indices]
    if any(reference.numel() == 0 for reference in selected_references):
        raise ValueError("OPCD SFT bootstrap selected an empty reference")

    lengths = [
        int(prompt.numel() + reference.numel())
        for prompt, reference in zip(
            selected_prompts, selected_references, strict=True
        )
    ]
    max_length = max(lengths)
    input_ids = torch.full(
        (len(lengths), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (prompt, reference) in enumerate(
        zip(selected_prompts, selected_references, strict=True)
    ):
        prompt_length = int(prompt.numel())
        sequence = torch.cat((prompt, reference))
        sequence_length = int(sequence.numel())
        input_ids[row, :sequence_length] = sequence
        attention_mask[row, :sequence_length] = 1
        position_ids[row, :sequence_length] = torch.arange(sequence_length)
        labels[row, prompt_length:sequence_length] = reference

    return {
        "ctx_ids": opcd_batch["ctx_ids"],
        "ctx_attn_mask": opcd_batch["ctx_attn_mask"],
        "n_ctx_chunks": opcd_batch["n_ctx_chunks"],
        "n_queries": selected_n_queries,
        "sample_indices": opcd_batch["sample_indices"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }


def packed_collate_hypernet(batch: list[dict]) -> dict:
    """Flash-Attention compatible packing: concatenate all sequences.

    For efficient training, multiple samples are packed into a single sequence
    using position_ids to separate them.

    Returns:
      ctx_ids: [1, total_ctx_len]  (packed)
      ctx_attn_mask: [1, total_ctx_len]
      ctx_position_ids: [1, total_ctx_len]
      input_ids: [1, total_seq_len]  (packed)
      attention_mask: [1, total_seq_len]
      position_ids: [1, total_seq_len]
      labels: [1, total_seq_len]
      n_ctx_chunks: [n_samples]
      n_queries: [n_samples]
    """
    all_ctx_ids = []
    all_ctx_pos = []
    all_input_ids = []
    all_pos_ids = []
    all_labels = []
    n_queries = []

    for s in batch:
        ctx = s["ctx_ids"]

        all_ctx_ids.append(ctx)
        all_ctx_pos.append(torch.arange(ctx.shape[0]))
        n_queries.append(len(s["input_ids"]))
        for inp, lab in zip(s["input_ids"], s["labels"]):
            all_input_ids.append(inp)
            all_pos_ids.append(torch.arange(inp.shape[0]))
            all_labels.append(lab)

    ctx_ids = torch.cat(all_ctx_ids).unsqueeze(0)
    ctx_position_ids = torch.cat(all_ctx_pos).unsqueeze(0)
    ctx_attn_mask = torch.ones_like(ctx_ids)
    input_ids = torch.cat(all_input_ids).unsqueeze(0)
    position_ids = torch.cat(all_pos_ids).unsqueeze(0)
    labels = torch.cat(all_labels).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    n_ctx_chunks = torch.ones(len(batch), dtype=torch.int32)
    n_queries = torch.tensor(n_queries, dtype=torch.int32)

    out = {
        "ctx_ids": ctx_ids,
        "ctx_attn_mask": ctx_attn_mask,
        "ctx_position_ids": ctx_position_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "n_ctx_chunks": n_ctx_chunks,
        "n_queries": n_queries,
    }

    if "logprobs_vals" in batch[0]:
        out["logprobs_vals"] = torch.cat([s["logprobs_vals"] for s in batch], dim=0)
        out["logprobs_indices"] = torch.cat(
            [s["logprobs_indices"] for s in batch], dim=0
        )

    return out


def load_parquet_samples(path: str) -> list[dict]:
    """Load training samples from a parquet file.

    Expected columns: context, prompts, responses
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    df = table.to_pandas()

    samples = []
    for _, row in df.iterrows():
        sample = {
            "context": row["context"],
            "prompts": row["prompts"] if "prompts" in row else [row.get("prompt", "")],
            "responses": row["responses"]
            if "responses" in row
            else [row.get("response", "")],
        }
        for optional_key in ("system_message", "logprobs_vals", "logprobs_indices"):
            if optional_key in row:
                sample[optional_key] = row[optional_key]
        samples.append(sample)
    return samples


def load_json_samples(path: str) -> list[dict]:
    """Load training samples from a JSON/JSONL file."""
    import json

    samples = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        else:
            samples = json.load(f)
    return samples
