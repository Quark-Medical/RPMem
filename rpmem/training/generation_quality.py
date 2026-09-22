"""Reference-based generation quality metrics for Phase 1 validation."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, fields
from typing import Iterable, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from rpmem.training.opcd import rotating_query_rows


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _trim_token_row(
    values: Sequence[int] | Tensor,
    *,
    eos_token_ids: set[int],
    special_token_ids: set[int],
) -> tuple[list[int], bool]:
    if isinstance(values, Tensor):
        values = values.detach().cpu().tolist()
    result: list[int] = []
    ended_with_eos = False
    for raw_value in values:
        value = int(raw_value)
        if value in eos_token_ids:
            ended_with_eos = True
            break
        if value not in special_token_ids:
            result.append(value)
    return result, ended_with_eos


def _token_f1(prediction: Sequence[int], reference: Sequence[int]) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_value in right:
        current = [0]
        for index, left_value in enumerate(left, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: Sequence[int], reference: Sequence[int]) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    overlap = _lcs_length(prediction, reference)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


@dataclass
class GenerationQualityStats:
    """Mergeable sufficient statistics for generation quality."""

    query_count: float = 0.0
    strict_exact_sum: float = 0.0
    normalized_exact_sum: float = 0.0
    reference_containment_sum: float = 0.0
    token_f1_sum: float = 0.0
    rouge_l_sum: float = 0.0
    eos_sum: float = 0.0
    generated_tokens_sum: float = 0.0
    reference_tokens_sum: float = 0.0

    def merge(self, other: "GenerationQualityStats") -> None:
        for field in fields(self):
            setattr(
                self, field.name, getattr(self, field.name) + getattr(other, field.name)
            )

    def raw_values(self) -> list[float]:
        return [float(getattr(self, field.name)) for field in fields(self)]

    @classmethod
    def from_raw_values(cls, values: Sequence[float]) -> "GenerationQualityStats":
        names = [field.name for field in fields(cls)]
        if len(values) != len(names):
            raise ValueError(
                f"expected {len(names)} generation quality values, got {len(values)}"
            )
        return cls(**dict(zip(names, map(float, values), strict=True)))

    def metrics(self, prefix: str = "quality") -> dict[str, float]:
        count = self.query_count
        if count <= 0:
            return {}
        return {
            f"{prefix}/strict_exact_match": self.strict_exact_sum / count,
            f"{prefix}/normalized_exact_match": self.normalized_exact_sum / count,
            f"{prefix}/reference_containment": (self.reference_containment_sum / count),
            f"{prefix}/token_f1": self.token_f1_sum / count,
            f"{prefix}/rouge_l": self.rouge_l_sum / count,
            f"{prefix}/eos_rate": self.eos_sum / count,
            f"{prefix}/mean_generated_tokens": self.generated_tokens_sum / count,
            f"{prefix}/mean_reference_tokens": self.reference_tokens_sum / count,
            f"{prefix}/query_count": count,
        }

    def raw_metrics(self, prefix: str = "quality") -> dict[str, float]:
        return {
            f"{prefix}/raw_{field.name}": float(getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def from_raw_metrics(
        cls, metrics: dict[str, float], prefix: str = "quality"
    ) -> "GenerationQualityStats":
        values = [
            float(metrics.get(f"{prefix}/raw_{field.name}", 0.0))
            for field in fields(cls)
        ]
        return cls.from_raw_values(values)


def is_raw_quality_metric(name: str, prefix: str = "quality") -> bool:
    return name.startswith(f"{prefix}/raw_")


def score_generation_rows(
    prediction_rows: Sequence[Sequence[int] | Tensor],
    reference_rows: Sequence[Sequence[int] | Tensor],
    *,
    tokenizer: PreTrainedTokenizerBase,
    eos_token_ids: int | Iterable[int],
) -> GenerationQualityStats:
    """Score generated token rows against fixed references."""

    if len(prediction_rows) != len(reference_rows):
        raise ValueError("prediction and reference row counts must match")
    if isinstance(eos_token_ids, int):
        eos_values = {int(eos_token_ids)}
    else:
        eos_values = {int(value) for value in eos_token_ids}
    if not eos_values:
        raise ValueError("at least one EOS token ID is required")
    special_values = {int(value) for value in tokenizer.all_special_ids}
    special_values.update(eos_values)

    stats = GenerationQualityStats()
    for prediction_row, reference_row in zip(
        prediction_rows, reference_rows, strict=True
    ):
        prediction, ended_with_eos = _trim_token_row(
            prediction_row,
            eos_token_ids=eos_values,
            special_token_ids=special_values,
        )
        reference, _ = _trim_token_row(
            reference_row,
            eos_token_ids=eos_values,
            special_token_ids=special_values,
        )
        if not reference:
            raise ValueError("generation quality references must not be empty")

        prediction_text = tokenizer.decode(
            prediction,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        reference_text = tokenizer.decode(
            reference,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        normalized_prediction = _normalize_text(prediction_text)
        normalized_reference = _normalize_text(reference_text)

        stats.query_count += 1
        stats.strict_exact_sum += float(prediction_text == reference_text)
        stats.normalized_exact_sum += float(
            normalized_prediction == normalized_reference
        )
        stats.reference_containment_sum += float(
            bool(normalized_reference) and normalized_reference in normalized_prediction
        )
        stats.token_f1_sum += _token_f1(prediction, reference)
        stats.rouge_l_sum += _rouge_l_f1(prediction, reference)
        stats.eos_sum += float(ended_with_eos)
        stats.generated_tokens_sum += len(prediction)
        stats.reference_tokens_sum += len(reference)
    return stats


def build_reference_generation_batch(
    *,
    prompt_rows: Sequence[Tensor],
    reference_rows: Sequence[Tensor],
    n_queries: Tensor,
    sample_indices: Tensor,
    queries_per_session: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
    """Select fixed probes and left-pad their thinking-disabled prompts."""

    total_queries = int(n_queries.sum().item())
    if len(prompt_rows) != total_queries or len(reference_rows) != total_queries:
        raise ValueError("quality prompt/reference rows do not match query counts")
    selected_rows, selected_n_queries, _ = rotating_query_rows(
        n_queries,
        sample_indices,
        corpus_pass=0,
        queries_per_session=queries_per_session,
    )
    prompts: list[Tensor] = []
    references: list[Tensor] = []
    for row in selected_rows.tolist():
        prompt = prompt_rows[row]
        reference = reference_rows[row]
        if prompt.numel() == 0:
            prompt = torch.tensor(
                [pad_token_id],
                dtype=torch.long,
                device=n_queries.device,
            )
        if reference.numel() == 0:
            raise ValueError("quality evaluation reference must not be empty")
        prompts.append(prompt.to(n_queries.device))
        references.append(reference.to(n_queries.device))

    max_prompt = max(prompt.numel() for prompt in prompts)
    prompt_ids = torch.full(
        (len(prompts), max_prompt),
        int(pad_token_id),
        dtype=prompts[0].dtype,
        device=n_queries.device,
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    for row, prompt in enumerate(prompts):
        width = prompt.numel()
        prompt_ids[row, -width:] = prompt
        prompt_mask[row, -width:] = 1
    return prompt_ids, prompt_mask, references, selected_n_queries


def validate_quality_eval_contract(
    *,
    quality_eval_sessions: int,
    quality_eval_queries_per_session: int,
    quality_eval_max_new_tokens: int,
) -> None:
    if quality_eval_sessions < 0:
        raise ValueError("quality_eval_sessions must be non-negative")
    if quality_eval_queries_per_session <= 0:
        raise ValueError("quality_eval_queries_per_session must be positive")
    if quality_eval_max_new_tokens <= 0:
        raise ValueError("quality_eval_max_new_tokens must be positive")
