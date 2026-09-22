"""Phase 1 objective functions shared by the RPMem trainers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _reduce_token_losses(losses: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def average_token_losses_by_session(
    token_losses: Tensor,
    labels: Tensor,
    position_ids: Tensor,
    n_queries: Tensor,
) -> Tensor:
    """Average response losses by token, then query, then session."""

    if token_losses.ndim != 1 or labels.ndim != 2 or position_ids.shape != labels.shape:
        raise ValueError("invalid shapes for session-balanced response loss")
    query_counts = [int(value) for value in n_queries.tolist()]
    total_queries = sum(query_counts)
    if labels.shape[0] == total_queries:
        token_counts = [int(value) for value in labels.ne(-100).sum(dim=1).tolist()]
    elif labels.shape[0] == 1:
        starts = torch.where(position_ids[0] == 0)[0].tolist()
        ends = starts[1:] + [labels.shape[1]]
        token_counts = [
            int(labels[0, start:end].ne(-100).sum().item())
            for start, end in zip(starts, ends)
        ]
    else:
        raise ValueError(
            "label rows must equal total queries or contain one packed row"
        )

    if len(token_counts) != total_queries or any(count <= 0 for count in token_counts):
        raise ValueError("each query must contain at least one response token")
    if sum(token_counts) != token_losses.numel():
        raise ValueError("response token losses do not align with query boundaries")

    query_losses = torch.stack(
        [loss.mean() for loss in torch.split(token_losses, token_counts)]
    )
    session_losses = torch.stack(
        [loss.mean() for loss in torch.split(query_losses, query_counts)]
    )
    return session_losses.mean()


def _prepare_teacher_topk(
    student_logits: Tensor,
    teacher_logprobs: Tensor,
    teacher_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [n_tokens, vocab_size]")
    if teacher_logprobs.ndim != 2 or teacher_indices.ndim != 2:
        raise ValueError("teacher top-k tensors must have shape [n_tokens, k]")
    if teacher_logprobs.shape != teacher_indices.shape:
        raise ValueError("teacher top-k values and indices must have matching shapes")
    if teacher_logprobs.shape[0] != student_logits.shape[0]:
        raise ValueError("teacher and student token counts must match")
    if teacher_indices.numel() and (
        teacher_indices.min() < 0 or teacher_indices.max() >= student_logits.shape[-1]
    ):
        raise ValueError("teacher top-k index is outside the student vocabulary")

    work_dtype = (
        torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    )
    return (
        student_logits.to(work_dtype),
        teacher_logprobs.to(device=student_logits.device, dtype=work_dtype),
        teacher_indices.to(device=student_logits.device, dtype=torch.long),
    )


def d2l_teacher_topk_cross_entropy(
    student_logits: Tensor,
    teacher_logprobs: Tensor,
    teacher_indices: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Reproduce upstream D2L's selected teacher-top-k cross entropy."""

    student_logits, teacher_logprobs, teacher_indices = _prepare_teacher_topk(
        student_logits, teacher_logprobs, teacher_indices
    )
    student_logprobs = student_logits.log_softmax(dim=-1)
    selected_student_logprobs = student_logprobs.gather(1, teacher_indices)
    teacher_probs = teacher_logprobs.exp()
    token_losses = -(teacher_probs * selected_student_logprobs).sum(dim=-1)
    return _reduce_token_losses(token_losses, reduction)


def teacher_topk_forward_kl(
    student_logits: Tensor,
    teacher_logprobs: Tensor,
    teacher_indices: Tensor,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Any]]:
    """Teacher-top-k plus tail forward KL on fixed response prefixes."""

    student_logits, teacher_logprobs, teacher_indices = _prepare_teacher_topk(
        student_logits, teacher_logprobs, teacher_indices
    )
    if teacher_indices.shape[1] == 0:
        raise ValueError("teacher top-k support must not be empty")
    if not bool(torch.isfinite(teacher_logprobs).all()):
        raise ValueError("teacher top-k log probabilities are not finite")
    sorted_indices = teacher_indices.sort(dim=-1).values
    if sorted_indices.shape[1] > 1 and bool(
        (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
    ):
        raise ValueError("teacher top-k support contains duplicate token IDs")
    student_logprobs = student_logits.log_softmax(dim=-1)
    selected_student_logprobs = student_logprobs.gather(1, teacher_indices)

    tiny = torch.finfo(student_logits.dtype).tiny
    teacher_probs = teacher_logprobs.exp()
    raw_teacher_mass = teacher_probs.sum(dim=-1)
    if not bool(torch.isfinite(raw_teacher_mass).all()):
        raise ValueError("teacher top-k probability mass is not finite")
    teacher_mass_excess = (raw_teacher_mass - 1.0).clamp_min(0.0)
    if bool((teacher_mass_excess > 1e-2).any()):
        raise ValueError(
            "teacher top-k probability mass exceeds one: "
            f"max={float(raw_teacher_mass.max()):.6f}"
        )

    # Teacher logprobs may be quantized before the remote payload reaches this
    # process. Project only small cumulative round-off back onto the simplex.
    teacher_probs = teacher_probs / raw_teacher_mass.clamp_min(1.0).unsqueeze(-1)
    teacher_mass = teacher_probs.sum(dim=-1)
    teacher_logprobs = teacher_probs.clamp_min(tiny).log()

    student_probs = selected_student_logprobs.exp()
    teacher_tail = (1.0 - teacher_mass).clamp_min(0.0)
    student_tail = (1.0 - student_probs.sum(dim=-1)).clamp_min(0.0)

    selected_kl = (teacher_probs * (teacher_logprobs - selected_student_logprobs)).sum(
        dim=-1
    )
    tail_kl = teacher_tail * (
        teacher_tail.clamp_min(tiny).log() - student_tail.clamp_min(tiny).log()
    )
    loss = _reduce_token_losses(selected_kl + tail_kl, reduction)
    stats = {
        "teacher_tail_mass": teacher_tail.detach().mean(),
        "student_tail_mass": student_tail.detach().mean(),
        "teacher_support_renormalization": teacher_mass_excess.detach().mean(),
        "teacher_support_renormalization_max": teacher_mass_excess.detach().max(),
    }
    return loss, stats


def student_topk_reverse_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    top_k: int = 32,
    chunk_size: int | None = None,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Any]]:
    """Student-top-k plus tail reverse KL on aligned response prefixes.

    The support is selected from the student because the reverse KL is
    weighted by the student distribution. All tokens outside that support are
    represented by one tail category, so the approximation remains normalized.
    Teacher logits are always treated as fixed targets.
    """

    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("student and teacher logits must have shape [n_tokens, vocab]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have matching shapes")
    if student_logits.shape[0] == 0:
        raise ValueError("reverse KL requires at least one response token")
    vocab_size = student_logits.shape[-1]
    if not 0 < top_k <= vocab_size:
        raise ValueError(f"top_k must be in [1, {vocab_size}], got {top_k}")
    if chunk_size is None:
        chunk_size = student_logits.shape[0]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    work_dtype = (
        torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    )
    tiny = torch.finfo(work_dtype).tiny
    token_loss_chunks = []
    student_tail_chunks = []
    teacher_tail_chunks = []
    student_entropy_chunks = []
    teacher_entropy_chunks = []

    for start in range(0, student_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, student_logits.shape[0])
        student_chunk = student_logits[start:stop].to(work_dtype)
        teacher_chunk = teacher_logits[start:stop].detach().to(work_dtype)

        student_logprobs = student_chunk.log_softmax(dim=-1)
        teacher_logprobs = teacher_chunk.log_softmax(dim=-1)
        selected_student_logprobs, selected_indices = student_logprobs.topk(
            top_k, dim=-1
        )
        selected_teacher_logprobs = teacher_logprobs.gather(
            dim=-1, index=selected_indices
        )

        selected_student_probs = selected_student_logprobs.exp()
        selected_teacher_probs = selected_teacher_logprobs.exp()
        student_tail = (1.0 - selected_student_probs.sum(dim=-1)).clamp(0.0, 1.0)
        teacher_tail = (1.0 - selected_teacher_probs.sum(dim=-1)).clamp(0.0, 1.0)

        selected_kl = (
            selected_student_probs
            * (selected_student_logprobs - selected_teacher_logprobs)
        ).sum(dim=-1)
        tail_kl = student_tail * (
            student_tail.clamp_min(tiny).log() - teacher_tail.clamp_min(tiny).log()
        )
        token_loss_chunks.append(selected_kl + tail_kl)

        student_probs = student_logprobs.exp()
        teacher_probs = teacher_logprobs.exp()
        student_entropy_chunks.append(
            -(student_probs * student_logprobs).sum(dim=-1).detach()
        )
        teacher_entropy_chunks.append(
            -(teacher_probs * teacher_logprobs).sum(dim=-1).detach()
        )
        student_tail_chunks.append(student_tail.detach())
        teacher_tail_chunks.append(teacher_tail.detach())

    token_losses = torch.cat(token_loss_chunks)
    stats = {
        "student_tail_mass": torch.cat(student_tail_chunks).mean(),
        "teacher_tail_mass": torch.cat(teacher_tail_chunks).mean(),
        "student_entropy": torch.cat(student_entropy_chunks).mean(),
        "teacher_entropy": torch.cat(teacher_entropy_chunks).mean(),
    }
    return _reduce_token_losses(token_losses, reduction), stats


def student_support_reverse_kl(
    student_logits: Tensor,
    support_indices: Tensor,
    teacher_support_logprobs: Tensor,
    teacher_tail_mass: Tensor,
    *,
    chunk_size: int | None = None,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Any]]:
    """Reverse KL using student-selected support scored by a remote teacher.

    The remote teacher returns full-vocabulary log probabilities for the
    selected IDs plus the probability mass outside that support. This is the
    same K+1 categorical approximation as :func:`student_topk_reverse_kl`
    without transferring full teacher logits to the training process.
    """

    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [n_tokens, vocab]")
    if support_indices.ndim != 2:
        raise ValueError("support_indices must have shape [n_tokens, k]")
    if teacher_support_logprobs.shape != support_indices.shape:
        raise ValueError("teacher support values and indices must have matching shapes")
    if support_indices.shape[0] != student_logits.shape[0]:
        raise ValueError("teacher and student token counts must match")
    if teacher_tail_mass.shape != (student_logits.shape[0],):
        raise ValueError("teacher_tail_mass must have shape [n_tokens]")
    if student_logits.shape[0] == 0:
        raise ValueError("reverse KL requires at least one response token")
    if support_indices.numel() and (
        support_indices.min() < 0 or support_indices.max() >= student_logits.shape[-1]
    ):
        raise ValueError("support index is outside the student vocabulary")
    if chunk_size is None:
        chunk_size = student_logits.shape[0]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    work_dtype = (
        torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    )
    tiny = torch.finfo(work_dtype).tiny
    support_indices = support_indices.long()
    teacher_support_logprobs = teacher_support_logprobs.detach().to(work_dtype)
    provided_teacher_tail = teacher_tail_mass.detach().to(work_dtype).clamp(0.0, 1.0)
    sorted_support, sort_order = support_indices.sort(dim=-1)
    unique_sorted = torch.ones_like(sorted_support, dtype=torch.bool)
    unique_sorted[:, 1:] = sorted_support[:, 1:] != sorted_support[:, :-1]
    support_unique_mask = torch.empty_like(unique_sorted)
    support_unique_mask.scatter_(1, sort_order, unique_sorted)
    support_unique_weight = support_unique_mask.to(work_dtype)
    raw_teacher_support_mass = (
        teacher_support_logprobs.exp() * support_unique_weight
    ).sum(dim=-1)
    if not bool(torch.isfinite(raw_teacher_support_mass).all()):
        raise ValueError("remote teacher support mass is not finite")
    teacher_support_mass_excess = (raw_teacher_support_mass - 1.0).clamp_min(0.0)
    if bool((teacher_support_mass_excess > 1e-2).any()):
        raise ValueError(
            "remote teacher unique support mass exceeds one: "
            f"max={float(raw_teacher_support_mass.max()):.6f}"
        )
    support_log_scale = torch.where(
        raw_teacher_support_mass > 1.0,
        raw_teacher_support_mass.clamp_min(tiny).log(),
        torch.zeros_like(raw_teacher_support_mass),
    )
    teacher_support_logprobs = teacher_support_logprobs - support_log_scale.unsqueeze(
        -1
    )
    teacher_support_mass = raw_teacher_support_mass.clamp_max(1.0)
    teacher_tail = (1.0 - teacher_support_mass).clamp(0.0, 1.0)
    teacher_tail_correction = (provided_teacher_tail - teacher_tail).abs()
    teacher_support_duplicate_fraction = 1.0 - support_unique_weight.mean()

    token_loss_chunks = []
    student_tail_chunks = []
    teacher_tail_chunks = []
    student_entropy_chunks = []
    teacher_entropy_chunks = []
    for start in range(0, student_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, student_logits.shape[0])
        indices = support_indices[start:stop]
        unique_weight = support_unique_weight[start:stop]
        teacher_selected = teacher_support_logprobs[start:stop]
        teacher_tail_chunk = teacher_tail[start:stop]
        student_chunk = student_logits[start:stop]
        selected_student_logits = student_chunk.gather(-1, indices).to(work_dtype)
        student_log_normalizer = student_chunk.to(work_dtype).logsumexp(
            dim=-1, keepdim=True
        )
        selected_student_logprobs = selected_student_logits - student_log_normalizer
        selected_student_probs = selected_student_logprobs.exp()
        student_tail = (
            1.0 - (selected_student_probs * unique_weight).sum(dim=-1)
        ).clamp(0.0, 1.0)

        selected_kl = (
            selected_student_probs
            * (selected_student_logprobs - teacher_selected)
            * unique_weight
        ).sum(dim=-1)
        tail_kl = student_tail * (
            student_tail.clamp_min(tiny).log()
            - teacher_tail_chunk.clamp_min(tiny).log()
        )
        token_loss_chunks.append(selected_kl + tail_kl)
        student_tail_chunks.append(student_tail.detach())
        teacher_tail_chunks.append(teacher_tail_chunk.detach())
        student_entropy_chunks.append(
            (
                -(
                    selected_student_probs * selected_student_logprobs * unique_weight
                ).sum(dim=-1)
                - student_tail * student_tail.clamp_min(tiny).log()
            ).detach()
        )
        teacher_entropy_chunks.append(
            -(teacher_selected.exp() * teacher_selected * unique_weight)
            .sum(dim=-1)
            .detach()
            - teacher_tail_chunk * teacher_tail_chunk.clamp_min(tiny).log()
        )

    token_losses = torch.cat(token_loss_chunks)
    stats = {
        "student_tail_mass": torch.cat(student_tail_chunks).mean(),
        "teacher_tail_mass": torch.cat(teacher_tail_chunks).mean(),
        "teacher_tail_correction": teacher_tail_correction.mean(),
        "teacher_tail_correction_max": teacher_tail_correction.max(),
        "teacher_support_duplicate_fraction": teacher_support_duplicate_fraction,
        "teacher_support_renormalization": teacher_support_mass_excess.mean(),
        "teacher_support_renormalization_max": teacher_support_mass_excess.max(),
        "student_entropy": torch.cat(student_entropy_chunks).mean(),
        "teacher_entropy": torch.cat(teacher_entropy_chunks).mean(),
    }
    return _reduce_token_losses(token_losses, reduction), stats


def sampled_token_reverse_kl_k3(
    student_logits: Tensor,
    sampled_token_ids: Tensor,
    teacher_sampled_logprobs: Tensor,
    *,
    use_k2_gradient: bool = False,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Any]]:
    """K3 reverse-KL estimator on teacher-scored student samples.

    The rollout token is sampled from the student, while the teacher only
    scores that token:

        log_ratio = log p_teacher(y) - log p_student(y)
        loss = exp(log_ratio) - log_ratio - 1

    Direct K3 backpropagation is biased. When ``use_k2_gradient`` is true the
    forward value remains K3, but a straight-through construction routes the
    backward pass through K2, matching VERL's ``k3+`` contract.
    """

    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [n_tokens, vocab]")
    if sampled_token_ids.shape != (student_logits.shape[0],):
        raise ValueError("sampled_token_ids must have shape [n_tokens]")
    if teacher_sampled_logprobs.shape != (student_logits.shape[0],):
        raise ValueError("teacher_sampled_logprobs must have shape [n_tokens]")
    if student_logits.shape[0] == 0:
        raise ValueError("K3 reverse KL requires at least one response token")
    if sampled_token_ids.numel() and (
        sampled_token_ids.min() < 0
        or sampled_token_ids.max() >= student_logits.shape[-1]
    ):
        raise ValueError("sampled token ID is outside the student vocabulary")
    if not bool(torch.isfinite(teacher_sampled_logprobs).all()):
        raise ValueError("teacher sampled-token log probabilities are not finite")

    work_dtype = (
        torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    )
    sampled_token_ids = sampled_token_ids.long()
    student_sampled_logprobs = (
        student_logits.to(work_dtype)
        .log_softmax(dim=-1)
        .gather(-1, sampled_token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    teacher_sampled_logprobs = teacher_sampled_logprobs.detach().to(work_dtype)
    log_ratio = (teacher_sampled_logprobs - student_sampled_logprobs).clamp(-20.0, 20.0)
    probability_ratio = log_ratio.exp()
    k3_losses = (probability_ratio - log_ratio - 1.0).clamp(-10.0, 10.0)
    if use_k2_gradient:
        k2_losses = 0.5 * (
            student_sampled_logprobs - teacher_sampled_logprobs
        ).square()
        token_losses = k2_losses - k2_losses.detach() + k3_losses.detach()
    else:
        token_losses = k3_losses

    stats = {
        "k3_loss": k3_losses.detach().mean(),
        "k3_plus": torch.tensor(
            float(use_k2_gradient),
            dtype=work_dtype,
            device=student_logits.device,
        ),
        "student_sampled_logprob": student_sampled_logprobs.detach().mean(),
        "teacher_sampled_logprob": teacher_sampled_logprobs.mean(),
        "sampled_log_ratio": log_ratio.detach().mean(),
        "sampled_probability_ratio": probability_ratio.detach().mean(),
    }
    return _reduce_token_losses(token_losses, reduction), stats
