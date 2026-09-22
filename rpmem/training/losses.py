"""Loss functions for CMP Gate training across different benchmarks."""

import torch
import torch.nn.functional as F
from torch import Tensor


def mcq_ce_loss(
    logits: Tensor,
    label_token_id: int,
) -> Tensor:
    """MCQ cross-entropy loss: CE on last-token logits vs gold label token.

    logits: [1, seq_len, vocab_size]
    label_token_id: int (token id of gold answer letter A-H)
    """
    last_logits = logits[:, -1, :]  # [1, vocab_size]
    target = torch.tensor([label_token_id], device=logits.device)
    return F.cross_entropy(last_logits, target)


def lm_ce_loss(
    logits: Tensor,
    input_ids: Tensor,
) -> Tensor:
    """Causal LM cross-entropy loss: predict next token for full sequence.

    logits: [1, seq_len, vocab_size]
    input_ids: [1, seq_len]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def qa_ce_loss(
    logits: Tensor,
    prompt_len: int,
    answer_ids: Tensor,
) -> Tensor:
    """QA cross-entropy loss: CE only on answer tokens (not prompt).

    logits: [1, prompt_len + answer_len, vocab_size]
    prompt_len: int
    answer_ids: [1, answer_len]
    """
    answer_logits = logits[:, prompt_len - 1:-1, :].contiguous()
    answer_labels = answer_ids.contiguous()
    return F.cross_entropy(
        answer_logits.view(-1, answer_logits.size(-1)),
        answer_labels.view(-1),
    )
