"""Frozen contracts for the first PersonaMem-v2 formal matrix."""

from __future__ import annotations

import hashlib
import json
from typing import Any


EVALUATION_PROTOCOL = "memlora_personamem_v2_mcq_no_thinking_v1"
LATENT_FORMAT = "memlora_personamem_v2_segment_latents_v1"
GATE_RESULT_FORMAT = "memlora_personamem_v2_cmp_gate_v1"
PHASE1_METHOD = "offline_fkl"
PHASE1_STEP = 51_885
PHASE1_PASSES = 5


GATE_CONTRACT = {
    "epochs": 5,
    "seed": 42,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "init_bias": -2.0,
    "selection": "fixed_final_epoch",
    "compiler": "frozen_offline_fkl_5pass",
    "optimizer_updates": "one_per_training_question",
    "training_order": "seeded_persona_then_question_shuffle_v1",
}


def _contract(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    result = dict(payload)
    result["sha256"] = hashlib.sha256(canonical).hexdigest()
    result["policy_id"] = f"{prefix}_{result['sha256'][:8]}"
    return result


def segmentation_contract(
    *, max_context_tokens: int = 4096, event_overlap: int = 1
) -> dict[str, Any]:
    return _contract(
        {
            "format": "memlora_personamem_v2_memory_segments_v1",
            "boundary": "chronological_message_stream_then_token_bounded_segment",
            "rendering": "formal_phase1_canonical_session",
            "input_fields": "chat_history_only",
            "question_visible_to_compiler": False,
            "max_context_tokens": max_context_tokens,
            "event_overlap": event_overlap,
            "oversize_event_policy": "lossless_token_chunks",
            "event_order": "source_order",
            "segment_order": "source_order",
        },
        f"personamem_message_stream_ctx{max_context_tokens}_overlap{event_overlap}",
    )
