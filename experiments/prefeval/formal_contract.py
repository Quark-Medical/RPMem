"""Frozen contracts for the formal PrefEval Phase-2 matrix."""

from __future__ import annotations

import hashlib
import json
from typing import Any


EVALUATION_PROTOCOL = "memlora_prefeval_mcq_no_thinking_v1"
LATENT_FORMAT = "memlora_prefeval_session_latents_v1"
ALL_HISTORY_LATENT_FORMAT = "memlora_prefeval_all_history_latents_v1"
GATE_RESULT_FORMAT = "memlora_prefeval_cmp_gate_v1"
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
    "training_scope": "all_train_topics_forms_and_primary_turn_counts",
    "test_scope": "held_out_topics_only",
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
            "format": "memlora_prefeval_session_segmentation_v1",
            "boundary": "source_session_then_message_bounded_segment",
            "rendering": "formal_phase1_canonical_session",
            "input_fields": "memory_sessions_only",
            "question_visible_to_compiler": False,
            "max_context_tokens": max_context_tokens,
            "event_overlap": event_overlap,
            "oversize_event_policy": "lossless_token_chunks",
            "session_order": "preference_then_official_noise_prefix",
            "cache_reuse": "shared_noise_session_latents",
        },
        f"prefeval_session_ctx{max_context_tokens}_overlap{event_overlap}",
    )
