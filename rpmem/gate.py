"""CMP (Continual Memory Parametrization) gate module."""

import torch
import torch.nn as nn
from torch import Tensor

from rpmem.config import GateConfig

FIRST_SESSION_RULES = ("direct", "gate_zero_state")


def checkpoint_first_session_rule(payload: dict) -> str:
    """Restore the recorded rule; unversioned historical gates used a zero state."""
    metadata = payload.get("metadata", {})
    sources = (payload, payload.get("args", {}), payload.get("release", {}),
               payload.get("gate_contract", {}), metadata, metadata.get("gate_contract", {}))
    rules = {source["first_session_rule"] for source in sources
             if source.get("first_session_rule") is not None}
    if len(rules) > 1 or not rules.issubset(FIRST_SESSION_RULES):
        raise ValueError(f"invalid or conflicting first_session_rule: {rules}")
    return next(iter(rules), "gate_zero_state")


def normalize_gate_contract(contract: dict) -> dict:
    """Give old resume contracts their explicit historical initialization rule."""
    return {"first_session_rule": "gate_zero_state", **contract}


class CMPGate(nn.Module):
    """Minimal GRU gate for recurrent LoRA embedding fusion.

    z = sigmoid(W @ [h_prev, q_new] + b)
    h_new = z * h_prev + (1-z) * q_new

    Operates on d_latent dimension, shared across all layers and ranks.
    """

    def __init__(self, config: GateConfig | None = None, *, d_latent: int = 512,
                 init_bias: float = -2.0, first_session_rule: str = "direct"):
        super().__init__()
        if config is not None:
            d_latent = config.d_latent
            init_bias = config.init_bias
            first_session_rule = config.first_session_rule
        if first_session_rule not in FIRST_SESSION_RULES:
            raise ValueError(f"unknown first_session_rule: {first_session_rule}")
        self.d_latent = d_latent
        self.first_session_rule = first_session_rule
        self.gate = nn.Linear(d_latent * 2, d_latent)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def update(self, h_prev: Tensor | None, q_new: Tensor) -> Tensor:
        """Start with q1 by default, then apply the recurrent gate from session 2."""
        q_new = q_new.to(self.gate.weight)
        if h_prev is None:
            if self.first_session_rule == "direct":
                return q_new
            h_prev = torch.zeros_like(q_new)
        if h_prev.shape != q_new.shape:
            raise ValueError("memory shape does not match the session compiler")
        return self(h_prev.to(q_new), q_new)

    def forward(self, h_prev: Tensor, q_new: Tensor) -> Tensor:
        """
        h_prev, q_new: [*, d_latent] (arbitrary leading dims, e.g. [32, 1, 8, 512])
        returns: h_new [*, d_latent]
        """
        z = torch.sigmoid(self.gate(torch.cat([h_prev, q_new], dim=-1)))
        return z * h_prev + (1 - z) * q_new


def run_cmp_sessions(gate: CMPGate, lora_embs: list[Tensor]) -> Tensor:
    """Recursively merge a sequence of session lora_embs through the CMP gate.

    lora_embs: list of [1, n_layers, n_modules, r, d_latent] tensors
    returns: h [1, n_layers, n_modules, r, d_latent] — final merged state
    """
    if not lora_embs:
        raise ValueError("at least one session embedding is required")
    h = None
    for emb in lora_embs:
        h = gate.update(h, emb)
    return h
