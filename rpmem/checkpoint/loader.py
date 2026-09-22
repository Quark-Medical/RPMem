"""Unified checkpoint loading for rpmem."""

import torch

from rpmem.gate import CMPGate, checkpoint_first_session_rule


def load_gate_checkpoint(path: str, device: str = "cpu") -> tuple[CMPGate, dict]:
    """Load a trusted PERMA, PersonaMem-v2, or PrefEval Gate checkpoint.

    Returns: (gate_module, metadata_dict)
    """
    ckpt = torch.load(path, weights_only=False, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Gate checkpoint must contain a dictionary")
    state_key = "gate_state_dict" if "gate_state_dict" in ckpt else "state_dict"
    state = ckpt.get(state_key)
    if not isinstance(state, dict) or not isinstance(state.get("gate.weight"), torch.Tensor):
        raise ValueError("Gate checkpoint is missing its gate.weight tensor")
    weight = state["gate.weight"]
    if weight.ndim != 2 or weight.shape[1] != 2 * weight.shape[0]:
        raise ValueError("Gate weight must have shape [d_latent, 2 * d_latent]")
    # Historical PERMA exports store state_dict and args, without d_latent.
    d_latent = int(ckpt.get("d_latent", weight.shape[0]))
    if d_latent != weight.shape[0]:
        raise ValueError("Gate d_latent metadata does not match its weights")
    init_bias = ckpt.get("init_bias", ckpt.get("args", {}).get("init_bias", -2.0))

    rule = checkpoint_first_session_rule(ckpt)
    gate = CMPGate(d_latent=d_latent, init_bias=init_bias, first_session_rule=rule).to(
        device=device, dtype=weight.dtype)
    gate.load_state_dict(state)

    metadata = {k: v for k, v in ckpt.items() if k != state_key}
    metadata.setdefault("d_latent", d_latent)
    metadata["first_session_rule"] = rule
    return gate, metadata
