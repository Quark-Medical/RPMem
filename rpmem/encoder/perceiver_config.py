"""Perceiver configuration (pure dataclass, no transformers dependency)."""

from dataclasses import dataclass


@dataclass
class PerceiverBlockConfig:
    """Configuration for a single Perceiver resampler block."""

    input_size: int
    hidden_size: int = 512
    n_latents: int = 8
    num_blocks: int = 9
    num_self_attn_per_block: int = 0
    shared_weights: bool = False
    intermediate_size_factor: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    n_heads: int = 8
    head_dim: int = 64
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
