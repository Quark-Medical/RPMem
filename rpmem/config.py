"""RPMem configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class LoRAConfig:
    """LoRA injection configuration."""

    r: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["down_proj"])


@dataclass
class PerceiverConfig:
    """Perceiver aggregator configuration."""

    input_size: int = 768
    hidden_size: int = 512
    n_latent_queries: int = 8
    num_key_value_heads: int = 8
    num_attention_heads: int = 8
    encoder_num_blocks: int = 9
    encoder_num_self_attn_per_block: int = 0
    decoder_num_blocks: int = 1
    decoder_num_self_attn_per_block: int = 0
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-5


@dataclass
class HeadConfig:
    """HyperLoRA head configuration (ResMLPBlock stack + PerLayerLinear)."""

    d_latent: int = 512
    n_layers: int = 32
    n_modules: int = 1
    r: int = 8
    num_pre_head_layers: int = 4
    dropout_rate: float = 0.0
    use_bias: bool = True
    per_layer_processing: bool = False
    target_modules: list[str] = field(default_factory=lambda: ["down_proj"])
    in_features: dict[str, int] = field(default_factory=dict)
    out_features: dict[str, int] = field(default_factory=dict)


@dataclass
class GateConfig:
    """CMP Gate configuration."""

    d_latent: int = 512
    init_bias: float = -2.0
    first_session_rule: str = "direct"


@dataclass
class RPMemConfig:
    """Top-level configuration aggregating all sub-configs."""

    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ctx_encoder_model_name: str = "answerdotai/ModernBERT-base"
    n_layers: int = 32
    layer_indices: list[int] = field(default_factory=lambda: list(range(32)))
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    def to_dict(self) -> dict:
        """Serialize without a Python package path embedded in the checkpoint."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RPMemConfig":
        value = dict(value)
        for name, config_type in (
            ("lora", LoRAConfig), ("perceiver", PerceiverConfig),
            ("head", HeadConfig), ("gate", GateConfig),
        ):
            if name in value and isinstance(value[name], dict):
                value[name] = config_type(**value[name])
        return cls(**value)
