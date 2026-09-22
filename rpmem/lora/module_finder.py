"""Module discovery for LoRA injection (replaces PEFT's get_peft_modules)."""

from operator import attrgetter

import torch.nn as nn

from rpmem.utils import get_layers

MODULE_PARENT_MAP = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "qkv_proj": "self_attn",
    "down_proj": "mlp",
    "up_proj": "mlp",
    "gate_proj": "mlp",
    "gate_up_proj": "mlp",
}

# Aliases whose public RPMem target name intentionally differs from the
# backbone attribute name. Qwen3.5 MoE stores routed experts as fused tensors,
# while its always-active shared expert exposes a conventional nn.Linear.
MODULE_PATH_MAP = {
    "shared_expert_down_proj": "mlp.shared_expert.down_proj",
}


def target_module_path(name: str) -> str:
    """Resolve a RPMem target name to its path inside one decoder layer."""

    explicit_path = MODULE_PATH_MAP.get(name)
    if explicit_path is not None:
        return explicit_path
    parent = MODULE_PARENT_MAP.get(name, "")
    return f"{parent}.{name}" if parent else name


def find_target_modules(
    model: nn.Module,
    layer_indices: list[int],
    target_module_names: list[str],
) -> dict[tuple[int, str], nn.Linear]:
    """Locate target nn.Linear modules by name, without PEFT.

    Returns: {(layer_idx, module_name): nn.Linear}
    """
    layers = get_layers(model)
    modules = {}
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for name in target_module_names:
            module = attrgetter(target_module_path(name))(layer)
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"LoRA target {name!r} in layer {layer_idx} must resolve "
                    f"to nn.Linear, found {type(module).__name__}"
                )
            modules[(layer_idx, name)] = module
    return modules


def get_in_out_features(
    model: nn.Module,
    layer_indices: list[int],
    target_module_names: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Get in/out feature dimensions for each target module type."""
    modules = find_target_modules(model, layer_indices[:1], target_module_names)
    in_features = {}
    out_features = {}
    for (_, name), module in modules.items():
        in_features[name] = module.in_features
        out_features[name] = module.out_features
    return in_features, out_features
