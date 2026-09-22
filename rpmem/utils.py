"""Utility functions for RPMem (no PEFT dependency)."""

import torch.nn as nn


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Get the transformer layer list from a HuggingFace causal LM model."""
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    for attribute in ("model", "language_model", "text_model"):
        nested = getattr(model, attribute, None)
        if isinstance(nested, nn.Module) and nested is not model:
            try:
                return get_layers(nested)
            except AttributeError:
                continue
    raise AttributeError(
        f"cannot locate transformer layers in model class {type(model).__name__}"
    )


def get_num_layers(model: nn.Module) -> int:
    return len(get_layers(model))
