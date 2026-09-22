"""LoRA forward pass and dynamic injection (no PEFT dependency)."""

from functools import partial
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rpmem.lora.module_finder import target_module_path
from rpmem.utils import get_layers


def lora_forward(
    x: Tensor,
    n_qs: Tensor,
    tot_q: int,
    A: Tensor,
    B: Tensor,
    lora_dropout_p: float,
    scaling: float,
    self: nn.Linear,
    *args,
    **kwargs,
) -> Tensor:
    """LoRA-augmented linear forward.

    x: [tot_q, seq_len, d_in] or [tot_q * seq_len, d_in]
    A: [n_ctx, r, d_in]
    B: [n_ctx, r, d_out]
    """
    n_qs = n_qs.to(A.device)
    A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    base_out = nn.Linear.forward(self, x, *args, **kwargs)
    if x.ndim == 2:
        if tot_q <= 0 or x.shape[0] % tot_q != 0:
            raise ValueError(
                "flattened LoRA input cannot be partitioned across queries: "
                f"rows={x.shape[0]} tot_q={tot_q}"
            )
        lora_input = x.reshape(tot_q, x.shape[0] // tot_q, x.shape[-1])
    elif x.ndim == 3:
        if x.shape[0] != tot_q:
            raise ValueError(
                "LoRA input batch does not match query count: "
                f"batch={x.shape[0]} tot_q={tot_q}"
            )
        lora_input = x
    else:
        raise ValueError(f"LoRA input must be rank 2 or 3, got rank {x.ndim}")

    lora_input = lora_input.to(A.dtype)
    delta_x = F.dropout(lora_input, p=lora_dropout_p, training=self.training)
    delta_x = torch.einsum("bsi, bri -> bsr", delta_x, A)
    delta_x = torch.einsum("bsr, bro -> bso", delta_x, B)
    delta_x = (delta_x * scaling).reshape(base_out.shape)
    return (base_out + delta_x).to(base_out.dtype)


def apply_lora(
    model: nn.Module,
    layer_indices: list[int],
    generated_loras: dict[str, dict[str, Tensor]],
    n_qs: Tensor,
) -> None:
    """Apply generated LoRA weights to model layers via partial forward binding.

    generated_loras: {module_name: {"A": [n_ctx, n_layers, r, d_in], "B": [n_ctx, n_layers, r, d_out]}}
    """
    layers = get_layers(model)
    tot_q = n_qs.sum().item()
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for mname in generated_loras:
            module = attrgetter(target_module_path(mname))(layer)
            A = generated_loras[mname]["A"][:, layer_idx].to(module.weight.device)
            B = generated_loras[mname]["B"][:, layer_idx].to(module.weight.device)
            if not hasattr(module, "_rpmem_base_forward"):
                raise RuntimeError("prepare the reader with patch_for_training before applying memory")
            module.forward = partial(
                lora_forward, self=module,
                lora_dropout_p=module._rpmem_lora_dropout,
                scaling=module._rpmem_lora_alpha,
                n_qs=n_qs, tot_q=tot_q, A=A, B=B,
            )


def patch_for_training(
    model: nn.Module,
    layer_indices: list[int],
    target_modules: list[str],
    lora_dropout: float = 0.0,
    lora_alpha: float = 32.0,
) -> None:
    """Prepare dynamic injection, leaving base inference usable without memory."""
    layers = get_layers(model)
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for name in target_modules:
            module = attrgetter(target_module_path(name))(layer)
            if not hasattr(module, "_rpmem_base_forward"):
                module._rpmem_base_forward = module.forward
            module._rpmem_lora_dropout = lora_dropout
            module._rpmem_lora_alpha = lora_alpha


def reset_lora_hooks(
    model: nn.Module,
    layer_indices: list[int],
    target_modules: list[str],
    lora_dropout: float = 0.0,
    lora_alpha: float = 32.0,
) -> None:
    """Restore original forwards; subsequent base inference needs no LoRA args."""
    layers = get_layers(model)
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for name in target_modules:
            module = attrgetter(target_module_path(name))(layer)
            if hasattr(module, "_rpmem_base_forward"):
                module.forward = module._rpmem_base_forward
