"""LoRA merging utilities for combining multi-chunk LoRA adapters."""

import torch
from torch import Tensor


def compute_rank(n_lora: int, rank: int) -> int:
    return (n_lora + 1) * rank


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Tensor,
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Tensor | None = None,
    bias_scaler: float | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Combine multi-chunk LoRA adapters by concatenating along rank dimension.

    generated_loras: {module: {"A": [tot_chunks, n_layers, r, d], "B": [tot_chunks, n_layers, r, d]}}
    n_chunks: [n_ctx] — number of chunks per context
    """
    total_chunks = int(n_chunks.sum())
    if bias_scaler is None:
        bias_scaler = 1

    first_module = next(iter(generated_loras))
    sampled_lora = generated_loras[first_module]["A"]
    base_rank = sampled_lora.shape[-2]
    device = sampled_lora.device
    dtype = sampled_lora.dtype
    max_rank_needed = int(compute_rank(n_chunks.max(), base_rank))

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras
    }
    rank_dim = 2
    num_groups = len(n_chunks)
    rank_per_group = (n_chunks * base_rank).tolist()
    bias_tensor = None

    for module_name, module_loras in generated_loras.items():
        for matrix_key in ("A", "B"):
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key]
            loras = module_loras[matrix_key]
            if (scalers is not None) and (matrix_key == "A"):
                loras = loras * scalers[:, None, None, None]

            # Equivalent to einops: rearrange(loras, "C L R D -> 1 L (C R) D")
            # Must transpose C and L first so chunks are grouped per-layer
            flat_loras = loras.transpose(0, 1).contiguous().reshape(
                1, loras.shape[1], -1, loras.shape[-1]
            )
            per_group_deltas = flat_loras.split(rank_per_group, dim=rank_dim)

            combined_shape = [num_groups, *per_group_deltas[0].shape[1:]]
            combined_shape[rank_dim] = max_rank_needed
            combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

            for g, deltas in enumerate(per_group_deltas):
                combined_rank = deltas.shape[rank_dim]
                combined[g, :, :combined_rank, :] = deltas

                if bias_tensor is not None:
                    combined[g, :, combined_rank:combined_rank + base_rank, :] = (
                        bias_tensor * bias_scaler
                    )

            combined_loras[module_name][matrix_key] = combined

    return combined_loras
