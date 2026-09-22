"""Context encoder: wraps a HuggingFace model to collect per-layer activations."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel


class PerLayerActivations(nn.Module):
    """Collects hidden states from specified layers of a HuggingFace model.

    Output: [bs, n_layers, seq_len, d_model] stacked hidden states.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_target_layers: int = 32,
        max_length: int = 8192,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model_name_or_path = model_name_or_path
        self.num_target_layers = num_target_layers
        self.max_length = max_length

        total_layers = self.base_model.config.num_hidden_layers
        self.layer_indices = self._compute_layer_indices(total_layers, num_target_layers)

    def _compute_layer_indices(self, total: int, target: int) -> list[int]:
        """Select evenly-spaced layer indices. Repeats if target > total."""
        if target <= total:
            step = total / target
            return [int(i * step) for i in range(target)]
        # target > total: map each target slot to nearest encoder layer
        return [min(int(i * total / target), total - 1) for i in range(target)]

    @property
    def hidden_size(self) -> int:
        return self.base_model.config.hidden_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        input_ids: [bs, seq_len]
        returns: [bs, n_layers, seq_len, d_model]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of (n_total_layers+1) x [bs, seq, d]

        selected = [hidden_states[i + 1] for i in self.layer_indices]
        return torch.stack(selected, dim=1)  # [bs, n_layers, seq_len, d_model]
