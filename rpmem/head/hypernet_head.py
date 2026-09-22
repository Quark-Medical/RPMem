"""HyperLoRA Head: transforms lora_emb into LoRA weight dictionaries."""

import torch
import torch.nn as nn
from torch import Tensor

from rpmem.config import HeadConfig
from rpmem.head.layers import (
    PerLayerLinear,
    PerLayerLinearMultiModule,
    ResMLPBlock,
    ResMLPBlockPerLayer,
)


class HyperLoRAHead(nn.Module):
    """Adapter decoder: lora_emb → LoRA weight dict.

    Pipeline: ResMLPBlock stack → L2 normalize → PerLayerLinear → split into A/B per module.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        self.target_modules = tuple(sorted(config.target_modules))
        self.n_modules = len(self.target_modules)
        self.d_in = config.in_features
        self.d_out = config.out_features

        if config.per_layer_processing:
            layer_blocks = [
                ResMLPBlockPerLayer(
                    config.n_layers,
                    config.d_latent,
                    config.d_latent * 4,
                    config.d_latent,
                )
                for _ in range(config.num_pre_head_layers)
            ]
        else:
            layer_blocks = [
                ResMLPBlock(
                    input_size=config.d_latent,
                    hidden_size=config.d_latent * 4,
                    output_size=config.d_latent,
                    dropout_rate=config.dropout_rate,
                )
                for _ in range(config.num_pre_head_layers)
            ]
        self.layers = nn.Sequential(*layer_blocks)

        self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

        if self.n_modules == 1:
            self.head = PerLayerLinear(config.n_layers, config.d_latent, self.d_lora)
        else:
            self.head = PerLayerLinearMultiModule(
                config.n_layers, self.n_modules, config.d_latent, self.d_lora
            )

        self.bias_A = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(config.n_layers, config.r, self.d_in[m]))
            for m in self.target_modules
        })
        self.bias_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(config.n_layers, config.r, self.d_out[m]))
            for m in self.target_modules
        })
        self.scaler_A = nn.ParameterDict({
            m: nn.Parameter(torch.ones(1, config.n_layers, config.r, 1))
            for m in self.target_modules
        })
        self.scaler_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, config.n_layers, config.r, 1))
            for m in self.target_modules
        })

    def get_head_bias(self) -> dict[str, dict[str, Tensor]]:
        return {
            m: {"A": self.bias_A[m], "B": self.bias_B[m]}
            for m in self.target_modules
        }

    def forward(self, lora_emb: Tensor) -> dict[str, dict[str, Tensor]]:
        """Transform lora_emb to LoRA weight dict.

        lora_emb: [bs, n_layers, n_modules, r, d_latent]
        returns: {module_name: {"A": [bs, n_layers, r, d_in], "B": [bs, n_layers, r, d_out]}}
        """
        with torch.autocast(device_type=lora_emb.device.type, dtype=torch.bfloat16,
                            enabled=lora_emb.device.type == "cuda"):
            h = self.layers(lora_emb)
            norm = torch.norm(h, dim=-1, keepdim=True)
            h = h / norm
            flat_loras = self.head(h)
        return self._to_lora_dict(flat_loras)

    def _to_lora_dict(self, flat_loras: Tensor) -> dict[str, dict[str, Tensor]]:
        """Split flat LoRA output into per-module A/B matrices with scalers.

        flat_loras: [bs, n_layers, n_modules, r, d_lora]
        """
        lora_dict = {}
        for i, module in enumerate(self.target_modules):
            if self.n_modules == 1:
                lora = flat_loras[:, :, 0]  # [bs, n_layers, r, d_lora]
            else:
                lora = flat_loras[:, :, i]  # [bs, n_layers, r, d_lora]

            d_in_m = self.d_in[module]
            d_out_m = self.d_out[module]
            A = lora[..., :d_in_m]  # [bs, n_layers, r, d_in]
            B = lora[..., d_in_m:d_in_m + d_out_m]  # [bs, n_layers, r, d_out]

            A = A * self.scaler_A[module]
            B = B * self.scaler_B[module]

            lora_dict[module] = {"A": A, "B": B}

        return lora_dict
