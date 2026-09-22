"""Head building blocks: PerLayerLinear (replaces EinMix) and ResMLPBlock."""

import torch
import torch.nn as nn
from torch import Tensor


class PerLayerLinear(nn.Module):
    """Per-layer linear transform (replaces einops EinMix).

    For n_modules == 1 (the common case):
        weight_shape: [n_layers, d_in, d_out]
        input:  [bs, n_layers, n_modules, r, d_in]
        output: [bs, n_layers, n_modules, r, d_out]

    Semantically equivalent to:
        EinMix("bs n_layers n_modules r d_in -> bs n_layers n_modules r d_out",
               weight_shape="n_layers d_in d_out", ...)
    """

    def __init__(self, n_layers: int, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(n_layers, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_layers, d_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, n_layers, ..., d_in]
        # weight: [n_layers, d_in, d_out]
        # We need to contract d_in and broadcast n_layers
        # x shape: [bs, n_layers, n_modules, r, d_in]
        # result:  [bs, n_layers, n_modules, r, d_out]
        out = torch.einsum("blmri, lio -> blmro", x, self.weight)
        if self.bias is not None:
            out = out + self.bias[None, :, None, None, :]
        return out


class PerLayerLinearMultiModule(nn.Module):
    """Per-layer, per-module linear transform (for n_modules > 1).

    weight_shape: [n_layers, n_modules, d_in, d_out]
    """

    def __init__(self, n_layers: int, n_modules: int, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_layers, n_modules, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_layers, n_modules, d_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, n_layers, n_modules, r, d_in]
        out = torch.einsum("blmri, lmio -> blmro", x, self.weight)
        if self.bias is not None:
            out = out + self.bias[None, :, :, None, :]
        return out


class ResMLPBlock(nn.Module):
    """Residual MLP block: LayerNorm → Linear → SiLU → Linear → LayerNorm + residual."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    """Per-layer ResMLPBlock using PerLayerLinear instead of nn.Linear."""

    def __init__(self, n_layers: int, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            PerLayerLinear(n_layers, input_size, hidden_size, bias=True),
            nn.SiLU(),
            PerLayerLinear(n_layers, hidden_size, output_size, bias=True),
            nn.LayerNorm(output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)
