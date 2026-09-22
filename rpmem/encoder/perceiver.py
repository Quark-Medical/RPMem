# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perceiver resampler (pure nn.Module, no PreTrainedModel inheritance).

Modified from the Idefics2 implementation distributed with Doc-to-LoRA:
pure nn.Module interfaces, local configuration, and an optional SDPA attention path.
See THIRD_PARTY_NOTICES.md for source attribution and license scope.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rpmem.encoder.perceiver_config import PerceiverBlockConfig

try:
    from flash_attn.bert_padding import unpad_input
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

ACT2FN = {
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PerceiverMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class PerceiverAttention(nn.Module):
    """Cross/Self attention for Perceiver, with SDPA fallback."""

    def __init__(self, config: PerceiverBlockConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.is_causal = False
        self.use_flash_attn = (
            FLASH_ATTN_AVAILABLE
            and torch.cuda.is_available()
            and os.getenv("RPMEM_PERCEIVER_FLASH_ATTN", os.getenv("MEMLORA_PERCEIVER_FLASH_ATTN", "0")) == "1"
        )

    def _sdpa_forward(self, latents: Tensor, kv_inp: Tensor) -> Tensor:
        """SDPA fallback path (no flash_attn dependency)."""
        bsz, q_len, _ = latents.shape
        kv_len = kv_inp.shape[1]

        q = self.q_proj(latents).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_inp).view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_inp).view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attention_dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_out)

    def _flash_forward(self, latents: Tensor, kv_inp: Tensor, **kwargs) -> Tensor:
        """Flash Attention 2 path."""
        bsz, q_len, _ = latents.shape

        query_states = self.q_proj(latents).view(*latents.shape[:2], self.num_heads, self.head_dim)
        key_states = self.k_proj(kv_inp).view(*kv_inp.shape[:2], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(kv_inp).view(*kv_inp.shape[:2], self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states,
            kwargs.get("attention_mask"),
            q_len,
            dropout=dropout_rate,
            position_ids=kwargs.get("position_ids"),
            sliding_window=None,
            is_causal=self.is_causal,
            use_top_left_mask=False,
            **{k: v for k, v in kwargs.items() if k.startswith("cu_seq_lens") or k.startswith("max_length")},
        )
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        return self.o_proj(attn_output)

    def forward(self, latents: Tensor, is_cross_attn: bool, context: Optional[Tensor] = None, **kwargs) -> Tensor:
        kv_inp = context if is_cross_attn else latents
        if self.use_flash_attn:
            return self._flash_forward(latents, kv_inp, **kwargs)
        return self._sdpa_forward(latents, kv_inp)


class PerceiverLayer(nn.Module):
    def __init__(self, config: PerceiverBlockConfig, is_cross_attn: bool):
        super().__init__()
        self.is_cross_attn = is_cross_attn
        self.input_latents_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_context_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if is_cross_attn
            else nn.Identity()
        )
        self.self_attn = PerceiverAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = PerceiverMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * config.intermediate_size_factor,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
        )

    def forward(self, latents: Tensor, context: Tensor, **kwargs) -> Tensor:
        residual = latents
        latents = self.input_latents_layernorm(latents)
        context = self.input_context_layernorm(context)

        attn_out = self.self_attn(
            latents=latents,
            is_cross_attn=self.is_cross_attn,
            context=context,
            **kwargs,
        )
        latents = self.post_attention_layernorm(attn_out)
        latents = residual + latents

        residual = latents
        latents = self.pre_ff_layernorm(latents)
        latents = self.mlp(latents)
        latents = self.post_ff_layernorm(latents)
        latents = residual + latents
        return latents


class PerceiverResampler(nn.Module):
    """Perceiver Resampler: compresses variable-length context into fixed latents."""

    def __init__(self, config: PerceiverBlockConfig):
        super().__init__()
        self.config = config
        self.n_latents = config.n_latents
        self.hidden_size = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_self_attn_per_block = config.num_self_attn_per_block
        self.shared_weights = config.shared_weights

        self.latents_q = nn.Parameter(torch.randn(self.n_latents, self.hidden_size))

        first_x_attn = [PerceiverLayer(config, is_cross_attn=True)]
        first_self_attn_block = [
            PerceiverLayer(config, is_cross_attn=False)
            for _ in range(config.num_self_attn_per_block)
        ]
        self.layers = nn.ModuleList(first_x_attn + first_self_attn_block)

        for layer_idx in range(1, config.num_blocks):
            if self.shared_weights:
                if layer_idx == 1:
                    second_x_attn = PerceiverLayer(config, is_cross_attn=True)
                x_attn = second_x_attn
            else:
                x_attn = PerceiverLayer(config, is_cross_attn=True)
            self.layers.append(x_attn)

            for i in range(config.num_self_attn_per_block):
                if self.shared_weights:
                    self_attn = first_self_attn_block[i]
                else:
                    self_attn = PerceiverLayer(config, is_cross_attn=False)
                self.layers.append(self_attn)

        self.layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.use_flash_attn = (
            FLASH_ATTN_AVAILABLE
            and torch.cuda.is_available()
            and os.getenv("RPMEM_PERCEIVER_FLASH_ATTN", os.getenv("MEMLORA_PERCEIVER_FLASH_ATTN", "0")) == "1"
        )

    def forward(
        self,
        context: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if position_ids is None:
            bsz = context.shape[0]
        else:
            bsz = torch.where(position_ids == 0, 1, 0).sum()

        latents = self.latents_q.unsqueeze(0).expand(bsz, -1, -1)
        compressed_context = latents

        if self.use_flash_attn:
            cu_seq_lens_q = (
                torch.tensor([self.n_latents] * (bsz + 1), device=context.device, dtype=torch.int32)
                * torch.arange(bsz + 1, device=context.device, dtype=torch.int32)
            )
            max_length_q = self.n_latents

            if attention_mask is not None:
                context, _, cu_seq_lens_k, max_length_k, _ = unpad_input(context, attention_mask)
                context = context.unsqueeze(0)
                position_ids = True
            elif position_ids is not None:
                position_ids_flat = position_ids.flatten()
                indices = torch.arange(position_ids_flat.size(0), device=context.device, dtype=torch.int32)
                cu_seq_lens_k = torch.cat([
                    indices[position_ids_flat == 0],
                    torch.tensor(position_ids_flat.size(), device=context.device, dtype=torch.int32),
                ])
                max_length_k = position_ids_flat.max() + 1
            else:
                raise ValueError("either position_ids or attention_mask is required")

            x_attn_kwargs = dict(
                position_ids=position_ids,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
            )
            self_attn_position_ids = torch.arange(self.n_latents, device=context.device, dtype=torch.int32).repeat(1, bsz)
            self_attn_kwargs = dict(
                position_ids=self_attn_position_ids,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_q,
                max_length_q=max_length_q,
                max_length_k=max_length_q,
            )
        else:
            x_attn_kwargs = {}
            self_attn_kwargs = {}

        for layer in self.layers:
            kwargs = x_attn_kwargs if layer.is_cross_attn else self_attn_kwargs
            compressed_context = layer(
                latents=compressed_context,
                context=context,
                **kwargs,
            )

        return self.layernorm(compressed_context)


class Perceiver(nn.Module):
    """Full Perceiver: modality projection → encoder → decoder.

    Produces lora_emb of shape [bs, n_layers, n_modules, r, d_latent].
    """

    def __init__(
        self,
        encoder_config: PerceiverBlockConfig,
        decoder_config: PerceiverBlockConfig,
        num_layers: int,
        num_modules: int = 1,
        lora_r: int = 8,
        per_rank_gen: bool = True,
        layer_to_layer: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if per_rank_gen else 1
        self.layer_to_layer = layer_to_layer

        self.modality_projection = PerceiverMLP(
            hidden_size=encoder_config.input_size,
            intermediate_size=encoder_config.intermediate_size_factor * encoder_config.input_size,
            output_size=encoder_config.hidden_size,
            hidden_act=encoder_config.hidden_act,
        )
        self.encoder = PerceiverResampler(encoder_config)
        self.decoder = PerceiverResampler(decoder_config)

    def forward(
        self,
        ctx_features: Tensor,
        ctx_attn_mask: Optional[Tensor] = None,
        ctx_position_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, None]:
        """
        ctx_features: [bs, seq_len, d] or [bs, n_layers, seq_len, d] (layer_to_layer)
        returns: lora_emb [bs, n_layers, n_modules, r, d_latent], None
        """
        if ctx_position_ids is None:
            bsz = ctx_features.shape[0]
        else:
            bsz = torch.where(ctx_position_ids == 0, 1, 0).sum()

        if self.layer_to_layer:
            if ctx_attn_mask is not None:
                ctx_attn_mask = ctx_attn_mask.repeat(self.num_layers, 1)
                ctx_features = ctx_features.reshape(
                    self.num_layers * bsz, ctx_features.shape[2], ctx_features.shape[3]
                )
            elif ctx_position_ids is not None:
                ctx_position_ids = ctx_position_ids.repeat(1, self.num_layers)
                ctx_features = ctx_features.reshape(
                    1, self.num_layers * ctx_features.shape[2], ctx_features.shape[3]
                )

        projected = self.modality_projection(ctx_features)
        latents = self.encoder(projected, ctx_attn_mask, ctx_position_ids)

        latent_position_ids = torch.arange(
            self.encoder.n_latents, device=ctx_features.device
        ).unsqueeze(0).tile(1, bsz if not self.layer_to_layer else bsz * self.num_layers)

        x = self.decoder(latents, position_ids=latent_position_ids)

        if self.layer_to_layer:
            per_layer_size = self.num_modules * self.r
            x = x.reshape(self.num_layers, bsz, per_layer_size, x.shape[-1])
            x = x.permute(1, 0, 2, 3)  # [bs, n_layers, per_layer_size, d]
            lora_x = x.reshape(bsz, self.num_layers, self.num_modules, self.r, x.shape[-1])
        else:
            total_lora = self.num_layers * self.num_modules * self.r
            lora_x = x[:, :total_lora].reshape(
                bsz, self.num_layers, self.num_modules, self.r, x.shape[-1]
            )

        return lora_x, None
