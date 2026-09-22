"""RPMemModel: unified interface for loading, encoding, merging, and inference."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from rpmem.checkpoint.loader import load_gate_checkpoint
from rpmem.config import RPMemConfig
from rpmem.encoder.ctx_encoder import PerLayerActivations
from rpmem.encoder.perceiver import Perceiver
from rpmem.gate import CMPGate, run_cmp_sessions
from rpmem.head.hypernet_head import HyperLoRAHead
from rpmem.lora.injection import apply_lora, patch_for_training, reset_lora_hooks
from rpmem.lora.merger import combine_lora


class RPMemModel(nn.Module):
    """Unified RPMem model: encodes sessions, merges via CMP Gate, applies LoRA."""

    def __init__(
        self,
        config: RPMemConfig,
        base_model: nn.Module,
        ctx_encoder: PerLayerActivations,
        perceiver: Perceiver,
        head: HyperLoRAHead,
        gate: CMPGate,
    ):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.ctx_encoder = ctx_encoder
        self.perceiver = perceiver
        self.head = head
        self.gate = gate
        self._tokenizer = None
        self._ctx_tokenizer = None
        patch_for_training(
            self.base_model, config.layer_indices, config.lora.target_modules,
            config.lora.lora_dropout, config.lora.lora_alpha,
        )

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        gate_ckpt_path: Optional[str] = None,
        device: str = "cuda",
        use_flash_attn: bool = True,
        base_model_path: Optional[str] = None,
        ctx_encoder_path: Optional[str] = None,
        first_session_rule: Optional[str] = None,
    ) -> "RPMemModel":
        """Load trusted compiler/gate checkpoints in evaluation mode.

        Omitting gate_ckpt_path creates an untrained gate, not the paper model.
        Path overrides support checkpoints produced on another machine.
        Fresh gates default to direct initialization. Saved gates restore their
        recorded rule (legacy zero-state if absent); an explicit rule must match.
        """
        from rpmem.training.hypernet_model import HypernetModel

        compiler = HypernetModel.from_checkpoint(
            checkpoint_path, use_flash_attn=use_flash_attn, train=False,
            base_model_path=base_model_path, ctx_encoder_path=ctx_encoder_path,
        )
        config = compiler.config
        gate = CMPGate(d_latent=config.gate.d_latent, init_bias=config.gate.init_bias,
                       first_session_rule=first_session_rule or "direct")
        if gate_ckpt_path:
            gate, _ = load_gate_checkpoint(gate_ckpt_path, device="cpu")
            if first_session_rule is not None and first_session_rule != gate.first_session_rule:
                raise ValueError("first_session_rule does not match the saved Gate")
            if gate.d_latent != config.gate.d_latent:
                raise ValueError("Gate latent dimension does not match the compiler")
        config.gate.first_session_rule = gate.first_session_rule
        model = cls(
            config, compiler.base_model, compiler.ctx_encoder,
            compiler.perceiver, compiler.head, gate,
        )
        return model.to(device).eval()

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        return self._tokenizer

    def get_ctx_tokenizer(self):
        if self._ctx_tokenizer is None:
            self._ctx_tokenizer = AutoTokenizer.from_pretrained(self.config.ctx_encoder_model_name)
        return self._ctx_tokenizer

    @torch.no_grad()
    def encode_session(self, text: str, max_tokens: int = 4096) -> Tensor:
        """Encode a single session text into lora_emb.

        Returns: [1, n_layers, n_modules, r, d_latent]
        """
        if not text.strip() or max_tokens < 1:
            raise ValueError("session text must be nonempty and max_tokens positive")
        tokenizer = self.get_ctx_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=True)
        reader_length = len(self.get_tokenizer().encode(text, add_special_tokens=False))
        if max(len(tokens), reader_length) > max_tokens:
            raise ValueError(
                f"session exceeds token budget {max_tokens}; split it before encoding "
                f"(encoder={len(tokens)}, reader={reader_length})"
            )

        input_ids = torch.tensor([tokens], device=self.device)
        attn_mask = torch.ones_like(input_ids)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" else nullcontext()
        )
        with autocast:
            ctx_features = self.ctx_encoder(input_ids=input_ids, attention_mask=attn_mask)
            lora_emb, _ = self.perceiver(ctx_features, attn_mask, None)

        return lora_emb

    @torch.no_grad()
    def update_memory(self, session: str, memory: Optional[Tensor] = None,
                      max_tokens: int = 4096) -> Tensor:
        """Compile and consolidate one session; state is explicit and user-owned.

        Pass the returned tensor to the next update or to apply_memory. This
        method does not change the memory currently injected into the reader.
        """
        embedding = self.encode_session(session, max_tokens=max_tokens)
        return self.gate.update(memory, embedding)

    def merge_sessions(self, lora_embs: list[Tensor]) -> Tensor:
        """Merge multiple session embeddings via CMP Gate.

        Returns: [1, n_layers, n_modules, r, d_latent]
        """
        return run_cmp_sessions(self.gate, lora_embs)

    def apply_memory(self, merged_emb: Tensor) -> None:
        """Convert merged lora_emb to LoRA weights and inject into base model."""
        if merged_emb.ndim != 5 or merged_emb.shape[0] != 1:
            raise ValueError("apply_memory expects one memory with shape [1, layers, modules, rank, latent]")
        parameter = next(self.head.parameters())
        lora_dict = self.head(merged_emb.to(device=parameter.device, dtype=parameter.dtype))
        n_qs = torch.tensor([1], device=self.device)
        lora_combined = combine_lora(
            lora_dict,
            n_qs,
            lora_bias=self.head.get_head_bias() if self.config.head.use_bias else None,
        )
        apply_lora(
            self.base_model,
            self.config.layer_indices,
            lora_combined,
            n_qs,
        )

    def reset(self) -> None:
        """Remove all LoRA hooks from base model."""
        reset_lora_hooks(
            self.base_model,
            self.config.layer_indices,
            self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            lora_alpha=self.config.lora.lora_alpha,
        )

    def setup_training(self) -> None:
        """Prepare model for CMP Gate training (freeze all except gate)."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.gate.parameters():
            param.requires_grad = True
        patch_for_training(
            self.base_model,
            self.config.layer_indices,
            self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            lora_alpha=self.config.lora.lora_alpha,
        )

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        """Generate one non-thinking chat response with the current memory.

        The reader has one memory bound to one sequence; beam expansion and
        multiple return sequences are not supported by this convenience API.
        """
        if kwargs.get("num_beams", 1) != 1 or kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("RPMemModel.generate supports one sequence without beam expansion")
        tokenizer = self.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("num_beams", 1)
        kwargs.setdefault("num_return_sequences", 1)
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
