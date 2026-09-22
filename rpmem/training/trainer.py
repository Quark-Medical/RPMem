"""Unified CMP Gate trainer supporting multiple loss functions."""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from rpmem.gate import CMPGate, run_cmp_sessions
from rpmem.head.hypernet_head import HyperLoRAHead
from rpmem.lora.injection import apply_lora, patch_for_training, reset_lora_hooks
from rpmem.lora.merger import combine_lora
from rpmem.utils import get_layers


def step_gate_optimizer(gate, optimizer, loss, *, num_sessions: int, max_grad_norm: float):
    """Return grad norm, or None when direct q1 bypasses the Gate entirely."""
    optimizer.zero_grad(set_to_none=True)
    if gate.first_session_rule == "direct" and num_sessions == 1:
        return None
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), max_grad_norm)
    optimizer.step()
    return float(grad_norm)


class CMPTrainer:
    """Unified CMP Gate trainer.

    Handles the full training loop: gate forward → head → LoRA → base model → loss → backprop.
    Only gate parameters are updated.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        head: HyperLoRAHead,
        gate: CMPGate,
        layer_indices: list[int],
        target_modules: list[str],
        lora_dropout: float = 0.0,
        lora_alpha: float = 32.0,
        use_bias: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.head = head
        self.gate = gate
        self.layer_indices = layer_indices
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self.use_bias = use_bias
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.base_model.requires_grad_(False)
        self.head.requires_grad_(False)
        self.gate.requires_grad_(True)
        patch_for_training(
            base_model, layer_indices, target_modules, lora_dropout, lora_alpha,
        )

        self.optimizer = torch.optim.AdamW(
            gate.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self._scheduler = None
        self.optimizer_updates = 0
        self.single_session_skips = 0

    def setup_scheduler(self, total_steps: int):
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps,
        )

    def gate_forward(self, embs: list[Tensor]) -> dict[str, dict[str, Tensor]]:
        """CMP Gate merge → Head → lora_dict (with gradients)."""
        h = run_cmp_sessions(self.gate, embs)

        lora_dict = self.head(h)
        return lora_dict

    def apply_and_forward(
        self,
        lora_dict: dict[str, dict[str, Tensor]],
        input_ids: Tensor,
    ) -> Tensor:
        """Apply LoRA and run base model forward.

        Returns: logits [bs, seq_len, vocab_size]
        """
        n_qs = torch.tensor([input_ids.shape[0]], device=self.device)
        lora_combined = combine_lora(
            lora_dict,
            torch.ones(1, dtype=torch.long, device=self.device),
            lora_bias=self.head.get_head_bias() if self.use_bias else None,
        )
        try:
            apply_lora(self.base_model, self.layer_indices, lora_combined, n_qs)
            return self.base_model(input_ids=input_ids.to(self.device)).logits
        finally:
            reset_lora_hooks(
                self.base_model, self.layer_indices, self.target_modules,
                self.lora_dropout, self.lora_alpha,
            )

    def train_step(
        self,
        embs: list[Tensor],
        input_ids: Tensor,
        loss_fn: Callable[[Tensor], Tensor],
    ) -> float:
        """Single training step: forward + backward + optimizer step.

        Args:
            embs: List of session embeddings
            input_ids: Input token IDs for the base model
            loss_fn: Function that takes logits and returns scalar loss

        Returns: loss value (float)
        """
        lora_dict = self.gate_forward(embs)
        logits = self.apply_and_forward(lora_dict, input_ids)
        loss = loss_fn(logits)

        grad_norm = step_gate_optimizer(
            self.gate, self.optimizer, loss,
            num_sessions=len(embs), max_grad_norm=self.max_grad_norm,
        )
        if grad_norm is None:
            self.single_session_skips += 1
        else:
            self.optimizer_updates += 1
            if self._scheduler:
                self._scheduler.step()
        torch.cuda.empty_cache()

        return loss.item()

    def train_epoch(
        self,
        samples: list[dict[str, Any]],
        build_inputs_fn: Callable[[dict], tuple[Tensor, Callable]],
    ) -> float:
        """Train one epoch.

        Args:
            samples: List of training samples (each has "embs" key)
            build_inputs_fn: Function(sample) → (input_ids, loss_fn)

        Returns: average epoch loss
        """
        self.gate.train()
        random.shuffle(samples)
        total_loss = 0.0
        n = 0

        for sample in samples:
            input_ids, loss_fn = build_inputs_fn(sample)
            if input_ids is None:
                continue
            loss = self.train_step(sample["embs"], input_ids, loss_fn)
            total_loss += loss
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(
        self,
        samples: list[dict[str, Any]],
        build_inputs_fn: Callable[[dict], tuple[Tensor, Callable]],
        metric_fn: Callable[[Tensor, dict], dict] | None = None,
    ) -> dict[str, float]:
        """Evaluate on a dataset.

        Args:
            samples: List of evaluation samples
            build_inputs_fn: Function(sample) → (input_ids, loss_fn)
            metric_fn: Optional function(logits, sample) → metrics dict

        Returns: dict with "loss" and any custom metrics
        """
        self.gate.eval()
        total_loss = 0.0
        n = 0
        all_metrics: dict[str, float] = {}

        for sample in samples:
            input_ids, loss_fn = build_inputs_fn(sample)
            if input_ids is None:
                continue

            lora_dict = self.gate_forward(sample["embs"])
            logits = self.apply_and_forward(lora_dict, input_ids)
            loss = loss_fn(logits)
            total_loss += loss.item()
            n += 1

            if metric_fn:
                metrics = metric_fn(logits, sample)
                for k, v in metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0) + v

            torch.cuda.empty_cache()

        result = {"loss": total_loss / max(n, 1), "n_samples": n}
        result.update(all_metrics)
        return result

    def save_checkpoint(self, path: str, epoch: int, **extra):
        """Save gate checkpoint."""
        torch.save({
            "gate_state_dict": self.gate.state_dict(),
            "d_latent": self.gate.d_latent,
            "epoch": epoch,
            **extra,
            "first_session_rule": self.gate.first_session_rule,
        }, path)
