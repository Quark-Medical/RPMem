"""HypernetModel: self-contained model for hypernetwork training.

Combines PerLayerActivations, Perceiver, and HyperLoRAHead.

Training pipeline:
  ctx_ids → CtxEncoder → Perceiver → Head → LoRA → apply to base_model → forward → loss

Only the hypernet (Perceiver + Head) is trained; ctx_encoder and base_model are frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from rpmem.config import RPMemConfig
from rpmem.checkpoint.compat import load_compiler_checkpoint
from rpmem.encoder.ctx_encoder import PerLayerActivations
from rpmem.encoder.perceiver import Perceiver
from rpmem.encoder.perceiver_config import PerceiverBlockConfig
from rpmem.head.hypernet_head import HyperLoRAHead
from rpmem.lora.injection import apply_lora, patch_for_training, reset_lora_hooks
from rpmem.lora.merger import combine_lora
from rpmem.training.checkpointing import atomic_torch_save
from rpmem.training.logit_projection import response_position_logits
from rpmem.training.model_compat import load_generation_model
from rpmem.training.objectives import (
    average_token_losses_by_session,
    d2l_teacher_topk_cross_entropy,
    sampled_token_reverse_kl_k3,
    student_support_reverse_kl,
    student_topk_reverse_kl,
    teacher_topk_forward_kl,
)
from rpmem.training.opcd import (
    append_rollout,
    response_mask_from_tokens,
    rotating_query_rows,
)
from rpmem.utils import get_layers


def _generated_lora_device(
    generated_loras: dict[str, dict[str, Tensor]],
) -> torch.device:
    first_lora = next(iter(generated_loras.values()), None)
    if first_lora is None:
        raise ValueError("generated LoRA weights must not be empty")
    reference = first_lora.get("A")
    if not isinstance(reference, Tensor):
        raise ValueError("generated LoRA weights must contain an A tensor")
    return reference.device


class HypernetModel(nn.Module):
    """Self-contained hypernetwork model for training.

    Wraps: ctx_encoder (frozen) + perceiver + head (trained) + base_model (frozen).
    The forward pass encodes context, generates LoRA, applies it, and runs the base model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        ctx_encoder: PerLayerActivations,
        perceiver: Perceiver,
        head: HyperLoRAHead,
        config: RPMemConfig,
        lora_dropout: float = 0.0,
        lora_alpha: float = 32.0,
        l1_reg_coef: float = 0.0,
        use_per_ctx_average_loss: bool = False,
        objective: str = "auto",
        opcd_loss_mode: str = "student_topk_reverse_kl",
        opcd_top_k: int = 32,
        opcd_kl_chunk_size: int = 16,
        opcd_rollout_max_new_tokens: int = 128,
        opcd_queries_per_session: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        self.ctx_encoder = ctx_encoder
        self.perceiver = perceiver
        self.head = head
        self.config = config
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self.l1_reg_coef = l1_reg_coef
        self.use_per_ctx_average_loss = use_per_ctx_average_loss
        if objective not in {"auto", "sft", "offline_fkl", "d2l_topk_ce", "opcd"}:
            raise ValueError(f"Unsupported Phase 1 objective: {objective}")
        self.objective = objective
        if opcd_loss_mode not in {
            "student_topk_reverse_kl",
            "offline_rkl",
            "online_fkl",
            "k3",
            "k3_plus",
        }:
            raise ValueError(f"Unsupported OPCD loss mode: {opcd_loss_mode}")
        self.opcd_loss_mode = opcd_loss_mode
        self.opcd_top_k = opcd_top_k
        self.opcd_kl_chunk_size = opcd_kl_chunk_size
        self.opcd_rollout_max_new_tokens = opcd_rollout_max_new_tokens
        self.opcd_queries_per_session = opcd_queries_per_session
        self.opcd_pad_token_id: int | None = None
        self.opcd_eos_token_ids: tuple[int, ...] = ()
        self.opcd_backend = None
        self.last_loss_metrics: dict[str, Tensor] = {}
        self.trainable_scope = "perceiver_and_head"
        self.checkpoint_metadata: dict[str, Any] = {}

        if opcd_top_k <= 0:
            raise ValueError("opcd_top_k must be positive")
        if opcd_kl_chunk_size <= 0:
            raise ValueError("opcd_kl_chunk_size must be positive")
        if opcd_rollout_max_new_tokens <= 0:
            raise ValueError("opcd_rollout_max_new_tokens must be positive")
        if opcd_queries_per_session <= 0:
            raise ValueError("opcd_queries_per_session must be positive")

        self._freeze_encoder_and_base()
        self._patch_lora_forward()

    @classmethod
    def from_config(
        cls,
        config: RPMemConfig,
        base_model_path: Optional[str] = None,
        ctx_encoder_path: Optional[str] = None,
        use_flash_attn: bool = True,
        base_model_device_map: Optional[str] = None,
        **kwargs,
    ) -> "HypernetModel":
        """Create a new HypernetModel from config (for training from scratch)."""
        model_path = base_model_path or config.base_model_name
        config.base_model_name = model_path
        base_model = load_generation_model(
            model_path,
            use_flash_attn=use_flash_attn,
            device_map=base_model_device_map,
        )
        base_model.requires_grad_(False)

        encoder_path = ctx_encoder_path or config.ctx_encoder_model_name
        config.ctx_encoder_model_name = encoder_path
        ctx_encoder = PerLayerActivations(
            encoder_path,
            num_target_layers=config.n_layers,
        )
        ctx_encoder.requires_grad_(False)

        perceiver = cls._build_perceiver(config)
        head = HyperLoRAHead(config.head)

        return cls(base_model, ctx_encoder, perceiver, head, config, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        base_model_path: Optional[str] = None,
        use_flash_attn: bool = True,
        base_model_device_map: Optional[str] = None,
        train: bool = True,
        ctx_encoder_path: Optional[str] = None,
        **kwargs,
    ) -> "HypernetModel":
        """Load a trusted RPMem compiler checkpoint (including legacy configs)."""
        state_dict = load_compiler_checkpoint(checkpoint_path)

        config = cls._checkpoint_config(state_dict)

        model_path = base_model_path or config.base_model_name
        config.base_model_name = model_path
        base_model = load_generation_model(
            model_path,
            use_flash_attn=use_flash_attn,
            device_map=base_model_device_map,
        )
        base_model.requires_grad_(False)

        encoder_path = ctx_encoder_path or config.ctx_encoder_model_name
        config.ctx_encoder_model_name = encoder_path
        ctx_encoder = PerLayerActivations(
            encoder_path,
            num_target_layers=config.n_layers,
        )
        ctx_encoder.requires_grad_(False)

        perceiver = cls._build_perceiver(config)
        head = HyperLoRAHead(config.head)

        model = cls(base_model, ctx_encoder, perceiver, head, config, **kwargs)

        hypernet_sd = {
            k: v
            for k, v in state_dict.items()
            if k
            not in (
                "config",
                "base_model_name_or_path",
                "hypernet_config",
                "ctx_encoder_args",
            )
        }
        # Remove compile prefixes if present
        cleaned = {}
        for k, v in hypernet_sd.items():
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod.") :]
            cleaned[k] = v

        model._load_hypernet_weights(cleaned)
        checkpoint_metadata = state_dict.get("checkpoint_metadata")
        if isinstance(checkpoint_metadata, dict):
            model.checkpoint_metadata = dict(checkpoint_metadata)
        return model.train(train)

    @staticmethod
    def _build_perceiver(config: RPMemConfig) -> Perceiver:
        pc = config.perceiver
        encoder_cfg = PerceiverBlockConfig(
            input_size=pc.input_size,
            hidden_size=pc.hidden_size,
            n_latents=pc.n_latent_queries,
            num_blocks=pc.encoder_num_blocks,
            num_self_attn_per_block=pc.encoder_num_self_attn_per_block,
            shared_weights=False,
            intermediate_size_factor=4,
            n_heads=pc.num_attention_heads,
            head_dim=pc.hidden_size // pc.num_attention_heads,
            num_key_value_heads=pc.num_key_value_heads,
        )
        n_output_queries = config.head.n_modules * config.head.r
        decoder_cfg = PerceiverBlockConfig(
            input_size=pc.hidden_size,
            hidden_size=pc.hidden_size,
            n_latents=n_output_queries,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            intermediate_size_factor=4,
            n_heads=pc.num_attention_heads,
            head_dim=pc.hidden_size // pc.num_attention_heads,
            num_key_value_heads=pc.num_key_value_heads,
        )
        return Perceiver(
            encoder_config=encoder_cfg,
            decoder_config=decoder_cfg,
            num_layers=config.n_layers,
            num_modules=config.head.n_modules,
            lora_r=config.head.r,
            per_rank_gen=True,
            layer_to_layer=True,
        )

    def _freeze_encoder_and_base(self):
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.ctx_encoder.parameters():
            p.requires_grad = False

    def configure_trainable_scope(self, scope: str) -> dict[str, Any]:
        """Apply and verify the trainable-component contract."""

        if scope not in {"perceiver_and_head", "head_only"}:
            raise ValueError(f"unsupported trainable scope: {scope}")
        self.base_model.requires_grad_(False)
        self.ctx_encoder.requires_grad_(False)
        self.perceiver.requires_grad_(scope == "perceiver_and_head")
        self.head.requires_grad_(True)
        self.trainable_scope = scope
        self.checkpoint_metadata["trainable_scope"] = scope

        report = self.trainable_parameter_report()
        expected = {"head"} if scope == "head_only" else {"perceiver", "head"}
        actual = {
            name.split(".", 1)[0]
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if actual != expected:
            raise RuntimeError(
                "trainable parameter scope differs from contract: "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )
        return report

    def trainable_parameter_report(self) -> dict[str, Any]:
        """Return a serializable parameter-level audit of the trainable scope."""

        modules = {}
        for name in ("base_model", "ctx_encoder", "perceiver", "head"):
            module = getattr(self, name)
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            modules[name] = {"parameters": total, "trainable_parameters": trainable}
        return {
            "format": "memlora_trainable_parameter_report_v1",
            "scope": self.trainable_scope,
            "total_parameters": sum(item["parameters"] for item in modules.values()),
            "trainable_parameters": sum(
                item["trainable_parameters"] for item in modules.values()
            ),
            "modules": modules,
            "trainable_parameter_names": [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ],
        }

    @staticmethod
    def _checkpoint_config(state_dict: dict[str, Any]) -> RPMemConfig:
        config = state_dict.get("config")
        if isinstance(config, RPMemConfig):
            return config
        if isinstance(config, dict):
            return RPMemConfig.from_dict(config)
        raise ValueError("Expected a native compiler checkpoint with a RPMemConfig")

    @staticmethod
    def _perceiver_state_dict(state_dict: dict[str, Any]) -> dict[str, Tensor]:
        """Extract Perceiver tensors from supported checkpoint tensor namespaces."""

        prefixes = (
            "aggregator.perceiver.",
            "perceiver.",
            "aggregator.",
        )
        extracted: dict[str, Tensor] = {}
        for raw_name, value in state_dict.items():
            name = raw_name
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
            for prefix in prefixes:
                if name.startswith(prefix):
                    canonical = name[len(prefix) :]
                    if canonical in extracted:
                        raise ValueError(
                            "checkpoint contains duplicate Perceiver tensor after "
                            f"canonicalization: {canonical}"
                        )
                    extracted[canonical] = value
                    break
        if not extracted:
            raise ValueError("checkpoint contains no Perceiver tensors")
        return extracted

    def initialize_perceiver_from_checkpoint(
        self, checkpoint_path: str | Path
    ) -> dict[str, Any]:
        """Strictly transfer the latent-space Perceiver from another backbone."""

        checkpoint_path = Path(checkpoint_path)
        state_dict = load_compiler_checkpoint(checkpoint_path)
        if not isinstance(state_dict, dict):
            raise TypeError("RPMem checkpoint must contain a dictionary")
        source_config = self._checkpoint_config(state_dict)
        perceiver_state = self._perceiver_state_dict(state_dict)
        expected_state = self.perceiver.state_dict()

        missing = sorted(set(expected_state) - set(perceiver_state))
        unexpected = sorted(set(perceiver_state) - set(expected_state))
        shape_mismatches = {
            name: {
                "source": list(perceiver_state[name].shape),
                "target": list(expected_state[name].shape),
            }
            for name in sorted(set(expected_state) & set(perceiver_state))
            if tuple(perceiver_state[name].shape) != tuple(expected_state[name].shape)
        }
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "source Perceiver is incompatible with the target latent space: "
                f"missing={missing} unexpected={unexpected} "
                f"shape_mismatches={shape_mismatches}"
            )
        self.perceiver.load_state_dict(perceiver_state, strict=True)

        report = {
            "format": "memlora_perceiver_transfer_report_v1",
            "checkpoint": str(checkpoint_path.resolve()),
            "transferred_tensors": len(perceiver_state),
            "transferred_parameters": sum(
                tensor.numel() for tensor in perceiver_state.values()
            ),
            "source": {
                "base_model_name": source_config.base_model_name,
                "n_layers": source_config.n_layers,
                "d_latent": source_config.perceiver.hidden_size,
                "n_latent_queries": source_config.perceiver.n_latent_queries,
                "lora_rank": source_config.head.r,
                "head_modules": source_config.head.n_modules,
            },
            "target": {
                "base_model_name": self.config.base_model_name,
                "n_layers": self.config.n_layers,
                "d_latent": self.config.perceiver.hidden_size,
                "n_latent_queries": self.config.perceiver.n_latent_queries,
                "lora_rank": self.config.head.r,
                "head_modules": self.config.head.n_modules,
            },
        }
        self.checkpoint_metadata["perceiver_transfer"] = report
        return report

    def _patch_lora_forward(self):
        patch_for_training(
            self.base_model, self.config.layer_indices,
            self.config.lora.target_modules, self.lora_dropout, self.lora_alpha,
        )

    def _load_hypernet_weights(self, state_dict: dict):
        """Load perceiver + head weights from a flat state dict."""
        perceiver_sd = {}
        head_sd = {}
        for k, v in state_dict.items():
            if k.startswith("aggregator."):
                perceiver_sd[k[len("aggregator.") :]] = v
            elif k.startswith("perceiver."):
                perceiver_sd[k[len("perceiver.") :]] = v
            elif (
                k.startswith("layers.")
                or k.startswith("head.")
                or k.startswith("bias_")
                or k.startswith("scaler_")
            ):
                head_sd[k] = v

        if perceiver_sd:
            self.perceiver.load_state_dict(perceiver_sd, strict=False)
        if head_sd:
            self._load_head_state_dict(head_sd)

    def _load_head_state_dict(self, sd: dict):
        """Load head weights, handling EinMix → PerLayerLinear mapping."""
        mapped = {}
        for k, v in sd.items():
            if k.startswith("head."):
                mapped[k[len("head.") :]] = v
            elif k.startswith("layers."):
                mapped[k] = v
            elif k.startswith("bias_A.") or k.startswith("bias_B."):
                mapped[k] = v
            elif k.startswith("scaler_A.") or k.startswith("scaler_B."):
                mapped[k] = v
        self.head.load_state_dict(mapped, strict=False)

    @property
    def device(self) -> torch.device:
        return next(self.perceiver.parameters()).device

    @property
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate_weights(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Optional[Tensor] = None,
        ctx_position_ids: Optional[Tensor] = None,
        ctx_features: Optional[Tensor] = None,
    ) -> dict[str, dict[str, Tensor]]:
        """Encode context → Perceiver → Head → LoRA dict.

        ctx_ids: [n_chunks, ctx_len]
        ctx_encoder is always frozen (no_grad), but perceiver+head have gradients during training.
        """
        lora_emb = self.encode_context(
            ctx_ids,
            ctx_attn_mask,
            ctx_position_ids,
            ctx_features=ctx_features,
        )
        lora_dict = self.head(lora_emb)
        return lora_dict

    @torch.no_grad()
    def generate_with_loras(
        self,
        *,
        generated_loras: dict[str, dict[str, Tensor]],
        n_ctx_chunks: Tensor,
        n_queries: Tensor,
        prompt_ids: Tensor,
        prompt_attention_mask: Tensor,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_ids: int | list[int] | tuple[int, ...],
    ) -> Tensor:
        """Generate deterministic validation responses from cached LoRA weights."""

        if prompt_ids.ndim != 2 or prompt_attention_mask.shape != prompt_ids.shape:
            raise ValueError(
                "quality generation prompts must be matching rank-2 tensors"
            )
        if int(n_queries.sum().item()) != prompt_ids.shape[0]:
            raise ValueError("quality generation query counts do not match prompt rows")
        if max_new_tokens <= 0:
            raise ValueError("quality generation max_new_tokens must be positive")
        eos_values = (
            [int(eos_token_ids)]
            if isinstance(eos_token_ids, int)
            else [int(value) for value in eos_token_ids]
        )
        if not eos_values:
            raise ValueError("quality generation requires at least one EOS token ID")

        detached_loras = {
            module: {name: value.detach() for name, value in matrices.items()}
            for module, matrices in generated_loras.items()
        }
        combined_loras = combine_lora(
            detached_loras,
            n_ctx_chunks,
            lora_bias=self.head.get_head_bias() if self.config.head.use_bias else None,
        )
        self._apply_lora_to_layers(combined_loras, n_queries)
        try:
            generated = self.base_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                eos_token_id=eos_values,
                pad_token_id=int(pad_token_id),
                use_cache=True,
                return_dict_in_generate=False,
            )
        finally:
            self._reset_lora_bindings()
        response_ids = generated[:, prompt_ids.shape[1] :]
        if response_ids.shape[1] == 0:
            raise RuntimeError("quality evaluation generated no response tokens")
        return response_ids

    def configure_opcd_token_ids(
        self,
        *,
        pad_token_id: int,
        eos_token_ids: int | list[int] | tuple[int, ...],
    ) -> None:
        """Set tokenizer-specific generation IDs after tokenizer loading."""

        if isinstance(eos_token_ids, int):
            eos_values = (eos_token_ids,)
        else:
            eos_values = tuple(int(value) for value in eos_token_ids)
        if pad_token_id is None or not eos_values:
            raise ValueError("OPCD requires pad and EOS token IDs")
        self.opcd_pad_token_id = int(pad_token_id)
        self.opcd_eos_token_ids = eos_values

    def configure_opcd_backend(self, backend) -> None:
        """Attach a non-module execution backend after distributed rank setup."""

        required = ("score_teacher", "last_metrics")
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise TypeError("OPCD backend is missing: " + ", ".join(missing))
        self.opcd_backend = backend

    def _select_opcd_batch(
        self,
        *,
        n_queries: Tensor,
        sample_indices: Optional[Tensor],
        student_prompt_ids: Optional[Tensor],
        student_prompt_attention_mask: Optional[Tensor],
        teacher_prompt_ids: Optional[Tensor],
        teacher_prompt_attention_mask: Optional[Tensor],
        corpus_pass: int,
    ) -> dict[str, Tensor]:
        required = {
            "sample_indices": sample_indices,
            "student_prompt_ids": student_prompt_ids,
            "student_prompt_attention_mask": student_prompt_attention_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_attention_mask": teacher_prompt_attention_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("OPCD batch is missing: " + ", ".join(missing))

        selected_rows, selected_n_queries, probe_offsets = rotating_query_rows(
            n_queries,
            sample_indices,
            corpus_pass=corpus_pass,
            queries_per_session=self.opcd_queries_per_session,
        )
        return {
            "selected_rows": selected_rows,
            "n_queries": selected_n_queries,
            "probe_offsets": probe_offsets,
            "student_prompt_ids": student_prompt_ids.index_select(0, selected_rows),
            "student_prompt_attention_mask": student_prompt_attention_mask.index_select(
                0, selected_rows
            ),
            "teacher_prompt_ids": teacher_prompt_ids.index_select(0, selected_rows),
            "teacher_prompt_attention_mask": teacher_prompt_attention_mask.index_select(
                0, selected_rows
            ),
        }

    def encode_context(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Optional[Tensor] = None,
        ctx_position_ids: Optional[Tensor] = None,
        *,
        ctx_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode context into the adapter latent space before the LoRA head.

        Returns lora_emb with shape [n_chunks, n_layers, n_modules, r, d_latent].
        The context encoder is frozen, while gradients can still flow through
        the perceiver during phase-1 hypernetwork training.
        """
        if ctx_features is None:
            ctx_features = self.extract_context_features(ctx_ids, ctx_attn_mask)
        elif ctx_features.requires_grad:
            raise ValueError("cached context features must be detached")

        # Perceiver is trainable in phase 1 and frozen explicitly by phase-2 code.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lora_emb, _ = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        return lora_emb

    def extract_context_features(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the frozen context encoder once for reuse within an OPCD step."""

        with torch.no_grad():
            return self.ctx_encoder(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
            ).detach()

    def forward(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Optional[Tensor] = None,
        ctx_position_ids: Optional[Tensor] = None,
        n_ctx_chunks: Optional[Tensor] = None,
        n_queries: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        logprobs_vals: Optional[Tensor] = None,
        logprobs_indices: Optional[Tensor] = None,
        sample_indices: Optional[Tensor] = None,
        opcd_student_prompt_ids: Optional[Tensor] = None,
        opcd_student_prompt_attention_mask: Optional[Tensor] = None,
        opcd_teacher_prompt_ids: Optional[Tensor] = None,
        opcd_teacher_prompt_attention_mask: Optional[Tensor] = None,
        opcd_corpus_pass: int = 0,
        opcd_rollout_seed: Optional[int] = None,
        opcd_rollout_ids: Optional[Tensor] = None,
        opcd_rollout_lengths: Optional[Tensor] = None,
        opcd_ctx_features: Optional[Tensor] = None,
        opcd_support_indices: Optional[Tensor] = None,
        opcd_teacher_support_logprobs: Optional[Tensor] = None,
        opcd_teacher_tail_mass: Optional[Tensor] = None,
        opcd_teacher_sampled_logprobs: Optional[Tensor] = None,
        opcd_teacher_entropy: Optional[Tensor] = None,
        opcd_teacher_response_projection: Optional[Tensor] = None,
        opcd_teacher_service_seconds: Optional[Tensor] = None,
        opcd_teacher_micro_batches: Optional[Tensor] = None,
        opcd_adapter_only: bool = False,
        opcd_sft_bootstrap: bool = False,
        opcd_measure_context_lift: bool = False,
        **kwargs,
    ):
        """Full forward pass: ctx → LoRA → apply → base model.

        If teacher top-K logprobs are provided, train against the in-context
        teacher distribution at response-token positions. Otherwise use hard
        label cross-entropy.

        Returns: (loss, logits, generated_loras) or (logits, generated_loras) if labels is None
        """
        generated_loras = self.generate_weights(
            ctx_ids,
            ctx_attn_mask,
            ctx_position_ids,
            ctx_features=opcd_ctx_features,
        )

        if n_ctx_chunks is None:
            n_ctx_chunks = torch.tensor([ctx_ids.shape[0]], device=ctx_ids.device)
        if n_queries is None:
            n_queries = torch.ones(
                n_ctx_chunks.shape[0], dtype=torch.int32, device=ctx_ids.device
            )

        combined_loras = combine_lora(
            generated_loras,
            n_ctx_chunks,
            lora_bias=self.head.get_head_bias() if self.config.head.use_bias else None,
        )

        objective = self.objective
        if objective == "auto":
            objective = "d2l_topk_ce" if logprobs_vals is not None else "sft"
        if opcd_sft_bootstrap:
            if objective != "opcd":
                raise ValueError("OPCD SFT bootstrap requires objective=opcd")
            objective = "sft"
        if objective == "opcd":
            selected = self._select_opcd_batch(
                n_queries=n_queries,
                sample_indices=sample_indices,
                student_prompt_ids=opcd_student_prompt_ids,
                student_prompt_attention_mask=opcd_student_prompt_attention_mask,
                teacher_prompt_ids=opcd_teacher_prompt_ids,
                teacher_prompt_attention_mask=opcd_teacher_prompt_attention_mask,
                corpus_pass=opcd_corpus_pass,
            )
            if opcd_adapter_only:
                return {
                    "combined_loras": combined_loras,
                    "generated_loras": generated_loras,
                    **selected,
                }
            return self._forward_opcd(
                generated_loras=generated_loras,
                combined_loras=combined_loras,
                ctx_attn_mask=ctx_attn_mask,
                selected=selected,
                rollout_seed=opcd_rollout_seed,
                rollout_ids=opcd_rollout_ids,
                rollout_lengths=opcd_rollout_lengths,
                support_indices=opcd_support_indices,
                teacher_support_logprobs=opcd_teacher_support_logprobs,
                teacher_tail_mass=opcd_teacher_tail_mass,
                teacher_sampled_logprobs=opcd_teacher_sampled_logprobs,
                teacher_entropy=opcd_teacher_entropy,
                teacher_response_projection=opcd_teacher_response_projection,
                teacher_service_seconds=opcd_teacher_service_seconds,
                teacher_micro_batches=opcd_teacher_micro_batches,
                used_cached_ctx_features=opcd_ctx_features is not None,
                measure_context_lift=opcd_measure_context_lift,
            )

        if input_ids is None:
            raise ValueError(f"{objective} requires input_ids")

        self._apply_lora_to_layers(combined_loras, n_queries, position_ids)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if labels is not None:
            objective_metrics: dict[str, Tensor] = {}
            if objective in {"offline_fkl", "d2l_topk_ce"}:
                if logprobs_vals is None:
                    raise ValueError(f"{objective} requires teacher top-k logprobs")
                if logprobs_indices is None:
                    raise ValueError("logprobs_indices is required with logprobs_vals")
                loss, objective_metrics = self._compute_distill_loss(
                    outputs.logits,
                    labels,
                    logprobs_vals,
                    logprobs_indices,
                    objective=objective,
                    position_ids=position_ids,
                    n_queries=n_queries,
                )
            elif objective == "sft":
                loss = self._compute_loss(outputs.logits, labels, n_queries)
            else:
                raise ValueError(f"Objective is not implemented yet: {objective}")
            l1_norm = self._compute_l1_reg(generated_loras)
            loss = loss.to(l1_norm.device)
            total_loss = loss + self.l1_reg_coef * l1_norm
            self.last_loss_metrics = {
                "task_loss": loss.detach(),
                "l1_norm": l1_norm.detach(),
                "l1_loss": (self.l1_reg_coef * l1_norm).detach(),
                "total_loss": total_loss.detach(),
                "distill_mode": torch.tensor(
                    1.0 if objective != "sft" else 0.0,
                    device=total_loss.device,
                ),
                **objective_metrics,
            }
            if opcd_sft_bootstrap:
                response_tokens = (labels != -100).sum().float()
                self.last_loss_metrics.update(
                    {
                        "sft_bootstrap": torch.tensor(
                            1.0, device=total_loss.device
                        ),
                        "selected_queries": n_queries.sum().float(),
                        "rollout_tokens": response_tokens,
                        "context_tokens": (
                            ctx_attn_mask.sum().float()
                            if ctx_attn_mask is not None
                            else torch.tensor(0.0, device=total_loss.device)
                        ),
                        "student_prompt_tokens": (
                            attention_mask.sum().float() - response_tokens
                        ),
                    }
                )
            return total_loss, outputs.logits, generated_loras
        return outputs.logits, generated_loras

    def _forward_opcd(
        self,
        *,
        generated_loras: dict[str, dict[str, Tensor]],
        combined_loras: dict[str, dict[str, Tensor]],
        ctx_attn_mask: Optional[Tensor],
        selected: dict[str, Tensor],
        rollout_seed: Optional[int],
        rollout_ids: Optional[Tensor],
        rollout_lengths: Optional[Tensor],
        support_indices: Optional[Tensor] = None,
        teacher_support_logprobs: Optional[Tensor] = None,
        teacher_tail_mass: Optional[Tensor] = None,
        teacher_sampled_logprobs: Optional[Tensor] = None,
        teacher_entropy: Optional[Tensor] = None,
        teacher_response_projection: Optional[Tensor] = None,
        teacher_service_seconds: Optional[Tensor] = None,
        teacher_micro_batches: Optional[Tensor] = None,
        used_cached_ctx_features: bool = False,
        measure_context_lift: bool,
    ):
        """Roll out the student, then replay teacher and student prefixes."""

        if self.opcd_pad_token_id is None or not self.opcd_eos_token_ids:
            raise ValueError("configure_opcd_token_ids must run before OPCD training")

        selected_n_queries = selected["n_queries"]
        probe_offsets = selected["probe_offsets"]
        student_prompt_ids = selected["student_prompt_ids"]
        student_prompt_attention_mask = selected["student_prompt_attention_mask"]
        teacher_prompt_ids = selected["teacher_prompt_ids"]
        teacher_prompt_attention_mask = selected["teacher_prompt_attention_mask"]

        generated_lengths = rollout_lengths
        if rollout_ids is not None:
            if rollout_ids.ndim != 2:
                raise ValueError("precomputed OPCD rollout IDs must be rank 2")
            if rollout_ids.shape[0] != student_prompt_ids.shape[0]:
                raise ValueError(
                    "precomputed OPCD rollout batch does not match selected queries"
                )
            if generated_lengths is not None and generated_lengths.shape != (
                rollout_ids.shape[0],
            ):
                raise ValueError("one OPCD rollout length is required per response")
        elif self.opcd_backend is None:
            if rollout_seed is None:
                raise ValueError("OPCD requires an explicit deterministic rollout seed")
            detached_loras = {
                module: {name: value.detach() for name, value in matrices.items()}
                for module, matrices in combined_loras.items()
            }
            self._apply_lora_to_layers(detached_loras, selected_n_queries)
            try:
                rollout_ids = self._sample_opcd_rollout(
                    student_prompt_ids,
                    student_prompt_attention_mask,
                    rollout_seed,
                )
            finally:
                self._reset_lora_bindings()
        else:
            if rollout_seed is None:
                raise ValueError("OPCD requires an explicit deterministic rollout seed")
            if not hasattr(self.opcd_backend, "sample_rollouts"):
                raise RuntimeError(
                    "OPCD rollout IDs were not supplied and the backend cannot sample"
                )
            rollout_ids, generated_lengths = self.opcd_backend.sample_rollouts(
                combined_loras=combined_loras,
                n_queries=selected_n_queries,
                prompt_ids=student_prompt_ids,
                prompt_attention_mask=student_prompt_attention_mask,
                seed=rollout_seed,
            )

        rollout_mask, ended_with_eos = response_mask_from_tokens(
            rollout_ids,
            self.opcd_eos_token_ids,
            padded_lengths=generated_lengths,
        )
        student_replay = append_rollout(
            student_prompt_ids,
            student_prompt_attention_mask,
            rollout_ids,
            rollout_mask,
        )
        student_positions = torch.arange(
            student_prompt_ids.shape[1],
            student_prompt_ids.shape[1] + rollout_ids.shape[1],
            device=student_prompt_ids.device,
        )
        teacher_logits = None
        base_logits = None
        teacher_projected: bool | Tensor = False
        context_lift = None
        remote_teacher_targets = any(
            value is not None
            for value in (
                support_indices,
                teacher_support_logprobs,
                teacher_tail_mass,
                teacher_sampled_logprobs,
            )
        )
        use_local_teacher = self.opcd_backend is None and not remote_teacher_targets
        if use_local_teacher:
            teacher_replay = append_rollout(
                teacher_prompt_ids,
                teacher_prompt_attention_mask,
                rollout_ids,
                rollout_mask,
            )
            teacher_positions = torch.arange(
                teacher_prompt_ids.shape[1],
                teacher_prompt_ids.shape[1] + rollout_ids.shape[1],
                device=teacher_prompt_ids.device,
            )
            with torch.no_grad():
                teacher_logits, teacher_projected = response_position_logits(
                    self.base_model,
                    teacher_replay[0],
                    teacher_positions,
                    squeeze_batch=False,
                    attention_mask=teacher_replay[1],
                    position_ids=teacher_replay[2],
                    use_cache=False,
                )
                if measure_context_lift:
                    base_logits, _ = response_position_logits(
                        self.base_model,
                        student_replay[0],
                        student_positions,
                        squeeze_batch=False,
                        attention_mask=student_replay[1],
                        position_ids=student_replay[2],
                        use_cache=False,
                    )
                    context_lift = self._sampled_token_context_lift(
                        teacher_logits,
                        base_logits,
                        rollout_ids,
                        rollout_mask,
                    )
                    del base_logits
        elif measure_context_lift:
            with torch.no_grad():
                base_logits, _ = response_position_logits(
                    self.base_model,
                    student_replay[0],
                    student_positions,
                    squeeze_batch=False,
                    attention_mask=student_replay[1],
                    position_ids=student_replay[2],
                    use_cache=False,
                )

        self._apply_lora_to_layers(combined_loras, selected_n_queries)
        try:
            student_logits, student_projected = response_position_logits(
                self.base_model,
                student_replay[0],
                student_positions,
                squeeze_batch=False,
                attention_mask=student_replay[1],
                position_ids=student_replay[2],
                use_cache=False,
            )
        finally:
            self._reset_lora_bindings()

        valid_student_logits = student_logits[rollout_mask]
        if use_local_teacher:
            valid_teacher_logits = teacher_logits[rollout_mask]
            if self.opcd_loss_mode in {"k3", "k3_plus"}:
                sampled_ids = rollout_ids[rollout_mask]
                teacher_sampled_logprobs = (
                    valid_teacher_logits.float()
                    .log_softmax(dim=-1)
                    .gather(-1, sampled_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                token_losses, objective_metrics = sampled_token_reverse_kl_k3(
                    valid_student_logits,
                    sampled_ids,
                    teacher_sampled_logprobs,
                    use_k2_gradient=self.opcd_loss_mode == "k3_plus",
                    reduction="none",
                )
            elif self.opcd_loss_mode == "online_fkl":
                teacher_logprobs = valid_teacher_logits.float().log_softmax(dim=-1)
                teacher_values, teacher_indices = teacher_logprobs.topk(
                    min(self.opcd_top_k, teacher_logprobs.shape[-1]),
                    dim=-1,
                )
                token_losses, objective_metrics = teacher_topk_forward_kl(
                    valid_student_logits,
                    teacher_values,
                    teacher_indices,
                    reduction="none",
                )
            else:
                token_losses, objective_metrics = student_topk_reverse_kl(
                    valid_student_logits,
                    valid_teacher_logits,
                    top_k=self.opcd_top_k,
                    chunk_size=self.opcd_kl_chunk_size,
                    reduction="none",
                )
        elif self.opcd_loss_mode in {"k3", "k3_plus"}:
            if teacher_sampled_logprobs is None:
                raise ValueError("K3 OPCD requires teacher sampled-token logprobs")
            if any(
                value is not None
                for value in (
                    support_indices,
                    teacher_support_logprobs,
                    teacher_tail_mass,
                )
            ):
                raise ValueError("K3 OPCD does not accept top-k support targets")
            sampled_ids = rollout_ids[rollout_mask]
            if teacher_sampled_logprobs.shape != sampled_ids.shape:
                raise ValueError(
                    "cached K3 teacher targets differ from student replay: "
                    f"{tuple(teacher_sampled_logprobs.shape)} != "
                    f"{tuple(sampled_ids.shape)}"
                )
            token_losses, objective_metrics = sampled_token_reverse_kl_k3(
                valid_student_logits,
                sampled_ids,
                teacher_sampled_logprobs,
                use_k2_gradient=self.opcd_loss_mode == "k3_plus",
                reduction="none",
            )
            if teacher_entropy is not None:
                objective_metrics["teacher_entropy"] = teacher_entropy.mean()
            if teacher_response_projection is not None:
                teacher_projected = teacher_response_projection
            if measure_context_lift:
                valid_base_logits = base_logits[rollout_mask].float()
                base_sampled_logprobs = (
                    valid_base_logits.log_softmax(dim=-1)
                    .gather(-1, sampled_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                context_lift = (teacher_sampled_logprobs - base_sampled_logprobs).mean()
                del base_logits
        elif self.opcd_loss_mode == "online_fkl":
            if support_indices is None or teacher_support_logprobs is None:
                raise ValueError(
                    "online FKL requires teacher top-k indices and logprobs"
                )
            if teacher_tail_mass is not None:
                raise ValueError(
                    "online FKL derives the teacher tail from teacher top-k targets"
                )
            token_losses, objective_metrics = teacher_topk_forward_kl(
                valid_student_logits,
                teacher_support_logprobs,
                support_indices,
                reduction="none",
            )
            teacher_projected = (
                teacher_response_projection
                if teacher_response_projection is not None
                else torch.tensor(False, device=valid_student_logits.device)
            )
            if base_logits is not None:
                del base_logits
        else:
            support_width = min(self.opcd_top_k, valid_student_logits.shape[-1])
            cached_target_values = (
                teacher_support_logprobs,
                teacher_tail_mass,
                teacher_sampled_logprobs,
                teacher_response_projection,
            )
            has_cached_targets = support_indices is not None
            if has_cached_targets and any(
                value is None for value in cached_target_values
            ):
                raise ValueError("cached OPCD teacher targets are incomplete")
            if not has_cached_targets and any(
                value is not None for value in cached_target_values
            ):
                raise ValueError("cached OPCD teacher targets require support indices")
            support_overlap = None
            if has_cached_targets:
                expected_support_shape = (
                    valid_student_logits.shape[0],
                    support_width,
                )
                if support_indices.shape != expected_support_shape:
                    raise ValueError(
                        "cached OPCD support shape differs from student replay: "
                        f"{tuple(support_indices.shape)} != "
                        f"{expected_support_shape}"
                    )
                overlap_rows = min(64, valid_student_logits.shape[0])
                overlap_indices = torch.linspace(
                    0,
                    valid_student_logits.shape[0] - 1,
                    steps=overlap_rows,
                    device=valid_student_logits.device,
                ).long()
                replay_support_sample = (
                    valid_student_logits.detach()
                    .index_select(0, overlap_indices)
                    .float()
                    .topk(support_width, dim=-1)
                    .indices
                )
                support_overlap = (
                    support_indices.index_select(0, overlap_indices)
                    .unsqueeze(-1)
                    .eq(replay_support_sample.unsqueeze(-2))
                    .any(dim=-1)
                    .float()
                    .mean()
                )
                teacher_targets = {
                    "support_logprobs": teacher_support_logprobs,
                    "tail_mass": teacher_tail_mass,
                    "sampled_logprobs": teacher_sampled_logprobs,
                    "response_projection": teacher_response_projection,
                }
                if teacher_entropy is not None:
                    teacher_targets["teacher_entropy"] = teacher_entropy
            else:
                support_indices = (
                    valid_student_logits.detach()
                    .float()
                    .topk(support_width, dim=-1)
                    .indices
                )
                teacher_targets = self.opcd_backend.score_teacher(
                    prompt_ids=teacher_prompt_ids,
                    prompt_attention_mask=teacher_prompt_attention_mask,
                    rollout_ids=rollout_ids,
                    rollout_mask=rollout_mask,
                    support_indices=support_indices,
                )
            token_losses, objective_metrics = student_support_reverse_kl(
                valid_student_logits,
                support_indices,
                teacher_targets["support_logprobs"],
                teacher_targets["tail_mass"],
                chunk_size=self.opcd_kl_chunk_size,
                reduction="none",
            )
            if "teacher_entropy" in teacher_targets:
                objective_metrics["teacher_entropy"] = teacher_targets[
                    "teacher_entropy"
                ].mean()
            if support_overlap is not None:
                objective_metrics["student_support_topk_overlap"] = support_overlap
            teacher_projected = teacher_targets["response_projection"]
            if measure_context_lift:
                valid_base_logits = base_logits[rollout_mask].float()
                sampled_ids = rollout_ids[rollout_mask].unsqueeze(-1)
                base_sampled_logprobs = (
                    valid_base_logits.log_softmax(dim=-1)
                    .gather(-1, sampled_ids)
                    .squeeze(-1)
                )
                context_lift = (
                    teacher_targets["sampled_logprobs"] - base_sampled_logprobs
                ).mean()
                del base_logits
        if self.use_per_ctx_average_loss:
            response_labels = rollout_ids.masked_fill(~rollout_mask, -100)
            response_positions = torch.arange(
                rollout_ids.shape[1], device=rollout_ids.device
            ).expand_as(rollout_ids)
            loss = average_token_losses_by_session(
                token_losses,
                response_labels,
                response_positions,
                selected_n_queries,
            )
        else:
            loss = token_losses.mean()

        l1_norm = self._compute_l1_reg(generated_loras)
        total_loss = loss + self.l1_reg_coef * l1_norm
        rollout_lengths = rollout_mask.sum(dim=1).float()
        metrics = {
            "task_loss": loss.detach(),
            "opcd_reverse_kl": (
                loss.detach()
                if self.opcd_loss_mode != "online_fkl"
                else torch.tensor(0.0, device=loss.device)
            ),
            "l1_norm": l1_norm.detach(),
            "l1_loss": (self.l1_reg_coef * l1_norm).detach(),
            "lora_l2_norm": self._compute_l2_norm(generated_loras).detach(),
            "total_loss": total_loss.detach(),
            "distill_mode": torch.tensor(1.0, device=total_loss.device),
            "context_encoder_cache_hit": torch.tensor(
                float(used_cached_ctx_features),
                device=total_loss.device,
            ),
            "rollout_length": rollout_lengths.mean(),
            "rollout_eos_rate": ended_with_eos.float().mean(),
            "rollout_tokens": rollout_mask.sum().float(),
            "context_tokens": (
                ctx_attn_mask.sum().float()
                if ctx_attn_mask is not None
                else torch.tensor(0.0, device=total_loss.device)
            ),
            "student_prompt_tokens": student_prompt_attention_mask.sum().float(),
            "teacher_scoring_tokens": (
                teacher_prompt_attention_mask.sum() + rollout_mask.sum()
            ).float(),
            "selected_queries": selected_n_queries.sum().float(),
            "probe_rotation_offset": probe_offsets.float().mean(),
            "teacher_response_projection": torch.tensor(
                float(teacher_projected), device=total_loss.device
            ),
            "student_response_projection": torch.tensor(
                float(student_projected), device=total_loss.device
            ),
            **objective_metrics,
        }
        if self.opcd_loss_mode == "online_fkl":
            metrics["opcd_forward_kl"] = loss.detach()
        if self.opcd_loss_mode in {"k3", "k3_plus"}:
            metrics["opcd_k3"] = torch.tensor(1.0, device=total_loss.device)
        if self.opcd_backend is not None or remote_teacher_targets:
            metrics["opcd_remote_backend"] = torch.tensor(1.0, device=total_loss.device)
            if teacher_service_seconds is not None:
                metrics["teacher_service_seconds"] = (
                    teacher_service_seconds.detach().float().mean()
                )
            elif self.opcd_backend is not None:
                for name, value in self.opcd_backend.last_metrics.items():
                    metrics[name] = torch.tensor(float(value), device=total_loss.device)
            if teacher_micro_batches is not None:
                metrics["teacher_micro_batches"] = (
                    teacher_micro_batches.detach().float().mean()
                )
        if context_lift is not None:
            metrics["teacher_context_lift"] = context_lift.detach()
        self.last_loss_metrics = metrics
        return total_loss, None, generated_loras

    def _sample_opcd_rollout(
        self,
        prompt_ids: Tensor,
        prompt_attention_mask: Tensor,
        seed: int,
    ) -> Tensor:
        devices = []
        if prompt_ids.device.type == "cuda":
            devices = [prompt_ids.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if prompt_ids.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            with torch.no_grad():
                generated = self.base_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    max_new_tokens=self.opcd_rollout_max_new_tokens,
                    eos_token_id=list(self.opcd_eos_token_ids),
                    pad_token_id=self.opcd_pad_token_id,
                    use_cache=True,
                    return_dict_in_generate=False,
                )
        rollout_ids = generated[:, prompt_ids.shape[1] :]
        if rollout_ids.shape[1] == 0:
            raise RuntimeError("OPCD student rollout generated no tokens")
        return rollout_ids

    @staticmethod
    def _sampled_token_context_lift(
        teacher_logits: Tensor,
        base_logits: Tensor,
        rollout_ids: Tensor,
        rollout_mask: Tensor,
    ) -> Tensor:
        teacher_logprobs = teacher_logits.float().log_softmax(dim=-1)
        base_logprobs = base_logits.float().log_softmax(dim=-1)
        labels = rollout_ids.unsqueeze(-1)
        teacher_values = teacher_logprobs.gather(-1, labels).squeeze(-1)
        base_values = base_logprobs.gather(-1, labels).squeeze(-1)
        return (teacher_values - base_values)[rollout_mask].mean()

    def _apply_lora_to_layers(
        self,
        combined_loras: dict[str, dict[str, Tensor]],
        n_queries: Tensor,
        position_ids: Optional[Tensor] = None,
    ):
        """Bind LoRA weights to model layers."""
        apply_lora(self.base_model, self.config.layer_indices, combined_loras, n_queries)

    def _reset_lora_bindings(self):
        """Restore unmodified reader forwards."""
        reset_lora_hooks(
            self.base_model, self.config.layer_indices, self.config.lora.target_modules,
        )

    def _compute_loss(
        self, logits: Tensor, labels: Tensor, n_queries: Tensor
    ) -> Tensor:
        """Cross-entropy loss with optional per-context averaging."""
        labels = labels.to(logits.device)
        n_queries = n_queries.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_per_token = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        if self.use_per_ctx_average_loss:
            # Average loss per context, then average across contexts
            mask = shift_labels.view(-1) != -100
            if not mask.any():
                raise ValueError("batch has no response labels after tokenization")
            loss = loss_per_token[mask].mean()
        else:
            mask = shift_labels.view(-1) != -100
            if not mask.any():
                raise ValueError("batch has no response labels after tokenization")
            loss = loss_per_token[mask].mean()

        return loss

    def _compute_distill_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        logprobs_vals: Tensor,
        logprobs_indices: Tensor,
        objective: str,
        position_ids: Optional[Tensor],
        n_queries: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Offline context distillation against teacher top-K probabilities.

        Rows in logprobs_vals/logprobs_indices must align with
        torch.where(labels != -100), in row-major order over the batch.
        """
        labels = labels.to(logits.device)
        logprobs_vals = logprobs_vals.to(logits.device)
        logprobs_indices = logprobs_indices.to(logits.device)
        if position_ids is not None:
            position_ids = position_ids.to(logits.device)
        n_queries = n_queries.to(logits.device)
        label_pos = torch.where(labels != -100)
        if label_pos[0].numel() == 0:
            raise ValueError("batch has no response labels after tokenization")
        student_logits = logits[label_pos[0], label_pos[1] - 1].float()

        if logprobs_vals.dim() == 3:
            logprobs_vals = logprobs_vals.squeeze(0)
            logprobs_indices = logprobs_indices.squeeze(0)

        if label_pos[0].shape[0] != logprobs_vals.shape[0]:
            raise ValueError(
                "Label positions and teacher logprobs must have the same number "
                f"of rows. Got {label_pos[0].shape[0]} and {logprobs_vals.shape[0]}."
            )

        if objective == "offline_fkl":
            token_losses, metrics = teacher_topk_forward_kl(
                student_logits,
                logprobs_vals,
                logprobs_indices,
                reduction="none",
            )
        elif objective == "d2l_topk_ce":
            token_losses = d2l_teacher_topk_cross_entropy(
                student_logits,
                logprobs_vals,
                logprobs_indices,
                reduction="none",
            )
            metrics = {}
        else:
            raise ValueError(f"Unsupported distillation objective: {objective}")

        if self.use_per_ctx_average_loss:
            if position_ids is None:
                raise ValueError("position_ids are required for per-context loss")
            loss = average_token_losses_by_session(
                token_losses,
                labels,
                position_ids,
                n_queries,
            )
        else:
            loss = token_losses.mean()
        return loss, metrics

    def _compute_l1_reg(self, generated_loras: dict) -> Tensor:
        """L1 regularization on generated LoRA weights."""
        device = _generated_lora_device(generated_loras)
        if self.l1_reg_coef == 0:
            return torch.tensor(0.0, device=device)
        l1_norm = torch.tensor(0.0, device=device)
        n_modules = len(generated_loras)
        for lora in generated_loras.values():
            l1_norm += lora["A"].abs().mean() + lora["B"].abs().mean()
        return l1_norm / n_modules

    def _compute_l2_norm(self, generated_loras: dict) -> Tensor:
        device = _generated_lora_device(generated_loras)
        squared_norm = torch.tensor(0.0, device=device)
        count = 0
        for lora in generated_loras.values():
            for matrix in (lora["A"], lora["B"]):
                squared_norm = squared_norm + matrix.float().square().sum()
                count += matrix.numel()
        return (squared_norm / max(count, 1)).sqrt()

    def state_dict_hypernet(self) -> dict:
        """Save only the trainable hypernet (perceiver + head)."""
        sd = {}
        for name, param in self.perceiver.named_parameters():
            sd[f"perceiver.{name}"] = param.data
        for name, param in self.head.named_parameters():
            sd[f"head.{name}"] = param.data
        for name, buf in self.perceiver.named_buffers():
            sd[f"perceiver.{name}"] = buf
        for name, buf in self.head.named_buffers():
            sd[f"head.{name}"] = buf
        sd["config"] = self.config.to_dict()
        sd["checkpoint_metadata"] = self.checkpoint_metadata
        return sd

    def save_checkpoint(self, path: str):
        """Save compiler tensors and a portable configuration dictionary."""
        atomic_torch_save(self.state_dict_hypernet(), path)
