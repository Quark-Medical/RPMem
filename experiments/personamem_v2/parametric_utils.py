"""Shared PersonaMem-v2 latent loading and LoRA binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

if __package__:
    from experiments.personamem_v2.evaluation_utils import history_cache_key
    from experiments.personamem_v2.formal_contract import LATENT_FORMAT
else:
    from evaluation_utils import history_cache_key
    from formal_contract import LATENT_FORMAT
from rpmem.lora.merger import combine_lora
from rpmem.gate import run_cmp_sessions


def load_latent_example(
    latent_root: Path,
    history_file: str,
    *,
    dataset_sha256: str,
    checkpoint_sha256: str,
    memory_selection: str,
) -> dict[str, Any]:
    directory = latent_root / history_cache_key(history_file)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if any(
        (
            meta.get("latent_format") != LATENT_FORMAT,
            meta.get("memory_selection") != memory_selection,
            meta.get("history_file") != history_file,
        )
    ):
        raise ValueError(f"incompatible latent format or history: {meta_path}")
    paths = sorted(directory.glob("memory_segment_*.pt"))
    if len(paths) != int(meta.get("num_memory_segments", -1)) or not paths:
        raise ValueError(f"incomplete latent cache: {directory}")
    return {"meta": meta, "latent_paths": paths}


def load_latents(example: dict[str, Any], device) -> list[torch.Tensor]:
    return [
        torch.load(path, weights_only=True, map_location=device).to(device)
        for path in example["latent_paths"]
    ]


def decode_rank_concatenated_lora(model, latents: list[torch.Tensor]):
    if not latents:
        raise ValueError("at least one latent is required")
    lora_dict = model.head(torch.cat(latents, dim=0))
    n_chunks = torch.tensor([len(latents)], device=model.device)
    return combine_lora(
        lora_dict,
        n_chunks,
        lora_bias=(
            model.head.get_head_bias() if model.config.head.use_bias else None
        ),
    )


def cmp_lora(model, gate, latents: list[torch.Tensor]):
    hidden = run_cmp_sessions(gate, latents)
    return decode_rank_concatenated_lora(model, [hidden])


def forward_with_lora(model, lora: dict, input_ids: torch.Tensor) -> torch.Tensor:
    n_queries = torch.tensor([input_ids.shape[0]], device=model.device)
    model._apply_lora_to_layers(lora, n_queries)
    try:
        return model.base_model(input_ids=input_ids).logits
    finally:
        model._reset_lora_bindings()


def tensor_tree_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_bytes(item) for item in value)
    return 0
