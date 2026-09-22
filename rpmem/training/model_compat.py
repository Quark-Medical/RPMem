"""Compatibility helpers for text backbones wrapped by multimodal models."""

from __future__ import annotations

from typing import Any

import torch


AUTO_GENERATION_MODEL_LOADERS = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForMultimodalLM",
)


def text_model_config(config: Any) -> Any:
    """Return the text decoder config from a plain or composite config."""

    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def load_generation_model(
    model_path: str,
    *,
    use_flash_attn: bool,
    device_map: str | None = None,
):
    """Load a text-generating model, including multimodal wrapper checkpoints."""

    import transformers

    attention_modes = ["flash_attention_2", "sdpa"] if use_flash_attn else ["sdpa"]
    failures: list[str] = []
    for loader_name in AUTO_GENERATION_MODEL_LOADERS:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        for attention_mode in attention_modes:
            try:
                load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attention_mode,
                }
                if device_map is not None:
                    load_kwargs["device_map"] = device_map
                return loader.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
            except (ImportError, ValueError, TypeError) as exc:
                failures.append(
                    f"{loader_name}/{attention_mode}: {type(exc).__name__}: {exc}"
                )
    raise RuntimeError(
        "no Transformers auto-model loader accepted the backbone checkpoint:\n"
        + "\n".join(failures)
    )
