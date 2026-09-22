"""Export trusted research checkpoints without private paths or training state.

This is an offline maintainer tool. It never downloads models or publishes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

from rpmem.checkpoint.compat import load_compiler_checkpoint
from rpmem.checkpoint.loader import load_gate_checkpoint
from rpmem.head.hypernet_head import HyperLoRAHead
from rpmem.training.checkpointing import atomic_torch_save
from rpmem.training.hypernet_model import HypernetModel


EXPORT_FORMAT = "rpmem_checkpoint_export_v1"
GATE_FORMATS = {"memlora_personamem_v2_cmp_gate_v1", "memlora_prefeval_cmp_gate_v1"}
PERMA_ARGS = {
    "variant", "seed", "epochs", "lr", "weight_decay", "max_grad_norm",
    "init_bias", "max_train_tasks", "max_eval_tasks", "use_flash_attn", "first_session_rule",
}
CONTRACT_FIELDS = {
    "epochs", "seed", "learning_rate", "weight_decay", "max_grad_norm",
    "init_bias", "training_examples", "training_histories", "selection", "compiler",
    "optimizer_updates", "training_order", "training_scope", "test_scope", "first_session_rule",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_destination(path: Path) -> Path:
    manifest = path.with_name(path.name + ".json")
    if path.exists() or manifest.exists():
        raise FileExistsError("choose a new checkpoint and sidecar destination")
    return manifest


def _public_metadata(value):
    # Exported fields are structured configuration, not free-form logs or paths.
    if isinstance(value, str):
        if any(part in value for part in ("/", "\\", "://")):
            raise ValueError("private path or URL in selected export metadata")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [_public_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _public_metadata(item) for key, item in value.items()}
    raise TypeError("export metadata must contain only JSON values")


def _write(payload: dict, destination: Path, manifest: dict) -> dict:
    sidecar = _new_destination(destination)
    atomic_torch_save(payload, destination)
    manifest = {
        "format": EXPORT_FORMAT, **manifest,
        "file": destination.name, "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def export_compiler(source: Path, destination: Path, *, base_model: str, ctx_encoder: str) -> dict:
    _new_destination(destination)
    for model_id in (base_model, ctx_encoder):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id) or ".." in model_id:
            raise ValueError("public model identifiers must be namespace/repository, not local paths")
    source_state = load_compiler_checkpoint(source)
    config = HypernetModel._checkpoint_config(source_state)
    config.base_model_name = base_model
    config.ctx_encoder_model_name = ctx_encoder
    state = {}
    for raw_key, value in source_state.items():
        key = raw_key.removeprefix("_orig_mod.")
        if key.startswith("aggregator."):
            key = "perceiver." + key.removeprefix("aggregator.")
        elif key.startswith(("layers.", "bias_A.", "bias_B.", "scaler_A.", "scaler_B.")):
            key = "head." + key
        if key.startswith(("perceiver.", "head.")):
            if key in state or not isinstance(value, torch.Tensor):
                raise ValueError(f"duplicate or non-tensor compiler weight: {key}")
            state[key] = value.detach().cpu()

    # Check completeness without allocating another full-size compiler or reader.
    with torch.device("meta"):
        perceiver = HypernetModel._build_perceiver(config)
        head = HyperLoRAHead(config.head)
    expected = {f"perceiver.{k}": v.shape for k, v in perceiver.state_dict().items()}
    expected.update({f"head.{k}": v.shape for k, v in head.state_dict().items()})
    if set(state) != set(expected):
        raise ValueError(f"compiler tensor keys differ: missing={sorted(set(expected)-set(state))}, "
                         f"unexpected={sorted(set(state)-set(expected))}")
    if any(state[key].shape != shape for key, shape in expected.items()):
        raise ValueError("compiler tensor shape does not match configuration")
    source_sha = sha256_file(source)
    state["config"] = config.to_dict()
    state["checkpoint_metadata"] = {"export_format": EXPORT_FORMAT, "source_sha256": source_sha}
    return _write(state, destination, {
        "kind": "compiler", "source_sha256": source_sha,
        "base_model": base_model, "ctx_encoder": ctx_encoder,
        "d_latent": config.gate.d_latent, "tensor_count": len(expected),
    })


def export_gate(source: Path, destination: Path, *, compiler_manifest: Path) -> dict:
    _new_destination(destination)
    compiler = json.loads(compiler_manifest.read_text())
    if compiler.get("format") != EXPORT_FORMAT or compiler.get("kind") != "compiler":
        raise ValueError("expected a compiler export sidecar")
    for key in ("source_sha256", "sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", compiler.get(key, "")):
            raise ValueError("compiler export sidecar is missing a file identity")
    payload = torch.load(source, weights_only=False, map_location="cpu")
    gate, _ = load_gate_checkpoint(source)
    if gate.d_latent != compiler["d_latent"]:
        raise ValueError("Gate and compiler latent dimensions differ")
    metadata = payload.get("metadata", {})
    source_compiler = payload.get("checkpoint_sha256", metadata.get("checkpoint_sha256"))
    if source_compiler != compiler["source_sha256"]:
        raise ValueError("Gate belongs to a different source compiler")
    if "gate_state_dict" in payload:
        if payload.get("format") not in GATE_FORMATS:
            raise ValueError("expected a final benchmark Gate export, not a training resume file")
        clean = {key: metadata[key] for key in (
            "dataset", "dataset_sha256", "phase1_method", "training_examples",
            "training_histories", "optimizer_updates",
        ) if key in metadata}
        clean["gate_contract"] = {key: value for key, value in metadata["gate_contract"].items()
                                  if key in CONTRACT_FIELDS}
        clean = _public_metadata(clean)
        clean.update(checkpoint_sha256=compiler["sha256"], source_checkpoint_sha256=source_compiler)
        exported = {"format": payload["format"], "gate_state_dict": payload["gate_state_dict"],
                    "d_latent": gate.d_latent, "metadata": clean}
    else:
        if payload.get("method", "cmp_gate") != "cmp_gate" or "control_contract" in payload:
            raise ValueError("PERMA control/ablation Gate is not a main-method export")
        clean = {key: payload[key] for key in (
            "train_users", "test_user", "eval_users", "phase1_method",
        ) if key in payload}
        # Original leave-one-user-out exports predate the explicit eval_users field.
        if "eval_users" not in clean and isinstance(clean.get("test_user"), int):
            clean["eval_users"] = [clean["test_user"]]
        clean["args"] = {key: value for key, value in payload.get("args", {}).items() if key in PERMA_ARGS}
        if not {"train_users", "test_user", "eval_users"}.issubset(clean) or "variant" not in clean["args"]:
            raise ValueError("PERMA Gate is missing fold or variant metadata")
        exported = {**_public_metadata(clean), "state_dict": payload["state_dict"],
                    "d_latent": gate.d_latent, "checkpoint_sha256": compiler["sha256"],
                    "source_checkpoint_sha256": source_compiler}
    source_sha = sha256_file(source)
    exported["first_session_rule"] = gate.first_session_rule
    exported["release"] = {"format": EXPORT_FORMAT, "source_sha256": source_sha,
                           "first_session_rule": gate.first_session_rule}
    return _write(exported, destination, {
        "kind": "gate", "source_sha256": source_sha,
        "compiler_sha256": compiler["sha256"], "source_compiler_sha256": source_compiler,
        "d_latent": gate.d_latent, "first_session_rule": gate.first_session_rule,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    compiler = commands.add_parser("compiler", help="export compiler tensors and public model identifiers")
    gate = commands.add_parser("gate", help="export a final Gate paired with the exported compiler")
    for command in (compiler, gate):
        command.add_argument("source", type=Path)
        command.add_argument("destination", type=Path)
    compiler.add_argument("--base-model", required=True)
    compiler.add_argument("--ctx-encoder", required=True)
    gate.add_argument("--compiler-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "compiler":
        result = export_compiler(args.source, args.destination,
                                 base_model=args.base_model, ctx_encoder=args.ctx_encoder)
    else:
        result = export_gate(args.source, args.destination, compiler_manifest=args.compiler_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
