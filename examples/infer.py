"""Generate a response from bounded sessions and locally supplied checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rpmem import RPMemModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--ctx-encoder", default=None)
    parser.add_argument("--sessions", nargs="+", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This pretrained-model example requires a CUDA GPU.")
    if args.max_context_tokens < 1 or args.max_new_tokens < 1:
        parser.error("token limits must be positive")
    model = RPMemModel.from_checkpoint(
        args.checkpoint, base_model_path=args.base_model, use_flash_attn=False,
        ctx_encoder_path=args.ctx_encoder, gate_ckpt_path=args.gate, device="cuda",
    ).requires_grad_(False)
    memory = None
    with torch.no_grad():
        for path in args.sessions:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"empty session: {path.name}")
            memory = model.update_memory(text, memory, max_tokens=args.max_context_tokens)
        try:
            model.apply_memory(memory)
            print(model.generate(args.query, max_new_tokens=args.max_new_tokens))
        finally:
            model.reset()


if __name__ == "__main__":
    main()
