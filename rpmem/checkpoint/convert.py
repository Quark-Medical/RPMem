"""Convert a trusted legacy compiler checkpoint to the portable RPMem format.

python -m rpmem.checkpoint.convert old.bin converted.bin
"""

import argparse
from pathlib import Path

import torch

from rpmem.checkpoint.compat import load_compiler_checkpoint
from rpmem.config import RPMemConfig
from rpmem.training.checkpointing import atomic_torch_save


def convert_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"choose a new output path: {destination}")
    state = load_compiler_checkpoint(source)
    config = state.get("config")
    if isinstance(config, RPMemConfig):
        state["config"] = config.to_dict()
    elif isinstance(config, dict):
        state["config"] = RPMemConfig.from_dict(config).to_dict()
    else:
        raise ValueError("not a native RPMem compiler checkpoint")
    # Validate the portable payload before writing the destination.
    import io
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    torch.load(buffer, weights_only=True, map_location="cpu")
    atomic_torch_save(state, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted legacy compiler checkpoint")
    parser.add_argument("destination", type=Path, help="new output file; never overwrite the source")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination)
    print(f"Converted compiler checkpoint: {args.destination}")


if __name__ == "__main__":
    main()
