"""Export model parameters from a resumable training checkpoint for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_checkpoint(source: str | Path, destination: str | Path) -> Path:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Store only tensors so inference can use weights_only=True and needs no
    # training module, optimizer, RNG state, or training data.
    torch.save(state_dict, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Resumable checkpoint .pt")
    parser.add_argument("destination", nargs="?", default="submission/weights/checkpoint.pt")
    args = parser.parse_args(argv)
    print(export_checkpoint(args.source, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
