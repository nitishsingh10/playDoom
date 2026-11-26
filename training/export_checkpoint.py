"""Export model parameters from a resumable training checkpoint for inference."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

import torch

from submission.checkpoint import (
    CHECKPOINT_VERSION,
    LEGACY_ARCHITECTURE_ID,
    STRUCTURED_ARCHITECTURE_ID,
    CheckpointCompatibilityError,
    checkpoint_architecture,
    extract_state_dict,
    load_policy_state,
)


DEFAULT_EXPORT_PATH = Path("training/runs/structured_v1/export/checkpoint.pt")
BASELINE_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "submission" / "weights"


def _guard_destination(destination: Path) -> None:
    if destination.resolve().is_relative_to(BASELINE_WEIGHTS_DIR.resolve()):
        raise ValueError(
            "Refusing to overwrite baseline submission/weights files. Export to a separate "
            "experiment directory; promotion requires a separately reviewed backup and copy."
        )


def export_checkpoint(
    source: str | Path, destination: str | Path | None = None, *, deterministic_eval: bool | None = None,
) -> Path:
    """Export versioned inference weights and matching config outside the baseline.

    Inference contents contain tensors and primitive metadata only, so loading
    with ``weights_only=True`` needs no optimizer, NumPy objects or training code.
    Raw legacy state dictionaries can be exported with their legacy identifier;
    a structured source must already have explicit version metadata.
    """
    from submission.model import DoomPolicy
    from submission.schema import PolicyConfig

    output = Path(destination) if destination is not None else DEFAULT_EXPORT_PATH
    _guard_destination(output)
    if output.name == "config.json":
        raise ValueError("Export destination cannot be named config.json (reserved for policy configuration).")
    if output.resolve() == Path(source).resolve():
        raise ValueError("Export destination must differ from the resumable source checkpoint.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    architecture_id = checkpoint_architecture(checkpoint)
    if architecture_id not in {STRUCTURED_ARCHITECTURE_ID, LEGACY_ARCHITECTURE_ID}:
        raise CheckpointCompatibilityError("Cannot export an unidentified DoomPolicy architecture.")

    saved_config = checkpoint.get("config", {})
    model_config = checkpoint.get("model_config", {})
    config_values = {
        key: saved_config[key]
        for key in (field.name for field in fields(PolicyConfig))
        if isinstance(saved_config, dict) and key in saved_config
    }
    # Raw legacy weights predate metadata; encoder dimensions recover the exact
    # model widths without changing, resizing or dropping a single parameter.
    for key, weight_key in (("hidden_size", "ray_encoder.0.weight"), ("fusion_size", "fusion.0.weight")):
        inferred = state_dict.get(weight_key)
        if inferred is None or inferred.ndim != 2:
            raise CheckpointCompatibilityError(f"Cannot recover DoomPolicy {key} from {weight_key}.")
        declared = model_config.get(key, config_values.get(key)) if isinstance(model_config, dict) else config_values.get(key)
        if declared is not None and declared != inferred.shape[0]:
            raise CheckpointCompatibilityError(f"Checkpoint {key} metadata disagrees with its weights.")
        config_values[key] = inferred.shape[0]
    config_values["architecture_id"] = architecture_id
    if deterministic_eval is not None:
        config_values["deterministic_eval"] = deterministic_eval
    model = DoomPolicy(
        hidden_size=config_values["hidden_size"], fusion_size=config_values["fusion_size"],
        architecture_id=architecture_id,
    )
    load_policy_state(model, checkpoint)
    inference_config = asdict(PolicyConfig(**config_values))
    export_data = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_id": architecture_id,
        "model_config": {
            key: config_values[key] for key in ("architecture_id", "hidden_size", "fusion_size")
        },
        "model_state_dict": state_dict,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_name("config.json")
    temporary_weights = output.with_suffix(output.suffix + ".tmp")
    temporary_config = config_path.with_suffix(".json.tmp")
    torch.save(export_data, temporary_weights)
    temporary_config.write_text(json.dumps(inference_config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_weights, output)
    os.replace(temporary_config, config_path)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Resumable checkpoint .pt")
    parser.add_argument("destination", nargs="?", default=str(DEFAULT_EXPORT_PATH))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stochastic", dest="deterministic_eval", action="store_false",
                      help="Export the stochastic inference mode used in sampled evaluation")
    mode.add_argument("--deterministic", dest="deterministic_eval", action="store_true",
                      help="Export deterministic structured-distribution modes")
    parser.set_defaults(deterministic_eval=None)
    args = parser.parse_args(argv)
    print(export_checkpoint(args.source, args.destination, deterministic_eval=args.deterministic_eval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
