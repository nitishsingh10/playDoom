"""Version checks shared by training and the standalone inference package.

The old nine-logit policy and the structured twelve-logit policy have different
action meanings. A state dictionary must never be silently converted between
them, even if a caller could otherwise ignore an incompatible actor head.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1
STRUCTURED_ARCHITECTURE_ID = "doom_structured_v1"
DIRECTIONAL_ARCHITECTURE_ID = "doom_structured_ray_v2"
SPATIAL_GRU_ARCHITECTURE_ID = "doom_spatial_gru_v3"
LEGACY_ARCHITECTURE_ID = "doom_bernoulli_v1"
STRUCTURED_ARCHITECTURES = frozenset({
    STRUCTURED_ARCHITECTURE_ID,
    DIRECTIONAL_ARCHITECTURE_ID,
    SPATIAL_GRU_ARCHITECTURE_ID,
})
SUPPORTED_ARCHITECTURES = STRUCTURED_ARCHITECTURES | {LEGACY_ARCHITECTURE_ID}


class CheckpointCompatibilityError(ValueError):
    """The saved policy architecture does not match the requested policy."""


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Read either a wrapped checkpoint or an explicitly supported legacy state."""
    if not isinstance(checkpoint, Mapping):
        raise CheckpointCompatibilityError("Checkpoint must contain a model state dictionary.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(torch.is_tensor(value) for value in state.values()):
        raise CheckpointCompatibilityError("Checkpoint model_state_dict must contain only tensors.")
    return state


def _architecture_from_weights(state: Mapping[str, torch.Tensor]) -> str | None:
    actor = state.get("actor.weight")
    if actor is None or actor.ndim != 2:
        return None
    if actor.shape[0] == 12 and "ray_conv.0.weight" in state and "gru.weight_ih_l0" in state:
        return SPATIAL_GRU_ARCHITECTURE_ID
    rays = state.get("ray_encoder.0.weight")
    if actor.shape[0] == 12 and rays is not None and rays.ndim == 2 and rays.shape[1] == 9:
        return DIRECTIONAL_ARCHITECTURE_ID
    return {
        9: LEGACY_ARCHITECTURE_ID,
        12: STRUCTURED_ARCHITECTURE_ID,
    }.get(actor.shape[0])


def _declared_architecture(checkpoint: Mapping[str, Any]) -> str | None:
    declarations = [checkpoint.get("architecture_id")]
    for key in ("model_config", "config"):
        config = checkpoint.get(key)
        if isinstance(config, Mapping):
            declarations.append(config.get("architecture_id"))
    declarations = [value for value in declarations if value is not None]
    if any(not isinstance(value, str) for value in declarations):
        raise CheckpointCompatibilityError("Checkpoint architecture_id must be a string.")
    if len(set(declarations)) > 1:
        raise CheckpointCompatibilityError("Checkpoint contains inconsistent architecture_id metadata.")
    return declarations[0] if declarations else None


def checkpoint_architecture(checkpoint: Mapping[str, Any]) -> str | None:
    """Return declared architecture, checking its agreement with actor dimensions.

    Unversioned nine-head checkpoints are identified for baseline comparisons.
    Recognizing an architecture here does not authorize a structured load;
    ``validate_checkpoint_compatibility`` also requires structured metadata.
    """
    state = extract_state_dict(checkpoint)
    declared = _declared_architecture(checkpoint)
    inferred = _architecture_from_weights(state)
    if declared is not None and declared not in SUPPORTED_ARCHITECTURES:
        raise CheckpointCompatibilityError(f"Unsupported policy architecture_id: {declared!r}.")
    if declared is not None and inferred is not None and declared != inferred:
        raise CheckpointCompatibilityError(
            f"Checkpoint declares {declared!r}, but its actor weights use {inferred!r}."
        )
    return declared or inferred


def validate_checkpoint_compatibility(checkpoint: Mapping[str, Any], model: nn.Module) -> None:
    """Reject incompatible DoomPolicy weights before mutating model parameters.

    Generic PyTorch modules retain normal state-dictionary compatibility, which
    keeps checkpoint utilities useful for small synthetic training tests.
    """
    state = extract_state_dict(checkpoint)
    expected = getattr(model, "architecture_id", None)
    if expected is None:
        return
    saved = checkpoint_architecture(checkpoint)
    if saved != expected:
        guidance = (
            "Legacy 9-logit Bernoulli weights cannot be loaded into the structured 12-logit policy; "
            "use an explicit legacy policy for baseline evaluation or a separately trained structured checkpoint."
            if LEGACY_ARCHITECTURE_ID in (saved, expected) else
            "Ray-direction v2 and structured v1 checkpoints are not interchangeable; build the exact saved variant."
        )
        raise CheckpointCompatibilityError(
            f"Incompatible checkpoint architecture: saved {saved or 'unidentified'!r}, "
            f"requested {expected!r}. {guidance}"
        )
    if expected in STRUCTURED_ARCHITECTURES:
        if _declared_architecture(checkpoint) is None or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise CheckpointCompatibilityError(
                f"Structured checkpoints require architecture_id={expected!r} "
                f"and checkpoint_version={CHECKPOINT_VERSION}; save/export a versioned checkpoint."
            )
    elif "checkpoint_version" in checkpoint and checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint_version: {checkpoint['checkpoint_version']!r}."
        )
    # Give a useful error for custom network widths before load_state_dict fails.
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, Mapping):
        for field in ("hidden_size", "fusion_size"):
            saved_value = model_config.get(field)
            actual_value = getattr(model, field, None)
            if saved_value is not None and saved_value != actual_value:
                raise CheckpointCompatibilityError(
                    f"Checkpoint {field}={saved_value!r} does not match policy {field}={actual_value!r}. "
                    "Build the policy using the checkpoint's model_config."
                )
    # Ensure malformed declared actor dimensions cannot pass metadata checks.
    expected_shape = model.state_dict().get("actor.weight")
    saved_shape = state.get("actor.weight")
    if expected_shape is not None and saved_shape is not None and expected_shape.shape != saved_shape.shape:
        raise CheckpointCompatibilityError(
            f"Checkpoint actor shape {tuple(saved_shape.shape)} does not match policy "
            f"{tuple(expected_shape.shape)}; no weights were converted."
        )


def load_policy_state(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Validate then strictly load inference or resumable checkpoint weights."""
    validate_checkpoint_compatibility(checkpoint, model)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
