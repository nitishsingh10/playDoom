"""Checkpoint management for DoomPolicy models and training state.

Owned by Member 2 (PPO and Optimization Engineer).
Provides deterministic saving and restoring of model parameters, optimizer state,
iteration/update steps, RNG states, and training configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from submission.checkpoint import CHECKPOINT_VERSION, load_policy_state


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    update: int = 0,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model, optimizer state, step counters, config, and RNG states.

    Args:
        path: Path where checkpoint will be saved (e.g., checkpoints/ckpt_100.pt).
        model: PyTorch model whose state_dict to save.
        optimizer: Optional optimizer whose state_dict to save.
        update: Current PPO update iteration index.
        global_step: Total environment steps taken so far.
        config: Optional configuration snapshot dictionary.
        extra: Any extra metadata (e.g. metrics, validation scores).

    Returns:
        Path to the saved checkpoint file.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng_state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    saved_config = dict(config or {})
    model_config: Dict[str, Any] = {}
    architecture_id = getattr(model, "architecture_id", None)
    if architecture_id is not None:
        model_config["architecture_id"] = architecture_id
        for field in ("hidden_size", "fusion_size"):
            if hasattr(model, field):
                model_config[field] = getattr(model, field)
        for field, value in model_config.items():
            if field in saved_config and saved_config[field] != value:
                raise ValueError(
                    f"Checkpoint config {field}={saved_config[field]!r} does not match model {value!r}."
                )
            saved_config[field] = value

    checkpoint_data: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "update": update,
        "global_step": global_step,
        "config": saved_config,
        "rng_state": rng_state,
        "extra": extra or {},
    }
    if architecture_id is not None:
        checkpoint_data["architecture_id"] = architecture_id
        checkpoint_data["model_config"] = model_config

    # Save to atomic temp file first to prevent corruption if interrupted
    tmp_path = save_path.with_suffix(f"{save_path.suffix}.tmp")
    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, save_path)
    return save_path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """Load model and optimizer states from checkpoint.

    Args:
        path: Path to checkpoint file.
        model: PyTorch model into which parameters will be loaded.
        optimizer: Optional optimizer into which states will be loaded.
        device: Device to map tensors onto ("cpu" or "cuda").
        restore_rng: Whether to restore PyTorch and NumPy RNG states.

    Returns:
        Checkpoint dictionary with update, global_step, config, and extra info.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    try:
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions where weights_only argument doesn't exist
        checkpoint = torch.load(load_path, map_location=device)

    # Validate architecture before loading any weights or optimizer state.
    load_policy_state(model, checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Optionally restore RNG states
    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        if "torch_cpu" in rng and rng["torch_cpu"] is not None:
            # map_location may place even the CPU RNG bytes on the training GPU.
            torch.set_rng_state(rng["torch_cpu"].cpu())
        if "numpy" in rng and rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
        if torch.cuda.is_available() and "torch_cuda" in rng and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])

    return {
        "update": checkpoint.get("update", 0),
        "global_step": checkpoint.get("global_step", 0),
        "config": checkpoint.get("config", {}),
        "extra": checkpoint.get("extra", {}),
        "architecture_id": checkpoint.get("architecture_id"),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "model_config": checkpoint.get("model_config", {}),
    }
