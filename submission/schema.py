"""Inference schema and helper utilities for DoomAgent evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from doom_env import spaces as doom_spaces
from submission.observation import ObservationEncoder

ACTION_DIM = len(doom_spaces.BUTTONS)
DEFAULT_CONFIG_FILENAME = "config.json"
DEFAULT_WEIGHTS_FILENAME = "checkpoint.pt"


@dataclass
class PolicyConfig:
    architecture_id: str = "doom_structured_v1"
    use_rgb: bool = False
    use_depth: bool = False
    deterministic_eval: bool = True
    action_threshold: float = 0.5  # Legacy baseline setting; structured modes use distribution modes.
    hidden_size: int = 32
    fusion_size: int = 128

    @classmethod
    def load_json(cls, path: Path | str) -> PolicyConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class EdgeTriggerTracker:
    """Tracks edge-triggered actions like USE and SELECT_NEXT_WEAPON."""

    def __init__(self):
        self.last_action = np.zeros(ACTION_DIM, dtype=np.int8)

    def reset(self) -> None:
        self.last_action.fill(0)

    def observe(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.int8).copy()


def format_action(raw_action: Any) -> np.ndarray:
    """Ensure raw action conforms to MultiBinary(9) int8 with values in {0, 1}."""
    action = np.asarray(raw_action, dtype=np.int8).reshape(ACTION_DIM)
    return np.clip(action, 0, 1)


_default_encoder = None


def preprocess_observation(
    observation: dict[str, Any],
    include_rgb: bool = False,
    include_depth: bool = False,
) -> dict[str, np.ndarray]:
    """Preprocess observation dict using ObservationEncoder, returning numpy arrays."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = ObservationEncoder(device="cpu")
    encoded = _default_encoder(observation)
    return {
        k: v.squeeze(0).cpu().numpy() if v.ndim > 0 and v.shape[0] == 1 else v.cpu().numpy()
        for k, v in encoded.items()
    }
