"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Owned by Member 1 (Tech Lead / Integrator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from submission.schema import (
    ACTION_DIM,
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    EdgeTriggerTracker,
    PolicyConfig,
    format_action,
    preprocess_observation,
)

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    """Contract-compliant stateful DOOM evaluation agent."""

    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load config/weights relative to submission/ onto `device`."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.rng = np.random.default_rng(0)

        self.config = self._load_config()
        self.model = None
        self.hidden_state = None
        self.edge_tracker = EdgeTriggerTracker()
        self.step_count = 0

        self._load_policy_if_available()

    def _load_config(self) -> PolicyConfig:
        cfg_path = WEIGHTS_DIR / DEFAULT_CONFIG_FILENAME
        if cfg_path.is_file():
            return PolicyConfig.load_json(cfg_path)
        return PolicyConfig()

    def _load_policy_if_available(self) -> None:
        weights_path = WEIGHTS_DIR / DEFAULT_WEIGHTS_FILENAME
        if not weights_path.is_file():
            # Fallback baseline before trained weights are promoted to submission/weights/.
            self.model = None
            return
        # Trained PyTorch checkpoint loading will be wired here once Member 3 publishes
        # the shared model definition and Member 2 produces a trained checkpoint.
        import torch  # noqa: PLC0415

        from training.model import build_policy  # noqa: PLC0415

        model = build_policy(self.config)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        model_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()
        self.model = model

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, edge trackers, and step counters."""
        self.hidden_state = None
        self.edge_tracker.reset()
        self.step_count = 0
        if self.model is not None and hasattr(self.model, "init_hidden"):
            self.hidden_state = self.model.init_hidden(batch_size=1, device=self.device)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a MultiBinary(9) int8 action array with values in {0, 1}."""
        processed = preprocess_observation(
            observation,
            include_rgb=self.config.use_rgb,
            include_depth=self.config.use_depth,
        )

        if self.model is None:
            raw_action = self.rng.integers(0, 2, size=(ACTION_DIM,), dtype=np.int8)
        else:
            raw_action = self._infer_action(processed)

        action = format_action(raw_action)
        self.edge_tracker.observe(action)
        self.step_count += 1
        return action

    def _infer_action(self, processed: dict[str, np.ndarray]) -> np.ndarray:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            batch = {
                k: torch.as_tensor(v, device=self.device).unsqueeze(0)
                for k, v in processed.items()
            }
            logits, _, next_hidden = self.model.step(batch, self.hidden_state)
            self.hidden_state = next_hidden
            probs = torch.sigmoid(logits.squeeze(0))
            if self.config.deterministic_eval:
                bits = (probs >= self.config.action_threshold).to(torch.int8)
            else:
                bits = torch.bernoulli(probs).to(torch.int8)
            return bits.cpu().numpy()
