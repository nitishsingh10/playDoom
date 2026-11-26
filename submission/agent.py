"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Owned by Member 1 (Tech Lead / Integrator).
"""

from __future__ import annotations

from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import numpy as np

from submission.schema import (
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

    def __init__(self, observation_space, action_space, device: str, *, weights_dir: Path | str | None = None):
        """Called once. Load config/weights relative to submission/ onto `device`."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        import torch

        torch.set_num_threads(1)
        self.weights_dir = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR

        self.config = self._load_config()
        self.model = None
        self.hidden_state = None
        self.edge_tracker = EdgeTriggerTracker()
        self.step_count = 0

        self._load_policy_if_available()

    def _load_config(self) -> PolicyConfig:
        cfg_path = self.weights_dir / DEFAULT_CONFIG_FILENAME
        if cfg_path.is_file():
            return PolicyConfig.load_json(cfg_path)
        return PolicyConfig()

    def _load_policy_if_available(self) -> None:
        weights_path = self.weights_dir / DEFAULT_WEIGHTS_FILENAME
        if not weights_path.is_file():
            # Fallback baseline before trained weights are promoted to submission/weights/.
            self.model = None
            return
        import torch  # noqa: PLC0415

        from submission.model import build_policy  # noqa: PLC0415
        from submission.checkpoint import (  # noqa: PLC0415
            CheckpointCompatibilityError, STRUCTURED_ARCHITECTURE_ID, load_policy_state,
        )

        if self.config.architecture_id != STRUCTURED_ARCHITECTURE_ID:
            raise CheckpointCompatibilityError(
                "DoomAgent requires doom_structured_v1. Use training.evaluate --legacy-baseline "
                "for an explicit comparison of preserved nine-logit weights."
            )

        model = build_policy(self.config)
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except UnpicklingError as error:
            raise CheckpointCompatibilityError(
                "DoomAgent requires a tensor-only, versioned inference export. Export resumable "
                "checkpoints with training.export_checkpoint; legacy nine-logit weights require "
                "training.evaluate --legacy-baseline and cannot load into this agent."
            ) from error
        load_policy_state(model, state)
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
            import torch
            from submission.actions import NUM_ACTION_LOGITS, StructuredActionDistribution

            distribution = StructuredActionDistribution(torch.zeros(NUM_ACTION_LOGITS))
            raw_action = (distribution.mode() if self.config.deterministic_eval else distribution.sample()).numpy()
        else:
            raw_action = self._infer_action(processed)

        action = format_action(raw_action)
        self.edge_tracker.observe(action)
        self.step_count += 1
        return action

    def _infer_action(self, processed: dict[str, np.ndarray]) -> np.ndarray:
        import torch  # noqa: PLC0415
        from submission.actions import StructuredActionDistribution

        with torch.inference_mode():
            batch = {
                k: torch.as_tensor(v, device=self.device).unsqueeze(0)
                for k, v in processed.items()
            }
            logits, _, next_hidden = self.model.step(batch, self.hidden_state)
            self.hidden_state = next_hidden
            distribution = StructuredActionDistribution(logits)
            action = distribution.mode() if self.config.deterministic_eval else distribution.sample()
            return action.squeeze(0).cpu().numpy()
