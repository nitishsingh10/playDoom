"""Compact spatial actor-critic for Track 1: Can It Play DOOM?"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from doom_env import spaces as doom_spaces
from submission.actions import NUM_ACTION_LOGITS
from submission.checkpoint import LEGACY_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID
from submission.observation import ObservationEncoder


class DoomPolicy(nn.Module):
    """Feed-forward spatial policy with 12 structured logits and a value head.

    ``hidden_state`` is reserved for a future recurrent variant. This first
    milestone is stateless and returns ``None`` as ``next_hidden``.
    """

    def __init__(
        self, hidden_size: int = 32, fusion_size: int = 128,
        architecture_id: str = STRUCTURED_ARCHITECTURE_ID,
    ):
        super().__init__()
        if architecture_id not in (STRUCTURED_ARCHITECTURE_ID, LEGACY_ARCHITECTURE_ID):
            raise ValueError(f"Unknown policy architecture: {architecture_id!r}")
        # The legacy architecture is available only for explicit baseline comparisons.
        self.architecture_id = architecture_id
        self.hidden_size = hidden_size
        self.fusion_size = fusion_size
        self.observation_encoder = ObservationEncoder()
        self.ray_encoder = nn.Sequential(
            nn.Linear(1 + len(doom_spaces.RAY_HIT_TYPES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        entity_input_size = len(doom_spaces.ENTITY_FEATURES) + len(doom_spaces.ENTITY_TYPES)
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.vitals_encoder = nn.Sequential(
            nn.Linear(len(doom_spaces.VITALS_FIELDS), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Mean and max pools preserve both the overall scene and its strongest cue.
        fused_size = hidden_size * 5
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, fusion_size),
            nn.ReLU(),
            nn.Linear(fusion_size, hidden_size * 2),
            nn.ReLU(),
        )
        output_size = NUM_ACTION_LOGITS if architecture_id == STRUCTURED_ARCHITECTURE_ID else len(doom_spaces.BUTTONS)
        self.actor = nn.Linear(hidden_size * 2, output_size)
        self.critic = nn.Linear(hidden_size * 2, 1)

    @staticmethod
    def _pool(features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mean = features.mean(dim=1)
            maximum = features.amax(dim=1)
        else:
            mask = mask.bool()
            expanded = mask.unsqueeze(-1)
            count = expanded.sum(dim=1).clamp_min(1)
            mean = (features * expanded).sum(dim=1) / count
            maximum = features.masked_fill(~expanded, torch.finfo(features.dtype).min).amax(dim=1)
            maximum = torch.where(expanded.any(dim=1), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, encoded_obs: dict[str, torch.Tensor], hidden_state=None):
        """Return ``(logits[B,12], value[B], next_hidden=None)`` for structured PPO."""
        # Training rollouts hold the official nested observation dictionary;
        # inference may pass the already encoded tensors from submission.schema.
        # Accept both forms so the same preprocessing is used end to end.
        if isinstance(encoded_obs.get("rays"), dict):
            encoded_obs = self.observation_encoder(encoded_obs)
        model_device = next(self.parameters()).device
        encoded_obs = {key: value.to(model_device) for key, value in encoded_obs.items()}

        rays = encoded_obs["rays"]
        entity_features = encoded_obs["entity_features"]
        entity_type = encoded_obs["entity_type"]
        entity_mask = encoded_obs["entity_mask"].bool()
        vitals = encoded_obs["vitals"]

        ray_embedding = self.ray_encoder(rays)
        ray_summary = self._pool(ray_embedding)

        entities = torch.cat((entity_features, entity_type), dim=-1)
        entity_embedding = self.entity_encoder(entities)
        entity_summary = self._pool(entity_embedding, entity_mask)

        vitals_embedding = self.vitals_encoder(vitals)
        fused = self.fusion(torch.cat((ray_summary, entity_summary, vitals_embedding), dim=-1))
        logits = self.actor(fused)
        value = self.critic(fused).squeeze(-1)
        return logits, value, None

    def step(self, encoded_obs: dict[str, torch.Tensor], hidden_state=None):
        """Evaluation step alias matching forward."""
        return self.forward(encoded_obs, hidden_state=hidden_state)

    def init_hidden(self, batch_size: int = 1, device: torch.device | str | None = None):
        """Return initial hidden state for agent (None for stateless feed-forward)."""
        return None


def build_policy(config=None, **kwargs) -> DoomPolicy:
    """Factory helper to instantiate DoomPolicy."""
    defaults = {"hidden_size": 32, "fusion_size": 128, "architecture_id": STRUCTURED_ARCHITECTURE_ID}
    if isinstance(config, Mapping):
        values = {key: config.get(key, default) for key, default in defaults.items()}
    elif config is not None:
        values = {key: getattr(config, key, default) for key, default in defaults.items()}
    else:
        values = {key: kwargs.get(key, default) for key, default in defaults.items()}
    return DoomPolicy(**values)
