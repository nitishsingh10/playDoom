"""Compact spatial actor-critic for Track 1: Can It Play DOOM?"""

from __future__ import annotations

import torch
from torch import nn

from doom_env import spaces as doom_spaces


class DoomPolicy(nn.Module):
    """Feed-forward spatial policy with 9 Bernoulli logits and a value head.

    ``hidden_state`` is reserved for a future recurrent variant. This first
    milestone is stateless and returns ``None`` as ``next_hidden``.
    """

    def __init__(self, hidden_size: int = 32, fusion_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
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
        self.actor = nn.Linear(hidden_size * 2, len(doom_spaces.BUTTONS))
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
        """Return ``(logits[B,9], value[B], next_hidden=None)``."""
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
    hidden_size = getattr(config, "hidden_size", 32) if config is not None else kwargs.get("hidden_size", 32)
    fusion_size = getattr(config, "fusion_size", 128) if config is not None else kwargs.get("fusion_size", 128)
    return DoomPolicy(hidden_size=hidden_size, fusion_size=fusion_size)

