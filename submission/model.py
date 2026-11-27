"""Compact spatial actor-critic models for Track 1: Can It Play DOOM?"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F
from torch import nn

from doom_env import spaces as doom_spaces
from submission.actions import ACTION_DIM, NUM_ACTION_LOGITS
from submission.checkpoint import (
    DIRECTIONAL_ARCHITECTURE_ID,
    SPATIAL_GRU_ARCHITECTURE_ID,
    STRUCTURED_ARCHITECTURE_ID,
    STRUCTURED_ARCHITECTURES,
    SUPPORTED_ARCHITECTURES,
)
from submission.observation import ObservationEncoder


def _init_linear(layer: nn.Linear, gain: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DoomPolicy(nn.Module):
    """Spatial actor-critic with 12 structured logits and a value head.

    Supports:
    - ``doom_structured_v1``: stateless MLP baseline
    - ``doom_structured_ray_v2``: stateless MLP with ray angles
    - ``doom_spatial_gru_v3``: circular 1D ray convolutions + 8-sector pooling,
      entity masked attention + max pooling, previous-action context, and GRU memory
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        fusion_size: int = 128,
        architecture_id: str = STRUCTURED_ARCHITECTURE_ID,
    ):
        super().__init__()
        if architecture_id not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unknown policy architecture: {architecture_id!r}")
        if hidden_size is None:
            hidden_size = 64 if architecture_id == SPATIAL_GRU_ARCHITECTURE_ID else 32
        self.architecture_id = architecture_id
        self.hidden_size = int(hidden_size)
        self.fusion_size = int(fusion_size)
        self.is_recurrent = architecture_id == SPATIAL_GRU_ARCHITECTURE_ID
        self.observation_encoder = ObservationEncoder()

        directional = architecture_id in (DIRECTIONAL_ARCHITECTURE_ID, SPATIAL_GRU_ARCHITECTURE_ID)
        if directional:
            # Match Sensors: ray k points at heading + k*2*pi/64. Positive
            # angles turn left; channels are sin/cos, as for visible entities.
            angles = torch.arange(doom_spaces.NUM_RAYS, dtype=torch.float64) * (math.tau / doom_spaces.NUM_RAYS)
            self.register_buffer(
                "ray_directions",
                torch.stack((angles.sin(), angles.cos()), -1).float(),
                persistent=False,
            )

        ray_in_channels = 1 + len(doom_spaces.RAY_HIT_TYPES) + (2 if directional else 0)
        entity_input_size = len(doom_spaces.ENTITY_FEATURES) + len(doom_spaces.ENTITY_TYPES)
        output_size = NUM_ACTION_LOGITS if architecture_id in STRUCTURED_ARCHITECTURES else len(doom_spaces.BUTTONS)

        if self.is_recurrent:
            self.ray_conv = nn.Sequential(
                nn.Conv1d(ray_in_channels, self.hidden_size, kernel_size=5, padding=2, padding_mode="circular"),
                nn.ReLU(),
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2, padding_mode="circular"),
                nn.ReLU(),
            )
            # 8 angular sectors x (mean + max) = 16 * hidden_size
            self.ray_proj = nn.Sequential(
                _init_linear(nn.Linear(self.hidden_size * 16, self.fusion_size)),
                nn.ReLU(),
            )
            self.entity_encoder = nn.Sequential(
                _init_linear(nn.Linear(entity_input_size, self.hidden_size)),
                nn.ReLU(),
                _init_linear(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.ReLU(),
            )
            self.entity_attn = _init_linear(nn.Linear(self.hidden_size, 1), gain=1.0)
            self.vitals_encoder = nn.Sequential(
                _init_linear(nn.Linear(len(doom_spaces.VITALS_FIELDS) + ACTION_DIM, self.hidden_size)),
                nn.ReLU(),
                _init_linear(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.ReLU(),
            )
            fused_in = self.fusion_size + self.hidden_size * 2 + self.hidden_size
            self.fusion = nn.Sequential(
                _init_linear(nn.Linear(fused_in, self.fusion_size)),
                nn.LayerNorm(self.fusion_size),
                nn.ReLU(),
            )
            self.gru = nn.GRU(self.fusion_size, self.fusion_size, batch_first=True)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)
            self.actor = _init_linear(nn.Linear(self.fusion_size, output_size), gain=0.01)
            # Initialize action logit priors so initial exploration walks forward
            # without randomly toggling to Fist or spamming USE on step 0.
            with torch.no_grad():
                self.actor.bias.copy_(torch.tensor([
                    0.0, 0.5, -0.5,   # forward/back: none, forward, backward
                    0.4, 0.0, 0.0,    # strafe: none, left, right
                    0.0, 0.0, 0.0,    # turn: none, left, right
                    -1.0, -1.8, -3.0, # ATTACK, USE, SELECT_NEXT_WEAPON
                ], dtype=self.actor.bias.dtype))
            self.critic = _init_linear(nn.Linear(self.fusion_size, 1), gain=1.0)
        else:
            self.ray_encoder = nn.Sequential(
                nn.Linear(ray_in_channels, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
            self.entity_encoder = nn.Sequential(
                nn.Linear(entity_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
            self.vitals_encoder = nn.Sequential(
                nn.Linear(len(doom_spaces.VITALS_FIELDS), self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
            # Mean and max pools preserve both the overall scene and its strongest cue.
            fused_size = self.hidden_size * 5
            self.fusion = nn.Sequential(
                nn.Linear(fused_size, self.fusion_size),
                nn.ReLU(),
                nn.Linear(self.fusion_size, self.hidden_size * 2),
                nn.ReLU(),
            )
            self.actor = nn.Linear(self.hidden_size * 2, output_size)
            self.critic = nn.Linear(self.hidden_size * 2, 1)

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

    def _pool_entities_attn(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        expanded = mask.unsqueeze(-1)
        has_any = expanded.any(dim=1)
        attn_scores = self.entity_attn(features).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1) * mask.to(features.dtype)
        attn_sum = torch.sum(features * attn_weights.unsqueeze(-1), dim=1)
        maximum = features.masked_fill(~expanded, torch.finfo(features.dtype).min).amax(dim=1)
        maximum = torch.where(has_any, maximum, torch.zeros_like(maximum))
        return torch.cat((attn_sum, maximum), dim=-1)

    def _prepare_encoded(self, encoded_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(encoded_obs.get("rays"), dict):
            prev = encoded_obs.get("prev_action")
            encoded_obs = self.observation_encoder(encoded_obs)
            if prev is not None:
                encoded_obs["prev_action"] = prev
        model_device = next(self.parameters()).device
        return {key: value.to(model_device) for key, value in encoded_obs.items()}

    def _encode_fused_spatial(self, encoded_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        rays = encoded_obs["rays"]
        entity_features = encoded_obs["entity_features"]
        entity_type = encoded_obs["entity_type"]
        entity_mask = encoded_obs["entity_mask"].bool()
        vitals = encoded_obs["vitals"]
        batch_size = rays.shape[0]

        directions = self.ray_directions.to(dtype=rays.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        rays_dir = torch.cat((rays, directions), dim=-1)  # [B, 64, 9]
        conv_out = self.ray_conv(rays_dir.transpose(1, 2))  # [B, C, 64]
        # Shift by +4 rays so Sector 0 is centered straight ahead ([-22.5, +22.5] deg).
        sectors = torch.roll(conv_out, shifts=4, dims=-1).reshape(batch_size, self.hidden_size, 8, 8)
        sector_mean = sectors.mean(dim=-1)
        sector_max = sectors.amax(dim=-1)
        ray_summary = self.ray_proj(torch.cat((sector_mean, sector_max), dim=1).flatten(start_dim=1))

        entities = torch.cat((entity_features, entity_type), dim=-1)
        entity_embedding = self.entity_encoder(entities)
        entity_summary = self._pool_entities_attn(entity_embedding, entity_mask)

        prev_action = encoded_obs.get("prev_action")
        if prev_action is None:
            prev_action = torch.zeros((batch_size, ACTION_DIM), dtype=vitals.dtype, device=vitals.device)
        else:
            prev_action = prev_action.to(dtype=vitals.dtype, device=vitals.device)
            if prev_action.ndim == 1:
                prev_action = prev_action.unsqueeze(0)
        vitals_in = torch.cat((vitals, prev_action), dim=-1)
        vitals_embedding = self.vitals_encoder(vitals_in)

        return self.fusion(torch.cat((ray_summary, entity_summary, vitals_embedding), dim=-1))

    def forward(self, encoded_obs: dict[str, torch.Tensor], hidden_state=None):
        """Return ``(logits[B,12], value[B], next_hidden)``."""
        encoded_obs = self._prepare_encoded(encoded_obs)

        if self.is_recurrent:
            fused = self._encode_fused_spatial(encoded_obs)
            batch_size = fused.shape[0]
            if hidden_state is None:
                h0 = self.init_hidden(batch_size=batch_size, device=fused.device)
            else:
                h0 = hidden_state.to(device=fused.device, dtype=fused.dtype)
            gru_out, next_hidden = self.gru(fused.unsqueeze(1), h0)
            core = gru_out.squeeze(1)
            logits = self.actor(core)
            value = self.critic(core).squeeze(-1)
            return logits, value, next_hidden

        rays = encoded_obs["rays"]
        entity_features = encoded_obs["entity_features"]
        entity_type = encoded_obs["entity_type"]
        entity_mask = encoded_obs["entity_mask"].bool()
        vitals = encoded_obs["vitals"]

        if self.architecture_id == DIRECTIONAL_ARCHITECTURE_ID:
            directions = self.ray_directions.to(dtype=rays.dtype).unsqueeze(0).expand(rays.shape[0], -1, -1)
            rays = torch.cat((rays, directions), dim=-1)
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

    def forward_sequence(
        self,
        encoded_obs: dict[str, torch.Tensor],
        dones: torch.Tensor,
        initial_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Vectorized sequence forward over ``[B, T, ...]`` chunks for recurrent PPO.

        Resets hidden state after step ``t`` whenever ``dones[b, t]`` is nonzero.
        """
        if not self.is_recurrent:
            raise RuntimeError("forward_sequence is only supported for recurrent architectures")

        dones = dones.to(dtype=torch.float32)
        batch_size, seq_len = dones.shape
        flat_obs: dict[str, torch.Tensor] = {}
        for key, value in encoded_obs.items():
            if isinstance(value, dict):
                flat_obs[key] = {sk: sv.reshape(batch_size * seq_len, *sv.shape[2:]) for sk, sv in value.items()}
            else:
                flat_obs[key] = value.reshape(batch_size * seq_len, *value.shape[2:])

        flat_obs = self._prepare_encoded(flat_obs)
        fused = self._encode_fused_spatial(flat_obs).reshape(batch_size, seq_len, self.fusion_size)

        if initial_hidden is None:
            h = self.init_hidden(batch_size=batch_size, device=fused.device)
        else:
            h = initial_hidden.to(device=fused.device, dtype=fused.dtype)

        dones_dev = dones.to(device=fused.device)
        outputs = []
        for t in range(seq_len):
            step_out, h = self.gru(fused[:, t : t + 1, :], h)
            outputs.append(step_out)
            reset_mask = (1.0 - dones_dev[:, t]).view(1, batch_size, 1)
            h = h * reset_mask

        core = torch.cat(outputs, dim=1)  # [B, T, fusion_size]
        logits = self.actor(core)
        values = self.critic(core).squeeze(-1)
        return logits, values, h

    def step(self, encoded_obs: dict[str, torch.Tensor], hidden_state=None):
        """Evaluation step alias matching forward."""
        return self.forward(encoded_obs, hidden_state=hidden_state)

    def init_hidden(self, batch_size: int = 1, device: torch.device | str | None = None):
        """Return initial hidden state for agent (zeros for GRU, None for feed-forward)."""
        if self.is_recurrent:
            return torch.zeros(1, batch_size, self.fusion_size, device=device)
        return None


def build_policy(config=None, **kwargs) -> DoomPolicy:
    """Factory helper to instantiate DoomPolicy."""
    arch = STRUCTURED_ARCHITECTURE_ID
    if isinstance(config, Mapping):
        arch = config.get("architecture_id", arch)
    elif config is not None:
        arch = getattr(config, "architecture_id", arch)
    elif "architecture_id" in kwargs:
        arch = kwargs["architecture_id"]
    default_hidden = 64 if arch == SPATIAL_GRU_ARCHITECTURE_ID else 32
    defaults = {"hidden_size": default_hidden, "fusion_size": 128, "architecture_id": arch}
    if isinstance(config, Mapping):
        values = {key: config.get(key, default) for key, default in defaults.items()}
    elif config is not None:
        values = {key: getattr(config, key, default) for key, default in defaults.items()}
    else:
        values = {key: kwargs.get(key, default) for key, default in defaults.items()}
    return DoomPolicy(**values)
