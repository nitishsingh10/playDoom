"""Tensor preprocessing for the official ego-centric Doom observations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from doom_env import spaces as doom_spaces


class ObservationEncoder:
    """Normalize policy inputs and return a consistently batched tensor mapping.

    Supported fields are rays, visible_entities and vitals. RGB and depth are
    intentionally omitted from the compact first model. Single observations and
    batches of observations are both accepted.
    """

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(device) if device is not None else None

    def _tensor(self, value, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _batch(value: torch.Tensor, unbatched_ndim: int, name: str) -> torch.Tensor:
        if value.ndim == unbatched_ndim:
            return value.unsqueeze(0)
        if value.ndim == unbatched_ndim + 1:
            return value
        raise ValueError(f"{name} must have {unbatched_ndim} or {unbatched_ndim + 1} dimensions")

    def __call__(self, observation: Mapping) -> dict[str, torch.Tensor]:
        """Return rays [B,64,7], entity tensors [B,32,*], and vitals [B,17]."""
        try:
            rays = observation["rays"]
            entities = observation["visible_entities"]
            vitals = observation["vitals"]
            ray_distance = self._tensor(rays["distance"], dtype=torch.float32)
            ray_type = self._tensor(rays["hit_type"], dtype=torch.long)
            entity_features = self._tensor(entities["features"], dtype=torch.float32)
            entity_type = self._tensor(entities["entity_type"], dtype=torch.long)
            entity_mask = self._tensor(entities["mask"], dtype=torch.bool)
            vitals = self._tensor(vitals, dtype=torch.float32)
        except (KeyError, TypeError) as exc:
            raise ValueError("observation must contain rays, visible_entities, and vitals") from exc

        ray_distance = self._batch(ray_distance, 1, "rays.distance")
        ray_type = self._batch(ray_type, 1, "rays.hit_type")
        entity_features = self._batch(entity_features, 2, "visible_entities.features")
        entity_type = self._batch(entity_type, 1, "visible_entities.entity_type")
        entity_mask = self._batch(entity_mask, 1, "visible_entities.mask")
        vitals = self._batch(vitals, 1, "vitals")

        expected = {
            "rays.distance": (ray_distance, (doom_spaces.NUM_RAYS,)),
            "rays.hit_type": (ray_type, (doom_spaces.NUM_RAYS,)),
            "visible_entities.features": (entity_features, (doom_spaces.MAX_ENTITIES, len(doom_spaces.ENTITY_FEATURES))),
            "visible_entities.entity_type": (entity_type, (doom_spaces.MAX_ENTITIES,)),
            "visible_entities.mask": (entity_mask, (doom_spaces.MAX_ENTITIES,)),
            "vitals": (vitals, (len(doom_spaces.VITALS_FIELDS),)),
        }
        batch_sizes = set()
        for name, (tensor, tail_shape) in expected.items():
            if tuple(tensor.shape[1:]) != tail_shape:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected [batch, {tail_shape}]")
            batch_sizes.add(tensor.shape[0])
        if len(batch_sizes) != 1:
            raise ValueError("all observation fields must have the same batch size")

        ray_distance = torch.nan_to_num(ray_distance, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        ray_type = ray_type.clamp(0, len(doom_spaces.RAY_HIT_TYPES) - 1)
        ray_features = torch.cat((ray_distance.unsqueeze(-1), F.one_hot(
            ray_type, num_classes=len(doom_spaces.RAY_HIT_TYPES)
        ).to(torch.float32)), dim=-1)

        entity_mask = entity_mask & (entity_type != 0)
        entity_features = torch.nan_to_num(entity_features, nan=0.0, posinf=0.0, neginf=0.0)
        # Distance is reported in 1024-unit increments (max 4); velocities are
        # map units/tic and are scaled to a useful bounded range.
        normalized_entities = torch.stack((
            entity_features[..., 0].clamp(0.0, 4.0) / 4.0,
            entity_features[..., 1].clamp(-1.0, 1.0),
            entity_features[..., 2].clamp(-1.0, 1.0),
            (entity_features[..., 3] / 20.0).clamp(-1.0, 1.0),
            (entity_features[..., 4] / 20.0).clamp(-1.0, 1.0),
        ), dim=-1)
        normalized_entities = normalized_entities * entity_mask.unsqueeze(-1)
        entity_type = entity_type.clamp(0, len(doom_spaces.ENTITY_TYPES) - 1)
        entity_type_one_hot = F.one_hot(entity_type, num_classes=len(doom_spaces.ENTITY_TYPES)).to(torch.float32)
        entity_type_one_hot = entity_type_one_hot * entity_mask.unsqueeze(-1)

        vitals = torch.nan_to_num(vitals, nan=0.0, posinf=0.0, neginf=0.0)
        vitals = vitals.clone()
        vitals[..., 0:2] = vitals[..., 0:2].clamp(0.0, 200.0) / 200.0
        vitals[..., 2] = vitals[..., 2].clamp(0.0, 7.0) / 7.0
        vitals[..., 3:8] = vitals[..., 3:8].clamp(0.0, 200.0) / 200.0
        vitals[..., 8:14] = vitals[..., 8:14].clamp(0.0, 1.0)
        vitals[..., 14:17] = (vitals[..., 14:17] / 20.0).clamp(-1.0, 1.0)

        return {
            "rays": ray_features,
            "entity_features": normalized_entities,
            "entity_type": entity_type_one_hot,
            "entity_mask": entity_mask,
            "vitals": vitals,
        }
