"""Shared policy distribution for the official nine-button DOOM action vector.

The policy makes three independent categorical decisions (none, first direction,
second direction), followed by independent attack, use, and weapon Bernoullis.
This is the policy's action space, so no action filtering is needed after sampling.
"""

from __future__ import annotations

import torch
from torch.distributions import Bernoulli, Categorical

ACTION_DIM = 9
NUM_ACTION_LOGITS = 12


class StructuredActionDistribution:
    """Map ``[..., 12]`` logits to valid ``[..., 9]`` binary button vectors.

    Logits 0:3, 3:6, and 6:9 describe longitudinal movement, strafing,
    and rotation. Within each group, choices 0, 1, and 2 mean neither button,
    the first button, and the second button. Logits 9:12 describe ATTACK,
    USE, and SELECT_NEXT_WEAPON. All leading dimensions are batch dimensions.
    """

    def __init__(self, logits: torch.Tensor):
        if logits.ndim < 1 or logits.shape[-1] != NUM_ACTION_LOGITS:
            raise ValueError(
                f"Structured action logits must have shape [..., {NUM_ACTION_LOGITS}], "
                f"got {tuple(logits.shape)}. Legacy nine-logit policies are incompatible."
            )
        if not logits.is_floating_point():
            raise ValueError("Structured action logits must be floating-point values")
        if not torch.isfinite(logits).all():
            raise ValueError("Structured action logits must contain only finite values")

        self.logits = logits
        self.batch_shape = logits.shape[:-1]
        self._directions = Categorical(
            logits=logits[..., :9].reshape(*self.batch_shape, 3, 3)
        )
        self._buttons = Bernoulli(logits=logits[..., 9:12])

    @staticmethod
    def _to_buttons(directions: torch.Tensor, buttons: torch.Tensor) -> torch.Tensor:
        pairs = torch.stack((directions == 1, directions == 2), dim=-1)
        return torch.cat((pairs.flatten(start_dim=-2), buttons), dim=-1).to(torch.int8)

    def sample(self) -> torch.Tensor:
        """Sample nine binary buttons, with at most one button per direction pair."""
        return self._to_buttons(self._directions.sample(), self._buttons.sample())

    def mode(self) -> torch.Tensor:
        """Choose the most likely value of each independent policy decision.

        Categorical ties choose the first maximal choice; a Bernoulli tie chooses on.
        """
        directions = self._directions.logits.argmax(dim=-1)
        return self._to_buttons(directions, self.logits[..., 9:12] >= 0)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return joint log probabilities of stored official button vectors.

        Invalid or contradictory actions raise an error rather than being silently
        mapped to a different action. Actions must have the same leading batch
        dimensions and device as the policy logits.
        """
        expected_shape = (*self.batch_shape, ACTION_DIM)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"Structured actions must have shape {expected_shape}, got {tuple(actions.shape)}"
            )
        if actions.device != self.logits.device:
            raise ValueError("Structured actions and logits must be on the same device")
        if not ((actions == 0) | (actions == 1)).all():
            raise ValueError("Structured actions must contain only binary values 0 or 1")

        pairs = actions[..., :6].reshape(*self.batch_shape, 3, 2)
        if (pairs.sum(dim=-1) > 1).any():
            raise ValueError("Structured actions cannot activate opposing direction buttons together")
        # 00 -> none (0), 10 -> first direction (1), 01 -> second direction (2).
        directions = pairs[..., 0].long() + 2 * pairs[..., 1].long()
        directional_log_prob = self._directions.log_prob(directions).sum(dim=-1)
        button_log_prob = self._buttons.log_prob(
            actions[..., 6:9].to(self.logits.dtype)
        ).sum(dim=-1)
        return directional_log_prob + button_log_prob

    def entropy(self) -> torch.Tensor:
        """Sum entropies of the three categorical and three Bernoulli decisions."""
        return self._directions.entropy().sum(dim=-1) + self._buttons.entropy().sum(dim=-1)
