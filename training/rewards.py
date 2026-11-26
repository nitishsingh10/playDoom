"""Reward shaping module for ViZDoom training.

Owned by Member 4 (Environment, rollouts, maps, and reward shaping).

CRITICAL ARCHITECTURAL BOUNDARY:
- Privileged state (`env.privileged_state()`) is strictly restricted to this module
  during training only.
- It must NEVER leak into the policy observation, `ObservationEncoder`, or `DoomAgent`.
- Evaluation runs strictly in a clean, isolated environment without this code.

Features & Anti-Exploit Design:
1. Spatial Exploration: Grid cell discretization (default 64 units) awarding one-time bonuses
   per newly discovered cell, encouraging agents to explore unfamiliar rooms.
2. Long-Horizon Progression: High bonuses for key acquisition (+5.0) and map completion (+10.0).
3. Tactical Combat: Positive rewards for kills (+2.0), penalties for taking damage (-0.02/hp).
4. Anti-Spinning: Detects rapid angle changes without spatial translation and penalizes
   spinning in place without exploring.
5. Anti-Door-Spam: ViZDoom doors reverse if USE is spammed while opening; penalizes repeated
   rapid USE button presses without position changes.
6. Anti-Stall / Living Cost: Small step penalty (-0.005) encouraging efficient progression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


@dataclass
class RewardConfig:
    """Hyperparameters and weights for training-only reward shaping."""

    # Progression & Goal
    exit_reward: float = 10.0
    death_penalty: float = 5.0
    key_reward: float = 5.0
    item_reward: float = 0.5
    kill_reward: float = 2.0
    damage_penalty_coef: float = 0.02

    # Exploration & Movement
    cell_size: float = 64.0  # Map units per spatial grid cell
    exploration_reward: float = 0.2  # Awarded once per unique cell visited per episode
    movement_reward_coef: float = 0.001  # Small reward for translation
    max_movement_reward_per_step: float = 0.05
    min_movement_threshold: float = 2.0  # Units of movement required to avoid stall penalty
    living_penalty: float = 0.005  # Step cost to discourage idling

    # Anti-Exploit Penalties
    anti_spin_penalty: float = 0.02  # Penalty for turning without translating
    spin_angle_threshold_deg: float = 15.0  # Degrees turn to check for spin
    anti_door_spam_penalty: float = 0.05  # Penalty for spamming USE button
    max_consecutive_use: int = 2  # Max consecutive USE steps before penalty triggers

    # Scale factor for raw environment reward
    env_reward_scale: float = 1.0


class RewardShaper:
    """Stateful reward shaper tracking per-episode progress and exploration."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.prev_privileged: Optional[Dict[str, Any]] = None
        self.consecutive_use_count: int = 0
        self.total_components: Dict[str, float] = self._zero_components()

    def _zero_components(self) -> Dict[str, float]:
        return {
            "env_raw": 0.0,
            "exit": 0.0,
            "death": 0.0,
            "kill": 0.0,
            "damage": 0.0,
            "key": 0.0,
            "item": 0.0,
            "exploration": 0.0,
            "movement": 0.0,
            "living": 0.0,
            "anti_spin": 0.0,
            "anti_door_spam": 0.0,
        }

    def reset(self, initial_privileged: Optional[Dict[str, Any]] = None) -> None:
        """Reset internal trackers at the beginning of an episode."""
        self.visited_cells.clear()
        self.consecutive_use_count = 0
        self.total_components = self._zero_components()
        self.prev_privileged = initial_privileged

        if initial_privileged is not None:
            # Mark starting grid cell as already visited so start pos isn't re-rewarded
            x = initial_privileged.get("x", 0.0)
            y = initial_privileged.get("y", 0.0)
            cell = (int(x // self.config.cell_size), int(y // self.config.cell_size))
            self.visited_cells.add(cell)

    def step(
        self,
        prev_privileged: Optional[Dict[str, Any]],
        curr_privileged: Dict[str, Any],
        action: Any,
        env_reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate shaped reward for a single environment step.

        Args:
            prev_privileged: Privileged state dict from before the step (or None if starting).
            curr_privileged: Privileged state dict from after the step.
            action: Action executed (array-like, length 9 MultiBinary).
            env_reward: Raw scalar reward returned by the environment wrapper.
            info: Optional dict returned by gym env.step.

        Returns:
            shaped_reward: Total shaped scalar reward for PPO training.
            components: Dictionary breaking down each reward term for metrics/logging.
        """
        cfg = self.config
        components = self._zero_components()
        components["env_raw"] = float(env_reward) * cfg.env_reward_scale

        if prev_privileged is None:
            prev_privileged = self.prev_privileged or curr_privileged

        # 1. Goal / Exit Reached
        if curr_privileged.get("exit_reached", False) and not prev_privileged.get("exit_reached", False):
            components["exit"] += cfg.exit_reward

        # 2. Death Penalty
        if curr_privileged.get("dead", False) and not prev_privileged.get("dead", False):
            components["death"] -= cfg.death_penalty

        # 3. Kills
        curr_kills = curr_privileged.get("kills", 0)
        prev_kills = prev_privileged.get("kills", 0)
        if curr_kills > prev_kills:
            components["kill"] += float(curr_kills - prev_kills) * cfg.kill_reward

        # 4. Damage Taken
        curr_dmg = curr_privileged.get("damage_taken", 0.0)
        prev_dmg = prev_privileged.get("damage_taken", 0.0)
        if curr_dmg > prev_dmg:
            components["damage"] -= float(curr_dmg - prev_dmg) * cfg.damage_penalty_coef

        # 5. Keys Picked Up
        curr_keys = curr_privileged.get("keys", [])
        prev_keys = prev_privileged.get("keys", [])
        if len(curr_keys) > 0 and len(prev_keys) > 0:
            new_keys = sum(curr_keys) - sum(prev_keys)
            if new_keys > 0:
                components["key"] += float(new_keys) * cfg.key_reward

        # 6. Items Picked Up (Ammo, Health, Armor)
        curr_items = curr_privileged.get("items", 0)
        prev_items = prev_privileged.get("items", 0)
        if curr_items > prev_items:
            components["item"] += float(curr_items - prev_items) * cfg.item_reward

        # 7. Spatial Discretization Exploration & Movement
        curr_x = curr_privileged.get("x", 0.0)
        curr_y = curr_privileged.get("y", 0.0)
        prev_x = prev_privileged.get("x", 0.0)
        prev_y = prev_privileged.get("y", 0.0)

        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dist_moved = math.hypot(dx, dy)

        # Unique grid cell exploration
        cell = (int(curr_x // cfg.cell_size), int(curr_y // cfg.cell_size))
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            components["exploration"] += cfg.exploration_reward

        # Movement reward (capped to prevent velocity exploits)
        if dist_moved >= cfg.min_movement_threshold:
            step_movement = min(dist_moved * cfg.movement_reward_coef, cfg.max_movement_reward_per_step)
            components["movement"] += step_movement

        # 8. Living Cost (anti-idling)
        components["living"] -= cfg.living_penalty

        # 9. Anti-Spinning Check
        curr_ang = curr_privileged.get("angle", 0.0)
        prev_ang = prev_privileged.get("angle", 0.0)
        d_ang = abs((curr_ang - prev_ang + 180.0) % 360.0 - 180.0)
        # If turning heavily without translating, penalize
        if d_ang > cfg.spin_angle_threshold_deg and dist_moved < cfg.min_movement_threshold:
            components["anti_spin"] -= cfg.anti_spin_penalty

        # 10. Anti-Door-Spam Check (action[7] is USE)
        use_pressed = False
        try:
            if hasattr(action, "__getitem__") and len(action) > 7:
                use_pressed = bool(action[7] == 1)
        except (IndexError, TypeError):
            use_pressed = False

        if use_pressed:
            self.consecutive_use_count += 1
            if self.consecutive_use_count > cfg.max_consecutive_use:
                components["anti_door_spam"] -= cfg.anti_door_spam_penalty
        else:
            self.consecutive_use_count = 0

        # Sum total shaped reward
        shaped_reward = sum(components.values())

        # Accumulate totals for episode-level metrics
        for k, v in components.items():
            self.total_components[k] += v

        # Update previous state
        self.prev_privileged = curr_privileged

        return float(shaped_reward), components

    def get_episode_summary(self) -> Dict[str, float]:
        """Return total accumulated reward components for the episode."""
        summary = dict(self.total_components)
        summary["total_shaped_reward"] = sum(self.total_components.values())
        summary["unique_cells_visited"] = float(len(self.visited_cells))
        return summary
