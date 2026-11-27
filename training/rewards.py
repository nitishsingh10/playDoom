"""Reward shaping module for ViZDoom training.

Owned by Member 4 (Environment, rollouts, maps, and reward shaping).

CRITICAL ARCHITECTURAL BOUNDARY:
- Privileged state (`env.privileged_state()`) is strictly restricted to this module
  during training only.
- It must NEVER leak into the policy observation, `ObservationEncoder`, or `DoomAgent`.
- Evaluation runs strictly in a clean, isolated environment without this code.

Features & Anti-Exploit Design:
1. Spatial Exploration: Grid cell discretization (default 64 units) awarding one-time bonuses
   per newly discovered cell, encouraging agents to explore unfamiliar rooms. When a new
   keycard is acquired, exploration memory can refresh once to guide backtracking to the
   locked door and exit.
2. Long-Horizon Progression: High bonuses for key acquisition (+5.0), opening doors/locked
   doors for the first time (+1.5 / +3.0), and map completion (+10.0).
3. Tactical Combat: Positive rewards for dealing damage (+0.03/hp) and kills (+2.0),
   penalties for taking damage (-0.02/hp).
4. Anti-Spinning: Detects rapid angle changes without spatial translation and penalizes
   spinning in place without exploring.
5. Anti-Door-Spam: ViZDoom doors reverse if USE is spammed while opening; penalizes repeated
   rapid USE button presses without position changes.
6. Anti-Stall / Living Cost: Small step penalty (-0.001) encouraging efficient progression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple


@dataclass
class RewardConfig:
    """Hyperparameters and weights for training-only reward shaping."""

    # Progression & Goal
    exit_reward: float = 15.0
    death_penalty: float = 5.0
    key_reward: float = 5.0
    door_open_reward: float = 1.5
    locked_door_open_reward: float = 6.0
    item_reward: float = 0.5
    kill_reward: float = 2.0
    damage_dealt_coef: float = 0.03
    damage_penalty_coef: float = 0.02

    # Exploration & Movement
    cell_size: float = 64.0  # Map units per spatial grid cell
    exploration_reward: float = 0.5  # Awarded once per unique cell visited per episode
    macro_cell_size: float = 128.0
    macro_exploration_reward: float = 0.0  # Extra bonus for entering a genuinely new room quadrant
    reset_exploration_on_key: bool = True
    movement_reward_coef: float = 0.001  # Small reward for translation
    max_movement_reward_per_step: float = 0.05
    min_movement_threshold: float = 2.0  # Units of movement required to avoid stall penalty
    living_penalty: float = 0.001  # Small step cost without drowning sparse progress rewards

    # Anti-Exploit Penalties
    anti_spin_penalty: float = 0.02  # Penalty for turning without translating
    spin_angle_threshold_deg: float = 15.0  # Degrees turn to check for spin
    wall_stall_penalty: float = 0.005  # Penalty for pushing into a wall with near-zero motion
    stagnation_grace_steps: int = 45
    stagnation_penalty: float = 0.004
    anti_door_spam_penalty: float = 0.02  # Penalty for premature repeated USE edges
    use_cooldown_steps: int = 8  # Do not press USE again while a door should still be opening

    # Scale factor for raw environment reward
    env_reward_scale: float = 1.0


class RewardShaper:
    """Stateful reward shaper tracking per-episode progress and exploration."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.visited_macro_cells: Set[Tuple[int, int]] = set()
        self.opened_doors: Set[int] = set()
        self.prev_privileged: Optional[Dict[str, Any]] = None
        self.step_count: int = 0
        self.previous_use_pressed: bool = False
        self.last_use_step: Optional[int] = None
        self.door_spam_count: int = 0
        self.steps_since_progress: int = 0
        self.total_components: Dict[str, float] = self._zero_components()

    def _zero_components(self) -> Dict[str, float]:
        return {
            "env_raw": 0.0,
            "exit": 0.0,
            "death": 0.0,
            "kill": 0.0,
            "damage_dealt": 0.0,
            "damage": 0.0,
            "key": 0.0,
            "door_open": 0.0,
            "item": 0.0,
            "exploration": 0.0,
            "movement": 0.0,
            "living": 0.0,
            "anti_spin": 0.0,
            "wall_stall": 0.0,
            "stagnation": 0.0,
            "anti_door_spam": 0.0,
        }

    def reset(self, initial_privileged: Optional[Dict[str, Any]] = None) -> None:
        """Reset internal trackers at the beginning of an episode."""
        self.visited_cells.clear()
        self.visited_macro_cells.clear()
        self.opened_doors.clear()
        self.step_count = 0
        self.previous_use_pressed = False
        self.last_use_step = None
        self.door_spam_count = 0
        self.steps_since_progress = 0
        self.total_components = self._zero_components()
        self.prev_privileged = initial_privileged

        if initial_privileged is not None:
            # Mark starting grid cell as already visited so start pos isn't re-rewarded
            x = initial_privileged.get("x", 0.0)
            y = initial_privileged.get("y", 0.0)
            cell = (int(x // self.config.cell_size), int(y // self.config.cell_size))
            self.visited_cells.add(cell)
            mcell = (int(x // self.config.macro_cell_size), int(y // self.config.macro_cell_size))
            self.visited_macro_cells.add(mcell)
            # Mark any initially open doors so they are never rewarded
            door_lock = initial_privileged.get("door_lock")
            sector_ceil = initial_privileged.get("sector_ceiling")
            if door_lock is not None and sector_ceil is not None:
                for idx, lock in enumerate(door_lock):
                    if int(lock) >= 0 and float(sector_ceil[idx]) > 24.0:
                        self.opened_doors.add(idx)

    def step(
        self,
        prev_privileged: Optional[Dict[str, Any]],
        curr_privileged: Dict[str, Any],
        action: Any,
        env_reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate shaped reward for a single environment step."""
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

        # 3. Kills & Damage Dealt (dense combat credit)
        curr_kills = curr_privileged.get("kills", 0)
        prev_kills = prev_privileged.get("kills", 0)
        if curr_kills > prev_kills:
            components["kill"] += float(curr_kills - prev_kills) * cfg.kill_reward

        curr_dealt = float(curr_privileged.get("damage_dealt", 0.0))
        prev_dealt = float(prev_privileged.get("damage_dealt", 0.0))
        if curr_dealt > prev_dealt and math.isfinite(curr_dealt) and math.isfinite(prev_dealt):
            components["damage_dealt"] += float(curr_dealt - prev_dealt) * cfg.damage_dealt_coef

        # 4. Damage Taken
        curr_dmg = curr_privileged.get("damage_taken", 0.0)
        prev_dmg = prev_privileged.get("damage_taken", 0.0)
        if curr_dmg > prev_dmg:
            components["damage"] -= float(curr_dmg - prev_dmg) * cfg.damage_penalty_coef

        # 5. Keys Picked Up (and optional exploration reset for backtracking)
        curr_keys = curr_privileged.get("keys", [])
        prev_keys = prev_privileged.get("keys", [])
        if len(curr_keys) > 0 and len(prev_keys) > 0:
            new_keys = sum(curr_keys) - sum(prev_keys)
            if new_keys > 0:
                components["key"] += float(new_keys) * cfg.key_reward
                if cfg.reset_exploration_on_key:
                    self.visited_cells.clear()
                    self.visited_macro_cells.clear()
                    # Allow re-opening currently closed normal doors once after key pickup
                    # so backtracking out of a keycard room gets guided properly.
                    dl = curr_privileged.get("door_lock")
                    sc = curr_privileged.get("sector_ceiling")
                    if dl is not None and sc is not None:
                        self.opened_doors = {
                            idx for idx in self.opened_doors
                            if idx >= len(dl) or int(dl[idx]) > 0 or float(sc[idx]) > 24.0
                        }

        # 5b. One-time Door / Locked Door Open Milestone
        door_lock = curr_privileged.get("door_lock")
        sector_ceil = curr_privileged.get("sector_ceiling")
        if door_lock is not None and sector_ceil is not None:
            for idx, lock in enumerate(door_lock):
                lock_val = int(lock)
                if lock_val >= 0 and idx not in self.opened_doors and float(sector_ceil[idx]) > 24.0:
                    self.opened_doors.add(idx)
                    components["door_open"] += (
                        cfg.locked_door_open_reward if lock_val > 0 else cfg.door_open_reward
                    )

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

        cell = (int(curr_x // cfg.cell_size), int(curr_y // cfg.cell_size))
        new_cell = cell not in self.visited_cells
        mcell = (int(curr_x // cfg.macro_cell_size), int(curr_y // cfg.macro_cell_size))
        new_mcell = mcell not in self.visited_macro_cells
        if new_mcell:
            self.visited_macro_cells.add(mcell)
            components["exploration"] += cfg.macro_exploration_reward
        if new_cell:
            self.visited_cells.add(cell)
            components["exploration"] += cfg.exploration_reward
        if new_cell and dist_moved >= cfg.min_movement_threshold:
            step_movement = min(dist_moved * cfg.movement_reward_coef, cfg.max_movement_reward_per_step)
            components["movement"] += step_movement

        # 8. Living Cost (anti-idling)
        components["living"] -= cfg.living_penalty

        # 9. Anti-Spinning & Wall-Stall Check
        curr_ang = curr_privileged.get("angle", 0.0)
        prev_ang = prev_privileged.get("angle", 0.0)
        d_ang = abs((curr_ang - prev_ang + 180.0) % 360.0 - 180.0)
        if d_ang > cfg.spin_angle_threshold_deg and dist_moved < cfg.min_movement_threshold:
            components["anti_spin"] -= cfg.anti_spin_penalty
        try:
            move_attempted = hasattr(action, "__getitem__") and any(int(action[i]) == 1 for i in range(4))
        except (IndexError, TypeError):
            move_attempted = False
        if move_attempted and dist_moved < 0.5 and d_ang < 3.0:
            components["wall_stall"] -= cfg.wall_stall_penalty

        # 10. Edge-triggered USE door-spam guard
        use_pressed = False
        try:
            if hasattr(action, "__getitem__") and len(action) > 7:
                use_pressed = bool(action[7] == 1)
        except (IndexError, TypeError):
            use_pressed = False

        use_edge = use_pressed and not self.previous_use_pressed
        if use_edge:
            if (
                self.last_use_step is not None
                and self.step_count - self.last_use_step <= cfg.use_cooldown_steps
                and dist_moved < cfg.min_movement_threshold
            ):
                components["anti_door_spam"] -= cfg.anti_door_spam_penalty
                self.door_spam_count += 1
            self.last_use_step = self.step_count
        self.previous_use_pressed = use_pressed

        made_progress = (
            new_cell
            or components["kill"] > 0
            or components["damage_dealt"] > 0
            or components["key"] > 0
            or components["door_open"] > 0
            or components["exit"] > 0
        )
        if made_progress:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
            if self.steps_since_progress > cfg.stagnation_grace_steps:
                components["stagnation"] -= cfg.stagnation_penalty

        shaped_reward = sum(components.values())
        for k, v in components.items():
            self.total_components[k] += v

        self.prev_privileged = curr_privileged
        self.step_count += 1
        return float(shaped_reward), components

    def get_episode_summary(self) -> Dict[str, float]:
        """Return total accumulated reward components for the episode."""
        summary = dict(self.total_components)
        summary["total_shaped_reward"] = sum(self.total_components.values())
        summary["unique_cells_visited"] = float(len(self.visited_cells))
        summary["doors_opened_count"] = float(len(self.opened_doors))
        return summary
