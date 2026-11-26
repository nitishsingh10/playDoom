"""Unit and integration tests for Member 4 workstream:
- Environment rollout collection
- Reward shaping & anti-exploit mechanisms
- Privileged state observation isolation
- Metrics and failure mode diagnostics

Run: python -m pytest tests/test_rollout.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path for direct pytest invocation
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.metrics import EpisodeRecord, FailureMode, MetricsLogger, compare_evaluations
from training.rewards import RewardConfig, RewardShaper
from training.rollout import RolloutBatch, RolloutCollector, collate_observations


# --------------------------------------------------------------------------- Mocks
class MockDoomPolicy(nn.Module):
    """Mock policy matching Member 3's DoomPolicy interface for testing."""

    def __init__(self, num_actions: int = 9):
        super().__init__()
        self.num_actions = num_actions
        self.linear = nn.Linear(17, 32)
        self.actor = nn.Linear(32, num_actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, obs: Dict[str, Any], hidden_state=None):
        # Use vitals tensor [batch, 17]
        vitals = obs["vitals"]
        h = torch.relu(self.linear(vitals))
        logits = self.actor(h)
        values = self.critic(h).squeeze(-1)
        return logits, values, None


class MockDoomEnv:
    """Mock Doom environment with predictable transitions and privileged state."""

    def __init__(self, max_ticks: int = 100):
        self.max_ticks = max_ticks
        self.tick = 0
        self.x = 100.0
        self.y = 100.0
        self.angle = 0.0
        self.kills = 0
        self.damage = 0.0
        self.keys = [0, 0, 0, 0, 0, 0]
        self.exit_reached = False
        self.dead = False
        self.timeout = False

    def reset(self, seed: int | None = None):
        self.tick = 0
        self.x = 100.0
        self.y = 100.0
        self.angle = 0.0
        self.kills = 0
        self.damage = 0.0
        self.keys = [0, 0, 0, 0, 0, 0]
        self.exit_reached = False
        self.dead = False
        self.timeout = False
        return self._make_obs(), {"tic": 0, "exit_reached": False}

    def _make_obs(self) -> Dict[str, Any]:
        return {
            "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
            "depth": np.zeros((120, 160), dtype=np.uint8),
            "vitals": np.zeros(17, dtype=np.float32),
            "rays": {
                "distance": np.ones(64, dtype=np.float32),
                "hit_type": np.zeros(64, dtype=np.int64),
            },
            "visible_entities": {
                "features": np.zeros((32, 5), dtype=np.float32),
                "entity_type": np.zeros(32, dtype=np.int64),
                "mask": np.zeros(32, dtype=np.int8),
            },
        }

    def step(self, action: np.ndarray):
        self.tick += 4
        terminated = False
        truncated = False

        if action[0] == 1:  # Move forward
            self.x += 10.0
            self.y += 10.0
        if action[4] == 1:  # Turn left
            self.angle = (self.angle + 30.0) % 360.0

        if self.tick >= self.max_ticks:
            truncated = True
            self.timeout = True

        reward = 1.0 if self.exit_reached else 0.0
        return self._make_obs(), reward, terminated, truncated, {"tic": self.tick, "exit_reached": self.exit_reached}

    def privileged_state(self) -> Dict[str, Any]:
        return {
            "tic": self.tick,
            "episode_ticks": self.max_ticks,
            "x": self.x,
            "y": self.y,
            "z": 0.0,
            "angle": self.angle,
            "health": 100.0,
            "kills": self.kills,
            "items": 0,
            "damage_taken": self.damage,
            "keys": list(self.keys),
            "dead": self.dead,
            "timeout": self.timeout,
            "exit_reached": self.exit_reached,
            "finished": self.dead or self.timeout or self.exit_reached,
        }


# --------------------------------------------------------------------------- Tests
def test_reward_shaper_components():
    """Verify reward shaper calculates accurate rewards for exit, kill, damage, key, and items."""
    cfg = RewardConfig(exit_reward=10.0, death_penalty=5.0, kill_reward=2.0, damage_penalty_coef=0.02, key_reward=5.0)
    shaper = RewardShaper(config=cfg)

    state0 = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 0, "damage_taken": 0.0, "keys": [0, 0, 0, 0, 0, 0], "exit_reached": False, "dead": False}
    shaper.reset(initial_privileged=state0)

    # 1. Kill reward
    state_kill = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 2, "damage_taken": 0.0, "keys": [0, 0, 0, 0, 0, 0], "exit_reached": False, "dead": False}
    r, comp = shaper.step(state0, state_kill, action=np.zeros(9))
    assert comp["kill"] == pytest.approx(4.0)

    # 2. Key reward
    state_key = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 2, "damage_taken": 0.0, "keys": [1, 0, 0, 0, 0, 0], "exit_reached": False, "dead": False}
    r, comp = shaper.step(state_kill, state_key, action=np.zeros(9))
    assert comp["key"] == pytest.approx(5.0)

    # 3. Damage penalty
    state_dmg = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 2, "damage_taken": 50.0, "keys": [1, 0, 0, 0, 0, 0], "exit_reached": False, "dead": False}
    r, comp = shaper.step(state_key, state_dmg, action=np.zeros(9))
    assert comp["damage"] == pytest.approx(-1.0)

    # 4. Exit reward
    state_exit = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 2, "damage_taken": 50.0, "keys": [1, 0, 0, 0, 0, 0], "exit_reached": True, "dead": False}
    r, comp = shaper.step(state_dmg, state_exit, action=np.zeros(9))
    assert comp["exit"] == pytest.approx(10.0)


def test_reward_shaper_exploration_and_anti_exploit():
    """Verify exploration grid cell bonus is one-time only, and anti-spinning/anti-spam triggers."""
    cfg = RewardConfig(cell_size=64.0, exploration_reward=1.0, anti_spin_penalty=0.5, anti_door_spam_penalty=0.5)
    shaper = RewardShaper(config=cfg)

    p0 = {"x": 0.0, "y": 0.0, "angle": 0.0, "kills": 0, "damage_taken": 0.0, "keys": [0]*6, "exit_reached": False, "dead": False}
    shaper.reset(initial_privileged=p0)

    # Move to cell (2, 0)
    p1 = {"x": 130.0, "y": 0.0, "angle": 0.0, "kills": 0, "damage_taken": 0.0, "keys": [0]*6, "exit_reached": False, "dead": False}
    _, comp1 = shaper.step(p0, p1, action=np.zeros(9))
    assert comp1["exploration"] == pytest.approx(1.0)

    # Move within same cell (2, 0) -> no second exploration reward
    p2 = {"x": 140.0, "y": 10.0, "angle": 0.0, "kills": 0, "damage_taken": 0.0, "keys": [0]*6, "exit_reached": False, "dead": False}
    _, comp2 = shaper.step(p1, p2, action=np.zeros(9))
    assert comp2["exploration"] == pytest.approx(0.0)

    # Spin in place: change angle by 45 degrees with 0 translation
    p_spin = {"x": 140.0, "y": 10.0, "angle": 45.0, "kills": 0, "damage_taken": 0.0, "keys": [0]*6, "exit_reached": False, "dead": False}
    _, comp_spin = shaper.step(p2, p_spin, action=np.zeros(9))
    assert comp_spin["anti_spin"] == pytest.approx(-0.5)

    # Repeated USE edges during the door opening interval, with a release in between.
    action_use = np.zeros(9, dtype=np.int8)
    action_use[7] = 1
    action_release = np.zeros(9, dtype=np.int8)
    shaper.step(p_spin, p_spin, action=action_use)  # first edge
    shaper.step(p_spin, p_spin, action=action_release)
    _, comp_spam = shaper.step(p_spin, p_spin, action=action_use)  # premature second edge
    assert comp_spam["anti_door_spam"] == pytest.approx(-0.5)


def test_revisiting_cells_does_not_earn_movement_reward():
    shaper = RewardShaper(RewardConfig(cell_size=64.0, living_penalty=0.0))
    start = {"x": 0.0, "y": 0.0, "angle": 0.0}
    shaper.reset(start)
    first_new_cell = {"x": 70.0, "y": 0.0, "angle": 0.0}
    _, first = shaper.step(start, first_new_cell, action=np.zeros(9))
    _, revisit = shaper.step(first_new_cell, start, action=np.zeros(9))
    assert first["exploration"] > 0
    assert revisit["exploration"] == 0
    assert revisit["movement"] == 0


def test_observation_isolation():
    """Verify that privileged state NEVER contaminates the official observation dict."""
    env = MockDoomEnv()
    policy = MockDoomPolicy()
    collector = RolloutCollector(env=env)

    batch = collector.collect(policy=policy, num_steps=10)
    obs = batch["obs"]

    # Official keys only
    expected_keys = {"rgb", "depth", "vitals", "rays", "visible_entities"}
    assert set(obs.keys()) == expected_keys

    # Confirm NO privileged state fields leaked
    forbidden_keys = {"x", "y", "z", "angle", "kills", "damage_taken", "dead", "timeout", "exit_reached"}
    assert not any(k in obs for k in forbidden_keys)
    assert not any(k in obs["rays"] for k in forbidden_keys)
    assert not any(k in obs["visible_entities"] for k in forbidden_keys)


def test_rollout_collector_shapes_and_boundaries():
    """Verify rollout collection tensor shapes and terminal transition handling."""
    env = MockDoomEnv(max_ticks=20)  # Will truncate after 5 steps (20 ticks / 4)
    policy = MockDoomPolicy()
    logger = MetricsLogger()
    collector = RolloutCollector(env=env, metrics_logger=logger)

    num_steps = 16
    batch = collector.collect(policy=policy, num_steps=num_steps)

    assert isinstance(batch, RolloutBatch)
    assert batch["actions"].shape == (num_steps, 9)
    assert batch["rewards"].shape == (num_steps,)
    assert batch["dones"].shape == (num_steps,)
    assert batch["values"].shape == (num_steps,)
    assert batch["log_probs"].shape == (num_steps,)
    assert batch["last_value"].shape == ()
    assert batch["truncations"].shape == (num_steps,)
    assert batch["bootstrap_values"].shape == (num_steps,)
    assert batch["bootstrap_values"].shape == (num_steps,)

    # Verify that truncations occurred and were logged
    assert batch["dones"].sum() > 0
    assert len(logger.episodes) >= 1

    # Verify tuple unpacking interface
    obs, acts, rews, dones, vals, logs = batch.to_tuple()
    assert acts.shape == (num_steps, 9)
    assert rews.shape == (num_steps,)


def test_metrics_and_failure_mode_diagnosis():
    """Verify failure mode classifier and checkpoint evaluation comparison."""
    # Success
    assert FailureMode.diagnose(True, False, False, 500, 2100, 20, 0, 100.0, 0.0) == FailureMode.SUCCESS
    # Early death
    assert FailureMode.diagnose(False, True, False, 200, 2100, 5, 0, 100.0, 100.0) == FailureMode.EARLY_DEATH
    # Combat death
    assert FailureMode.diagnose(False, True, False, 1200, 2100, 15, 0, 50.0, 100.0) == FailureMode.COMBAT_DEATH
    # Door bounce
    assert FailureMode.diagnose(False, False, True, 2100, 2100, 4, 15, 100.0, 0.0) == FailureMode.DOOR_BOUNCE
    # Stuck nav
    assert FailureMode.diagnose(False, False, True, 2100, 2100, 2, 0, 100.0, 0.0) == FailureMode.STUCK_NAV

    # Evaluation comparison
    base = {"exit_rate": 0.1, "survival_rate": 0.3, "mean_kills": 0.5, "mean_unique_cells": 10.0, "mean_ticks": 1800.0}
    cand = {"exit_rate": 0.4, "survival_rate": 0.6, "mean_kills": 1.5, "mean_unique_cells": 25.0, "mean_ticks": 1200.0}
    comp = compare_evaluations(base, cand)

    assert comp["exit_rate"]["delta"] == pytest.approx(0.3)
    assert comp["exit_rate"]["improved"] is True
    assert comp["mean_ticks"]["improved"] is True  # Less ticks to finish is improvement


def test_real_vizdoom_environment_rollout():
    """Verify RolloutCollector and smoke test with the actual ViZDoom DoomEnv wrapper."""
    from doom_env import make_env
    from training.rollout import run_environment_smoke_test

    # Run smoke diagnostic on two_rooms.wad
    env = make_env("maps/two_rooms.wad", "MAP01", render_buffers=False, episode_ticks=2100)
    smoke_results = run_environment_smoke_test(env, num_steps=20)
    assert smoke_results["steps_run"] == 20
    assert smoke_results["steps_per_sec"] > 0
    assert smoke_results["vitals_shape"] == (17,)
    assert smoke_results["rays_distance_shape"] == (64,)

    # Run RolloutCollector on real environment
    policy = MockDoomPolicy()
    logger = MetricsLogger()
    collector = RolloutCollector(env=env, metrics_logger=logger)

    num_steps = 16
    batch = collector.collect(policy=policy, num_steps=num_steps)

    assert batch["actions"].shape == (num_steps, 9)
    assert batch["rewards"].shape == (num_steps,)
    assert batch["dones"].shape == (num_steps,)
    assert batch["values"].shape == (num_steps,)
    assert batch["log_probs"].shape == (num_steps,)
    assert torch.isfinite(batch["rewards"]).all()
    assert torch.isfinite(batch["values"]).all()

    # Observation isolation on real environment
    obs = batch["obs"]
    assert "vitals" in obs and obs["vitals"].shape == (num_steps, 17)
    assert "rays" in obs and obs["rays"]["distance"].shape == (num_steps, 64)
    assert "visible_entities" in obs and obs["visible_entities"]["features"].shape == (num_steps, 32, 5)

    env.close()


def test_real_rollout_connects_spatial_policy_to_one_ppo_update(tmp_path):
    """Exercise nested official observations through collection and a real optimizer step."""
    from doom_env import make_env
    from submission.model import DoomPolicy
    from training.train import train_loop

    env = make_env("maps/training/tier2_door.wad", "MAP01", render_buffers=False, episode_ticks=40)
    try:
        policy = DoomPolicy()
        collector = RolloutCollector(env, device="cpu", seed=17)
        result = train_loop(
            policy,
            collector,
            config={
                "total_updates": 1,
                "rollout_steps": 4,
                "minibatch_size": 4,
                "update_epochs": 1,
                "checkpoint_interval": 1,
                "save_dir": str(tmp_path),
                "device": "cpu",
            },
        )
        assert result["total_steps"] == 4
        assert (tmp_path / "checkpoint_1.pt").is_file()
    finally:
        env.close()

