"""Rollout collection, environment interaction, and rollout batch buffering.

Owned by Member 4 (Environment, rollouts, maps, and reward shaping).

CRITICAL ARCHITECTURAL CONTRACT:
- Implements `RolloutCollector` compatible with Member 2's `train_loop` / `compute_gae`
  and Member 3's `DoomPolicy`.
- Expected return schema for `train_loop`:
    {
        "obs": collated batched observation dict [T, ...],
        "actions": torch.Tensor [T, 9] in {0, 1},
        "rewards": torch.Tensor [T] float32,
        "dones": torch.Tensor [T] float32,
        "values": torch.Tensor [T] float32,
        "log_probs": torch.Tensor [T] float32,
        "last_value": torch.Tensor scalar or [1] float32,
        "truncations": torch.Tensor [T] bool (optional, for GAE boundary handling)
    }
- STRICT PRIVILEGED STATE ISOLATION:
  `env.privileged_state()` is consumed solely by `RewardShaper` and `MetricsLogger`.
  It NEVER touches `obs` or enters the policy network / DoomAgent.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from training.metrics import EpisodeRecord, FailureMode, MetricsLogger
from training.rewards import RewardConfig, RewardShaper

try:
    from training.ppo import sample_actions
except ImportError:
    # Standalone fallback if training.ppo is not available
    def sample_actions(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Bernoulli(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs


def collate_observations(obs_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    """Collate a sequence of T observation dictionaries into a batched dictionary of PyTorch tensors.

    Args:
        obs_list: List of T observation dictionaries, each matching docs/CONTRACT.md §2.
        device: PyTorch device to place tensors on.

    Returns:
        batched_obs: Dictionary where every leaf array is stacked into a Tensor along dim 0.
    """
    T = len(obs_list)
    if T == 0:
        raise ValueError("Cannot collate empty observation list")

    sample = obs_list[0]

    # Vitals [T, 17]
    vitals = torch.as_tensor(np.stack([o["vitals"] for o in obs_list]), dtype=torch.float32, device=device)

    # Rays
    rays_dist = torch.as_tensor(
        np.stack([o["rays"]["distance"] for o in obs_list]), dtype=torch.float32, device=device
    )
    rays_hit = torch.as_tensor(
        np.stack([o["rays"]["hit_type"] for o in obs_list]), dtype=torch.long, device=device
    )
    rays = {"distance": rays_dist, "hit_type": rays_hit}

    # Visible entities
    ent_feat = torch.as_tensor(
        np.stack([o["visible_entities"]["features"] for o in obs_list]), dtype=torch.float32, device=device
    )
    ent_type = torch.as_tensor(
        np.stack([o["visible_entities"]["entity_type"] for o in obs_list]), dtype=torch.long, device=device
    )
    ent_mask = torch.as_tensor(
        np.stack([o["visible_entities"]["mask"] for o in obs_list]), dtype=torch.float32, device=device
    )
    visible_entities = {"features": ent_feat, "entity_type": ent_type, "mask": ent_mask}

    collated: Dict[str, Any] = {
        "vitals": vitals,
        "rays": rays,
        "visible_entities": visible_entities,
    }

    # Optional RGB & Depth (only stack if populated to save memory)
    if "rgb" in sample and sample["rgb"] is not None:
        collated["rgb"] = torch.as_tensor(
            np.stack([o["rgb"] for o in obs_list]), dtype=torch.uint8, device=device
        )
    if "depth" in sample and sample["depth"] is not None:
        collated["depth"] = torch.as_tensor(
            np.stack([o["depth"] for o in obs_list]), dtype=torch.uint8, device=device
        )

    return collated


class RolloutBatch(dict):
    """Dictionary holding collected trajectory data.

    Supports dict indexing (for Member 2's train_loop), named properties, and tuple unpacking.
    """

    @property
    def obs(self) -> Any:
        return self["obs"]

    @property
    def actions(self) -> torch.Tensor:
        return self["actions"]

    @property
    def rewards(self) -> torch.Tensor:
        return self["rewards"]

    @property
    def dones(self) -> torch.Tensor:
        return self["dones"]

    @property
    def values(self) -> torch.Tensor:
        return self["values"]

    @property
    def log_probs(self) -> torch.Tensor:
        return self["log_probs"]

    @property
    def last_value(self) -> torch.Tensor:
        return self["last_value"]

    @property
    def truncations(self) -> Optional[torch.Tensor]:
        return self.get("truncations")

    @property
    def bootstrap_values(self) -> Optional[torch.Tensor]:
        return self.get("bootstrap_values")

    def to_tuple(self) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack the 6 canonical rollout fields according to shared Python interfaces."""
        return (
            self["obs"],
            self["actions"],
            self["rewards"],
            self["dones"],
            self["values"],
            self["log_probs"],
        )


class RolloutCollector:
    """Manages environment stepping, action sampling, reward shaping, and rollout buffering."""

    def __init__(
        self,
        env: Any,
        shaper: Optional[RewardShaper] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        self.env = env
        self.shaper = shaper or RewardShaper()
        self.metrics_logger = metrics_logger
        self.device = torch.device(device)
        self.seed = seed
        self._has_reset = False

        self._current_obs: Optional[Dict[str, Any]] = None
        self._current_info: Optional[Dict[str, Any]] = None
        self._episode_id: int = 0
        self._episode_step_count: int = 0
        self._episode_raw_return: float = 0.0
        self._episode_start_time: float = time.time()
        self._door_spam_count: int = 0

    def _ensure_env_reset(self) -> None:
        """Ensure environment is active and observation is available."""
        if self._current_obs is None:
            if not self._has_reset and self.seed is not None:
                self._current_obs, self._current_info = self.env.reset(seed=self.seed)
            else:
                self._current_obs, self._current_info = self.env.reset()
            self._has_reset = True
            self._episode_id += 1
            self._episode_step_count = 0
            self._episode_raw_return = 0.0
            self._episode_start_time = time.time()
            self._door_spam_count = 0

            # Initialize shaper with initial privileged state if available
            init_priv = None
            if hasattr(self.env, "privileged_state"):
                try:
                    init_priv = self.env.privileged_state()
                except Exception:
                    init_priv = None
            self.shaper.reset(initial_privileged=init_priv)

    def _step_policy(
        self,
        policy: Any,
        obs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run policy forward pass on a single observation.

        Returns:
            action: Binary tensor of shape [1, 9] in {0, 1}.
            value: Scalar value tensor [1].
            log_prob: Scalar log probability tensor [1].
        """
        # Batch single observation to shape [1, ...]
        batched_obs = collate_observations([obs], device=self.device)

        with torch.no_grad():
            if hasattr(policy, "forward"):
                out = policy(batched_obs)
            elif callable(policy):
                out = policy(batched_obs)
            else:
                raise TypeError(f"Unsupported policy object {type(policy)}")

            # Unpack policy outputs: Member 3 returns (logits [batch, 9], value [batch], next_hidden)
            if isinstance(out, tuple):
                logits = out[0]
                values = out[1]
            elif isinstance(out, dict):
                logits = out["logits"]
                values = out["values"]
            else:
                logits = out
                values = torch.zeros(logits.shape[0], device=self.device)

            if values.dim() > 1:
                values = values.squeeze(-1)

            actions, log_probs = sample_actions(logits)

        return actions.squeeze(0), values.squeeze(0), log_probs.squeeze(0)

    def collect(
        self,
        policy: Any,
        num_steps: int = 256,
    ) -> RolloutBatch:
        """Collect trajectory transitions of length num_steps from the environment.

        Args:
            policy: Policy model (e.g. DoomPolicy from Member 3).
            num_steps: Number of environment steps to collect.

        Returns:
            RolloutBatch containing collated obs, actions, rewards, dones, values, log_probs, last_value.
        """
        self._ensure_env_reset()

        obs_buffer: List[Dict[str, Any]] = []
        action_buffer: List[torch.Tensor] = []
        reward_buffer: List[float] = []
        done_buffer: List[float] = []
        value_buffer: List[torch.Tensor] = []
        log_prob_buffer: List[torch.Tensor] = []
        truncation_buffer: List[bool] = []
        bootstrap_value_buffer: List[float] = []

        last_transition_terminated = False
        last_transition_truncated = False

        for _ in range(num_steps):
            obs = self._current_obs
            obs_buffer.append(obs)

            # Query policy
            action_tensor, val_tensor, log_prob_tensor = self._step_policy(policy, obs)
            action_np = action_tensor.cpu().numpy().astype(np.int8)

            # Read privileged state before step (for delta reward shaping)
            prev_priv = None
            if hasattr(self.env, "privileged_state"):
                try:
                    prev_priv = self.env.privileged_state()
                except Exception:
                    prev_priv = None

            # Step environment
            next_obs, env_reward, terminated, truncated, info = self.env.step(action_np)
            self._episode_step_count += 1
            self._episode_raw_return += float(env_reward)

            # Read privileged state after step
            curr_priv = None
            if hasattr(self.env, "privileged_state"):
                try:
                    curr_priv = self.env.privileged_state()
                except Exception:
                    curr_priv = None

            # Calculate shaped reward
            if curr_priv is not None:
                shaped_reward, comp_dict = self.shaper.step(
                    prev_privileged=prev_priv,
                    curr_privileged=curr_priv,
                    action=action_np,
                    env_reward=env_reward,
                    info=info,
                )
                self._door_spam_count = self.shaper.door_spam_count
            else:
                shaped_reward = float(env_reward)

            is_done = terminated or truncated
            last_transition_terminated = bool(terminated)
            last_transition_truncated = bool(truncated)

            # Capture the final timeout-state value before resetting. Using the
            # next rollout state's value here would leak across episodes.
            if truncated:
                _, truncation_value, _ = self._step_policy(policy, next_obs)
                bootstrap_value_buffer.append(float(truncation_value.item()))
            else:
                bootstrap_value_buffer.append(0.0)

            action_buffer.append(action_tensor)
            reward_buffer.append(float(shaped_reward))
            done_buffer.append(1.0 if is_done else 0.0)
            value_buffer.append(val_tensor)
            log_prob_buffer.append(log_prob_tensor)
            truncation_buffer.append(bool(truncated))

            if is_done:
                # Record metrics
                if self.metrics_logger is not None:
                    summary = self.shaper.get_episode_summary()
                    exit_reached = False
                    dead = False
                    timeout = bool(truncated)
                    kills = 0
                    damage_taken = 0.0
                    keys_held = 0
                    ammo_remaining = float(np.asarray(next_obs["vitals"])[4:8].sum())
                    max_ticks = 2100

                    if curr_priv is not None:
                        exit_reached = bool(curr_priv.get("exit_reached", False))
                        dead = bool(curr_priv.get("dead", False))
                        kills = int(curr_priv.get("kills", 0))
                        damage_taken = float(curr_priv.get("damage_taken", 0.0))
                        keys_held = int(sum(curr_priv.get("keys", [])))
                        max_ticks = int(curr_priv.get("episode_ticks", 2100))
                    elif info is not None:
                        exit_reached = bool(info.get("exit_reached", False))

                    failure = FailureMode.diagnose(
                        exit_reached=exit_reached,
                        dead=dead,
                        timeout=timeout,
                        ticks=self._episode_step_count * 4,  # ViZDoom frame skip = 4
                        max_ticks=max_ticks,
                        unique_cells=int(summary.get("unique_cells_visited", 0)),
                        door_spam_count=self._door_spam_count,
                        ammo_remaining=ammo_remaining,
                        damage_taken=damage_taken,
                    )

                    rec = EpisodeRecord(
                        episode_id=self._episode_id,
                        map_name=getattr(self.env, "map_name", "UNKNOWN"),
                        seed=None,
                        ticks=self._episode_step_count * 4,
                        max_ticks=max_ticks,
                        raw_return=self._episode_raw_return,
                        shaped_return=summary.get("total_shaped_reward", 0.0),
                        exit_reached=exit_reached,
                        dead=dead,
                        timeout=timeout,
                        kills=kills,
                        damage_taken=damage_taken,
                        keys_held=keys_held,
                        ammo_remaining=ammo_remaining,
                        unique_cells=int(summary.get("unique_cells_visited", 0)),
                        failure_mode=failure,
                        reward_components=summary,
                        duration_sec=time.time() - self._episode_start_time,
                    )
                    self.metrics_logger.record_episode(rec)

                # Reset environment for subsequent steps
                self._current_obs, self._current_info = self.env.reset()
                self._episode_id += 1
                self._episode_step_count = 0
                self._episode_raw_return = 0.0
                self._episode_start_time = time.time()
                self._door_spam_count = 0

                init_priv = None
                if hasattr(self.env, "privileged_state"):
                    try:
                        init_priv = self.env.privileged_state()
                    except Exception:
                        init_priv = None
                self.shaper.reset(initial_privileged=init_priv)
            else:
                self._current_obs = next_obs
                self._current_info = info

        # Compute bootstrap value for GAE:
        # If last step was true terminal, bootstrap value is 0.0.
        # Otherwise, bootstrap value is policy value estimate on self._current_obs.
        if last_transition_terminated or last_transition_truncated:
            last_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            _, last_value, _ = self._step_policy(policy, self._current_obs)

        # Collate batch
        collated_obs = collate_observations(obs_buffer, device=self.device)
        actions = torch.stack(action_buffer).to(self.device)
        rewards = torch.tensor(reward_buffer, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_buffer, dtype=torch.float32, device=self.device)
        values = torch.stack(value_buffer).to(self.device)
        log_probs = torch.stack(log_prob_buffer).to(self.device)
        truncations = torch.tensor(truncation_buffer, dtype=torch.bool, device=self.device)

        return RolloutBatch(
            {
                "obs": collated_obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "values": values,
                "log_probs": log_probs,
                "last_value": last_value,
                "truncations": truncations,
                "bootstrap_values": torch.tensor(bootstrap_value_buffer, dtype=torch.float32, device=self.device),
            }
        )


def run_environment_smoke_test(
    env: Any,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """Diagnostic environment smoke test to verify step execution, shapes, and FPS."""
    obs, info = env.reset()
    start_t = time.time()
    steps_run = 0

    for step_idx in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps_run += 1
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_t
    fps = (steps_run * 4) / max(elapsed, 1e-4)  # 4 tics per step
    steps_per_sec = steps_run / max(elapsed, 1e-4)

    return {
        "steps_run": steps_run,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "game_ticks_per_sec": fps,
        "last_obs_keys": list(obs.keys()),
        "vitals_shape": obs["vitals"].shape,
        "rays_distance_shape": obs["rays"]["distance"].shape,
        "entities_mask_shape": obs["visible_entities"]["mask"].shape,
    }
