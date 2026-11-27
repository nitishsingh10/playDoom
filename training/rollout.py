"""Rollout collection, environment interaction, and vectorized rollout batch buffering.

Owned by Member 4 (Environment, rollouts, maps, and reward shaping).

CRITICAL ARCHITECTURAL CONTRACT:
- Implements `RolloutCollector` and `VecRolloutCollector` compatible with
  Member 2's `train_loop` / `compute_gae` and Member 3's `DoomPolicy`.
- STRICT PRIVILEGED STATE ISOLATION:
  `env.privileged_state()` is consumed solely by `RewardShaper` and `MetricsLogger`.
  It NEVER touches `obs` or enters the policy network / DoomAgent.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from training.metrics import EpisodeRecord, FailureMode, MetricsLogger
from training.ppo import sample_actions
from training.rewards import RewardConfig, RewardShaper


def collate_observations(obs_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    """Collate a sequence of T observation dictionaries into a batched dictionary of PyTorch tensors."""
    T = len(obs_list)
    if T == 0:
        raise ValueError("Cannot collate empty observation list")

    sample = obs_list[0]
    vitals = torch.as_tensor(np.stack([o["vitals"] for o in obs_list]), dtype=torch.float32, device=device)

    rays_dist = torch.as_tensor(
        np.stack([o["rays"]["distance"] for o in obs_list]), dtype=torch.float32, device=device
    )
    rays_hit = torch.as_tensor(
        np.stack([o["rays"]["hit_type"] for o in obs_list]), dtype=torch.long, device=device
    )
    rays = {"distance": rays_dist, "hit_type": rays_hit}

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

    if "rgb" in sample and sample["rgb"] is not None:
        collated["rgb"] = torch.as_tensor(
            np.stack([o["rgb"] for o in obs_list]), dtype=torch.uint8, device=device
        )
    if "depth" in sample and sample["depth"] is not None:
        collated["depth"] = torch.as_tensor(
            np.stack([o["depth"] for o in obs_list]), dtype=torch.uint8, device=device
        )

    return collated


def _collate_time_env(step_batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack T batched [N, ...] observation dicts into [T, N, ...]."""
    first = step_batches[0]
    out: Dict[str, Any] = {}
    for k, v in first.items():
        if isinstance(v, dict):
            out[k] = _collate_time_env([b[k] for b in step_batches])
        elif torch.is_tensor(v):
            out[k] = torch.stack([b[k] for b in step_batches], dim=0)
    return out


class RolloutBatch(dict):
    """Dictionary holding collected trajectory data."""

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
        return (
            self["obs"],
            self["actions"],
            self["rewards"],
            self["dones"],
            self["values"],
            self["log_probs"],
        )


def _record_episode(
    metrics_logger: Optional[MetricsLogger],
    episode_id: int,
    map_name: str,
    seed: Optional[int],
    step_count: int,
    raw_return: float,
    shaper: RewardShaper,
    next_obs: Dict[str, Any],
    curr_priv: Optional[Dict[str, Any]],
    info: Optional[Dict[str, Any]],
    truncated: bool,
    start_time: float,
) -> None:
    if metrics_logger is None:
        return
    summary = shaper.get_episode_summary()
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
        ticks=step_count * 4,
        max_ticks=max_ticks,
        unique_cells=int(summary.get("unique_cells_visited", 0)),
        door_spam_count=shaper.door_spam_count,
        ammo_remaining=ammo_remaining,
        damage_taken=damage_taken,
    )
    rec = EpisodeRecord(
        episode_id=episode_id,
        map_name=map_name,
        seed=seed,
        ticks=step_count * 4,
        max_ticks=max_ticks,
        raw_return=raw_return,
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
        duration_sec=time.time() - start_time,
    )
    metrics_logger.record_episode(rec)


class RolloutCollector:
    """Manages single-environment stepping, action sampling, reward shaping, and rollout buffering."""

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
        self._hidden_state: Optional[torch.Tensor] = None
        self._prev_action: np.ndarray = np.zeros(9, dtype=np.float32)

    def _ensure_env_reset(self) -> None:
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
            self._hidden_state = None
            self._prev_action.fill(0.0)

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
        hidden_state: Optional[torch.Tensor] = None,
        prev_action: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batched_obs = collate_observations([obs], device=self.device)
        if getattr(policy, "is_recurrent", False):
            pa = self._prev_action if prev_action is None else prev_action
            batched_obs["prev_action"] = torch.as_tensor(pa, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if hasattr(policy, "forward"):
                out = policy(batched_obs, hidden_state)
            elif callable(policy):
                out = policy(batched_obs)
            else:
                raise TypeError(f"Unsupported policy object {type(policy)}")

            next_hidden = None
            if isinstance(out, tuple):
                logits = out[0]
                values = out[1]
                if len(out) > 2:
                    next_hidden = out[2]
            elif isinstance(out, dict):
                logits = out["logits"]
                values = out["values"]
            else:
                logits = out
                values = torch.zeros(logits.shape[0], device=self.device)

            if values.dim() > 1:
                values = values.squeeze(-1)

            actions, log_probs = sample_actions(logits)

        return actions.squeeze(0), values.squeeze(0), log_probs.squeeze(0), next_hidden

    def collect(
        self,
        policy: Any,
        num_steps: int = 256,
    ) -> RolloutBatch:
        self._ensure_env_reset()
        is_recurrent = bool(getattr(policy, "is_recurrent", False))
        if is_recurrent and self._hidden_state is None and hasattr(policy, "init_hidden"):
            self._hidden_state = policy.init_hidden(batch_size=1, device=self.device)

        obs_buffer: List[Dict[str, Any]] = []
        prev_action_buffer: List[np.ndarray] = []
        hidden_buffer: List[torch.Tensor] = []
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
            if is_recurrent:
                prev_action_buffer.append(self._prev_action.copy())
                hidden_buffer.append(self._hidden_state.squeeze(0).squeeze(0).detach().clone())

            action_tensor, val_tensor, log_prob_tensor, next_hidden = self._step_policy(
                policy, obs, hidden_state=self._hidden_state, prev_action=self._prev_action
            )
            action_np = action_tensor.cpu().numpy().astype(np.int8)

            prev_priv = None
            if hasattr(self.env, "privileged_state"):
                try:
                    prev_priv = self.env.privileged_state()
                except Exception:
                    prev_priv = None

            next_obs, env_reward, terminated, truncated, info = self.env.step(action_np)
            self._episode_step_count += 1
            self._episode_raw_return += float(env_reward)

            curr_priv = None
            if hasattr(self.env, "privileged_state"):
                try:
                    curr_priv = self.env.privileged_state()
                except Exception:
                    curr_priv = None

            if curr_priv is not None:
                shaped_reward, _ = self.shaper.step(
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

            if truncated:
                _, truncation_value, _, _ = self._step_policy(
                    policy, next_obs, hidden_state=next_hidden, prev_action=action_np.astype(np.float32)
                )
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
                _record_episode(
                    self.metrics_logger,
                    self._episode_id,
                    getattr(self.env, "episode_map_spec", None) or getattr(self.env, "map_name", "UNKNOWN"),
                    getattr(self.env, "current_episode_seed", None),
                    self._episode_step_count,
                    self._episode_raw_return,
                    self.shaper,
                    next_obs,
                    curr_priv,
                    info,
                    bool(truncated),
                    self._episode_start_time,
                )
                self._current_obs, self._current_info = self.env.reset()
                self._episode_id += 1
                self._episode_step_count = 0
                self._episode_raw_return = 0.0
                self._episode_start_time = time.time()
                self._door_spam_count = 0
                self._prev_action.fill(0.0)
                if is_recurrent and hasattr(policy, "init_hidden"):
                    self._hidden_state = policy.init_hidden(batch_size=1, device=self.device)
                else:
                    self._hidden_state = None

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
                self._prev_action = action_np.astype(np.float32)
                self._hidden_state = next_hidden

        if last_transition_terminated or last_transition_truncated:
            last_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            _, last_value, _, _ = self._step_policy(
                policy, self._current_obs, hidden_state=self._hidden_state, prev_action=self._prev_action
            )

        collated_obs = collate_observations(obs_buffer, device=self.device)
        if is_recurrent:
            collated_obs["prev_action"] = torch.as_tensor(
                np.stack(prev_action_buffer), dtype=torch.float32, device=self.device
            )

        batch_data: Dict[str, Any] = {
            "obs": collated_obs,
            "actions": torch.stack(action_buffer).to(self.device),
            "rewards": torch.tensor(reward_buffer, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(done_buffer, dtype=torch.float32, device=self.device),
            "values": torch.stack(value_buffer).to(self.device),
            "log_probs": torch.stack(log_prob_buffer).to(self.device),
            "last_value": last_value,
            "truncations": torch.tensor(truncation_buffer, dtype=torch.bool, device=self.device),
            "bootstrap_values": torch.tensor(bootstrap_value_buffer, dtype=torch.float32, device=self.device),
        }
        if is_recurrent:
            batch_data["init_hiddens"] = torch.stack(hidden_buffer).to(self.device)
        return RolloutBatch(batch_data)


class VecRolloutCollector:
    """Vectorized rollout collector over SubprocVecEnv producing [T, N, ...] trajectories."""

    def __init__(
        self,
        vec_env: Any,
        shaper_factory: Optional[Callable[[], RewardShaper]] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.env = vec_env
        self.num_envs = int(vec_env.num_envs)
        factory = shaper_factory or RewardShaper
        self.shapers = [factory() for _ in range(self.num_envs)]
        self.metrics_logger = metrics_logger
        self.device = torch.device(device)

        self._obs_list: Optional[List[Dict[str, Any]]] = None
        self._prev_privs: List[Optional[Dict[str, Any]]] = [None] * self.num_envs
        self._map_specs: List[str] = ["UNKNOWN"] * self.num_envs
        self._ep_seeds: List[Optional[int]] = [None] * self.num_envs
        self._ep_ids: List[int] = list(range(1, self.num_envs + 1))
        self._next_ep_id: int = self.num_envs + 1
        self._ep_steps: List[int] = [0] * self.num_envs
        self._ep_raw_returns: List[float] = [0.0] * self.num_envs
        self._ep_start_times: List[float] = [time.time()] * self.num_envs
        self._hidden_state: Optional[torch.Tensor] = None
        self._prev_actions: np.ndarray = np.zeros((self.num_envs, 9), dtype=np.float32)

    def _ensure_reset(self) -> None:
        if self._obs_list is None:
            obs_list, priv_list, map_list, seed_list = self.env.reset()
            self._obs_list = obs_list
            self._prev_privs = list(priv_list)
            self._map_specs = [m or "UNKNOWN" for m in map_list]
            self._ep_seeds = list(seed_list)
            now = time.time()
            for i in range(self.num_envs):
                self.shapers[i].reset(initial_privileged=priv_list[i])
                self._ep_steps[i] = 0
                self._ep_raw_returns[i] = 0.0
                self._ep_start_times[i] = now
            self._prev_actions.fill(0.0)
            self._hidden_state = None

    def collect(self, policy: Any, num_steps: int = 128) -> RolloutBatch:
        self._ensure_reset()
        is_recurrent = bool(getattr(policy, "is_recurrent", False))
        if is_recurrent and self._hidden_state is None and hasattr(policy, "init_hidden"):
            self._hidden_state = policy.init_hidden(batch_size=self.num_envs, device=self.device)

        step_obs_buffers: List[Dict[str, Any]] = []
        hidden_buffer: List[torch.Tensor] = []
        action_buffer: List[torch.Tensor] = []
        reward_buffer: List[torch.Tensor] = []
        done_buffer: List[torch.Tensor] = []
        value_buffer: List[torch.Tensor] = []
        log_prob_buffer: List[torch.Tensor] = []
        truncation_buffer: List[torch.Tensor] = []
        bootstrap_value_buffer: List[torch.Tensor] = []

        for _ in range(num_steps):
            batched_obs = collate_observations(self._obs_list, device=self.device)
            if is_recurrent:
                batched_obs["prev_action"] = torch.as_tensor(
                    self._prev_actions, dtype=torch.float32, device=self.device
                )
                hidden_buffer.append(self._hidden_state.squeeze(0).detach().clone())
            step_obs_buffers.append(batched_obs)

            with torch.no_grad():
                logits, values, next_hidden = policy.step(batched_obs, self._hidden_state)
                if values.dim() > 1:
                    values = values.squeeze(-1)
                actions, log_probs = sample_actions(logits)

            actions_np = actions.cpu().numpy().astype(np.int8)
            results = self.env.step(actions_np)

            step_rewards = np.zeros(self.num_envs, dtype=np.float32)
            step_dones = np.zeros(self.num_envs, dtype=np.float32)
            step_truncs = np.zeros(self.num_envs, dtype=bool)
            step_bootstraps = np.zeros(self.num_envs, dtype=np.float32)
            next_obs_list: List[Dict[str, Any]] = []

            trunc_indices: List[int] = []
            trunc_obs_list: List[Dict[str, Any]] = []
            trunc_prev_actions: List[np.ndarray] = []

            for i, res in enumerate(results):
                next_obs_i, rew_i, term_i, trunc_i, info_i, term_obs_i, curr_priv_i, reset_priv_i, map_i, seed_i = res
                next_obs_list.append(next_obs_i)
                self._ep_steps[i] += 1
                self._ep_raw_returns[i] += float(rew_i)

                shaped_i, _ = self.shapers[i].step(
                    prev_privileged=self._prev_privs[i],
                    curr_privileged=curr_priv_i,
                    action=actions_np[i],
                    env_reward=rew_i,
                    info=info_i,
                )
                step_rewards[i] = float(shaped_i)
                is_done = bool(term_i or trunc_i)
                step_dones[i] = 1.0 if is_done else 0.0
                step_truncs[i] = bool(trunc_i)

                if trunc_i and term_obs_i is not None:
                    trunc_indices.append(i)
                    trunc_obs_list.append(term_obs_i)
                    trunc_prev_actions.append(actions_np[i].astype(np.float32))

                if is_done:
                    _record_episode(
                        self.metrics_logger,
                        self._ep_ids[i],
                        map_i or self._map_specs[i],
                        seed_i if seed_i is not None else self._ep_seeds[i],
                        self._ep_steps[i],
                        self._ep_raw_returns[i],
                        self.shapers[i],
                        term_obs_i if term_obs_i is not None else next_obs_i,
                        curr_priv_i,
                        info_i,
                        bool(trunc_i),
                        self._ep_start_times[i],
                    )
                    self._ep_ids[i] = self._next_ep_id
                    self._next_ep_id += 1
                    self._ep_steps[i] = 0
                    self._ep_raw_returns[i] = 0.0
                    self._ep_start_times[i] = time.time()
                    self._map_specs[i] = map_i or "UNKNOWN"
                    self._ep_seeds[i] = seed_i
                    self._prev_privs[i] = reset_priv_i
                    self.shapers[i].reset(initial_privileged=reset_priv_i)
                    self._prev_actions[i] = 0.0
                else:
                    self._prev_privs[i] = curr_priv_i
                    self._prev_actions[i] = actions_np[i].astype(np.float32)

            if trunc_indices:
                trunc_batch = collate_observations(trunc_obs_list, device=self.device)
                trunc_h = None
                if is_recurrent:
                    trunc_batch["prev_action"] = torch.as_tensor(
                        np.stack(trunc_prev_actions), dtype=torch.float32, device=self.device
                    )
                    trunc_h = next_hidden[:, trunc_indices, :]
                with torch.no_grad():
                    _, trunc_vals, _ = policy.step(trunc_batch, trunc_h)
                    if trunc_vals.dim() > 1:
                        trunc_vals = trunc_vals.squeeze(-1)
                for local_idx, env_idx in enumerate(trunc_indices):
                    step_bootstraps[env_idx] = float(trunc_vals[local_idx].item())

            if next_hidden is not None:
                done_mask = torch.as_tensor(step_dones, dtype=next_hidden.dtype, device=self.device)
                next_hidden = next_hidden * (1.0 - done_mask).view(1, self.num_envs, 1)
            self._hidden_state = next_hidden
            self._obs_list = next_obs_list

            action_buffer.append(actions)
            reward_buffer.append(torch.as_tensor(step_rewards, dtype=torch.float32, device=self.device))
            done_buffer.append(torch.as_tensor(step_dones, dtype=torch.float32, device=self.device))
            value_buffer.append(values)
            log_prob_buffer.append(log_probs)
            truncation_buffer.append(torch.as_tensor(step_truncs, dtype=torch.bool, device=self.device))
            bootstrap_value_buffer.append(torch.as_tensor(step_bootstraps, dtype=torch.float32, device=self.device))

        # Bootstrap last_value from self._obs_list
        last_obs_batch = collate_observations(self._obs_list, device=self.device)
        if is_recurrent:
            last_obs_batch["prev_action"] = torch.as_tensor(
                self._prev_actions, dtype=torch.float32, device=self.device
            )
        with torch.no_grad():
            _, last_value, _ = policy.step(last_obs_batch, self._hidden_state)
            if last_value.dim() > 1:
                last_value = last_value.squeeze(-1)

        # Mask last_value to 0 for any env that finished on step T-1
        last_done_mask = done_buffer[-1]
        last_value = last_value * (1.0 - last_done_mask)

        collated_obs = _collate_time_env(step_obs_buffers)
        batch_data: Dict[str, Any] = {
            "obs": collated_obs,
            "actions": torch.stack(action_buffer, dim=0),
            "rewards": torch.stack(reward_buffer, dim=0),
            "dones": torch.stack(done_buffer, dim=0),
            "values": torch.stack(value_buffer, dim=0),
            "log_probs": torch.stack(log_prob_buffer, dim=0),
            "last_value": last_value,
            "truncations": torch.stack(truncation_buffer, dim=0),
            "bootstrap_values": torch.stack(bootstrap_value_buffer, dim=0),
        }
        if is_recurrent:
            batch_data["init_hiddens"] = torch.stack(hidden_buffer, dim=0)  # [T, N, H]
        return RolloutBatch(batch_data)


def run_environment_smoke_test(
    env: Any,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """Diagnostic environment smoke test to verify step execution, shapes, and FPS."""
    obs, info = env.reset()
    start_t = time.time()
    steps_run = 0

    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps_run += 1
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_t
    fps = (steps_run * 4) / max(elapsed, 1e-4)
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
