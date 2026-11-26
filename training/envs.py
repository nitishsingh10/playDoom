"""Training-only environment factories and map curriculum helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from doom_env import make_env, spaces as doom_spaces


class MapCyclerEnv:
    """Select a training WAD per episode while preserving the doom_env contract.

    This wrapper never changes evaluation behavior. It only chooses among the
    explicitly supplied training maps when a rollout episode resets.
    """

    def __init__(self, maps: Iterable[tuple[str, str]], episode_ticks: int = 2100, seed: int | None = None):
        self.maps = [(str(Path(wad)), map_name.upper()) for wad, map_name in maps]
        if not self.maps:
            raise ValueError("MapCyclerEnv needs at least one WAD/map pair")
        self.episode_ticks = int(episode_ticks)
        self.rng = np.random.default_rng(seed)
        self.observation_space = doom_spaces.observation_space()
        self.action_space = doom_spaces.action_space()
        self.env = None
        self._active_spec = None
        self.map_name = "UNKNOWN"
        self.episode_map_spec = None
        self.current_episode_seed = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        index = int(self.rng.integers(len(self.maps)))
        wad, map_name = self.maps[index]
        spec = (wad, map_name)
        if self.env is not None and spec != self._active_spec:
            self.env.close()
            self.env = None
        if self.env is None:
            self.env = make_env(wad, map_name, episode_ticks=self.episode_ticks, render_buffers=False)
            self._active_spec = spec
        self.map_name = map_name
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self.episode_map_spec = f"{wad}:{map_name}"
        self.current_episode_seed = episode_seed
        return self.env.reset(seed=episode_seed, options=options)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("call reset() before step()")
        return self.env.step(action)

    def privileged_state(self) -> dict:
        if self.env is None:
            raise RuntimeError("call reset() before privileged_state()")
        return self.env.privileged_state()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self._active_spec = None


def default_training_maps() -> list[tuple[str, str]]:
    """Return the bundled low-cost synthetic progression curriculum."""
    return [
        ("maps/training/tier1_projectile.wad", "MAP01"),
        ("maps/training/tier2_door.wad", "MAP01"),
        ("maps/training/tier3_red_key.wad", "MAP01"),
    ]
