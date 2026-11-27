"""Training-only environment factories, curriculum maps, random map cycling, and vectorized workers."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import vizdoom as vd

from doom_env import make_env, spaces as doom_spaces
from training.mapgen import generate_random_map, write_udmf_wad


class MapCyclerEnv:
    """Select a training WAD (or generate a random UDMF map) per episode.

    This wrapper never changes evaluation behavior. It only chooses among the
    explicitly supplied training maps or freshly generated random maps on reset.
    """

    def __init__(
        self,
        maps: Iterable[tuple[str, str]],
        episode_ticks: int = 2100,
        seed: int | None = None,
        episodes_per_map: int = 1,
        random_map_prob: float = 0.0,
        random_tier_probs: tuple[float, float, float, float] = (0.10, 0.25, 0.35, 0.30),
        render_buffers: bool = False,
    ):
        self.maps = [(str(Path(wad)), map_name.upper()) for wad, map_name in maps]
        if not self.maps:
            raise ValueError("MapCyclerEnv needs at least one WAD/map pair")
        self.episode_ticks = int(episode_ticks)
        self.episodes_per_map = max(1, int(episodes_per_map))
        self.random_map_prob = float(random_map_prob)
        self.random_tier_probs = tuple(float(x) for x in random_tier_probs)
        self.render_buffers = bool(render_buffers)
        self._episodes_on_current_map = 0
        self.rng = np.random.default_rng(seed)
        self.observation_space = doom_spaces.observation_space()
        self.action_space = doom_spaces.action_space()
        self.env = None
        self._active_spec = None
        self._tmp_wad: Path | None = None
        self.map_name = "UNKNOWN"
        self.episode_map_spec = None
        self.current_episode_seed = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        need_new_map = self._active_spec is None or self._episodes_on_current_map >= self.episodes_per_map
        if need_new_map:
            self._episodes_on_current_map = 0
            if self.random_map_prob > 0.0 and self.rng.random() < self.random_map_prob:
                tier = int(self.rng.choice([1, 2, 3, 4], p=self.random_tier_probs))
                spec_obj = generate_random_map(self.rng, tier=tier)
                if self._tmp_wad is None:
                    self._tmp_wad = Path(f"/tmp/playdoom_rand_{os.getpid()}_{id(self)}.wad")
                if self.env is not None:
                    self.env.close()
                    self.env = None
                write_udmf_wad(self._tmp_wad, spec_obj.grid, player_angle=spec_obj.player_angle)
                wad, map_name = str(self._tmp_wad), "MAP01"
                spec = (wad, map_name)
                label = f"random:{spec_obj.name}"
                self._active_spec = None  # Force fresh DoomEnv load if another random map follows
                self.env = make_env(wad, map_name, episode_ticks=self.episode_ticks, render_buffers=self.render_buffers)
                self._active_spec = spec
                self._label = label
            else:
                index = int(self.rng.integers(len(self.maps)))
                wad, map_name = self.maps[index]
                spec = (wad, map_name)
                if self.env is not None and spec != self._active_spec:
                    self.env.close()
                    self.env = None
                if self.env is None:
                    self.env = make_env(wad, map_name, episode_ticks=self.episode_ticks, render_buffers=self.render_buffers)
                    self._active_spec = spec
                self._label = f"{wad}:{map_name}"
        self._episodes_on_current_map += 1
        self.map_name = self._active_spec[1]
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self.episode_map_spec = getattr(self, "_label", f"{self._active_spec[0]}:{self.map_name}")
        self.current_episode_seed = episode_seed
        return self.env.reset(seed=episode_seed, options=options)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("call reset() before step()")
        return self.env.step(action)

    def privileged_state(self) -> dict:
        if self.env is None:
            raise RuntimeError("call reset() before privileged_state()")
        state = dict(self.env.privileged_state())
        game = self.env._game
        state["damage_dealt"] = float(game.get_game_variable(vd.GameVariable.DAMAGECOUNT))
        state["hits"] = float(game.get_game_variable(vd.GameVariable.HITCOUNT))
        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self._active_spec = None
        if self._tmp_wad is not None and self._tmp_wad.exists():
            try:
                self._tmp_wad.unlink()
            except OSError:
                pass


def _vec_worker(
    remote,
    parent_remote,
    maps: list[tuple[str, str]],
    episode_ticks: int,
    seed: int,
    random_map_prob: float,
    render_buffers: bool = False,
) -> None:
    parent_remote.close()
    env = MapCyclerEnv(
        maps,
        episode_ticks=episode_ticks,
        seed=seed,
        episodes_per_map=2 if random_map_prob > 0 else 3,
        random_map_prob=random_map_prob,
        render_buffers=render_buffers,
    )
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                priv = env.privileged_state()
                remote.send((obs, priv, env.episode_map_spec, env.current_episode_seed))
            elif cmd == "step":
                next_obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    term_obs = next_obs
                    term_priv = env.privileged_state()
                    term_map = env.episode_map_spec
                    term_seed = env.current_episode_seed
                    reset_obs, _ = env.reset()
                    reset_priv = env.privileged_state()
                    remote.send((
                        reset_obs,
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        info,
                        term_obs,
                        term_priv,
                        reset_priv,
                        term_map,
                        term_seed,
                    ))
                else:
                    curr_priv = env.privileged_state()
                    remote.send((
                        next_obs,
                        float(reward),
                        False,
                        False,
                        info,
                        None,
                        curr_priv,
                        None,
                        env.episode_map_spec,
                        env.current_episode_seed,
                    ))
            elif cmd == "close":
                remote.close()
                break
    finally:
        env.close()


class SubprocVecEnv:
    """Multi-process vectorized training environment running N MapCyclerEnv workers."""

    def __init__(
        self,
        maps: Iterable[tuple[str, str]],
        num_envs: int = 8,
        episode_ticks: int = 2100,
        seed: int = 0,
        random_map_prob: float = 0.0,
        render_buffers: bool = False,
    ):
        self.maps = [(str(Path(wad)), map_name.upper()) for wad, map_name in maps]
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.episode_ticks = int(episode_ticks)
        self.random_map_prob = float(random_map_prob)
        self.observation_space = doom_spaces.observation_space()
        self.action_space = doom_spaces.action_space()
        self._closed = False

        ctx = mp.get_context("spawn")
        self._remotes: list[Any] = []
        self._work_remotes: list[Any] = []
        self._procs: list[Any] = []
        for idx in range(self.num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_vec_worker,
                args=(child_conn, parent_conn, self.maps, self.episode_ticks, int(seed) + idx * 1009, self.random_map_prob, bool(render_buffers)),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._work_remotes.append(child_conn)
            self._procs.append(proc)

    def reset(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str | None], list[int | None]]:
        for remote in self._remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self._remotes]
        obs_list = [r[0] for r in results]
        priv_list = [r[1] for r in results]
        map_list = [r[2] for r in results]
        seed_list = [r[3] for r in results]
        return obs_list, priv_list, map_list, seed_list

    def step(self, actions: Sequence[np.ndarray]):
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", np.asarray(action, dtype=np.int8)))
        return [remote.recv() for remote in self._remotes]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()


def default_training_maps() -> list[tuple[str, str]]:
    """Return the bundled low-cost synthetic progression curriculum."""
    return [
        ("maps/training/tier1_projectile.wad", "MAP01"),
        ("maps/training/tier2_door.wad", "MAP01"),
        ("maps/training/tier3_red_key.wad", "MAP01"),
    ]


def freedoom_training_maps() -> list[tuple[str, str]]:
    """Return real DOOM-geometry maps from the bundled Freedoom2 IWAD."""
    iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
    return [(iwad, m) for m in ("MAP01", "MAP02", "MAP03", "MAP04", "MAP05")]


def default_curriculum_maps(include_freedoom: bool = False) -> list[tuple[str, str]]:
    """Return all synthetic and procedural multi-room training WADs."""
    wads = sorted(Path("maps/training").glob("*.wad"))
    maps = [(str(p), "MAP01") for p in wads] if wads else default_training_maps()
    if include_freedoom:
        maps.extend(freedoom_training_maps())
    return maps


def default_heldout_maps() -> list[tuple[str, str]]:
    """Return held-out evaluation WADs strictly separate from training maps."""
    maps: list[tuple[str, str]] = [("maps/two_rooms.wad", "MAP01")]
    for p in sorted(Path("maps/heldout").glob("*.wad")):
        maps.append((str(p), "MAP01"))
    return maps
