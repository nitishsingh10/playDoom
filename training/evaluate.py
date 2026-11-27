"""Evaluate checkpoints with combat diagnostics kept outside policy inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from doom_env import make_env, spaces as doom_spaces
from submission.actions import StructuredActionDistribution
from submission.checkpoint import (
    LEGACY_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURES,
    checkpoint_architecture, extract_state_dict, load_policy_state,
)
from submission.model import DoomPolicy
from submission.observation import ObservationEncoder
from training.metrics import FailureMode
from training.rewards import RewardConfig, RewardShaper

CONFLICT_PAIRS = {
    "forward_backward": (0, 1), "left_right": (2, 3), "turn_left_right": (4, 5),
}
ENEMY_TYPES = tuple(i for i, name in enumerate(doom_spaces.ENTITY_TYPES) if name.startswith("enemy_"))


def _map_spec(spec: str) -> tuple[str, str]:
    path, separator, map_name = spec.rpartition(":")
    if separator and map_name.upper().startswith(("MAP", "E")):
        return path, map_name.upper()
    return spec, "MAP01"


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


@dataclass
class ActionDiagnostics:
    """Count actions against the observations on which they were selected.

    Useful range is a distance proxy: visible enemy center <= useful_range_units.
    Raw entity distance is divided by 1024, not by the 4096 maximum visibility
    distance. It does not imply aim, weapon reach, loaded ammunition or a hit.
    """

    useful_range_units: float = doom_spaces.RAY_RANGE
    decisions: int = 0
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.int64))
    conflict_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CONFLICT_PAIRS, 0))
    enemy_visible_steps: int = 0
    enemy_useful_range_steps: int = 0
    attack_while_enemy_visible: int = 0
    attack_while_enemy_in_useful_range: int = 0
    forward_ray_enemy_steps: int = 0
    attack_while_forward_ray_enemy: int = 0
    selected_weapon_ammo_zero_steps: int = 0
    attack_with_zero_selected_weapon_ammo: int = 0
    use_button_down_edges: int = 0
    weapon_select_button_down_edges: int = 0
    weapon_selection_changes: int = 0
    _previous_action: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.int8))

    def __post_init__(self):
        if not np.isfinite(self.useful_range_units) or self.useful_range_units <= 0:
            raise ValueError("useful_range_units must be finite and positive")

    def observe_action(self, observation: dict, action: np.ndarray) -> None:
        action = np.asarray(action)
        if action.shape != (len(doom_spaces.BUTTONS),) or not np.all((action == 0) | (action == 1)):
            raise ValueError("diagnostic action must be binary with shape (9,)")
        self.decisions += 1
        self.action_counts += action.astype(np.int64)
        for name, (first, second) in CONFLICT_PAIRS.items():
            self.conflict_counts[name] += int(bool(action[first]) and bool(action[second]))
        entities = observation["visible_entities"]
        enemy = np.asarray(entities["mask"], dtype=bool) & np.isin(entities["entity_type"], ENEMY_TYPES)
        distances = np.asarray(entities["features"])[..., 0] * doom_spaces.RAY_RANGE
        visible = bool(np.any(enemy))
        in_range = bool(np.any(enemy & np.isfinite(distances) & (distances >= 0) & (distances <= self.useful_range_units)))
        forward_enemy = int(np.asarray(observation["rays"]["hit_type"])[0]) == doom_spaces.RAY_HIT_TYPES.index("enemy")
        empty = float(np.asarray(observation["vitals"])[doom_spaces.VITALS_INDEX["selected_weapon_ammo"]]) <= 0
        attack = bool(action[6])
        self.enemy_visible_steps += int(visible)
        self.enemy_useful_range_steps += int(in_range)
        self.attack_while_enemy_visible += int(attack and visible)
        self.attack_while_enemy_in_useful_range += int(attack and in_range)
        self.forward_ray_enemy_steps += int(forward_enemy)
        self.attack_while_forward_ray_enemy += int(attack and forward_enemy)
        self.selected_weapon_ammo_zero_steps += int(empty)
        self.attack_with_zero_selected_weapon_ammo += int(attack and empty)
        self.use_button_down_edges += int(bool(action[7]) and not bool(self._previous_action[7]))
        self.weapon_select_button_down_edges += int(bool(action[8]) and not bool(self._previous_action[8]))
        self._previous_action = action.copy()

    def observe_transition(self, previous_weapon: float, current_weapon: float) -> None:
        # Includes automatic switches and switches delayed by weapon animations.
        self.weapon_selection_changes += int(previous_weapon != current_weapon)

    def summary(self) -> dict[str, Any]:
        return {
            "action_activations": {
                name: {"count": int(count), "percentage": _percentage(int(count), self.decisions)}
                for name, count in zip(doom_spaces.BUTTONS, self.action_counts)
            },
            "action_conflicts": {
                name: {"count": count, "percentage": _percentage(count, self.decisions)}
                for name, count in self.conflict_counts.items()
            },
            "enemy_visible_steps": self.enemy_visible_steps,
            "enemy_useful_range_steps": self.enemy_useful_range_steps,
            "attack_activations": int(self.action_counts[6]),
            "attack_while_enemy_visible": self.attack_while_enemy_visible,
            "attack_while_enemy_visible_percentage": _percentage(self.attack_while_enemy_visible, self.enemy_visible_steps),
            "attack_while_enemy_in_useful_range": self.attack_while_enemy_in_useful_range,
            "attack_while_enemy_in_useful_range_percentage": _percentage(self.attack_while_enemy_in_useful_range, self.enemy_useful_range_steps),
            "forward_ray_enemy_steps": self.forward_ray_enemy_steps,
            "attack_while_forward_ray_enemy": self.attack_while_forward_ray_enemy,
            "selected_weapon_ammo_zero_steps": self.selected_weapon_ammo_zero_steps,
            "attack_with_zero_selected_weapon_ammo": self.attack_with_zero_selected_weapon_ammo,
            "use_activations": int(self.action_counts[7]),
            "use_button_down_edges": self.use_button_down_edges,
            "weapon_select_activations": int(self.action_counts[8]),
            "weapon_select_button_down_edges": self.weapon_select_button_down_edges,
            "weapon_selection_changes": self.weapon_selection_changes,
        }


def load_policy(checkpoint_path: str | Path, device: str, *, legacy_baseline: bool = False) -> DoomPolicy:
    # Training checkpoints contain optimizer/RNG state and are local trusted artifacts.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint_architecture(checkpoint)
    requested = (LEGACY_ARCHITECTURE_ID if legacy_baseline else
                 architecture if architecture in STRUCTURED_ARCHITECTURES else STRUCTURED_ARCHITECTURE_ID)
    if architecture != requested:
        hint = " Pass --legacy-baseline explicitly for a historical 9-logit baseline." if architecture == LEGACY_ARCHITECTURE_ID else ""
        raise ValueError(f"Checkpoint architecture {architecture!r} does not match requested {requested!r}.{hint}")
    state = extract_state_dict(checkpoint)
    # Recover dimensions even for historical state_dict-only baselines.
    hidden_size = (
        int(state["ray_conv.0.weight"].shape[0])
        if "ray_conv.0.weight" in state
        else int(state["actor.weight"].shape[1] // 2)
    )
    model = DoomPolicy(
        hidden_size=hidden_size,
        fusion_size=int(state["fusion.0.weight"].shape[0]), architecture_id=requested,
    )
    load_policy_state(model, checkpoint)
    return model.to(device).eval()


def _choose_action(logits: torch.Tensor, *, legacy_baseline: bool, stochastic: bool, action_threshold: float) -> np.ndarray:
    if legacy_baseline:
        # Isolated historical distribution only for explicit baseline comparison.
        distribution = torch.distributions.Bernoulli(logits=logits)
        bits = distribution.sample() if stochastic else (distribution.probs >= action_threshold)
    else:
        distribution = StructuredActionDistribution(logits)
        bits = distribution.sample() if stochastic else distribution.mode()
    return bits[0].to(torch.int8).cpu().numpy()


def evaluate_checkpoint(
    checkpoint_path: str | Path, maps: list[str], seeds: list[int], device: str = "cpu",
    episode_ticks: int = 2100, action_threshold: float = 0.5, *, legacy_baseline: bool = False,
    stochastic: bool = False, useful_range_units: float = doom_spaces.RAY_RANGE,
) -> dict[str, Any]:
    if not maps or not seeds:
        raise ValueError("at least one map and one seed are required")
    if not 0 <= action_threshold <= 1:
        raise ValueError("action_threshold must be between 0 and 1")
    if not legacy_baseline and action_threshold != 0.5:
        raise ValueError("--action-threshold only applies to --legacy-baseline; structured inference uses the shared mode")
    ActionDiagnostics(useful_range_units=useful_range_units)
    torch.set_num_threads(1)
    model = load_policy(checkpoint_path, device, legacy_baseline=legacy_baseline)
    encoder = ObservationEncoder(device=device, include_depth=bool(getattr(model, "use_depth", False)))
    records = []
    for spec in maps:
        wad, map_name = _map_spec(spec)
        for seed in seeds:
            torch.manual_seed(seed)  # Reproducible sampled actions per episode.
            env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=bool(getattr(model, "use_depth", False)))
            try:
                observation, _ = env.reset(seed=seed)
                state = env.privileged_state()
                initial_health = float(state["health"])
                initial_kills = int(state["kills"])
                initial_damage = float(state["damage_taken"])
                previous_weapon = float(np.asarray(observation["vitals"])[2])
                diagnostics = ActionDiagnostics(useful_range_units=useful_range_units)
                shaper = RewardShaper()  # Metrics only; no effect on evaluation actions.
                shaper.reset(state)
                raw_return = 0.0
                done = False
                visited_cells = set()
                hidden_state = model.init_hidden(batch_size=1, device=device) if hasattr(model, "init_hidden") else None
                prev_action = np.zeros(9, dtype=np.float32)
                while not done:
                    encoded = encoder(observation)
                    if getattr(model, "is_recurrent", False):
                        encoded["prev_action"] = torch.as_tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.inference_mode():
                        logits, _, hidden_state = model.step(encoded, hidden_state) if hasattr(model, "step") else model(encoded)
                        action = _choose_action(logits, legacy_baseline=legacy_baseline, stochastic=stochastic, action_threshold=action_threshold)
                    prev_action = action.astype(np.float32)
                    diagnostics.observe_action(observation, action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    raw_return += float(reward)
                    previous_state = state
                    state = env.privileged_state()
                    shaper.step(previous_state, state, action, env_reward=reward, info=info)
                    visited_cells.add((int(state["x"] // 64.0), int(state["y"] // 64.0)))
                    diagnostics.observe_transition(previous_weapon, float(state["selected_weapon"]))
                    previous_weapon = float(state["selected_weapon"])
                    done = bool(terminated or truncated)
                unique_cells = len(visited_cells)
                ammo = float(np.asarray(observation["vitals"])[4:8].sum())
                keys = int(np.asarray(state["keys"]).sum())
                ticks = int(state["tic"])
                record = {
                    "map": f"{wad}:{map_name}", "seed": seed,
                    "decisions": diagnostics.decisions, "ticks": ticks, "raw_return": raw_return,
                    "exit_reached": bool(state["exit_reached"]), "dead": bool(state["dead"]), "timeout": bool(state["timeout"]),
                    "termination_reason": "death" if state["dead"] else "exit" if state["exit_reached"] else "timeout" if state["timeout"] else "other",
                    "kills": int(state["kills"]) - initial_kills,
                    "damage_taken": float(state["damage_taken"]) - initial_damage,
                    "initial_health": initial_health,
                    # Terminal observations are stale; engine variables include final tics.
                    "final_health": float(state["health"]),
                    "final_selected_weapon": float(state["selected_weapon"]),
                    "final_selected_weapon_ammo": float(state["selected_weapon_ammo"]),
                    "keycards": keys, "ammo_remaining": ammo, "unique_cells": unique_cells,
                    "doors_opened": None, "reward_components": shaper.get_episode_summary(),
                    **diagnostics.summary(),
                }
                record["failure_mode"] = FailureMode.diagnose(
                    exit_reached=record["exit_reached"], dead=record["dead"], timeout=record["timeout"],
                    ticks=ticks, max_ticks=episode_ticks, unique_cells=unique_cells,
                    door_spam_count=shaper.door_spam_count, ammo_remaining=ammo, damage_taken=record["damage_taken"],
                )
                records.append(record)
            finally:
                env.close()
    total_decisions = sum(r["decisions"] for r in records)
    return {
        "checkpoint": str(checkpoint_path),
        "architecture_id": model.architecture_id,
        "action_mode": "stochastic" if stochastic else "deterministic",
        "action_threshold": action_threshold if legacy_baseline else None,
        "episode_ticks": episode_ticks, "episodes": len(records),
        "exit_rate": mean(float(r["exit_reached"]) for r in records),
        "survival_rate": mean(float(not r["dead"]) for r in records),
        "mean_raw_return": mean(r["raw_return"] for r in records),
        "mean_kills": mean(r["kills"] for r in records),
        "mean_damage_taken": mean(r["damage_taken"] for r in records),
        "mean_keycards": mean(r["keycards"] for r in records),
        "mean_ammo_remaining": mean(r["ammo_remaining"] for r in records),
        "mean_unique_cells": mean(r["unique_cells"] for r in records),
        "action_conflicts": {
            name: {"count": sum(r["action_conflicts"][name]["count"] for r in records),
                   "percentage": _percentage(sum(r["action_conflicts"][name]["count"] for r in records), total_decisions)}
            for name in CONFLICT_PAIRS
        },
        "reward_config_for_diagnostics": asdict(RewardConfig()),
        "metric_definitions": {
            "conflict_percentage": "Exact simultaneous opposing-button decisions / all decisions * 100.",
            "enemy_visible": "Mask=1 and enemy entity type 1, 2 or 3 in the action's input observation (FOV and line of sight).",
            "useful_range_units": useful_range_units,
            "useful_range": "Visible enemy center distance = raw entity feature[0] * 1024 <= useful_range_units. Distance proxy only; not weapon-specific reach, aim or successful damage.",
            "attack_percentage": "Attack+enemy condition decisions / decisions satisfying that condition * 100.",
            "forward_ray_enemy": "Observation ray index 0 (straight ahead) has enemy hit type; a cue, not confirmed damage.",
            "edge_counts": "Button=1 after previous decision button=0, with previous=0 at reset. Activation counts include held decisions.",
            "weapon_selection_changes": "Changes in selected weapon ID across decisions, including automatic and delayed switches; distinct from button activations and edges.",
            "zero_selected_weapon_ammo": "Reported selected weapon ammo <=0; melee weapons can attack without ammo.",
            "reward_components": "Default training reward-shaper totals recomputed for diagnosis only; no effect on evaluation actions.",
        },
        "episodes_detail": records,
        "unavailable_metrics": {
            "doors_opened": "The privileged state does not expose a door-open counter.",
            "damage_dealt": "The wrapper does not expose damage dealt; visibility and ATTACK do not prove a hit.",
            "ammo_remaining": "Reserve ammunition uses the last returned observation, which may precede the terminal tics.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map", action="append", dest="maps", required=True, help="WAD path[:MAPxx], repeatable")
    parser.add_argument("--seed", action="append", dest="seeds", type=int, default=None, help="Repeatable seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--legacy-baseline", action="store_true", help="Explicitly evaluate a preserved 9-logit Bernoulli checkpoint")
    parser.add_argument("--stochastic", action="store_true", help="Seeded sampling; default is deterministic shared mode")
    parser.add_argument("--action-threshold", type=float, default=0.5, help="Historical Bernoulli baseline only")
    parser.add_argument("--useful-range-units", type=float, default=doom_spaces.RAY_RANGE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint, args.maps, args.seeds if args.seeds is not None else [0],
        device=args.device, episode_ticks=args.episode_ticks, action_threshold=args.action_threshold,
        legacy_baseline=args.legacy_baseline, stochastic=args.stochastic, useful_range_units=args.useful_range_units,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
