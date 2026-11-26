"""Opt-in, paired combat-credit pilot. Existing training/submission behavior is unchanged.

Both arms resume one structured checkpoint with identical PPO settings. Only the
treatment receives bounded credit for native DAMAGECOUNT deltas. Privileged
counters never enter observations. All outputs go to a new experiment directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np
import torch
import vizdoom as vd

from doom_env import make_env
from submission.checkpoint import STRUCTURED_ARCHITECTURE_ID, checkpoint_architecture
from submission.model import build_policy
from submission.observation import ObservationEncoder
from training.envs import MapCyclerEnv, default_training_maps
from training.evaluate import ActionDiagnostics, _choose_action, load_policy
from training.metrics import MetricsLogger
from training.rewards import RewardShaper
from training.rollout import RolloutCollector
from training.train import default_train_config, train_loop


class CombatTelemetryEnv:
    """Training/scorer-only native counters; reset/step return original observations."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        self.privileged_state()  # Fail before collection if native counters cannot be read.
        return result

    def step(self, action):
        result = self._env.step(action)
        self.privileged_state()  # Collector's optional privileged-state fallback must not hide errors.
        return result

    def privileged_state(self):
        state = dict(self._env.privileged_state())
        owner = getattr(self._env, "env", self._env)  # MapCyclerEnv or direct DoomEnv.
        game = owner._game
        for name, variable in (
            ("damage_dealt", vd.GameVariable.DAMAGECOUNT),
            ("hits", vd.GameVariable.HITCOUNT),
            ("ammo_bullets", vd.GameVariable.AMMO2),
            ("attack_ready", vd.GameVariable.ATTACK_READY),
        ):
            state[name] = float(game.get_game_variable(variable))
            if not math.isfinite(state[name]):
                raise ValueError(f"Invalid native combat counter: {name}")
        return state


class CombatRewardShaper(RewardShaper):
    """Base rewards plus coefficient * positive damage delta, capped per episode.

    Coefficient zero is the unchanged-reward control. ATTACK, visibility, hit
    counts and ammo expenditure themselves earn no reward. Delayed projectile
    damage may earn credit even when ATTACK is no longer pressed.
    """

    def __init__(self, coefficient=0.0, episode_cap=2.0):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("damage coefficient must be finite and nonnegative")
        if not math.isfinite(episode_cap) or episode_cap <= 0:
            raise ValueError("episode reward cap must be finite and positive")
        self.coefficient = float(coefficient)
        self.episode_cap = float(episode_cap)
        super().__init__()
        self.reset()

    def _zero_components(self):
        return {**super()._zero_components(), "damage_dealt_credit": 0.0}

    def reset(self, initial_privileged=None):
        super().reset(initial_privileged)
        self.combat = dict(hits=0.0, damage_dealt=0.0, bullets_spent=0.0,
                           pistol_ready_decisions=0, attack_when_pistol_ready=0,
                           first_kill_decision=None)

    def step(self, prev_privileged, curr_privileged, action, env_reward=0.0, info=None):
        prev = prev_privileged if prev_privileged is not None else self.prev_privileged
        prev = curr_privileged if prev is None else prev
        for state in (prev, curr_privileged):
            for key in ("damage_dealt", "hits", "ammo_bullets", "attack_ready"):
                if key not in state or not math.isfinite(float(state[key])):
                    raise ValueError(f"Missing or invalid native combat counter: {key}")
        damage = max(0.0, curr_privileged["damage_dealt"] - prev["damage_dealt"])
        remaining = max(0.0, self.episode_cap - self.total_components["damage_dealt_credit"])
        credit = min(self.coefficient * damage, remaining)
        reward, components = super().step(prev, curr_privileged, action, env_reward, info)
        components["damage_dealt_credit"] = credit
        self.total_components["damage_dealt_credit"] += credit
        self.combat["damage_dealt"] += damage
        self.combat["hits"] += max(0.0, curr_privileged["hits"] - prev["hits"])
        self.combat["bullets_spent"] += max(0.0, prev["ammo_bullets"] - curr_privileged["ammo_bullets"])
        if prev.get("selected_weapon") == 2 and prev["ammo_bullets"] > 0 and prev["attack_ready"]:
            self.combat["pistol_ready_decisions"] += 1
            self.combat["attack_when_pistol_ready"] += int(bool(action[6]))
        if curr_privileged.get("kills", 0) > prev.get("kills", 0) and self.combat["first_kill_decision"] is None:
            self.combat["first_kill_decision"] = self.step_count
        return reward + credit, components

    def get_episode_summary(self):
        return {**super().get_episode_summary(), **{f"combat_{k}": v for k, v in self.combat.items()}}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentCollector(RolloutCollector):
    def collect(self, policy, num_steps=256):
        # train_loop restores the parent's RNG on resume. Establish this pilot's
        # explicit seed after restoration, identically in both experimental arms.
        if not getattr(self, "_experiment_seeded", False):
            seed_all(self.seed)
            self._experiment_seeded = True
        return super().collect(policy, num_steps)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Protected input changed: {path}")


def evaluate(checkpoint, maps, seeds, stochastic):
    """Same learned structured actions as standard evaluation, with native telemetry."""
    model = load_policy(checkpoint, "cpu")
    encoder = ObservationEncoder(device="cpu")
    records = []
    for wad, map_name in maps:
        for seed in seeds:
            torch.manual_seed(seed)
            env = CombatTelemetryEnv(make_env(wad, map_name, episode_ticks=2100, render_buffers=False))
            try:
                observation, _ = env.reset(seed=seed)
                state = env.privileged_state()
                initial = dict(state)
                diagnostics = ActionDiagnostics()
                shaper = CombatRewardShaper()  # Common base rewards for evaluation.
                shaper.reset(state)
                raw_return = 0.0
                done = False
                first_kill_tick = None
                while not done:
                    with torch.inference_mode():
                        logits, _, _ = model(encoder(observation))
                        action = _choose_action(logits, legacy_baseline=False, stochastic=stochastic, action_threshold=0.5)
                    diagnostics.observe_action(observation, action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    previous = state
                    state = env.privileged_state()
                    diagnostics.observe_transition(previous["selected_weapon"], state["selected_weapon"])
                    shaper.step(previous, state, action, reward, info)
                    if state["kills"] > initial["kills"] and first_kill_tick is None:
                        first_kill_tick = state["tic"]
                    raw_return += reward
                    done = terminated or truncated
                records.append({
                    "map": f"{wad}:{map_name}", "seed": seed,
                    "ticks": state["tic"], "decisions": diagnostics.decisions,
                    "kills": state["kills"] - initial["kills"],
                    "damage_taken": state["damage_taken"] - initial["damage_taken"],
                    "final_health": state["health"], "dead": state["dead"],
                    "exit_reached": state["exit_reached"], "raw_return": raw_return,
                    "final_selected_weapon": state["selected_weapon"],
                    "time_to_first_kill_ticks": first_kill_tick,
                    "reward_components": shaper.get_episode_summary(),
                    **diagnostics.summary(),
                })
            finally:
                env.close()
    return {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
            "stochastic": stochastic, "seeds": seeds, "episode_ticks": 2100,
            "action_intervention": None, "episodes_detail": records}


def print_comparison(results):
    print("POLICY/MODE | MAP | EXITS | ANY KILL | KILLS/EP | HITS/EP | DEALT/EP | SURVIVED", flush=True)
    for label, result in results.items():
        groups = sorted({r["map"] for r in result["episodes_detail"]})
        for map_name in groups:
            rows = [r for r in result["episodes_detail"] if r["map"] == map_name]
            n = len(rows)
            exits = "N/A" if "tier1_projectile" in map_name else f"{sum(r['exit_reached'] for r in rows)}/{n}"
            kills = sum(r["kills"] for r in rows) / n
            hits = sum(r["reward_components"]["combat_hits"] for r in rows) / n
            dealt = sum(r["reward_components"]["combat_damage_dealt"] for r in rows) / n
            print(f"{label} | {map_name} | {exits} | {sum(r['kills'] > 0 for r in rows)}/{n} | "
                  f"{kills:.2f} | {hits:.2f} | {dealt:.1f} | {sum(not r['dead'] for r in rows)}/{n}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--additional-updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--damage-coefficient", type=float, default=0.02)
    parser.add_argument("--episode-credit-cap", type=float, default=2.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("training/runs"))
    args = parser.parse_args(argv)
    if args.additional_updates < 1 or args.eval_episodes < 1 or args.seed < 0:
        parser.error("updates/episodes must be positive and seed nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use the VM's .venv/bin/python")
    CombatRewardShaper(args.damage_coefficient, args.episode_credit_cap)
    if args.damage_coefficient <= 0:
        parser.error("treatment coefficient must be positive")
    parent = args.parent.resolve(strict=True)
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if checkpoint_architecture(saved) != STRUCTURED_ARCHITECTURE_ID:
        parser.error("parent must be a structured policy checkpoint")
    if not saved.get("optimizer_state_dict") or int(saved.get("update", 0)) < 1:
        parser.error("parent must contain trained weights, Adam state and an update counter")
    if saved.get("model_config", {}).get("architecture_id") != STRUCTURED_ARCHITECTURE_ID:
        parser.error("parent lacks structured model configuration")
    training_maps = [("maps/training/tier1_projectile.wad", "MAP01")]
    maps = default_training_maps() + [("maps/two_rooms.wad", "MAP01")]
    protected = [parent, Path("submission/weights/checkpoint.pt"), Path("submission/weights/config.json")]
    previous = Path("training/runs/structured_combat_seed17_SFhSzA/checkpoints/checkpoint_2500.pt")
    if previous.is_file():
        protected.append(previous)
    protected += [Path(wad) for wad, _ in maps]
    hashes = {str(p.resolve(strict=True)): sha256(p) for p in protected}
    torch.set_num_threads(1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"combat_credit_seed{args.seed}_", dir=args.output_root)).resolve()
    config = default_train_config()
    config.update(saved.get("config", {}))
    target = int(saved["update"]) + args.additional_updates
    config.update(device=args.device, total_updates=target, experiment_seed=args.seed,
                  training_maps=training_maps, episode_ticks=2100,
                  parent_checkpoint=str(parent), parent_checkpoint_sha256=sha256(parent))
    manifest = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parent": str(parent), "protected_sha256": hashes, "seed": args.seed,
        "start_update": saved["update"], "final_update": target,
        "additional_decisions_per_arm": args.additional_updates * config["rollout_steps"],
        "ppo_config": config, "training_maps": training_maps, "evaluation_maps": maps,
        "evaluation_seeds": list(range(100, 100 + args.eval_episodes)),
        "episode_ticks": 2100, "damage_coefficient": args.damage_coefficient,
        "episode_credit_cap": args.episode_credit_cap, "torch": torch.__version__,
        "rng_protocol": "Restore parent optimizer/weights, then reseed policy and fresh environment identically per arm.",
        "status": "running", "action_intervention": None,
    }
    write_json(output / "manifest.json", manifest)
    print(f"EXPERIMENT DIRECTORY: {output}", flush=True)
    print(f"Each arm: {args.additional_updates} additional updates; {manifest['additional_decisions_per_arm']} decisions.", flush=True)
    candidates = {"parent": parent}
    try:
        for name, coefficient in (("control", 0.0), ("damage_credit", args.damage_coefficient)):
            arm = output / name
            checkpoints = arm / "checkpoints"
            logger = MetricsLogger(log_dir=arm / "logs")
            env = CombatTelemetryEnv(MapCyclerEnv(training_maps, episode_ticks=2100, seed=args.seed))
            collector = ExperimentCollector(env, shaper=CombatRewardShaper(coefficient, args.episode_credit_cap),
                                            metrics_logger=logger, device=args.device, seed=args.seed)
            arm_config = {**config, "save_dir": str(checkpoints), "combat_damage_coefficient": coefficient,
                          "combat_episode_credit_cap": args.episode_credit_cap}
            print(f"TRAINING ARM: {name}", flush=True)
            try:
                train_loop(build_policy(saved["model_config"]), collector, config=arm_config,
                           resume_path=str(parent), metrics_logger=logger)
            finally:
                env.close()
                logger.save_json("training_metrics.json")
                write_json(arm / "all_completed_episodes.json", [asdict(r) for r in logger.episodes])
            candidates[name] = checkpoints / f"checkpoint_{target}.pt"
            verify_hashes(hashes)
        results = {}
        for name, checkpoint in candidates.items():
            for mode in ("stochastic", "deterministic"):
                label = f"{name}_{mode}"
                print(f"EVALUATING: {label}", flush=True)
                results[label] = evaluate(checkpoint, maps, manifest["evaluation_seeds"], mode == "stochastic")
                write_json(output / f"{label}.json", results[label])
        print_comparison(results)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest["protected_inputs_unchanged"] = all(sha256(p) == h for p, h in hashes.items())
        write_json(output / "manifest.json", manifest)
        verify_hashes(hashes)
    print(f"DONE. No weights promoted. Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
