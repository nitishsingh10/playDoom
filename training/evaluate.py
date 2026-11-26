"""Evaluate a local learned checkpoint on explicitly selected Doom maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from doom_env import make_env, spaces as doom_spaces
from submission.model import DoomPolicy
from submission.observation import ObservationEncoder
from training.metrics import FailureMode


def _map_spec(spec: str) -> tuple[str, str]:
    path, separator, map_name = spec.rpartition(":")
    if separator and map_name.upper().startswith(("MAP", "E")):
        return path, map_name.upper()
    return spec, "MAP01"


def load_policy(checkpoint_path: str | Path, device: str) -> DoomPolicy:
    # Training checkpoints contain optimizer/RNG state; evaluation is a local
    # workflow and only loads checkpoints produced by this repository.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DoomPolicy()
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    maps: list[str],
    seeds: list[int],
    device: str = "cpu",
    episode_ticks: int = 2100,
    action_threshold: float = 0.5,
) -> dict[str, Any]:
    if not maps or not seeds:
        raise ValueError("at least one map and one seed are required")
    model = load_policy(checkpoint_path, device)
    encoder = ObservationEncoder(device=device)
    records = []

    for spec in maps:
        wad, map_name = _map_spec(spec)
        for seed in seeds:
            env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=False)
            try:
                observation, _ = env.reset(seed=seed)
                raw_return = 0.0
                decisions = 0
                done = False
                visited_cells = set()
                while not done:
                    encoded = encoder(observation)
                    with torch.inference_mode():
                        logits, _, _ = model(encoded)
                        action = (torch.sigmoid(logits[0]) >= action_threshold).to(torch.int8).cpu().numpy()
                    observation, reward, terminated, truncated, _ = env.step(action)
                    raw_return += float(reward)
                    decisions += 1
                    metric_state = env.privileged_state()
                    visited_cells.add((
                        int(metric_state["x"] // 64.0),
                        int(metric_state["y"] // 64.0),
                    ))
                    done = bool(terminated or truncated)

                # Privileged values are read only after the policy episode ends
                # and only for local evaluation metrics, never for action choice.
                state = env.privileged_state()
                unique_cells = len(visited_cells)
                ammo = float(np.asarray(observation["vitals"])[4:8].sum())
                keys = int(np.asarray(observation["vitals"])[8:14].sum())
                ticks = int(state["tic"])
                record = {
                    "map": f"{wad}:{map_name}",
                    "seed": seed,
                    "decisions": decisions,
                    "ticks": ticks,
                    "raw_return": raw_return,
                    "exit_reached": bool(state["exit_reached"]),
                    "dead": bool(state["dead"]),
                    "timeout": bool(state["timeout"]),
                    "kills": int(state["kills"]),
                    "damage_taken": float(state["damage_taken"]),
                    "keycards": keys,
                    "ammo_remaining": ammo,
                    "unique_cells": unique_cells,
                    "doors_opened": None,
                }
                record["failure_mode"] = FailureMode.diagnose(
                    exit_reached=record["exit_reached"],
                    dead=record["dead"],
                    timeout=record["timeout"],
                    ticks=ticks,
                    max_ticks=episode_ticks,
                    unique_cells=unique_cells,
                    door_spam_count=0,
                    ammo_remaining=ammo,
                    damage_taken=record["damage_taken"],
                )
                records.append(record)
            finally:
                env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "episodes": len(records),
        "exit_rate": mean(float(r["exit_reached"]) for r in records),
        "survival_rate": mean(float(not r["dead"]) for r in records),
        "mean_raw_return": mean(r["raw_return"] for r in records),
        "mean_kills": mean(r["kills"] for r in records),
        "mean_damage_taken": mean(r["damage_taken"] for r in records),
        "mean_keycards": mean(r["keycards"] for r in records),
        "mean_ammo_remaining": mean(r["ammo_remaining"] for r in records),
        "mean_unique_cells": mean(r["unique_cells"] for r in records),
        "episodes_detail": records,
        "unavailable_metrics": {
            "doors_opened": "The official privileged state does not expose a door-open counter.",
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map", action="append", dest="maps", required=True, help="WAD path[:MAPxx], repeatable")
    parser.add_argument("--seed", action="append", dest="seeds", type=int, default=None, help="Repeatable seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--action-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint,
        args.maps,
        args.seeds if args.seeds is not None else [0],
        device=args.device,
        episode_ticks=args.episode_ticks,
        action_threshold=args.action_threshold,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
