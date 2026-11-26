"""Metrics, diagnostics, evaluation summaries, and failure mode analysis.

Owned by Member 4 (Environment, rollouts, maps, and reward shaping).

Provides:
  - EpisodeTracker: per-episode telemetry and statistics.
  - FailureMode: automated diagnosis of failed runs (stuck, door bounce, early death, ammo, etc.).
  - MetricsLogger: rolling averages, PPO logging integration, and JSON reporting.
  - compare_evaluations: quantitative comparison between two checkpoints or baseline runs.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple


class FailureMode:
    """Diagnoses the primary cause of an episode ending without reaching the exit."""

    SUCCESS = "success"
    EARLY_DEATH = "early_death"          # Died in first 20% of episode ticks
    COMBAT_DEATH = "combat_death"        # Died later in combat
    DOOR_BOUNCE = "door_bounce"          # High consecutive USE presses in doorway without opening
    STUCK_NAV = "stuck_nav"              # Low unique cells / minimal movement despite living long
    AMMO_DEPLETION = "ammo_depleted"     # Out of ammo while enemies are still present
    TIMEOUT = "timeout"                  # Ran out of time without finding exit

    @classmethod
    def diagnose(
        cls,
        exit_reached: bool,
        dead: bool,
        timeout: bool,
        ticks: int,
        max_ticks: int,
        unique_cells: int,
        door_spam_count: int,
        ammo_remaining: float,
        damage_taken: float,
    ) -> str:
        if exit_reached:
            return cls.SUCCESS

        if dead:
            if ticks < max_ticks * 0.25:
                return cls.EARLY_DEATH
            return cls.COMBAT_DEATH

        if door_spam_count >= 10:
            return cls.DOOR_BOUNCE

        if ammo_remaining <= 0.0 and damage_taken > 0:
            return cls.AMMO_DEPLETION

        if timeout:
            if unique_cells <= 3:
                return cls.STUCK_NAV
            return cls.TIMEOUT

        return cls.TIMEOUT


@dataclass
class EpisodeRecord:
    """Complete summary of a single episode rollout."""

    episode_id: int
    map_name: str
    seed: Optional[int]
    ticks: int
    max_ticks: int
    raw_return: float
    shaped_return: float
    exit_reached: bool
    dead: bool
    timeout: bool
    kills: int
    damage_taken: float
    keys_held: int
    unique_cells: int
    failure_mode: str
    reward_components: Dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0


class MetricsLogger:
    """Rollout and training metrics tracker with rolling statistics and PPO integration."""

    def __init__(self, window_size: int = 100, log_dir: Optional[str | Path] = None):
        self.window_size = window_size
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episodes: List[EpisodeRecord] = []
        self.recent_episodes: Deque[EpisodeRecord] = deque(maxlen=window_size)
        self.training_steps_log: List[Dict[str, Any]] = []

    def record_episode(self, record: EpisodeRecord) -> None:
        """Add a completed episode record to metrics."""
        self.episodes.append(record)
        self.recent_episodes.append(record)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Integration with Member 2's train_loop."""
        entry = dict(metrics)
        entry["step"] = step
        entry["timestamp"] = time.time()
        self.training_steps_log.append(entry)

    def get_summary(self, recent_only: bool = True) -> Dict[str, Any]:
        """Compute aggregate statistics over recent or all episodes."""
        records = list(self.recent_episodes) if recent_only else self.episodes
        n = len(records)
        if n == 0:
            return {
                "num_episodes": 0,
                "exit_rate": 0.0,
                "survival_rate": 0.0,
                "mean_raw_return": 0.0,
                "mean_shaped_return": 0.0,
                "mean_kills": 0.0,
                "mean_unique_cells": 0.0,
                "mean_ticks": 0.0,
                "failure_modes": {},
            }

        exit_count = sum(1 for r in records if r.exit_reached)
        dead_count = sum(1 for r in records if r.dead)

        failure_counts: Dict[str, int] = {}
        for r in records:
            failure_counts[r.failure_mode] = failure_counts.get(r.failure_mode, 0) + 1

        summary = {
            "num_episodes": n,
            "exit_rate": exit_count / n,
            "survival_rate": (n - dead_count) / n,
            "mean_raw_return": statistics.mean(r.raw_return for r in records),
            "mean_shaped_return": statistics.mean(r.shaped_return for r in records),
            "mean_kills": statistics.mean(r.kills for r in records),
            "mean_unique_cells": statistics.mean(r.unique_cells for r in records),
            "mean_ticks": statistics.mean(r.ticks for r in records),
            "mean_damage_taken": statistics.mean(r.damage_taken for r in records),
            "failure_modes": {k: v / n for k, v in failure_counts.items()},
        }
        return summary

    def format_summary_table(self, title: str = "Rollout Metrics Summary") -> str:
        """Return formatted human-readable summary table."""
        s = self.get_summary(recent_only=True)
        lines = [
            f"=== {title} (Last {s['num_episodes']} episodes) ===",
            f"  Exit Rate:           {s['exit_rate'] * 100:.1f}%",
            f"  Survival Rate:       {s['survival_rate'] * 100:.1f}%",
            f"  Mean Raw Return:     {s['mean_raw_return']:.2f}",
            f"  Mean Shaped Return:  {s['mean_shaped_return']:.2f}",
            f"  Mean Kills:          {s['mean_kills']:.2f}",
            f"  Mean Unique Cells:   {s['mean_unique_cells']:.1f}",
            f"  Mean Ticks/Episode:  {s['mean_ticks']:.1f}",
            f"  Mean Damage Taken:   {s['mean_damage_taken']:.1f}",
            "  Failure Modes:",
        ]
        for mode, frac in s["failure_modes"].items():
            lines.append(f"    - {mode:15s}: {frac * 100:.1f}%")
        lines.append("=" * len(lines[0]))
        return "\n".join(lines)

    def save_json(self, filename: str = "rollout_metrics.json") -> Path:
        """Save episode history and summary to JSON."""
        if not self.log_dir:
            raise ValueError("log_dir must be configured to save json")
        path = self.log_dir / filename
        data = {
            "summary": self.get_summary(recent_only=False),
            "recent_summary": self.get_summary(recent_only=True),
            "episodes": [asdict(r) for r in self.episodes[-200:]],  # store last 200
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path


def compare_evaluations(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare performance metrics between two evaluation checkpoints.

    Computes deltas and percentage changes for exit rate, survival, kills, and exploration.
    """
    keys = [
        "exit_rate",
        "survival_rate",
        "mean_raw_return",
        "mean_shaped_return",
        "mean_kills",
        "mean_unique_cells",
        "mean_ticks",
    ]
    comparison = {}
    for k in keys:
        base_val = baseline_summary.get(k, 0.0)
        cand_val = candidate_summary.get(k, 0.0)
        delta = cand_val - base_val
        comparison[k] = {
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta,
            "improved": delta > 0 if k != "mean_ticks" else delta < 0,
        }
    return comparison
