"""Evaluate checkpoints across the held-out multi-tier benchmark, unseen random maps, and baseline maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import tempfile

import numpy as np
import torch

from training.envs import default_heldout_maps, default_training_maps
from training.evaluate import evaluate_checkpoint
from training.mapgen import generate_random_map, write_udmf_wad


def summarize_by_map(report: dict) -> list[dict]:
    by_map: dict[str, list[dict]] = {}
    for ep in report["episodes_detail"]:
        by_map.setdefault(ep["map"], []).append(ep)
    rows = []
    for map_key, eps in sorted(by_map.items()):
        rows.append({
            "map": map_key,
            "episodes": len(eps),
            "exits": sum(int(e["exit_reached"]) for e in eps),
            "exit_rate": mean(float(e["exit_reached"]) for e in eps),
            "survival_rate": mean(float(not e["dead"]) for e in eps),
            "mean_kills": mean(e["kills"] for e in eps),
            "mean_damage_taken": mean(e["damage_taken"] for e in eps),
            "mean_keycards": mean(e["keycards"] for e in eps),
            "mean_unique_cells": mean(e["unique_cells"] for e in eps),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--include-baseline-maps", action="store_true")
    parser.add_argument("--random-maps", type=int, default=0, help="Also evaluate on N unseen random Tier 1-4 maps")
    parser.add_argument("--random-seed", type=int, default=9999)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    maps = [f"{w}:{m}" for w, m in default_heldout_maps()]
    if args.include_baseline_maps:
        maps.extend(f"{w}:{m}" for w, m in default_training_maps())

    with tempfile.TemporaryDirectory(prefix="playdoom_eval_rand_") as tmp_dir:
        if args.random_maps > 0:
            rng = np.random.default_rng(args.random_seed)
            for idx in range(args.random_maps):
                tier = (idx % 4) + 1
                spec = generate_random_map(rng, tier=tier)
                wad_path = write_udmf_wad(
                    Path(tmp_dir) / f"eval_rand_{idx:02d}_t{tier}.wad",
                    spec.grid,
                    player_angle=spec.player_angle,
                )
                maps.append(f"{wad_path}:MAP01")

        all_results = []
        for ckpt in args.checkpoints:
            rep = evaluate_checkpoint(
                ckpt,
                maps,
                args.seeds,
                device=args.device,
                episode_ticks=args.episode_ticks,
                stochastic=args.stochastic,
            )
            rows = summarize_by_map(rep)
            heldout_rows = [r for r in rows if "heldout" in r["map"] or "two_rooms" in r["map"]]
            rand_rows = [r for r in rows if "eval_rand_" in r["map"]]
            print(f"\n=== {ckpt} ({'stochastic' if args.stochastic else 'deterministic'}) ===")
            print(
                f"Overall held-out exit_rate={mean(r['exit_rate'] for r in heldout_rows):.1%}, "
                f"survival={mean(r['survival_rate'] for r in heldout_rows):.1%}, "
                f"kills/ep={mean(r['mean_kills'] for r in heldout_rows):.2f}"
            )
            if rand_rows:
                print(
                    f"Overall unseen random ({len(rand_rows)} maps) exit_rate={mean(r['exit_rate'] for r in rand_rows):.1%}, "
                    f"survival={mean(r['survival_rate'] for r in rand_rows):.1%}, "
                    f"kills/ep={mean(r['mean_kills'] for r in rand_rows):.2f}"
                )
            for r in rows:
                name = Path(r["map"].rsplit(":", 1)[0]).stem
                print(
                    f"  {name:26s} | exits={r['exits']}/{r['episodes']} "
                    f"| surv={r['survival_rate']:.0%} | kills={r['mean_kills']:.2f} "
                    f"| dmg={r['mean_damage_taken']:.1f} | keys={r['mean_keycards']:.2f} | cells={r['mean_unique_cells']:.1f}"
                )
            all_results.append({"checkpoint": str(ckpt), "summary": rows, "report": rep})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
