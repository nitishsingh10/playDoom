"""Fresh v1/v2 comparison with matched initialization, budgets and training seeds.

No pretrained checkpoint is loaded or migrated. Common parameters start equal;
v2's two extra ray-angle columns start at zero and can learn through PPO. Both
arms use the existing rewards. Evaluation keeps native counters out of inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
import subprocess
import tempfile

import torch

from submission.checkpoint import DIRECTIONAL_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID
from submission.model import DoomPolicy
from training.checkpoint import save_checkpoint
from training.combat_experiment import (
    CombatRewardShaper, CombatTelemetryEnv, ExperimentCollector, evaluate,
    seed_all, sha256, verify_hashes, write_json,
)
from training.envs import MapCyclerEnv, default_training_maps
from training.metrics import MetricsLogger
from training.train import default_train_config, train_loop


def matched_fresh_models(seed):
    """Initialize fresh paired models; this function accepts no saved weights."""
    seed_all(seed)
    v1 = DoomPolicy(architecture_id=STRUCTURED_ARCHITECTURE_ID)
    v2 = DoomPolicy(architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
    common = dict(v1.named_parameters())
    with torch.no_grad():
        for name, parameter in v2.named_parameters():
            if name == "ray_encoder.0.weight":
                parameter.zero_()
                parameter[:, :common[name].shape[1]].copy_(common[name])
            elif parameter.shape == common[name].shape:
                parameter.copy_(common[name])
            else:
                raise ValueError(f"Unexpected architecture difference: {name}")
    return {"v1": v1, "v2": v2}


def episode_metrics(rows):
    return {
        "exit_rate": mean(float(r["exit_reached"]) for r in rows),
        "survival_rate": mean(float(not r["dead"]) for r in rows),
        "mean_kills": mean(r["kills"] for r in rows),
        "mean_hits": mean(r["reward_components"]["combat_hits"] for r in rows),
        "mean_damage_dealt": mean(r["reward_components"]["combat_damage_dealt"] for r in rows),
        "mean_damage_taken": mean(r["damage_taken"] for r in rows),
        "mean_ticks": mean(r["ticks"] for r in rows),
    }


def summarize(results, training_seeds):
    rows = [
        {**row, "variant": result["variant"], "training_seed": result["training_seed"],
         "mode": "stochastic" if result["stochastic"] else "deterministic"}
        for result in results.values() for row in result["episodes_detail"]
    ]
    per_map, paired = [], []
    for mode, map_name in sorted({(r["mode"], r["map"]) for r in rows}):
        means = {}
        for variant in ("v1", "v2"):
            selected = [r for r in rows if r["variant"] == variant and r["mode"] == mode and r["map"] == map_name]
            if not selected:
                raise ValueError(f"Missing evaluation group: {variant}, {mode}, {map_name}")
            metrics = episode_metrics(selected)
            if Path(map_name.rsplit(":", 1)[0]).stem == "tier1_projectile":
                metrics["exit_rate"] = None
            per_map.append({"variant": variant, "mode": mode, "map": map_name,
                            "episodes": len(selected), "exits": sum(r["exit_reached"] for r in selected),
                            "survived": sum(not r["dead"] for r in selected), **metrics})
            for seed in training_seeds:
                group = [r for r in selected if r["training_seed"] == seed]
                if not group:
                    raise ValueError(f"Missing paired training seed: {variant}, {seed}")
                means[variant, seed] = episode_metrics(group)
        for metric in means["v1", training_seeds[0]]:
            if metric == "exit_rate" and Path(map_name.rsplit(":", 1)[0]).stem == "tier1_projectile":
                continue
            differences = {seed: means["v2", seed][metric] - means["v1", seed][metric]
                           for seed in training_seeds}
            values = list(differences.values())
            paired.append({"mode": mode, "map": map_name, "metric": metric,
                           "v2_minus_v1_by_training_seed": differences,
                           "mean_difference": mean(values),
                           "sample_sd_of_training_seed_differences": stdev(values) if len(values) > 1 else None})
    return {"per_map": per_map, "paired_training_seed_differences": paired,
            "statistical_limitations": "Pilot across independent training seeds; evaluation episodes from one checkpoint are clustered. Raw paired seed effects and SD are reported, without a significance claim or population confidence interval.",
            "combat_exit_rate_note": "tier1_projectile has no exit; its exit metric is not applicable."}


def print_summary(summary):
    print("POLICY/MODE | MAP | EXITS | KILLS/EP | DEALT/EP | SURVIVED", flush=True)
    for r in summary["per_map"]:
        name = Path(r["map"].rsplit(":", 1)[0]).stem
        exits = "N/A" if name == "tier1_projectile" else f"{r['exits']}/{r['episodes']}"
        print(f"{r['variant']}/{r['mode']} | {name} | {exits} | {r['mean_kills']:.2f} | "
              f"{r['mean_damage_dealt']:.1f} | {r['survived']}/{r['episodes']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("training/runs"))
    args = parser.parse_args(argv)
    if args.updates < 1 or args.eval_episodes < 1:
        parser.error("updates and evaluation episode counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("training seeds must be distinct integers between 0 and 2**32-1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this interpreter")
    maps = default_training_maps()
    evaluation_maps = maps + [("maps/two_rooms.wad", "MAP01")]
    protected = [Path("submission/weights/checkpoint.pt"), Path("submission/weights/config.json")]
    protected += [Path(wad) for wad, _ in evaluation_maps]
    for path in (
        "training/runs/structured_l4_seed17_iXoMX8/checkpoints/checkpoint_500.pt",
        "training/runs/structured_combat_seed17_SFhSzA/checkpoints/checkpoint_2500.pt",
        "training/runs/combat_credit_seed17_v43syh5u/control/checkpoints/checkpoint_1000.pt",
        "training/runs/combat_credit_seed17_v43syh5u/damage_credit/checkpoints/checkpoint_1000.pt",
        "training/runs/ray_v2_smoke_tCxcTk/checkpoints/checkpoint_2.pt",
    ):
        if Path(path).is_file():
            protected.append(Path(path))
    hashes = {str(p.resolve(strict=True)): sha256(p) for p in protected}
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="directional_comparison_", dir=args.output_root)).resolve()
    torch.set_num_threads(1)
    config = {**default_train_config(), "rollout_steps": 128, "total_updates": args.updates, "device": args.device}
    config.pop("save_dir", None)  # Actual output paths are per-arm, never the v1 default directory.
    manifest = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_seeds": args.seeds, "evaluation_seeds": list(range(100, 100 + args.eval_episodes)),
        "ppo_config": config, "training_maps": maps, "evaluation_maps": evaluation_maps,
        "decisions_per_model": args.updates * config["rollout_steps"],
        "total_training_decisions": args.updates * config["rollout_steps"] * 2 * len(args.seeds),
        "episode_ticks": 2100, "reward": "Existing RewardShaper; damage-credit coefficient zero in both arms",
        "initialization": "Fresh v1 common parameters copied into fresh v2; new angle weights zero. No pretrained weights loaded. Actual initial checkpoints retained.",
        "rng_protocol": "Same explicit policy/environment seed per paired arm. Map sequence per episode reset is seeded; episode lengths and map decision exposure can diverge on-policy.",
        "generalization_limit": "two_rooms is excluded from this training curriculum but has been evaluated before; these fixed fixtures do not establish unseen-layout generalization.",
        "protected_sha256": hashes, "initial_checkpoints": {}, "trained_checkpoints": {},
        "status": "running", "weights_promoted": False, "torch": torch.__version__,
    }
    write_json(output / "manifest.json", manifest)
    print(f"COMPARISON DIRECTORY: {output}", flush=True)
    print(f"Fresh models: {2 * len(args.seeds)}; decisions each: {manifest['decisions_per_model']}; total: {manifest['total_training_decisions']}", flush=True)
    trained = {}
    try:
        for index, seed in enumerate(args.seeds):
            models = matched_fresh_models(seed)
            order = ("v1", "v2") if index % 2 == 0 else ("v2", "v1")
            for variant in order:
                label = f"{variant}_seed{seed}"
                arm = output / label
                model = models[variant]
                cfg = {**config, "save_dir": str(arm / "checkpoints"), "experiment_seed": seed,
                       "training_maps": maps, "episode_ticks": 2100, "combat_damage_coefficient": 0.0}
                initial = save_checkpoint(arm / "initial.pt", model, config=cfg)
                manifest["initial_checkpoints"][label] = {"path": str(initial), "sha256": sha256(initial)}
                write_json(output / "manifest.json", manifest)
                logger = MetricsLogger(log_dir=arm / "logs")
                env = CombatTelemetryEnv(MapCyclerEnv(maps, episode_ticks=2100, seed=seed))
                collector = ExperimentCollector(env, shaper=CombatRewardShaper(), metrics_logger=logger,
                                                device=args.device, seed=seed)
                print(f"TRAINING: {label} ({model.architecture_id}), fresh update 1", flush=True)
                try:
                    train_loop(model, collector, config=cfg, resume_path=None, metrics_logger=logger)
                finally:
                    env.close()
                    logger.save_json("training_metrics.json")
                    write_json(arm / "all_completed_episodes.json", [asdict(r) for r in logger.episodes])
                final = arm / "checkpoints" / f"checkpoint_{args.updates}.pt"
                saved = torch.load(final, map_location="cpu", weights_only=False)
                if saved.get("update") != args.updates or saved.get("global_step") != manifest["decisions_per_model"]:
                    raise RuntimeError(f"Training budget mismatch: {label}")
                if saved.get("architecture_id") != model.architecture_id:
                    raise RuntimeError(f"Wrong architecture saved: {label}")
                if not all(torch.isfinite(t).all() for t in saved["model_state_dict"].values()):
                    raise RuntimeError(f"Nonfinite weights: {label}")
                if sha256(initial) != manifest["initial_checkpoints"][label]["sha256"]:
                    raise RuntimeError(f"Initial checkpoint was modified: {label}")
                trained[label] = (final, variant, seed)
                initial_weights = torch.load(initial, map_location="cpu", weights_only=False)["model_state_dict"]
                final_weights = saved["model_state_dict"]
                optimizer_states = (saved.get("optimizer_state_dict") or {}).get("state", {})
                manifest["trained_checkpoints"][label] = {
                    "path": str(final), "sha256": sha256(final),
                    "changed_parameter_tensors": sum(not torch.equal(t, final_weights[k]) for k, t in initial_weights.items()),
                    "actor_weight_row_max_abs_change": (final_weights["actor.weight"] - initial_weights["actor.weight"]).abs().amax(1).tolist(),
                    "angle_column_max_abs_change": float((final_weights["ray_encoder.0.weight"][:, -2:] -
                        initial_weights["ray_encoder.0.weight"][:, -2:]).abs().max()) if variant == "v2" else None,
                    "adam_steps": sorted({float(s["step"]) for s in optimizer_states.values() if "step" in s}),
                }
                verify_hashes(hashes)
                write_json(output / "manifest.json", manifest)
        results = {}
        for label, (checkpoint, variant, seed) in trained.items():
            for mode in ("stochastic", "deterministic"):
                key = f"{label}_{mode}"
                print(f"EVALUATING: {key}", flush=True)
                result = evaluate(checkpoint, evaluation_maps, manifest["evaluation_seeds"], mode == "stochastic")
                result.update(variant=variant, training_seed=seed)
                results[key] = result
                write_json(output / f"{key}.json", result)
                if sha256(checkpoint) != manifest["trained_checkpoints"][label]["sha256"]:
                    raise RuntimeError(f"Evaluated checkpoint changed: {label}")
        summary = summarize(results, args.seeds)
        write_json(output / "comparison.json", summary)
        print_summary(summary)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest["protected_inputs_unchanged"] = all(sha256(p) == h for p, h in hashes.items())
        write_json(output / "manifest.json", manifest)
        verify_hashes(hashes)
    print(f"COMPARISON COMPLETE. No weights promoted. Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
