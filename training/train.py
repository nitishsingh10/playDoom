"""Training loop orchestration for Doom RL agent.

Owned by Member 2 (PPO and Optimization Engineer).
Coordinates:
  - DoomPolicy model (Member 3)
  - RolloutCollector / VecRolloutCollector (Member 4)
  - compute_gae & ppo_update (Member 2)
  - Checkpointing (Member 2)
  - Metrics logging and checkpoint evaluation (Member 4 / Member 1)
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from submission.checkpoint import SPATIAL_GRU_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURES
from training.checkpoint import load_checkpoint, save_checkpoint
from training.ppo import compute_gae, ppo_update


def _flatten_time_batch(tree: Any) -> Any:
    """Flatten a rollout's [T, B, ...] leaves while preserving nested dicts."""
    if isinstance(tree, dict):
        return {key: _flatten_time_batch(value) for key, value in tree.items()}
    if torch.is_tensor(tree) and tree.ndim >= 2:
        return tree.flatten(0, 1)
    return tree


def default_train_config() -> Dict[str, Any]:
    """Default hyperparameters for PPO training on ViZDoom."""
    return {
        "rollout_steps": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": 3e-4,
        "eps_adam": 1e-5,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "minibatch_size": 64,
        "seq_len": 32,
        "clip_vloss": True,
        "normalize_advantages": True,
        "total_updates": 500,
        "checkpoint_interval": 25,
        "eval_interval": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "training/runs/structured_v1/checkpoints",
    }


def train_loop(
    model: torch.nn.Module,
    collector: Any,
    optimizer: Optional[optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    resume_path: Optional[str] = None,
    metrics_logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run PPO training loop."""
    cfg = default_train_config()
    if config is not None:
        cfg.update(config)

    device = torch.device(cfg["device"])
    model = model.to(device)
    if hasattr(collector, "device"):
        collector.device = device

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=cfg["eps_adam"])

    start_update = 1
    global_step = 0

    if resume_path and Path(resume_path).exists():
        ckpt_info = load_checkpoint(resume_path, model, optimizer, device=device, restore_rng=True)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg["lr"]
        start_update = ckpt_info.get("update", 0) + 1
        global_step = ckpt_info.get("global_step", 0)
        print(f"Resumed from {resume_path} at update {start_update}, global step {global_step}")

    total_updates = cfg["total_updates"]
    rollout_steps = cfg["rollout_steps"]
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting PPO training on {device} for {total_updates} updates...")
    t0 = time.time()

    for update in range(start_update, total_updates + 1):
        t_update_start = time.time()

        rollout_data = collector.collect(model, num_steps=rollout_steps)

        rewards = rollout_data["rewards"].to(device)
        values = rollout_data["values"].to(device)
        dones = rollout_data["dones"].to(device)
        last_value = rollout_data["last_value"].to(device)
        truncations = rollout_data.get("truncations")
        if truncations is not None:
            truncations = truncations.to(device)
        bootstrap_values = rollout_data.get("bootstrap_values")
        if bootstrap_values is not None:
            bootstrap_values = bootstrap_values.to(device)

        with torch.no_grad():
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_value=last_value,
                gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"],
                truncations=truncations,
                bootstrap_values=bootstrap_values,
            )

        is_recurrent_batch = (
            getattr(model, "is_recurrent", False)
            and hasattr(model, "forward_sequence")
            and "init_hiddens" in rollout_data
        )
        num_transitions = rewards.numel()

        if is_recurrent_batch:
            batch = {
                "obs": rollout_data["obs"],
                "actions": torch.as_tensor(rollout_data["actions"], device=device),
                "old_log_probs": torch.as_tensor(rollout_data["log_probs"], device=device),
                "returns": returns,
                "advantages": advantages,
                "old_values": values,
                "dones": dones,
                "init_hiddens": rollout_data["init_hiddens"].to(device),
            }
        elif rewards.dim() > 1:
            batch = {
                "obs": _flatten_time_batch(rollout_data["obs"]),
                "actions": torch.as_tensor(rollout_data["actions"], device=device).flatten(0, 1),
                "old_log_probs": torch.as_tensor(rollout_data["log_probs"], device=device).flatten(0, 1),
                "returns": returns.flatten(0, 1),
                "advantages": advantages.flatten(0, 1),
                "old_values": values.flatten(0, 1),
            }
        else:
            batch = {
                "obs": rollout_data["obs"],
                "actions": torch.as_tensor(rollout_data["actions"], device=device),
                "old_log_probs": torch.as_tensor(rollout_data["log_probs"], device=device),
                "returns": returns,
                "advantages": advantages,
                "old_values": values,
            }

        metrics = ppo_update(model, optimizer, batch, cfg)

        global_step += num_transitions
        dt = time.time() - t_update_start
        fps = num_transitions / max(dt, 1e-4)

        metrics["fps"] = fps
        metrics["global_step"] = global_step
        metrics["update"] = update
        metrics["mean_reward"] = float(rewards.mean().item())

        if metrics_logger is not None:
            metrics_logger.log_metrics(metrics, step=global_step)

        if update % max(1, (total_updates // 20)) == 0 or update == 1 or update == total_updates:
            ep_str = ""
            if metrics_logger is not None and getattr(metrics_logger, "episodes", None):
                recent = metrics_logger.episodes[-20:]
                ex_rate = sum(1.0 for e in recent if e.exit_reached) / len(recent)
                kills = sum(e.kills for e in recent) / len(recent)
                ret = sum(e.shaped_return for e in recent) / len(recent)
                ep_str = f" | EpRet={ret:.2f} Exit={ex_rate:.0%} Kills={kills:.2f}"
            print(
                f"[Update {update}/{total_updates}] Step={global_step} "
                f"Reward={metrics['mean_reward']:.3f} PLoss={metrics['policy_loss']:.4f} "
                f"VLoss={metrics['value_loss']:.4f} Ent={metrics['entropy']:.3f} "
                f"KL={metrics['approx_kl']:.4f} FPS={fps:.1f}{ep_str}",
                flush=True,
            )

        if update % cfg["checkpoint_interval"] == 0 or update == total_updates:
            ckpt_path = save_checkpoint(
                path=save_dir / f"checkpoint_{update}.pt",
                model=model,
                optimizer=optimizer,
                update=update,
                global_step=global_step,
                config=cfg,
                extra=metrics,
            )
            save_checkpoint(
                path=save_dir / "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                update=update,
                global_step=global_step,
                config=cfg,
                extra=metrics,
            )

    total_time = time.time() - t0
    print(f"Training finished in {total_time:.2f}s for {global_step} steps.")
    return {"total_steps": global_step, "total_updates": total_updates, "total_time": total_time}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PPO training runner for DoomPolicy")
    parser.add_argument("--updates", type=int, default=50, help="Number of PPO updates")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout per env")
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence chunk length for recurrent PPO")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel ViZDoom workers")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wad", action="append", default=None, help="Training WAD; repeat to cycle among maps")
    parser.add_argument("--curriculum", action="store_true", help="Use full multi-tier procedural + synthetic map suite")
    parser.add_argument("--include-freedoom", action="store_true", help="Include Freedoom2 MAP01-MAP05 in curriculum")
    parser.add_argument("--random-map-prob", type=float, default=0.0, help="Probability of generating a fresh random UDMF map per reset")
    parser.add_argument("--map", type=str, default="MAP01")
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--fusion-size", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=sorted(STRUCTURED_ARCHITECTURES),
        default=STRUCTURED_ARCHITECTURE_ID,
        help="Explicit model variant.",
    )
    args = parser.parse_args(argv)
    run_root = Path("training/runs") / args.architecture.removeprefix("doom_")
    args.save_dir = args.save_dir or str(run_root / "checkpoints")
    args.log_dir = args.log_dir or str(run_root / "logs")

    import numpy as np
    from submission.model import DoomPolicy
    from training.envs import MapCyclerEnv, SubprocVecEnv, default_curriculum_maps, default_training_maps
    from training.metrics import MetricsLogger
    from training.rewards import RewardShaper
    from training.rollout import RolloutCollector, VecRolloutCollector

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    if args.wad:
        maps = [(wad, args.map) for wad in args.wad]
    elif args.curriculum:
        maps = default_curriculum_maps(include_freedoom=args.include_freedoom)
    else:
        maps = default_training_maps()

    logger = MetricsLogger(log_dir=args.log_dir)
    if args.num_envs > 1:
        env = SubprocVecEnv(
            maps,
            num_envs=args.num_envs,
            episode_ticks=args.episode_ticks,
            seed=args.seed,
            random_map_prob=args.random_map_prob,
        )
        collector = VecRolloutCollector(env, shaper_factory=RewardShaper, metrics_logger=logger, device=args.device)
    else:
        env = MapCyclerEnv(
            maps,
            episode_ticks=args.episode_ticks,
            seed=args.seed,
            random_map_prob=args.random_map_prob,
        )
        collector = RolloutCollector(env, shaper=RewardShaper(), metrics_logger=logger, device=args.device, seed=args.seed)

    config = default_train_config()
    config.update({
        "total_updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "minibatch_size": args.minibatch_size,
        "seq_len": args.seq_len,
        "update_epochs": args.epochs,
        "lr": args.lr,
        "ent_coef": args.ent_coef,
        "checkpoint_interval": args.checkpoint_interval,
        "device": args.device,
        "save_dir": args.save_dir,
    })
    policy = DoomPolicy(
        hidden_size=args.hidden_size,
        fusion_size=args.fusion_size,
        architecture_id=args.architecture,
    )
    try:
        summary = train_loop(policy, collector, config=config, resume_path=args.resume, metrics_logger=logger)
        logger.save_json("training_metrics.json")
        print(f"Training summary: {summary}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
