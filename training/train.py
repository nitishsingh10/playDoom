"""Training loop orchestration for Doom RL agent.

Owned by Member 2 (PPO and Optimization Engineer).
Coordinates:
  - DoomPolicy model (Member 3)
  - RolloutCollector (Member 4)
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
        # Rollout settings
        "rollout_steps": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        # Optimization settings
        "lr": 3e-4,
        "eps_adam": 1e-5,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "minibatch_size": 64,
        "clip_vloss": True,
        "normalize_advantages": True,
        # Loop settings
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
    """Run PPO training loop.

    Args:
        model: DoomPolicy instance (Member 3).
        collector: RolloutCollector instance (Member 4) implementing .collect(policy, num_steps).
        optimizer: PyTorch optimizer (if None, Adam is created with config['lr']).
        config: Dict of hyperparams.
        resume_path: Path to checkpoint to resume training from.
        metrics_logger: Optional logger (Member 4) with .log_metrics(dict, step).

    Returns:
        Summary dictionary with training results.
    """
    cfg = default_train_config()
    if config is not None:
        cfg.update(config)

    device = torch.device(cfg["device"])
    model = model.to(device)
    # Keep observations/actions and model outputs on one device throughout PPO.
    # This also avoids CPU/GPU mismatches in structured log-probability updates.
    if hasattr(collector, "device"):
        collector.device = device

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=cfg["eps_adam"])

    start_update = 1
    global_step = 0

    if resume_path and Path(resume_path).exists():
        ckpt_info = load_checkpoint(resume_path, model, optimizer, device=device, restore_rng=True)
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

        # 1. Collect rollouts using Member 4's collector
        # Expected return: dict with obs, actions, rewards, dones, values, log_probs, last_value
        rollout_data = collector.collect(model, num_steps=rollout_steps)

        # 2. Extract rollout tensors onto target device
        rewards = torch.as_tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        values = torch.as_tensor(rollout_data["values"], device=device, dtype=torch.float32)
        dones = torch.as_tensor(rollout_data["dones"], device=device, dtype=torch.float32)
        last_value = torch.as_tensor(rollout_data["last_value"], device=device, dtype=torch.float32)
        truncations = (
            torch.as_tensor(rollout_data["truncations"], device=device, dtype=torch.bool)
            if "truncations" in rollout_data
            else None
        )
        bootstrap_values = (
            torch.as_tensor(rollout_data["bootstrap_values"], device=device, dtype=torch.float32)
            if "bootstrap_values" in rollout_data
            else None
        )

        # 3. Compute GAE advantages and target returns
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

        # Prepare batch for PPO update
        # Flatten rollout trajectory [T, B] into batch [N] if batched
        if rewards.dim() > 1:
            batch_obs = _flatten_time_batch(rollout_data["obs"])
            batch_actions = torch.as_tensor(rollout_data["actions"], device=device).flatten(0, 1)
            batch_old_log_probs = torch.as_tensor(rollout_data["log_probs"], device=device).flatten(0, 1)
            batch_returns = returns.flatten(0, 1)
            batch_advantages = advantages.flatten(0, 1)
            batch_old_values = values.flatten(0, 1)
            num_transitions = rewards.numel()
        else:
            batch_obs = rollout_data["obs"]
            batch_actions = torch.as_tensor(rollout_data["actions"], device=device)
            batch_old_log_probs = torch.as_tensor(rollout_data["log_probs"], device=device)
            batch_returns = returns
            batch_advantages = advantages
            batch_old_values = values
            num_transitions = rewards.shape[0]

        batch = {
            "obs": batch_obs,
            "actions": batch_actions,
            "old_log_probs": batch_old_log_probs,
            "returns": batch_returns,
            "advantages": batch_advantages,
            "old_values": batch_old_values,
        }

        # 4. Perform PPO update
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

        # Periodic logging to stdout
        if update % max(1, (total_updates // 20)) == 0 or update == 1:
            print(
                f"[Update {update}/{total_updates}] Step={global_step} "
                f"Reward={metrics['mean_reward']:.3f} PLoss={metrics['policy_loss']:.4f} "
                f"VLoss={metrics['value_loss']:.4f} Ent={metrics['entropy']:.3f} "
                f"KL={metrics['approx_kl']:.4f} FPS={fps:.1f}"
            )

        # 5. Periodic Checkpoint save
        if update % cfg["checkpoint_interval"] == 0 or update == total_updates:
            ckpt_path = save_dir / f"checkpoint_{update}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                update=update,
                global_step=global_step,
                config=cfg,
                extra=metrics,
            )
            # Maintain latest pointer
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
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wad", action="append", default=None, help="Training WAD; repeat to cycle among maps")
    parser.add_argument("--map", type=str, default="MAP01")
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="training/runs/structured_v1/checkpoints")
    parser.add_argument("--log-dir", type=str, default="training/runs/structured_v1/logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args(argv)

    import numpy as np
    from submission.model import DoomPolicy
    from training.envs import MapCyclerEnv, default_training_maps
    from training.metrics import MetricsLogger
    from training.rewards import RewardShaper
    from training.rollout import RolloutCollector

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    maps = [(wad, args.map) for wad in args.wad] if args.wad else default_training_maps()
    env = MapCyclerEnv(maps, episode_ticks=args.episode_ticks, seed=args.seed)
    logger = MetricsLogger(log_dir=args.log_dir)
    collector = RolloutCollector(
        env,
        shaper=RewardShaper(),
        metrics_logger=logger,
        device=args.device,
        seed=args.seed,
    )
    config = default_train_config()
    config.update({
        "total_updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "minibatch_size": args.minibatch_size,
        "update_epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "save_dir": args.save_dir,
    })
    try:
        summary = train_loop(
            DoomPolicy(), collector, config=config, resume_path=args.resume, metrics_logger=logger
        )
        logger.save_json("training_metrics.json")
        print(f"Training summary: {summary}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
