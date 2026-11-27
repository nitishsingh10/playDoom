"""PPO learning algorithm, Generalized Advantage Estimation (GAE), and action utilities.

Owned by Member 2 (PPO and Optimization Engineer).
Provides:
  - compute_gae: Generalized Advantage Estimation with terminal vs truncation distinction
  - evaluate_actions: Computes structured action log probabilities and entropy
  - sample_actions: Samples valid nine-button actions from twelve policy logits
  - ppo_update: Mini-batch clipped surrogate objective (feed-forward and recurrent BPTT)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from submission.actions import StructuredActionDistribution


def sample_actions(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample structured actions and compute their joint log probabilities."""
    dist = StructuredActionDistribution(logits)
    actions = dist.sample()
    return actions, dist.log_prob(actions)


def evaluate_actions(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate log probabilities and entropy of given actions under policy logits."""
    dist = StructuredActionDistribution(logits)
    return dist.log_prob(actions), dist.entropy()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    truncations: Optional[torch.Tensor] = None,
    bootstrap_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and returns."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        current_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        if truncations is not None:
            is_truncated = truncations[t].bool()
            if bootstrap_values is None:
                truncation_next_value = next_value
            else:
                truncation_next_value = bootstrap_values[t]
            truncation_delta = rewards[t] + gamma * truncation_next_value - values[t]
            last_gae = torch.where(is_truncated, truncation_delta, current_gae)
        else:
            last_gae = current_gae

        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def _slice_tree(tree: Any, indices: torch.Tensor) -> Any:
    if isinstance(tree, dict):
        return {k: _slice_tree(v, indices) for k, v in tree.items()}
    return tree[indices]


def _unsqueeze_env_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _unsqueeze_env_tree(v) for k, v in tree.items()}
    if torch.is_tensor(tree):
        return tree.unsqueeze(1)
    return tree


def _chunk_time_env(tensor: torch.Tensor, num_seq: int, seq_len: int, num_envs: int) -> torch.Tensor:
    rest = tensor.shape[2:]
    reshaped = tensor.reshape(num_seq, seq_len, num_envs, *rest)
    perm = (2, 0, 1, *range(3, reshaped.ndim))
    return reshaped.permute(*perm).reshape(num_envs * num_seq, seq_len, *rest)


def _chunk_tree(tree: Any, num_seq: int, seq_len: int, num_envs: int) -> Any:
    if isinstance(tree, dict):
        return {k: _chunk_tree(v, num_seq, seq_len, num_envs) for k, v in tree.items()}
    if torch.is_tensor(tree):
        return _chunk_time_env(tree, num_seq, seq_len, num_envs)
    return tree


def _ppo_update_recurrent(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    clip_eps = float(config.get("clip_eps", 0.2))
    vf_coef = float(config.get("vf_coef", 0.5))
    ent_coef = float(config.get("ent_coef", 0.01))
    max_grad_norm = float(config.get("max_grad_norm", 0.5))
    update_epochs = int(config.get("update_epochs", 4))
    minibatch_size = int(config.get("minibatch_size", 64))
    clip_vloss = bool(config.get("clip_vloss", True))
    normalize_advantages = bool(config.get("normalize_advantages", True))

    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    old_values = batch.get("old_values", returns - advantages)
    dones = batch["dones"]
    init_hiddens = batch["init_hiddens"]

    if old_log_probs.ndim == 1:
        obs = _unsqueeze_env_tree(obs)
        actions = actions.unsqueeze(1)
        old_log_probs = old_log_probs.unsqueeze(1)
        returns = returns.unsqueeze(1)
        advantages = advantages.unsqueeze(1)
        old_values = old_values.unsqueeze(1)
        dones = dones.unsqueeze(1)
        init_hiddens = init_hiddens.unsqueeze(1)

    T, N = old_log_probs.shape
    seq_len = min(int(config.get("seq_len", 32)), T)
    while seq_len > 1 and T % seq_len != 0:
        seq_len -= 1
    num_seq = T // seq_len
    num_chunks = N * num_seq

    chunked_obs = _chunk_tree(obs, num_seq, seq_len, N)
    chunked_actions = _chunk_time_env(actions, num_seq, seq_len, N)
    chunked_old_lp = _chunk_time_env(old_log_probs, num_seq, seq_len, N)
    chunked_returns = _chunk_time_env(returns, num_seq, seq_len, N)
    chunked_adv = _chunk_time_env(advantages, num_seq, seq_len, N)
    chunked_old_val = _chunk_time_env(old_values, num_seq, seq_len, N)
    chunked_dones = _chunk_time_env(dones, num_seq, seq_len, N)
    # Initial hidden at start of each chunk: [num_seq, N, H] -> [1, num_chunks, H]
    chunk_init_h = (
        init_hiddens[::seq_len]
        .permute(1, 0, 2)
        .reshape(num_chunks, init_hiddens.shape[-1])
    )

    mb_chunks = max(1, min(num_chunks, minibatch_size // seq_len))

    policy_losses, value_losses, entropies, grad_norms, kl_divs, clip_fractions = [], [], [], [], [], []
    for _ in range(update_epochs):
        perm = torch.randperm(num_chunks, device=actions.device)
        for start in range(0, num_chunks, mb_chunks):
            idx = perm[start : start + mb_chunks]
            mb_obs = _slice_tree(chunked_obs, idx)
            mb_actions = chunked_actions[idx]
            mb_old_lp = chunked_old_lp[idx]
            mb_returns = chunked_returns[idx]
            mb_adv = chunked_adv[idx]
            mb_old_val = chunked_old_val[idx]
            mb_dones = chunked_dones[idx]
            mb_h0 = chunk_init_h[idx].unsqueeze(0)

            if normalize_advantages and mb_adv.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            logits, values, _ = model.forward_sequence(mb_obs, mb_dones, mb_h0)
            new_lp, entropy = evaluate_actions(logits, mb_actions)

            log_ratio = new_lp - mb_old_lp
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipped = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()
                kl_divs.append(approx_kl.item())
                clip_fractions.append(clipped.item())

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            if clip_vloss:
                v_clipped = mb_old_val + torch.clamp(values - mb_old_val, -clip_eps, clip_eps)
                v_loss1 = (values - mb_returns) ** 2
                v_loss2 = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

            entropy_bonus = entropy.mean()
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_bonus.item())
            grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm))

    return {
        "policy_loss": float(sum(policy_losses) / max(len(policy_losses), 1)),
        "value_loss": float(sum(value_losses) / max(len(value_losses), 1)),
        "entropy": float(sum(entropies) / max(len(entropies), 1)),
        "grad_norm": float(sum(grad_norms) / max(len(grad_norms), 1)),
        "approx_kl": float(sum(kl_divs) / max(len(kl_divs), 1)),
        "clip_fraction": float(sum(clip_fractions) / max(len(clip_fractions), 1)),
    }


def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Perform PPO optimization over a rollout batch using mini-batches and multiple epochs."""
    if getattr(model, "is_recurrent", False) and hasattr(model, "forward_sequence") and "init_hiddens" in batch:
        return _ppo_update_recurrent(model, optimizer, batch, config)

    clip_eps = float(config.get("clip_eps", 0.2))
    vf_coef = float(config.get("vf_coef", 0.5))
    ent_coef = float(config.get("ent_coef", 0.01))
    max_grad_norm = float(config.get("max_grad_norm", 0.5))
    update_epochs = int(config.get("update_epochs", 4))
    minibatch_size = int(config.get("minibatch_size", 64))
    clip_vloss = bool(config.get("clip_vloss", True))
    normalize_advantages = bool(config.get("normalize_advantages", True))

    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    old_values = batch.get("old_values", returns - advantages)
    obs = batch["obs"]

    batch_size = actions.shape[0]
    minibatch_size = min(minibatch_size, batch_size)

    old_log_probs = old_log_probs.view(-1)
    returns = returns.view(-1)
    advantages = advantages.view(-1)
    old_values = old_values.view(-1)

    policy_losses = []
    value_losses = []
    entropies = []
    grad_norms = []
    kl_divs = []
    clip_fractions = []

    for _ in range(update_epochs):
        indices = torch.randperm(batch_size, device=actions.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = indices[start:end]

            if isinstance(obs, dict):
                mb_obs = _slice_tree(obs, mb_inds)
            else:
                mb_obs = obs[mb_inds]

            mb_actions = actions[mb_inds]
            mb_old_log_probs = old_log_probs[mb_inds]
            mb_returns = returns[mb_inds]
            mb_advantages = advantages[mb_inds]
            mb_old_values = old_values[mb_inds]

            if normalize_advantages and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            forward_out = model(mb_obs)
            if isinstance(forward_out, (tuple, list)):
                logits = forward_out[0]
                values = forward_out[1]
            else:
                raise ValueError("Model output must be tuple (logits, values) or (logits, values, next_hidden)")

            values = values.view(-1)
            new_log_probs, entropy = evaluate_actions(logits, mb_actions)

            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipped = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()
                kl_divs.append(approx_kl.item())
                clip_fractions.append(clipped.item())

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if clip_vloss:
                v_clipped = mb_old_values + torch.clamp(values - mb_old_values, -clip_eps, clip_eps)
                v_loss1 = (values - mb_returns) ** 2
                v_loss2 = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

            entropy_bonus = entropy.mean()
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_bonus.item())
            grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm))

    return {
        "policy_loss": float(sum(policy_losses) / max(len(policy_losses), 1)),
        "value_loss": float(sum(value_losses) / max(len(value_losses), 1)),
        "entropy": float(sum(entropies) / max(len(entropies), 1)),
        "grad_norm": float(sum(grad_norms) / max(len(grad_norms), 1)),
        "approx_kl": float(sum(kl_divs) / max(len(kl_divs), 1)),
        "clip_fraction": float(sum(clip_fractions) / max(len(clip_fractions), 1)),
    }
