"""PPO learning algorithm, Generalized Advantage Estimation (GAE), and action utilities.

Owned by Member 2 (PPO and Optimization Engineer).
Provides:
  - compute_gae: Generalized Advantage Estimation with terminal vs truncation distinction
  - evaluate_actions: Computes log probs and entropy for 9 independent Bernoulli action dimensions
  - sample_actions: Samples binary actions from logits and returns log probs
  - ppo_update: Mini-batch clipped surrogate objective, value loss, entropy bonus, gradient clipping
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli


def sample_actions(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample binary actions from Bernoulli logits and compute their log probabilities.

    Each action dimension is treated as an independent Bernoulli variable:
      P(action_i = 1) = sigmoid(logits_i)

    Args:
        logits: Tensor of shape [batch, 9] or [..., 9] containing unnormalized Bernoulli logits.

    Returns:
        actions: Binary tensor of shape [batch, 9] with values in {0, 1} (int8 or float32).
        log_prob: Tensor of shape [batch] containing sum of log probabilities across 9 dimensions:
                  sum_{i=0}^8 log P(action_i | logits_i)
    """
    dist = Bernoulli(logits=logits)
    actions = dist.sample()  # shape [batch, 9] in {0.0, 1.0}
    # Log prob per dimension is shape [batch, 9], sum over the 9 action dimensions
    log_probs = dist.log_prob(actions).sum(dim=-1)  # shape [batch]
    return actions, log_probs


def evaluate_actions(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate log probabilities and entropy of given actions under policy logits.

    Args:
        logits: Tensor of shape [batch, 9] with unnormalized Bernoulli logits.
        actions: Tensor of shape [batch, 9] with binary action values in {0, 1}.

    Returns:
        log_prob: Tensor of shape [batch], joint log probability of actions.
        entropy: Tensor of shape [batch], sum of Bernoulli entropies over 9 dimensions.
    """
    dist = Bernoulli(logits=logits)
    log_prob = dist.log_prob(actions.float()).sum(dim=-1)  # shape [batch]
    entropy = dist.entropy().sum(dim=-1)  # shape [batch]
    return log_prob, entropy


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    truncations: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and returns.

    Terminal handling rules:
    - If a step is a true terminal (player death, goal reached, terminated=True):
      The episode truly ended; future value does NOT bootstrap: next_non_terminal = 0.
    - If a step is truncated (time limit cap reached, truncated=True):
      The episode stopped due to horizon, but the underlying state is not terminal.
      If truncations mask is provided, we can bootstrap value at truncation while
      cutting the trajectory horizon, or handle standard dones.
    - When dones is [T, B] boolean or float where 1.0 indicates episode end (terminated):
      next_non_terminal = 1.0 - dones[t].

    Args:
        rewards: [T, B] or [T] tensor of rewards.
        values: [T, B] or [T] tensor of value estimates V(s_t).
        dones: [T, B] or [T] tensor of done flags (1.0 = terminal/done, 0.0 = not done).
        last_value: [B] or scalar tensor of value estimate for the step after T (V(s_{T})).
        gamma: Discount factor (e.g., 0.99).
        gae_lambda: GAE smoothing parameter (e.g., 0.95).
        truncations: Optional [T, B] tensor indicating time-limit truncation.

    Returns:
        advantages: [T, B] or [T] tensor of GAE advantages A_t.
        returns: [T, B] or [T] tensor of target returns R_t = A_t + V(s_t).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)

    # If truncations are provided, a done that is truncated still has a continuation value
    # for bootstrapping the TD error, but last_gae does not propagate across episode boundary.
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = values[t + 1]

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        # Note: if truncated at step t, V(s_{t+1}) is valid bootstrap for r_t, but GAE resets
        if truncations is not None and truncations[t]:
            # Truncation: bootstrap value from next_value, but don't propagate GAE backwards
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta
        else:
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Perform PPO optimization over a rollout batch using mini-batches and multiple epochs.

    Batch schema expected:
      - 'obs': Encoded observations or raw obs dict acceptable by model.forward
               (can be a Tensor [N, ...] or Dict of Tensors).
      - 'actions': Tensor [N, 9] binary actions.
      - 'old_log_probs': Tensor [N] log probabilities under the behavior policy.
      - 'returns': Tensor [N] target returns.
      - 'advantages': Tensor [N] estimated advantages.
      - 'old_values': Optional Tensor [N] value predictions from rollout.
      - 'hidden_state': Optional initial hidden state if recurrent policy used.

    Config parameters:
      - 'clip_eps': PPO clip parameter epsilon (default: 0.2).
      - 'vf_coef': Value function loss coefficient (default: 0.5).
      - 'ent_coef': Entropy bonus coefficient (default: 0.01).
      - 'max_grad_norm': Max gradient norm for clipping (default: 0.5).
      - 'update_epochs': Number of passes over the batch (default: 4).
      - 'minibatch_size': Size of minibatches (default: 64 or len(batch)//4).
      - 'clip_vloss': Boolean, whether to clip value loss (default: True).
      - 'normalize_advantages': Boolean (default: True).

    Returns:
        metrics: Dictionary containing mean policy_loss, value_loss, entropy, grad_norm, approx_kl, clip_fraction.
    """
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

    # Flatten / ensure 1D shape for scalar tensors
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

            # Slice observations (handle Tensor or Dict of Tensors)
            if isinstance(obs, dict):
                mb_obs = {}
                for k, v in obs.items():
                    if isinstance(v, dict):
                        mb_obs[k] = {sk: sv[mb_inds] for sk, sv in v.items()}
                    else:
                        mb_obs[k] = v[mb_inds]
            else:
                mb_obs = obs[mb_inds]

            mb_actions = actions[mb_inds]
            mb_old_log_probs = old_log_probs[mb_inds]
            mb_returns = returns[mb_inds]
            mb_advantages = advantages[mb_inds]
            mb_old_values = old_values[mb_inds]

            if normalize_advantages and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Model forward call
            # Policy forward is expected to return (action_logits [B, 9], value [B], next_hidden)
            forward_out = model(mb_obs)
            if isinstance(forward_out, (tuple, list)):
                logits = forward_out[0]
                values = forward_out[1]
            else:
                raise ValueError("Model output must be tuple (logits, values) or (logits, values, next_hidden)")

            values = values.view(-1)

            # Evaluate new log_probs and entropy for the 9 Bernoulli action dims
            new_log_probs, entropy = evaluate_actions(logits, mb_actions)

            # Ratio = exp(new_log_prob - old_log_prob)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            # Approx KL divergence for early stopping / diagnostics: (ratio - 1) - log_ratio
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipped = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()
                kl_divs.append(approx_kl.item())
                clip_fractions.append(clipped.item())

            # Clipped surrogate policy loss
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (with optional clipping)
            if clip_vloss:
                v_clipped = mb_old_values + torch.clamp(values - mb_old_values, -clip_eps, clip_eps)
                v_loss1 = (values - mb_returns) ** 2
                v_loss2 = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

            # Total loss: Policy loss + vf_coef * Value loss - ent_coef * Entropy
            entropy_bonus = entropy.mean()
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            # Backpropagation
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
