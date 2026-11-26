"""Unit tests for Member 2 PPO, GAE, action distribution, and checkpointing.

Verifies:
  1. GAE terminal boundary handling: true terminal transitions do not bootstrap next value.
  2. Action sampling & log-prob shapes: 9 independent Bernoulli action dimensions.
  3. PPO loss finiteness and optimizer parameter update.
  4. Deterministic checkpoint save and restore consistency.
  5. Mini training loop integration smoke test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from training.checkpoint import load_checkpoint, save_checkpoint
from training.ppo import compute_gae, evaluate_actions, ppo_update, sample_actions
from training.train import default_train_config, train_loop


class DummyDoomPolicy(nn.Module):
    """Minimal policy/value network matching Member 3's DoomPolicy interface for testing."""

    def __init__(self, input_dim: int = 16, num_actions: int = 9):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 32)
        self.actor_head = nn.Linear(32, num_actions)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, x, hidden_state=None):
        if isinstance(x, dict):
            # If dictionary observation, concatenate or select first tensor
            first_val = next(iter(x.values()))
            if isinstance(first_val, dict):
                first_val = next(iter(first_val.values()))
            features = first_val.float()
        else:
            features = x.float()

        if features.dim() > 2:
            features = features.flatten(1)

        h = torch.relu(self.encoder(features))
        logits = self.actor_head(h)  # [batch, 9]
        values = self.critic_head(h).squeeze(-1)  # [batch]
        return logits, values, None


class DummyRolloutCollector:
    """Mock RolloutCollector for smoke testing training orchestration."""

    def __init__(self, obs_dim: int = 16, num_actions: int = 9):
        self.obs_dim = obs_dim
        self.num_actions = num_actions

    def collect(self, policy: nn.Module, num_steps: int = 32):
        obs = torch.randn(num_steps, self.obs_dim)
        with torch.no_grad():
            logits, values, _ = policy(obs)
            actions, log_probs = sample_actions(logits)
            last_logits, last_val, _ = policy(torch.randn(1, self.obs_dim))

        rewards = torch.randn(num_steps)
        dones = torch.zeros(num_steps)
        dones[num_steps // 2] = 1.0  # One mid-rollout termination

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
            "last_value": last_val.squeeze(0),
        }


def test_bernoulli_action_sampling_and_shapes():
    """Verify that 9 Bernoulli actions sample valid {0, 1} and return joint log_probs."""
    batch_size = 10
    logits = torch.randn(batch_size, 9)

    actions, log_probs = sample_actions(logits)

    assert actions.shape == (batch_size, 9), f"Expected [10, 9], got {actions.shape}"
    assert log_probs.shape == (batch_size,), f"Expected [10], got {log_probs.shape}"

    # Actions must be binary (0 or 1)
    unique_vals = set(actions.flatten().int().tolist())
    assert unique_vals.issubset({0, 1}), f"Non-binary actions found: {unique_vals}"

    # Evaluate actions
    eval_log_probs, entropy = evaluate_actions(logits, actions)
    assert eval_log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)

    # Log probs from evaluation should match sampling log_probs
    assert torch.allclose(log_probs, eval_log_probs, atol=1e-5), "Mismatch between sample and eval log_prob"
    # Entropies must be strictly positive for non-degenerate logits
    assert (entropy > 0).all(), "Entropy should be positive"


def test_gae_terminal_handling():
    """Verify GAE: terminal state does NOT bootstrap next value, advantages are finite."""
    T = 4
    gamma = 0.99
    gae_lambda = 0.95

    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # Step 1 is terminal (done=1.0)
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0])
    last_value = torch.tensor(0.5)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert torch.isfinite(advantages).all(), "Advantages contain NaN/inf"
    assert torch.isfinite(returns).all(), "Returns contain NaN/inf"
    assert advantages.shape == (T,)
    assert returns.shape == (T,)

    # Hand verification for step 1 (where done=1.0):
    # delta_1 = reward[1] + gamma * values[2] * (1 - done[1]) - values[1]
    # Since done[1] == 1, next value bootstrap is 0!
    # delta_1 = 1.0 + 0 - 0.5 = 0.5
    # adv[1] = delta_1 = 0.5 (since done[1] == 1 resets forward GAE)
    expected_adv_1 = 1.0 - 0.5
    assert torch.isclose(advantages[1], torch.tensor(expected_adv_1), atol=1e-5), (
        f"Step 1 advantage {advantages[1]} != expected {expected_adv_1} on terminal boundary"
    )

    # Step 0 bootstraps step 1:
    # delta_0 = 1.0 + gamma * values[1] - values[0] = 1.0 + 0.99 * 0.5 - 0.5 = 0.995
    # adv[0] = delta_0 + gamma * lambda * (1 - done[0]) * adv[1]
    expected_delta_0 = 1.0 + gamma * 0.5 - 0.5
    expected_adv_0 = expected_delta_0 + gamma * gae_lambda * expected_adv_1
    assert torch.isclose(advantages[0], torch.tensor(expected_adv_0), atol=1e-5)


def test_gae_truncation_bootstraps_terminal_observation_and_cuts_episode():
    rewards = torch.tensor([1.0, 5.0, 1.0])
    values = torch.tensor([0.5, 10.0, 0.25])
    dones = torch.tensor([1.0, 0.0, 0.0])
    truncations = torch.tensor([True, False, False])
    bootstrap_values = torch.tensor([2.0, 0.0, 0.0])

    advantages, _ = compute_gae(
        rewards, values, dones, torch.tensor(0.0), gamma=0.99, gae_lambda=0.95,
        truncations=truncations, bootstrap_values=bootstrap_values,
    )

    assert advantages[0].item() == pytest.approx(1.0 + 0.99 * 2.0 - 0.5)
    # The large value in the reset episode must not leak across the boundary.
    assert advantages[0].item() != pytest.approx(1.0 + 0.99 * 10.0 - 0.5)


def test_ppo_synthetic_batch_update_and_parameter_changes():
    """Verify PPO update produces finite metrics and modifies model parameters."""
    model = DummyDoomPolicy(input_dim=16, num_actions=9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Copy initial parameter values to detect updates
    initial_params = [p.clone() for p in model.parameters()]

    batch_size = 64
    obs = torch.randn(batch_size, 16)
    actions = torch.randint(0, 2, (batch_size, 9)).float()
    old_log_probs = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    old_values = returns - advantages

    batch = {
        "obs": obs,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "returns": returns,
        "advantages": advantages,
        "old_values": old_values,
    }

    config = {
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "update_epochs": 2,
        "minibatch_size": 16,
        "normalize_advantages": True,
    }

    metrics = ppo_update(model, optimizer, batch, config)

    # Check that metrics are finite numbers
    for k, v in metrics.items():
        assert np.isfinite(v), f"Metric {k} is not finite: {v}"

    # Check that parameters changed after optimizer step
    param_changed = False
    for p_init, p_new in zip(initial_params, model.parameters()):
        if not torch.allclose(p_init, p_new):
            param_changed = True
            break
    assert param_changed, "Model parameters did not change during PPO update"


def test_checkpoint_save_and_load():
    """Verify deterministic checkpoint saving and loading."""
    model = DummyDoomPolicy(input_dim=16, num_actions=9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Make one step to give optimizer non-empty state
    loss = model(torch.randn(2, 16))[0].sum()
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
        saved = save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            update=42,
            global_step=1337,
            config={"lr": 1e-3},
            extra={"test_metric": 0.99},
        )
        assert saved.exists()

        # Create fresh model with different initial weights
        restored_model = DummyDoomPolicy(input_dim=16, num_actions=9)
        restored_optimizer = optim.Adam(restored_model.parameters(), lr=1e-3)

        info = load_checkpoint(
            path=ckpt_path,
            model=restored_model,
            optimizer=restored_optimizer,
            device="cpu",
        )

        assert info["update"] == 42
        assert info["global_step"] == 1337
        assert info["config"]["lr"] == 1e-3
        assert info["extra"]["test_metric"] == 0.99

        # Ensure model weights match exactly
        for p1, p2 in zip(model.parameters(), restored_model.parameters()):
            assert torch.allclose(p1, p2), "Restored model weights differ from original"


def test_training_smoke_loop():
    """Verify training loop runs a few updates with mock collector without crash or NaN."""
    model = DummyDoomPolicy(input_dim=16, num_actions=9)
    collector = DummyRolloutCollector(obs_dim=16, num_actions=9)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = default_train_config()
        cfg["total_updates"] = 3
        cfg["rollout_steps"] = 16
        cfg["minibatch_size"] = 8
        cfg["update_epochs"] = 2
        cfg["checkpoint_interval"] = 2
        cfg["save_dir"] = tmp_dir
        cfg["device"] = "cpu"

        summary = train_loop(
            model=model,
            collector=collector,
            config=cfg,
        )

        assert summary["total_updates"] == 3
        assert summary["total_steps"] == 3 * 16
        assert (Path(tmp_dir) / "checkpoint_2.pt").exists()
        assert (Path(tmp_dir) / "checkpoint_latest.pt").exists()


if __name__ == "__main__":
    print("Running test_bernoulli_action_sampling_and_shapes...")
    test_bernoulli_action_sampling_and_shapes()
    print("Running test_gae_terminal_handling...")
    test_gae_terminal_handling()
    print("Running test_ppo_synthetic_batch_update_and_parameter_changes...")
    test_ppo_synthetic_batch_update_and_parameter_changes()
    print("Running test_checkpoint_save_and_load...")
    test_checkpoint_save_and_load()
    print("Running test_training_smoke_loop...")
    test_training_smoke_loop()
    print("All Member 2 PPO tests passed successfully!")
