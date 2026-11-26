"""Pure unit checks for action parity and rollout replay; no engine or optimizer."""

import numpy as np
import pytest
import torch

from doom_env import spaces as doom_spaces
from submission.actions import StructuredActionDistribution
from submission.agent import DoomAgent
from submission.model import DoomPolicy
from test_model import make_observation
from training.ppo import evaluate_actions, sample_actions
from training.rollout import RolloutCollector, collate_observations


@pytest.mark.parametrize("deterministic", [False, True])
def test_agent_and_training_choose_identical_actions_on_same_observation(tmp_path, deterministic):
    """Match the complete inference path to training's raw-observation path."""
    with torch.random.fork_rng():
        torch.manual_seed(317)
        model = DoomPolicy().eval()
        # An empty temporary directory avoids loading or writing any checkpoint.
        agent = DoomAgent(
            doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=tmp_path,
        )
        agent.model = model
        agent.config.deterministic_eval = deterministic
        agent.reset()
        observation = make_observation(3)
        with torch.no_grad():
            logits, _, _ = model(collate_observations([observation], torch.device("cpu")))

        # Several seeds also check exact stochastic RNG consumption parity.
        for seed in (0, 1, 17, 310):
            torch.manual_seed(seed)
            expected = (
                StructuredActionDistribution(logits).mode()
                if deterministic else sample_actions(logits)[0]
            )[0].numpy()
            torch.manual_seed(seed)
            np.testing.assert_array_equal(agent.act(observation), expected)


class _RecordingObservationEnv:
    """Fake transitions record submitted buttons without emulating Doom physics."""

    def __init__(self, truncated):
        self.truncated = truncated
        self.episode = -1
        self.step_in_episode = 0
        self.reset_seeds = []
        self.executed_actions = []
        self.executed_episodes = []

    def _observation(self):
        observation = make_observation(1 + self.episode % 3)
        # Distinguish decisions and resets so replay also checks observation alignment.
        observation["vitals"][0] = 100 - 3 * self.episode - self.step_in_episode
        observation["vitals"][14] = self.step_in_episode
        return observation

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode += 1
        self.step_in_episode = 0
        return self._observation(), {}

    def step(self, action):
        assert action.shape == (9,)
        assert action.dtype == np.int8
        assert np.all((action == 0) | (action == 1))
        self.executed_actions.append(action.copy())
        self.executed_episodes.append(self.episode)
        self.step_in_episode += 1
        done = self.step_in_episode == 2
        return self._observation(), 0.0, done and not self.truncated, done and self.truncated, {}


@pytest.mark.parametrize("truncated", [False, True])
def test_collector_executes_stored_buttons_and_replays_probability_across_resets(truncated):
    """Check real policy sampling/storage/replay through fake episode boundaries."""
    with torch.random.fork_rng():
        torch.manual_seed(819)
        model = DoomPolicy().eval()
        original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        env = _RecordingObservationEnv(truncated=truncated)
        collector = RolloutCollector(env, device="cpu", seed=27)
        batch = collector.collect(model, num_steps=7)

        np.testing.assert_array_equal(np.stack(env.executed_actions), batch["actions"].numpy())
        assert env.executed_episodes == [0, 0, 1, 1, 2, 2, 3]
        assert env.reset_seeds == [27, None, None, None]
        assert batch["dones"].tolist() == [0, 1, 0, 1, 0, 1, 0]
        assert batch["truncations"].tolist() == (
            [False, True, False, True, False, True, False] if truncated else [False] * 7
        )
        assert (batch["actions"][:, :6].reshape(-1, 3, 2).sum(-1) <= 1).all()
        assert not batch["log_probs"].requires_grad

        with torch.no_grad():
            logits, _, _ = model(batch["obs"])
            replay_log_prob, entropy = evaluate_actions(logits, batch["actions"])
        torch.testing.assert_close(replay_log_prob, batch["log_probs"])
        torch.testing.assert_close((replay_log_prob - batch["log_probs"]).exp(), torch.ones(7))
        assert torch.isfinite(entropy).all()
        assert (entropy >= 0).all()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)
