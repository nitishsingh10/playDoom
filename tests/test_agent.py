"""Exercise the submission entry point with separately exported structured weights."""

import json

import numpy as np
import pytest
import torch

from doom_env import spaces as doom_spaces
from submission.actions import StructuredActionDistribution
from submission.agent import DoomAgent
from submission.checkpoint import CheckpointCompatibilityError, LEGACY_ARCHITECTURE_ID
from submission.model import DoomPolicy
from submission.schema import preprocess_observation
from training.checkpoint import save_checkpoint
from training.export_checkpoint import export_checkpoint


def export_agent(tmp_path, deterministic):
    source = tmp_path / "training.pt"
    weights = tmp_path / "export"
    save_checkpoint(source, DoomPolicy(), update=0, global_step=0)
    export_checkpoint(source, weights / "checkpoint.pt")
    config_path = weights / "config.json"
    config = json.loads(config_path.read_text())
    config["deterministic_eval"] = deterministic
    config_path.write_text(json.dumps(config))
    return DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=weights)


def assert_valid(action):
    assert action.shape == (9,)
    assert action.dtype == np.int8
    assert set(action.tolist()) <= {0, 1}
    assert np.all(action[:6].reshape(3, 2).sum(axis=-1) <= 1)


@pytest.mark.parametrize("deterministic", [True, False])
def test_exported_agent_actions_and_reset(tmp_path, deterministic):
    agent = export_agent(tmp_path, deterministic)
    observation = doom_spaces.observation_space().sample()
    # Equal logits give broad support for every decision during stochastic inference.
    with torch.no_grad():
        agent.model.actor.weight.zero_()
        agent.model.actor.bias.zero_()
    actions = [agent.act(observation) for _ in range(40)]
    for action in actions:
        assert_valid(action)
    if deterministic:
        assert all(np.array_equal(action, actions[0]) for action in actions)
    else:
        assert len({tuple(action) for action in actions}) > 1
    assert torch.get_num_threads() == 1
    agent.reset()
    assert agent.step_count == 0
    assert not agent.edge_tracker.last_action.any()


def test_agent_uses_exact_training_distribution_mode(tmp_path):
    agent = export_agent(tmp_path, True)
    observation = doom_spaces.observation_space().sample()
    processed = preprocess_observation(observation)
    with torch.inference_mode():
        batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in processed.items()}
        logits, _, _ = agent.model(batch)
        expected = StructuredActionDistribution(logits).mode()[0].numpy()
    np.testing.assert_array_equal(agent.act(observation), expected)


def test_agent_rejects_unversioned_legacy_weights(tmp_path):
    torch.save(DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID).state_dict(), tmp_path / "checkpoint.pt")
    with pytest.raises(CheckpointCompatibilityError, match="(?i)(legacy|nine|9|incompatible|architecture)"):
        DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=tmp_path)


def test_agent_rejects_resumable_checkpoint_with_export_guidance(tmp_path):
    save_checkpoint(tmp_path / "checkpoint.pt", DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID))
    with pytest.raises(CheckpointCompatibilityError, match="inference export"):
        DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=tmp_path)


def test_agent_without_weights_still_has_valid_structured_actions(tmp_path):
    agent = DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=tmp_path)
    assert_valid(agent.act(doom_spaces.observation_space().sample()))


def test_use_and_weapon_outputs_are_not_rewritten(tmp_path):
    agent = export_agent(tmp_path, True)
    with torch.no_grad():
        agent.model.actor.weight.zero_()
        agent.model.actor.bias.zero_()
        agent.model.actor.bias[9:] = 20
    observation = doom_spaces.observation_space().sample()
    # Held signals must reach vanilla Doom unchanged; the engine handles their edges.
    for _ in range(3):
        action = agent.act(observation)
        np.testing.assert_array_equal(action[6:], [1, 1, 1])
