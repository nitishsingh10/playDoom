"""Direction-aware variant checks; no optimizer steps or engine execution."""
import copy

import pytest
import torch

from submission.checkpoint import (
    CHECKPOINT_VERSION, DIRECTIONAL_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID,
    CheckpointCompatibilityError, checkpoint_architecture, load_policy_state,
)
from submission.model import DoomPolicy
from test_model import make_observation
from training.checkpoint import save_checkpoint
from training.check_submission import check_export
from training.export_checkpoint import export_checkpoint
from training.ppo import evaluate_actions, sample_actions


def direction_observation(index):
    observation = make_observation(0)
    observation["rays"]["distance"][:] = 1
    observation["rays"]["hit_type"][:] = 1  # wall
    observation["rays"]["hit_type"][index] = 2  # one door cue
    return observation


def test_v1_is_still_default_and_direction_conventions_match_sensors():
    assert DoomPolicy().architecture_id == STRUCTURED_ARCHITECTURE_ID
    assert DoomPolicy().ray_encoder[0].in_features == 7
    model = DoomPolicy(architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
    assert model.ray_encoder[0].in_features == 9
    torch.testing.assert_close(model.ray_directions[[0, 16, 32, 48]], torch.tensor(
        [[0., 1.], [1., 0.], [0., -1.], [-1., 0.]]), atol=1e-7, rtol=0)
    assert "ray_directions" not in model.state_dict()


def test_v2_can_distinguish_same_ray_content_at_front_and_left():
    """Construct an angle-sensitive parameterization; this is not learned behavior."""
    model = DoomPolicy(hidden_size=1, fusion_size=2, architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # ReLU(2*door + cos(angle) - 2.5): active only for a front-facing door.
        model.ray_encoder[0].weight[0, 3] = 2
        model.ray_encoder[0].weight[0, 8] = 1
        model.ray_encoder[0].bias[0] = -2.5
        model.ray_encoder[2].weight[0, 0] = 1
        model.fusion[0].weight[0, 1] = 1  # ray max pool
        model.fusion[2].weight[0, 0] = 1
        model.actor.weight[0, 0] = 1
        front = model(direction_observation(0))[0][0, 0]
        left = model(direction_observation(16))[0][0, 0]
    assert front.item() == pytest.approx(0.5)
    assert left.item() == 0
    # v1 remains permutation invariant, retaining exact old architecture behavior.
    original = DoomPolicy().eval()
    with torch.no_grad():
        a = original(direction_observation(0))[0]
        b = original(direction_observation(16))[0]
    torch.testing.assert_close(a, b, atol=1e-7, rtol=0)


def test_directional_gradient_flow_and_ppo_replay_without_optimizer_steps():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = DoomPolicy(architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
        observation = make_observation(3)
        original = copy.deepcopy(model.state_dict())
        logits, values, hidden = model(observation)
        assert logits.shape == (1, 12) and values.shape == (1,) and hidden is None
        actions, stored = sample_actions(logits.detach())
        replay, entropy = evaluate_actions(logits, actions)
        torch.testing.assert_close(stored, replay.detach(), atol=0, rtol=0)
        torch.testing.assert_close((replay.detach() - stored).exp(), torch.ones(1))
        loss = -replay.mean() - 0.01 * entropy.mean() + values.square().mean()
        loss.backward()
        gradient = model.ray_encoder[0].weight.grad[:, -2:]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert torch.isfinite(model.actor.weight.grad).all()
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, original[name])


@pytest.mark.parametrize("mode", ["stochastic", "deterministic"])
def test_v2_export_loads_through_actual_agent_and_keeps_metadata(tmp_path, mode):
    model = DoomPolicy(hidden_size=16, fusion_size=48, architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
    source = save_checkpoint(tmp_path / "v2.pt", model)
    destination = tmp_path / "export" / "checkpoint.pt"
    export_checkpoint(source, destination, deterministic_eval=mode == "deterministic")
    result = check_export(destination.parent, mode=mode, source_checkpoint=source)
    assert result["architecture_id"] == DIRECTIONAL_ARCHITECTURE_ID
    assert result["exact_source_parameters"] and result["synthetic_action_parity_checks"] == 6


@pytest.mark.parametrize("saved_arch,requested_arch", [
    (STRUCTURED_ARCHITECTURE_ID, DIRECTIONAL_ARCHITECTURE_ID),
    (DIRECTIONAL_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID),
])
def test_v1_v2_cross_loading_rejected_before_any_parameter_changes(saved_arch, requested_arch):
    source = DoomPolicy(architecture_id=saved_arch)
    destination = DoomPolicy(architecture_id=requested_arch)
    before = copy.deepcopy(destination.state_dict())
    checkpoint = {"model_state_dict": source.state_dict(), "architecture_id": saved_arch,
                  "checkpoint_version": CHECKPOINT_VERSION}
    with pytest.raises(CheckpointCompatibilityError, match="not interchangeable"):
        load_policy_state(destination, checkpoint)
    assert all(torch.equal(tensor, destination.state_dict()[name]) for name, tensor in before.items())


def test_mislabeled_and_unversioned_v2_weights_are_rejected():
    model = DoomPolicy(architecture_id=DIRECTIONAL_ARCHITECTURE_ID)
    checkpoint = {"model_state_dict": model.state_dict(), "architecture_id": STRUCTURED_ARCHITECTURE_ID,
                  "checkpoint_version": CHECKPOINT_VERSION}
    with pytest.raises(CheckpointCompatibilityError, match="declares"):
        checkpoint_architecture(checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="Structured checkpoints require"):
        load_policy_state(model, model.state_dict())


def test_default_v2_export_does_not_overwrite_v1_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "training/runs/structured_v1/export/checkpoint.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"preserve existing v1 export")
    source = save_checkpoint(tmp_path / "v2.pt", DoomPolicy(architecture_id=DIRECTIONAL_ARCHITECTURE_ID))
    output = export_checkpoint(source)
    assert str(output) == "training/runs/structured_ray_v2/export/checkpoint.pt"
    assert old.read_bytes() == b"preserve existing v1 export"


@pytest.mark.parametrize("architecture,directory", [
    (STRUCTURED_ARCHITECTURE_ID, "structured_v1"),
    (DIRECTIONAL_ARCHITECTURE_ID, "structured_ray_v2"),
])
def test_training_cli_opt_in_uses_separate_output_paths(tmp_path, monkeypatch, architecture, directory):
    from training import train
    monkeypatch.chdir(tmp_path)
    calls = []

    def no_training(model, collector, config, **kwargs):
        assert model.architecture_id == architecture
        assert config["save_dir"] == f"training/runs/{directory}/checkpoints"
        calls.append(True)
        return {}

    monkeypatch.setattr(train, "train_loop", no_training)
    assert train.main(["--architecture", architecture, "--device", "cpu"]) == 0
    assert calls
    assert (tmp_path / f"training/runs/{directory}/logs/training_metrics.json").is_file()
