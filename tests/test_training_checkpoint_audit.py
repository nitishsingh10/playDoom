"""Checkpoint wiring checks without engine execution or optimizer updates."""

import pytest
import torch

from submission.checkpoint import LEGACY_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID
from submission.model import DoomPolicy
from test_model import make_observation
from training.checkpoint import load_checkpoint, save_checkpoint
from training.evaluate import load_policy
from training.train import train_loop


@pytest.mark.parametrize("architecture", [LEGACY_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID])
@pytest.mark.parametrize("dimensions", [(32, 128), (16, 48)])
def test_evaluator_restores_every_tensor_and_exact_predictions(tmp_path, architecture, dimensions):
    """Shape compatibility alone cannot detect evaluating fresh random weights."""
    with torch.random.fork_rng():
        torch.manual_seed(41)
        original = DoomPolicy(*dimensions, architecture_id=architecture).eval()
        checkpoint = save_checkpoint(tmp_path / "candidate.pt", original)
        torch.manual_seed(999)  # Loader construction must not retain these fresh weights.
        restored = load_policy(checkpoint, "cpu", legacy_baseline=architecture == LEGACY_ARCHITECTURE_ID)
        assert not restored.training
        for name, value in original.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name]), name
        with torch.inference_mode():
            expected_logits, expected_values, _ = original(make_observation(3))
            logits, values, _ = restored(make_observation(3))
        assert torch.equal(logits, expected_logits)
        assert torch.equal(values, expected_values)


def test_train_resume_restores_model_and_adam_before_first_collection(tmp_path, monkeypatch, capsys):
    """Exercise the actual train-loop resume branch; stop before any rollout."""
    class CollectionReached(Exception):
        pass

    with torch.random.fork_rng():
        torch.manual_seed(41)
        original = DoomPolicy()
        original_optimizer = torch.optim.Adam(original.parameters(), lr=3e-4)
        # Synthetic Adam state tests restoration without performing a training step.
        for parameter in original.parameters():
            original_optimizer.state[parameter] = {
                "step": torch.tensor(4000.0),
                "exp_avg": torch.full_like(parameter, 0.01),
                "exp_avg_sq": torch.full_like(parameter, 0.02),
            }
        path = save_checkpoint(tmp_path / "parent.pt", original, original_optimizer,
                               update=500, global_step=64000)
        loaded = []

        def inspect_load(checkpoint_path, model, optimizer, **kwargs):
            metadata = load_checkpoint(checkpoint_path, model, optimizer, **kwargs)
            assert optimizer.param_groups[0]["lr"] == 3e-4
            assert set(optimizer.param_groups[0]["params"]) == set(model.parameters())
            for parameter in model.parameters():
                state = optimizer.state[parameter]
                assert state["step"].item() == 4000
                assert torch.equal(state["exp_avg"], torch.full_like(parameter, 0.01))
                assert torch.equal(state["exp_avg_sq"], torch.full_like(parameter, 0.02))
            loaded.append(metadata)
            return metadata

        class Collector:
            device = None

            def collect(self, model, num_steps):
                assert loaded and loaded[0]["update"] == 500
                assert loaded[0]["global_step"] == 64000
                assert num_steps == 128
                for name, tensor in original.state_dict().items():
                    assert torch.equal(tensor, model.state_dict()[name]), name
                raise CollectionReached

        monkeypatch.setattr("training.train.load_checkpoint", inspect_load)
        torch.manual_seed(999)
        with pytest.raises(CollectionReached):
            train_loop(DoomPolicy(), Collector(), resume_path=str(path), config={
                "device": "cpu", "total_updates": 501, "rollout_steps": 128,
                "save_dir": str(tmp_path / "unused_outputs"),
            })
        assert "at update 501, global step 64000" in capsys.readouterr().out
