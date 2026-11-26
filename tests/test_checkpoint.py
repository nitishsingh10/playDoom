"""Architecture metadata, baseline isolation and inference-export checks."""

import json

import pytest
import torch
from torch import nn

from submission.checkpoint import (
    CHECKPOINT_VERSION,
    LEGACY_ARCHITECTURE_ID,
    STRUCTURED_ARCHITECTURE_ID,
    CheckpointCompatibilityError,
    load_policy_state,
)
from submission.model import DoomPolicy, build_policy
from submission.schema import PolicyConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from training.export_checkpoint import export_checkpoint


def test_structured_checkpoint_round_trip_with_model_configuration(tmp_path):
    model = DoomPolicy(hidden_size=16, fusion_size=48)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.actor.weight.sum().backward()
    optimizer.step()
    path = save_checkpoint(
        tmp_path / "structured.pt", model, optimizer, update=3, global_step=64,
        config={"lr": 1e-3}, extra={"entropy": 5.0},
    )
    saved = torch.load(path, weights_only=False)
    assert saved["checkpoint_version"] == CHECKPOINT_VERSION
    assert saved["architecture_id"] == STRUCTURED_ARCHITECTURE_ID
    assert saved["model_config"] == {
        "architecture_id": STRUCTURED_ARCHITECTURE_ID, "hidden_size": 16, "fusion_size": 48,
    }
    restored = build_policy(saved["model_config"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1.0)
    metadata = load_checkpoint(path, restored, restored_optimizer)
    assert metadata["update"] == 3
    assert metadata["global_step"] == 64
    assert metadata["config"]["architecture_id"] == STRUCTURED_ARCHITECTURE_ID
    assert metadata["extra"] == {"entropy": 5.0}
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_legacy_weights_rejected_before_structured_model_changes(tmp_path):
    legacy = DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID)
    path = tmp_path / "old_baseline.pt"
    torch.save(legacy.state_dict(), path)
    structured = DoomPolicy()
    before = {key: value.clone() for key, value in structured.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="Legacy 9-logit.*structured 12-logit"):
        load_checkpoint(path, structured)
    assert all(torch.equal(value, structured.state_dict()[key]) for key, value in before.items())


def test_raw_legacy_weights_can_load_only_when_legacy_is_explicit():
    legacy = DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID)
    restored = DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID)
    load_policy_state(restored, legacy.state_dict())
    assert torch.equal(legacy.actor.weight, restored.actor.weight)


@pytest.mark.parametrize("metadata", [{}, {"architecture_id": STRUCTURED_ARCHITECTURE_ID}, {
    "architecture_id": STRUCTURED_ARCHITECTURE_ID, "checkpoint_version": 999,
}])
def test_structured_checkpoint_requires_supported_version_metadata(metadata):
    model = DoomPolicy()
    checkpoint = {"model_state_dict": model.state_dict(), **metadata}
    with pytest.raises(CheckpointCompatibilityError, match="Structured checkpoints require"):
        load_policy_state(model, checkpoint)


def test_mislabeled_legacy_actor_is_rejected():
    legacy = DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID)
    checkpoint = {
        "model_state_dict": legacy.state_dict(), "architecture_id": STRUCTURED_ARCHITECTURE_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
    }
    with pytest.raises(CheckpointCompatibilityError, match="actor weights"):
        load_policy_state(DoomPolicy(), checkpoint)


def test_checkpoint_rejects_inconsistent_configuration(tmp_path):
    with pytest.raises(ValueError, match="does not match model"):
        save_checkpoint(tmp_path / "bad.pt", DoomPolicy(), config={"architecture_id": LEGACY_ARCHITECTURE_ID})
    path = save_checkpoint(tmp_path / "good.pt", DoomPolicy(hidden_size=16))
    with pytest.raises(CheckpointCompatibilityError, match="hidden_size=16.*hidden_size=32"):
        load_checkpoint(path, DoomPolicy())


def test_generic_pytorch_checkpoint_remains_supported(tmp_path):
    original = nn.Linear(3, 2)
    path = save_checkpoint(tmp_path / "generic.pt", original, config={"lr": 0.001})
    restored = nn.Linear(3, 2)
    info = load_checkpoint(path, restored)
    assert info["config"] == {"lr": 0.001}
    assert torch.equal(original.weight, restored.weight)


def test_export_is_weights_only_loadable_and_preserves_configuration(tmp_path):
    model = DoomPolicy(hidden_size=16, fusion_size=48)
    path = save_checkpoint(
        tmp_path / "training.pt", model,
        config={"deterministic_eval": False, "rollout_steps": 64},
    )
    exported = export_checkpoint(path, tmp_path / "experiment" / "checkpoint.pt")
    weights = torch.load(exported, weights_only=True)
    assert weights["architecture_id"] == STRUCTURED_ARCHITECTURE_ID
    assert weights["checkpoint_version"] == CHECKPOINT_VERSION
    assert "rng_state" not in weights and "optimizer_state_dict" not in weights
    config = PolicyConfig.load_json(exported.with_name("config.json"))
    assert config.hidden_size == 16 and config.fusion_size == 48
    assert not config.deterministic_eval
    restored = build_policy(config)
    load_policy_state(restored, weights)
    assert torch.equal(model.actor.weight, restored.actor.weight)


def test_export_defaults_to_a_separate_experiment_directory(tmp_path, monkeypatch):
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy())
    monkeypatch.chdir(tmp_path)
    output = export_checkpoint(source)
    assert not output.is_absolute()
    assert output.parts[:3] == ("training", "runs", "structured_v1")
    assert output.is_file() and output.with_name("config.json").is_file()


def test_export_preserves_baseline_checkpoint_and_config(tmp_path, monkeypatch):
    import training.export_checkpoint as exporter

    baseline = tmp_path / "submission" / "weights"
    baseline.mkdir(parents=True)
    checkpoint_path = baseline / "checkpoint.pt"
    config_path = baseline / "config.json"
    checkpoint_path.write_bytes(b"preserve baseline weights")
    config_path.write_text(json.dumps({"preserve": "baseline config"}))
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy())
    monkeypatch.setattr(exporter, "BASELINE_WEIGHTS_DIR", baseline)
    with pytest.raises(ValueError, match="Refusing to overwrite baseline"):
        export_checkpoint(source, checkpoint_path)
    assert checkpoint_path.read_bytes() == b"preserve baseline weights"
    assert json.loads(config_path.read_text()) == {"preserve": "baseline config"}


def test_export_refuses_to_replace_resumable_source(tmp_path):
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy())
    with pytest.raises(ValueError, match="must differ"):
        export_checkpoint(source, source)


def test_export_refuses_config_filename_collision(tmp_path):
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy())
    with pytest.raises(ValueError, match="reserved for policy configuration"):
        export_checkpoint(source, tmp_path / "export" / "config.json")


def test_export_legacy_baseline_keeps_legacy_semantics(tmp_path):
    legacy = DoomPolicy(architecture_id=LEGACY_ARCHITECTURE_ID)
    source = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), source)
    output = export_checkpoint(source, tmp_path / "legacy_export" / "checkpoint.pt")
    checkpoint = torch.load(output, weights_only=True)
    config = PolicyConfig.load_json(output.with_name("config.json"))
    assert config.architecture_id == LEGACY_ARCHITECTURE_ID
    restored = build_policy(config)
    load_policy_state(restored, checkpoint)
    assert restored.actor.out_features == 9
    assert torch.equal(legacy.actor.weight, restored.actor.weight)
