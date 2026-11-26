"""Export/readiness checks use synthetic weights, not gameplay or training."""
import json
import subprocess

import pytest
import torch

from submission.model import DoomPolicy
from training.checkpoint import save_checkpoint
from training.check_submission import check_export, main
from training.export_checkpoint import export_checkpoint, main as export_main


def make_export(tmp_path, deterministic=False):
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy(), update=500, global_step=64000)
    destination = tmp_path / "export" / "checkpoint.pt"
    export_checkpoint(source, destination, deterministic_eval=deterministic)
    return source, destination.parent


@pytest.mark.parametrize("mode", ["stochastic", "deterministic"])
def test_loaded_model_exact_source_weights_action_parity_and_mode(tmp_path, mode):
    source, directory = make_export(tmp_path, deterministic=mode == "deterministic")
    result = check_export(directory, mode=mode, source_checkpoint=source)
    assert result["model_loaded"]
    assert result["exact_export_parameters"] and result["exact_source_parameters"]
    assert result["synthetic_action_parity_checks"] == 6
    assert result["files_unchanged"]
    assert not result["gameplay_quality_verified"]
    assert not result["current_commit_files_verified"]


def test_empty_weights_directory_fails_instead_of_accepting_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="fallback"):
        check_export(tmp_path, mode="stochastic")


def test_wrong_inference_mode_is_not_silently_accepted(tmp_path):
    source, directory = make_export(tmp_path)
    with pytest.raises(ValueError, match="deterministic inference"):
        check_export(directory, mode="deterministic", source_checkpoint=source)


def test_wrong_training_checkpoint_is_not_silently_accepted(tmp_path):
    _, directory = make_export(tmp_path)
    other = save_checkpoint(tmp_path / "other.pt", DoomPolicy())
    with pytest.raises(ValueError, match="selected training checkpoint"):
        check_export(directory, mode="stochastic", source_checkpoint=other)


def test_nonfinite_export_is_rejected(tmp_path):
    _, directory = make_export(tmp_path)
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
    checkpoint["model_state_dict"]["actor.bias"][0] = float("nan")
    torch.save(checkpoint, directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="nonfinite"):
        check_export(directory, mode="stochastic")


@pytest.mark.parametrize("flag,deterministic", [("--stochastic", False), ("--deterministic", True)])
def test_export_cli_selects_explicit_inference_mode(tmp_path, flag, deterministic):
    source = save_checkpoint(tmp_path / "training.pt", DoomPolicy())
    destination = tmp_path / "export" / "checkpoint.pt"
    assert export_main([str(source), str(destination), flag]) == 0
    assert json.loads(destination.with_name("config.json").read_text())["deterministic_eval"] is deterministic


def test_commit_requirement_rejects_untracked_and_modified_weights(tmp_path):
    source, directory = make_export(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="absent from the current Git commit"):
        check_export(directory, mode="stochastic", require_committed=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "export"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "Test export"], check=True)
    result = check_export(directory, mode="stochastic", source_checkpoint=source, require_committed=True)
    assert result["current_commit_files_verified"]
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
    checkpoint["model_state_dict"]["actor.bias"].add_(0.1)
    torch.save(checkpoint, directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="differs from the current Git commit"):
        check_export(directory, mode="stochastic", require_committed=True)


def test_cli_missing_weights_returns_nonzero_with_guidance(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--mode", "stochastic"])
    assert error.value.code == 1
    assert "fallback" in capsys.readouterr().err
