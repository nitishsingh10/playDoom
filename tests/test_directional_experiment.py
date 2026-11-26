"""Comparison protocol tests; fake training/evaluation, no optimizer or engine."""
import copy
import json

import pytest
import torch

import training.directional_experiment as experiment
from test_model import make_observation
from training.checkpoint import save_checkpoint
from training.ppo import evaluate_actions, sample_actions


def test_fresh_pair_has_identical_common_weights_and_can_learn_angle_columns():
    with torch.random.fork_rng():
        models = experiment.matched_fresh_models(17)
        v1, v2 = models["v1"], models["v2"]
        first, second = v1.state_dict(), v2.state_dict()
        for name, tensor in first.items():
            comparison = second[name][:, :7] if name == "ray_encoder.0.weight" else second[name]
            assert torch.equal(tensor, comparison), name
        assert not v2.ray_encoder[0].weight[:, -2:].any()
        observation = make_observation(3)
        with torch.no_grad():
            a, av, _ = v1(observation)
            b, bv, _ = v2(observation)
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(av, bv, atol=1e-7, rtol=1e-6)
        logits, values, _ = v2(observation)
        actions, _ = sample_actions(logits.detach())
        log_prob, entropy = evaluate_actions(logits, actions)
        (-log_prob.mean() - 0.01*entropy.mean() + values.square().mean()).backward()
        assert v2.ray_encoder[0].weight.grad[:, -2:].abs().sum() > 0
        assert torch.isfinite(v2.ray_encoder[0].weight.grad).all()
        # Copying initialization must not share parameter storage across arms.
        before = v2.actor.bias.detach().clone()
        with torch.no_grad():
            v1.actor.bias.add_(1)
        assert torch.equal(before, v2.actor.bias)


def test_initialization_reproducible_and_training_seeds_are_independent():
    first = experiment.matched_fresh_models(17)
    repeat = experiment.matched_fresh_models(17)
    other = experiment.matched_fresh_models(23)
    assert all(torch.equal(v, repeat["v1"].state_dict()[k]) for k, v in first["v1"].state_dict().items())
    assert not torch.equal(first["v1"].actor.weight, other["v1"].actor.weight)


def record(seed, kills):
    return {"map": "maps/training/tier1_projectile.wad:MAP01", "seed": seed,
            "exit_reached": False, "dead": kills == 0, "kills": kills,
            "damage_taken": 100 if kills == 0 else 0, "ticks": 100,
            "reward_components": {"combat_hits": kills, "combat_damage_dealt": kills * 60}}


def test_summary_pairs_training_seeds_and_does_not_invent_combat_completion_rate():
    results = {}
    for seed, base, candidate in ((17, 0, 1), (23, 1, 0), (41, 0, 1)):
        for variant, kills in (("v1", base), ("v2", candidate)):
            results[f"{variant}_{seed}"] = dict(variant=variant, training_seed=seed, stochastic=True,
                                              episodes_detail=[record(100, kills), record(101, kills)])
    summary = experiment.summarize(results, [17, 23, 41])
    assert len(summary["per_map"]) == 2
    assert all(r["episodes"] == 6 and r["exit_rate"] is None for r in summary["per_map"])
    assert not any(r["metric"] == "exit_rate" for r in summary["paired_training_seed_differences"])
    kills = next(r for r in summary["paired_training_seed_differences"] if r["metric"] == "mean_kills")
    assert kills["v2_minus_v1_by_training_seed"] == {17: 1, 23: -1, 41: 1}
    assert kills["mean_difference"] == pytest.approx(1/3)
    assert kills["sample_sd_of_training_seed_differences"] == pytest.approx(1.154700538)


@pytest.mark.parametrize("seeds", [[17, 17], [-1], [2**32]])
def test_bad_seed_protocol_fails_before_creating_outputs(tmp_path, monkeypatch, seeds):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        experiment.main(["--device", "cpu", "--seeds", *map(str, seeds)])
    assert not (tmp_path / "training/runs").exists()


def test_runner_trains_fresh_equal_budget_arms_and_keeps_initial_snapshots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ["submission/weights/checkpoint.pt", "submission/weights/config.json",
                 "maps/training/tier1_projectile.wad", "maps/training/tier2_door.wad",
                 "maps/training/tier3_red_key.wad", "maps/two_rooms.wad"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected")
    calls = []

    def fake_train(model, collector, config, resume_path, metrics_logger):
        assert resume_path is None
        assert collector.shaper.coefficient == 0
        calls.append((copy.deepcopy(config), model.architecture_id))
        save_checkpoint(tmp_path / config["save_dir"] / "checkpoint_2.pt", model, update=2,
                        global_step=256, config=config)

    monkeypatch.setattr(experiment, "train_loop", fake_train)
    monkeypatch.setattr(experiment, "MapCyclerEnv", lambda *a, **kw: type("NoEngine", (), {"close": lambda self: None})())
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **kw: "test-revision\n")
    monkeypatch.setattr(experiment, "evaluate", lambda *a, **kw: dict(stochastic=a[3], episodes_detail=[]))
    monkeypatch.setattr(experiment, "summarize", lambda results, seeds: {"per_map": [], "pairs": len(results)})
    monkeypatch.setattr(experiment, "print_summary", lambda summary: None)
    assert experiment.main(["--device", "cpu", "--seeds", "17", "--updates", "2", "--eval-episodes", "1"]) == 0
    assert len(calls) == 2
    first, second = (c[0].copy() for c in calls)
    assert first.pop("save_dir") != second.pop("save_dir")
    assert first == second
    manifest_path = next((tmp_path / "training/runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["total_training_decisions"] == 512
    assert manifest["decisions_per_model"] == 256
    assert manifest["protected_inputs_unchanged"] and manifest["status"] == "complete"
    assert not manifest["weights_promoted"]
    assert len(manifest["initial_checkpoints"]) == len(manifest["trained_checkpoints"]) == 2
    for entry in manifest["initial_checkpoints"].values():
        assert experiment.sha256(entry["path"]) == entry["sha256"]
        initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
        assert initial["update"] == initial["global_step"] == 0


def test_map_cycler_records_actual_wad_and_engine_seed_without_changing_selection(monkeypatch):
    from training import envs
    observed = []

    class FakeEnv:
        def reset(self, seed=None, options=None):
            observed.append(seed)
            return {}, {}

        def close(self):
            pass

    monkeypatch.setattr(envs, "make_env", lambda *a, **kw: FakeEnv())
    maps = [("a.wad", "MAP01"), ("b.wad", "MAP01")]
    first, repeat = envs.MapCyclerEnv(maps, seed=17), envs.MapCyclerEnv(maps, seed=17)
    for _ in range(5):
        first.reset()
        repeat.reset()
        assert first.episode_map_spec == repeat.episode_map_spec
        assert first.current_episode_seed == repeat.current_episode_seed
        assert first.current_episode_seed == observed[-2] == observed[-1]


def test_collector_saves_episode_map_and_engine_seed_in_completed_records():
    from test_rollout import MockDoomEnv, MockDoomPolicy
    from training.metrics import MetricsLogger
    from training.rollout import RolloutCollector
    env = MockDoomEnv(max_ticks=8)
    env.episode_map_spec = "maps/example.wad:MAP01"
    env.current_episode_seed = 123456
    logger = MetricsLogger()
    collector = RolloutCollector(env, metrics_logger=logger, device="cpu", seed=17)
    collector.collect(MockDoomPolicy(), num_steps=3)
    assert len(logger.episodes) == 1
    assert logger.episodes[0].map_name == "maps/example.wad:MAP01"
    assert logger.episodes[0].seed == 123456
