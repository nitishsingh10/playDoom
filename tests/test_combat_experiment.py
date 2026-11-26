"""Pure checks for the opt-in experiment; no gameplay or optimizer steps."""
import copy
import json

import numpy as np
import pytest
import torch

import training.combat_experiment as experiment
from submission.model import DoomPolicy
from test_model import make_observation
from training.checkpoint import save_checkpoint
from training.rewards import RewardShaper


def state(**overrides):
    return {"x": 0, "y": 0, "angle": 0, "kills": 0, "damage_taken": 0, "dead": False,
            "keys": [], "items": 0, "selected_weapon": 2, "damage_dealt": 0.0,
            "hits": 0.0, "ammo_bullets": 50.0, "attack_ready": 1.0, **overrides}


@pytest.mark.parametrize("attack", [0, 1])
def test_credit_requires_actual_damage_and_allows_delayed_damage(attack):
    shaper = experiment.CombatRewardShaper(0.02, 2.0)
    prev = state()
    shaper.reset(prev)
    action = np.zeros(9, dtype=np.int8)
    action[6] = attack
    _, components = shaper.step(prev, state(), action)
    assert components["damage_dealt_credit"] == 0
    _, components = shaper.step(prev, state(damage_dealt=20.0, hits=1.0), action)
    assert components["damage_dealt_credit"] == pytest.approx(0.4)
    assert shaper.get_episode_summary()["combat_hits"] == 1


def test_credit_is_capped_resets_and_cannot_reward_counter_decreases():
    shaper = experiment.CombatRewardShaper(0.02, 2.0)
    previous = state()
    shaper.reset(previous)
    for damage, expected in [(80, 1.6), (120, 0.4), (400, 0.0), (0, 0.0)]:
        current = state(damage_dealt=damage)
        _, components = shaper.step(previous, current, np.zeros(9))
        assert components["damage_dealt_credit"] == pytest.approx(expected)
        previous = current
    assert shaper.total_components["damage_dealt_credit"] == 2
    shaper.reset(state())
    _, components = shaper.step(state(), state(damage_dealt=20), np.zeros(9))
    assert components["damage_dealt_credit"] == pytest.approx(0.4)


def test_control_matches_existing_rewards_and_counts_terminal_combat():
    base, control = RewardShaper(), experiment.CombatRewardShaper()
    prev = state()
    base.reset(prev)
    control.reset(prev)
    for overrides in [dict(x=80, damage_dealt=20, hits=2, ammo_bullets=48),
                      dict(x=80, damage_dealt=80, hits=5, ammo_bullets=45,
                           kills=1, dead=True, damage_taken=110)]:
        curr = {**state(), **overrides}
        expected, _ = base.step(prev, curr, np.ones(9), env_reward=-1.0)
        actual, components = control.step(prev, curr, np.ones(9), env_reward=-1.0)
        assert actual == expected
        assert actual == pytest.approx(sum(components.values()))
        prev = curr
    summary = control.get_episode_summary()
    assert summary["combat_damage_dealt"] == 80
    assert summary["combat_hits"] == 5
    assert summary["combat_bullets_spent"] == 5
    assert summary["combat_first_kill_decision"] == 2
    assert summary["total_shaped_reward"] == base.get_episode_summary()["total_shaped_reward"]


@pytest.mark.parametrize("coefficient,cap", [(-1, 2), (float("nan"), 2), (0.02, 0), (0.02, float("inf"))])
def test_invalid_reward_configuration_fails(coefficient, cap):
    with pytest.raises(ValueError):
        experiment.CombatRewardShaper(coefficient, cap)


def test_missing_native_counters_fail_instead_of_silently_training_without_credit():
    previous = state()
    current = state()
    del current["damage_dealt"]
    with pytest.raises(ValueError, match="damage_dealt"):
        experiment.CombatRewardShaper(0.02).step(previous, current, np.zeros(9))


def test_native_telemetry_never_changes_observation_and_reads_terminal_counters():
    observation = make_observation(1)

    class FakeGame:
        damage = 0

        def get_game_variable(self, variable):
            return self.damage if variable == experiment.vd.GameVariable.DAMAGECOUNT else 1

    class FakeEnv:
        _game = FakeGame()

        def reset(self, **kwargs):
            return observation, {"seed": kwargs.get("seed")}

        def step(self, action):
            self._game.damage = 60
            return observation, 0, True, False, {}

        def privileged_state(self):
            return {"kills": int(self._game.damage > 0)}

    env = experiment.CombatTelemetryEnv(FakeEnv())
    assert env.reset(seed=17)[0] is observation
    assert "damage_dealt" not in observation
    assert env.privileged_state()["damage_dealt"] == 0
    assert env.step(np.zeros(9))[0] is observation
    assert env.privileged_state()["damage_dealt"] == 60
    assert env.privileged_state()["kills"] == 1


def test_protected_checkpoint_mutation_is_detected(tmp_path):
    path = tmp_path / "protected"
    path.write_text("original")
    hashes = {str(path): experiment.sha256(path)}
    experiment.verify_hashes(hashes)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Protected input changed"):
        experiment.verify_hashes(hashes)


def test_missing_parent_fails_without_creating_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        experiment.main(["--parent", "missing.pt", "--device", "cpu"])
    assert not (tmp_path / "training/runs").exists()


def test_each_arm_reseeds_after_parent_rng_restoration_only_once(monkeypatch):
    monkeypatch.setattr(experiment.RolloutCollector, "collect", lambda *a, **kw: torch.rand(8))
    with torch.random.fork_rng():
        first = experiment.ExperimentCollector(object(), seed=17)
        second = experiment.ExperimentCollector(object(), seed=17)
        torch.manual_seed(999)  # Simulate a restored parent RNG.
        a = first.collect(None)
        continuation = first.collect(None)
        assert not torch.equal(a, continuation)
        torch.manual_seed(111)  # Different inherited state must not affect this arm's seed.
        assert torch.equal(a, second.collect(None))
        assert torch.equal(continuation, second.collect(None))


def test_evaluation_reads_terminal_native_counters_without_leaking_them(monkeypatch, tmp_path):
    observation = make_observation(1)

    class Policy:
        def __call__(self, encoded):
            assert set(encoded) == {"rays", "entity_features", "entity_type", "entity_mask", "vitals"}
            assert "damage_dealt" not in encoded
            return torch.zeros(1, 12), torch.zeros(1), None

    class Env:
        finished = False

        def reset(self, seed=None):
            return observation, {}

        def step(self, action):
            assert action.shape == (9,)
            self.finished = True
            return observation, 0.1, True, False, {}

        def privileged_state(self):
            return {**state(), "tic": 4 if self.finished else 0, "health": 100,
                    "damage_dealt": 60 if self.finished else 0,
                    "hits": 5 if self.finished else 0,
                    "kills": int(self.finished), "dead": False, "exit_reached": self.finished}

        def close(self):
            pass

    checkpoint = tmp_path / "fake_checkpoint"
    checkpoint.write_text("never loaded by the fake evaluator")
    monkeypatch.setattr(experiment, "load_policy", lambda *a, **kw: Policy())
    monkeypatch.setattr(experiment, "CombatTelemetryEnv", lambda env: env)
    monkeypatch.setattr(experiment, "make_env", lambda *a, **kw: Env())
    result = experiment.evaluate(checkpoint, [("fake.wad", "MAP01")], [100], stochastic=False)
    row = result["episodes_detail"][0]
    assert row["kills"] == 1
    assert row["time_to_first_kill_ticks"] == 4
    assert row["reward_components"]["combat_damage_dealt"] == 60
    assert row["reward_components"]["combat_hits"] == 5
    assert result["action_intervention"] is None


def test_paired_runner_uses_same_parent_settings_and_distinct_outputs(tmp_path, monkeypatch):
    """Exercise orchestration with fake training/evaluation; no engine or updates."""
    monkeypatch.chdir(tmp_path)
    for path in ["submission/weights/checkpoint.pt", "submission/weights/config.json",
                 "maps/training/tier1_projectile.wad", "maps/training/tier2_door.wad",
                 "maps/training/tier3_red_key.wad", "maps/two_rooms.wad"]:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("protected")
    model = DoomPolicy()
    optimizer = torch.optim.Adam(model.parameters())
    parent = save_checkpoint(tmp_path / "parent.pt", model, optimizer, update=500, global_step=64000,
                             config={"rollout_steps": 128, "minibatch_size": 64, "update_epochs": 4})
    # Adam state dictionaries with empty state still have their param_groups and are valid.
    calls = []

    def fake_train(model, collector, config, resume_path, metrics_logger):
        calls.append((config.copy(), resume_path, collector.shaper.coefficient))
        path = tmp_path / config["save_dir"] / "checkpoint_1000.pt"
        save_checkpoint(path, model, update=1000, global_step=128000)

    monkeypatch.setattr(experiment, "train_loop", fake_train)
    monkeypatch.setattr(experiment, "MapCyclerEnv", lambda *a, **kw: type("NoEngine", (), {"close": lambda self: None})())
    monkeypatch.setattr(experiment, "evaluate", lambda *a, **kw: {"episodes_detail": []})
    monkeypatch.setattr(experiment, "print_comparison", lambda results: None)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **kw: "test-revision\n")
    original = experiment.sha256(parent)
    assert experiment.main(["--parent", str(parent), "--device", "cpu"]) == 0
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == str(parent)
    assert [c[2] for c in calls] == [0, 0.02]
    first, second = (copy.deepcopy(c[0]) for c in calls)
    assert first.pop("save_dir") != second.pop("save_dir")
    first.pop("combat_damage_coefficient")
    second.pop("combat_damage_coefficient")
    assert first == second
    assert first["total_updates"] == 1000
    assert experiment.sha256(parent) == original
    manifest = next((tmp_path / "training/runs").glob("*/manifest.json"))
    data = json.loads(manifest.read_text())
    assert data["additional_decisions_per_arm"] == 64000
    assert data["evaluation_seeds"] == list(range(100, 110))
    assert data["protected_inputs_unchanged"]
    assert data["status"] == "complete"
