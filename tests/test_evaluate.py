"""Diagnostic counts and policy/privileged-state separation."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from doom_env import spaces as S
from submission.actions import StructuredActionDistribution
from submission.checkpoint import LEGACY_ARCHITECTURE_ID, STRUCTURED_ARCHITECTURE_ID
from submission.model import DoomPolicy
from test_model import make_observation
from training.checkpoint import save_checkpoint
from training.evaluate import ActionDiagnostics, _choose_action, evaluate_checkpoint, load_policy


def diagnostic_observation(distance=0.25, masked=True, ammo=50):
    obs = make_observation()
    obs["visible_entities"]["entity_type"][0] = 1
    obs["visible_entities"]["mask"][0] = int(masked)
    obs["visible_entities"]["features"][0, 0] = distance
    obs["vitals"][3] = ammo
    obs["rays"]["hit_type"][:] = 0
    return obs


def test_exact_conflicts_combat_conditions_and_button_down_edges():
    diagnostics = ActionDiagnostics()
    near = diagnostic_observation()
    near["rays"]["hit_type"][0] = S.RAY_HIT_TYPES.index("enemy")
    original = deepcopy(near)
    diagnostics.observe_action(near, np.ones(9, np.int8))
    diagnostics.observe_action(diagnostic_observation(2.0), np.array([1, 0, 0, 1, 0, 1, 1, 1, 1]))
    diagnostics.observe_action(diagnostic_observation(masked=False), np.zeros(9))
    diagnostics.observe_action(diagnostic_observation(1.0, ammo=0), np.array([0, 1, 0, 0, 1, 1, 1, 1, 1]))
    for before, after in ((2, 2), (2, 1), (1, 1), (1, 2)):
        diagnostics.observe_transition(before, after)
    result = diagnostics.summary()
    assert result["action_conflicts"] == {
        "forward_backward": {"count": 1, "percentage": 25.0},
        "left_right": {"count": 1, "percentage": 25.0},
        "turn_left_right": {"count": 2, "percentage": 50.0},
    }
    assert result["enemy_visible_steps"] == result["attack_while_enemy_visible"] == 3
    assert result["enemy_useful_range_steps"] == result["attack_while_enemy_in_useful_range"] == 2
    assert result["attack_while_enemy_visible_percentage"] == 100.0
    assert result["forward_ray_enemy_steps"] == result["attack_while_forward_ray_enemy"] == 1
    assert result["attack_with_zero_selected_weapon_ammo"] == 1
    assert result["use_activations"] == result["weapon_select_activations"] == 3
    assert result["use_button_down_edges"] == result["weapon_select_button_down_edges"] == 2
    assert result["weapon_selection_changes"] == 2
    np.testing.assert_array_equal(near["visible_entities"]["features"], original["visible_entities"]["features"])


def test_useful_range_uses_raw_entity_distance_times_1024_and_mask():
    diagnostics = ActionDiagnostics(useful_range_units=512)
    attack = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    for distance, masked in ((0.5, True), (0.51, True), (0.01, False)):
        diagnostics.observe_action(diagnostic_observation(distance, masked), attack)
    assert diagnostics.enemy_visible_steps == 2
    assert diagnostics.enemy_useful_range_steps == 1
    assert diagnostics.summary()["attack_while_enemy_visible_percentage"] == 100
    empty = ActionDiagnostics().summary()
    assert empty["attack_while_enemy_visible_percentage"] == 0
    assert all(value["percentage"] == 0 for value in empty["action_conflicts"].values())


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_useful_range_is_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        ActionDiagnostics(useful_range_units=bad)


def test_evaluator_uses_shared_structured_mode_and_seeded_sampler():
    logits = torch.linspace(-2, 2, 12).unsqueeze(0)
    chosen = _choose_action(logits, legacy_baseline=False, stochastic=False, action_threshold=0.5)
    np.testing.assert_array_equal(chosen, StructuredActionDistribution(logits).mode()[0].numpy())
    torch.manual_seed(17)
    chosen = _choose_action(logits, legacy_baseline=False, stochastic=True, action_threshold=0.5)
    torch.manual_seed(17)
    np.testing.assert_array_equal(chosen, StructuredActionDistribution(logits).sample()[0].numpy())


def test_evaluator_requires_explicit_legacy_loading_and_keeps_dimensions(tmp_path):
    legacy = DoomPolicy(hidden_size=16, fusion_size=48, architecture_id=LEGACY_ARCHITECTURE_ID)
    path = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), path)
    with pytest.raises(ValueError, match="legacy-baseline"):
        load_policy(path, "cpu")
    restored = load_policy(path, "cpu", legacy_baseline=True)
    assert restored.hidden_size == 16 and restored.fusion_size == 48
    assert restored.actor.out_features == 9
    structured = DoomPolicy(hidden_size=16, fusion_size=48)
    path = save_checkpoint(tmp_path / "structured.pt", structured)
    restored = load_policy(path, "cpu")
    assert restored.architecture_id == STRUCTURED_ARCHITECTURE_ID
    assert restored.actor.out_features == 12


def test_terminal_metrics_use_final_engine_state_without_policy_leakage(monkeypatch):
    class ObservationOnlyPolicy:
        def __call__(self, encoded):
            assert set(encoded) == {"rays", "entity_features", "entity_type", "entity_mask", "vitals"}
            # The policy must see its public health=100, never privileged terminal health=-8.
            assert encoded["vitals"][0, 0].item() == pytest.approx(0.5)
            return torch.zeros(1, 12), torch.zeros(1), None

    class TerminalEnv:
        def __init__(self):
            self.done = False
            self.closed = False
            self.observation = diagnostic_observation()
        def reset(self, seed):
            return self.observation, {}
        def privileged_state(self):
            return {
                "tic": 4 if self.done else 0, "health": -8 if self.done else 100,
                "x": 0, "y": 0, "kills": 0, "damage_taken": 108 if self.done else 0,
                "selected_weapon": 1 if self.done else 2, "selected_weapon_ammo": 0 if self.done else 50,
                "dead": self.done, "timeout": False, "exit_reached": False, "keys": [0] * 6,
                "private_sentinel": "must not enter encoded observation",
            }
        def step(self, action):
            self.done = True
            return self.observation, -1.0, True, False, {}
        def close(self):
            self.closed = True
    env = TerminalEnv()
    monkeypatch.setattr("training.evaluate.load_policy", lambda *args, **kwargs: ObservationOnlyPolicy())
    monkeypatch.setattr("training.evaluate.make_env", lambda *args, **kwargs: env)
    result = evaluate_checkpoint("unused.pt", ["unused.wad"], [0])
    episode = result["episodes_detail"][0]
    assert episode["final_health"] == -8
    assert episode["damage_taken"] == 108
    assert episode["weapon_selection_changes"] == 1
    assert episode["termination_reason"] == "death"
    assert episode["reward_components"]["death"] == -5
    assert episode["reward_components"]["damage"] == pytest.approx(-2.16)
    assert env.closed
