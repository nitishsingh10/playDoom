"""Structured policy probabilities and the official nine-button action encoding."""

from itertools import product
import math

import pytest
import torch

from submission.actions import ACTION_DIM, NUM_ACTION_LOGITS, StructuredActionDistribution


def _buttons(choices, switches):
    return [int(choice == direction) for choice in choices for direction in (1, 2)] + list(switches)


@pytest.mark.parametrize("batch_shape", [(), (1,), (64,), (2, 3, 4)])
def test_sample_shapes_binary_values_conflicts_and_replay(batch_shape):
    logits = torch.randn(*batch_shape, NUM_ACTION_LOGITS)
    distribution = StructuredActionDistribution(logits)
    actions = distribution.sample()

    assert actions.shape == (*batch_shape, ACTION_DIM)
    assert actions.dtype == torch.int8
    assert ((actions == 0) | (actions == 1)).all()
    assert (actions[..., :6].reshape(*batch_shape, 3, 2).sum(dim=-1) <= 1).all()
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    assert log_prob.shape == batch_shape
    assert entropy.shape == batch_shape
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.equal(log_prob, StructuredActionDistribution(logits).log_prob(actions))


def test_sampling_preserves_diagonal_movement_and_all_six_direction_buttons():
    with torch.random.fork_rng():
        torch.manual_seed(310)
        actions = StructuredActionDistribution(torch.zeros(4096, NUM_ACTION_LOGITS)).sample()
    pairs = actions[:, :6].reshape(-1, 3, 2)
    assert (pairs.sum(dim=-1) <= 1).all()
    assert (actions[:, :6].sum(dim=0) > 0).all()
    assert ((pairs[:, 0].sum(-1) == 1) & (pairs[:, 1].sum(-1) == 1)).any()
    # Both a direction-free no-op and simultaneous attack/interaction/selection
    # remain valid choices of the factorized policy.
    assert (actions == 0).all(dim=-1).any()
    assert (actions[:, 6:] == 1).all(dim=-1).any()


@pytest.mark.parametrize("choices", list(product(range(3), repeat=3)))
def test_mode_maps_every_direction_choice_to_the_correct_button(choices):
    logits = torch.full((NUM_ACTION_LOGITS,), -10.0)
    for group, choice in enumerate(choices):
        logits[group * 3 + choice] = 10.0
    logits[9:] = torch.tensor([2.0, -2.0, 2.0])

    actions = StructuredActionDistribution(logits).mode()

    assert actions.tolist() == _buttons(choices, (1, 0, 1))


def test_mode_ties_are_repeatable():
    distribution = StructuredActionDistribution(torch.zeros(2, NUM_ACTION_LOGITS))
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.int8).expand(2, -1)
    assert torch.equal(distribution.mode(), expected)
    assert torch.equal(distribution.mode(), distribution.mode())


def test_joint_probability_matches_manual_factorization_and_sums_to_one():
    logits = torch.tensor([0.5, -1.0, 2.0, 0.2, 1.7, -0.5, -1.0, 0.4, 0.1, 1.3, -0.8, 0.2], dtype=torch.float64)
    combinations = list(product(product(range(3), repeat=3), product(range(2), repeat=3)))
    actions = torch.tensor([_buttons(directions, switches) for directions, switches in combinations])
    distribution = StructuredActionDistribution(logits.expand(len(actions), -1))
    probabilities = distribution.log_prob(actions).exp()

    expected = []
    for directions, switches in combinations:
        probability = 1.0
        for group, choice in enumerate(directions):
            probability *= torch.softmax(logits[group * 3:group * 3 + 3], dim=-1)[choice].item()
        for index, switch in enumerate(switches):
            p_on = torch.sigmoid(logits[9 + index]).item()
            probability *= p_on if switch else 1.0 - p_on
        expected.append(probability)

    assert torch.allclose(probabilities, torch.tensor(expected, dtype=logits.dtype), atol=1e-12)
    assert probabilities.sum().item() == pytest.approx(1.0)
    expected_entropy = -(probabilities * probabilities.log()).sum()
    assert torch.allclose(distribution.entropy(), expected_entropy.expand(len(actions)), atol=1e-12)


def test_uniform_entropy_is_log_of_216_valid_actions():
    entropy = StructuredActionDistribution(torch.zeros(7, NUM_ACTION_LOGITS)).entropy()
    assert torch.allclose(entropy, torch.full((7,), math.log(3 ** 3 * 2 ** 3)))


def test_extreme_logits_have_finite_sample_log_prob_and_entropy():
    logits = torch.tensor([1e4, -1e4, -1e4, -1e4, 1e4, -1e4, -1e4, -1e4, 1e4, 1e4, -1e4, 1e4])
    distribution = StructuredActionDistribution(logits)
    assert torch.isfinite(distribution.log_prob(distribution.sample()))
    assert torch.isfinite(distribution.entropy())


@pytest.mark.parametrize("objective", ["log_prob", "entropy"])
def test_gradients_reach_all_twelve_policy_logits(objective):
    logits = torch.tensor([[0.2, -0.7, 0.9, 0.3, 0.8, -0.9, -0.2, 0.5, 1.1, -0.4, 0.7, 0.3]], requires_grad=True)
    distribution = StructuredActionDistribution(logits)
    if objective == "log_prob":
        loss = distribution.log_prob(torch.tensor([_buttons((1, 2, 0), (1, 0, 1))])).sum()
    else:
        loss = distribution.entropy().sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs() > 0).all()


@pytest.mark.parametrize("pair", [0, 2, 4])
def test_replay_rejects_each_opposing_button_pair(pair):
    actions = torch.zeros(2, ACTION_DIM)
    actions[1, pair:pair + 2] = 1
    with pytest.raises(ValueError, match="opposing direction"):
        StructuredActionDistribution(torch.zeros(2, NUM_ACTION_LOGITS)).log_prob(actions)


@pytest.mark.parametrize("value", [-1.0, 0.25, 2.0, float("nan"), float("inf")])
def test_replay_rejects_nonbinary_actions(value):
    actions = torch.zeros(2, ACTION_DIM)
    actions[0, 6] = value
    with pytest.raises(ValueError, match="binary values"):
        StructuredActionDistribution(torch.zeros(2, NUM_ACTION_LOGITS)).log_prob(actions)


@pytest.mark.parametrize("shape", [(9,), (2, 9), (2, 1, 9), ()])
def test_replay_rejects_incorrect_batch_or_action_dimensions(shape):
    with pytest.raises(ValueError, match="actions must have shape"):
        StructuredActionDistribution(torch.zeros(3, NUM_ACTION_LOGITS)).log_prob(torch.zeros(shape))


@pytest.mark.parametrize("shape", [(), (9,), (2, 9), (2, 13)])
def test_rejects_legacy_or_malformed_logits(shape):
    with pytest.raises(ValueError, match="Legacy nine-logit policies are incompatible"):
        StructuredActionDistribution(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_logits(value):
    logits = torch.zeros(NUM_ACTION_LOGITS)
    logits[4] = value
    with pytest.raises(ValueError, match="finite values"):
        StructuredActionDistribution(logits)


def test_rejects_integer_logits():
    with pytest.raises(ValueError, match="floating-point"):
        StructuredActionDistribution(torch.zeros(NUM_ACTION_LOGITS, dtype=torch.int64))
