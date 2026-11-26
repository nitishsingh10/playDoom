"""Shape and masking checks for Member 3's observation encoder and policy."""

import numpy as np
import pytest
import torch

from doom_env import spaces as doom_spaces
from submission.model import DoomPolicy
from submission.observation import ObservationEncoder


def make_observation(entity_count: int = 0) -> dict:
    entities = doom_spaces.MAX_ENTITIES
    features = np.zeros((entities, len(doom_spaces.ENTITY_FEATURES)), dtype=np.float32)
    entity_type = np.zeros((entities,), dtype=np.int64)
    mask = np.zeros((entities,), dtype=np.int8)
    if entity_count:
        features[:entity_count, 0] = np.linspace(0.05, 3.5, entity_count, dtype=np.float32)
        features[:entity_count, 2] = 1.0
        features[:entity_count, 3:5] = 2.0
        entity_type[:entity_count] = np.arange(1, entity_count + 1) % (len(doom_spaces.ENTITY_TYPES) - 1) + 1
        mask[:entity_count] = 1
    return {
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.zeros((120, 160), dtype=np.uint8),
        "vitals": np.array([100, 0, 2, 50, 100, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "rays": {
            "distance": np.linspace(0.0, 1.0, doom_spaces.NUM_RAYS, dtype=np.float32),
            "hit_type": np.arange(doom_spaces.NUM_RAYS, dtype=np.int64) % len(doom_spaces.RAY_HIT_TYPES),
        },
        "visible_entities": {"features": features, "entity_type": entity_type, "mask": mask},
    }


@pytest.mark.parametrize("entity_count", [0, 1, doom_spaces.MAX_ENTITIES])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_encoder_and_policy_shapes_are_finite(entity_count, batch_size):
    encoder = ObservationEncoder(device="cpu")
    observation = make_observation(entity_count)
    if batch_size > 1:
        observation = {
            "vitals": np.repeat(observation["vitals"][None, :], batch_size, axis=0),
            "rays": {key: np.repeat(value[None, ...], batch_size, axis=0) for key, value in observation["rays"].items()},
            "visible_entities": {
                key: np.repeat(value[None, ...], batch_size, axis=0)
                for key, value in observation["visible_entities"].items()
            },
        }

    encoded = encoder(observation)
    model = DoomPolicy().eval()
    with torch.inference_mode():
        logits, value, next_hidden = model(encoded)

    assert logits.shape == (batch_size, len(doom_spaces.BUTTONS))
    assert value.shape == (batch_size,)
    assert next_hidden is None
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert encoded["entity_mask"].sum().item() == entity_count * batch_size


def test_masked_padding_is_zeroed_before_model():
    observation = make_observation(0)
    observation["visible_entities"]["features"][:] = 99.0
    observation["visible_entities"]["entity_type"][:] = len(doom_spaces.ENTITY_TYPES) - 1
    encoded = ObservationEncoder()(observation)
    assert torch.count_nonzero(encoded["entity_features"]) == 0
    assert torch.count_nonzero(encoded["entity_type"]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_policy_runs_on_available_devices(device):
    encoded = ObservationEncoder(device=device)(make_observation(2))
    model = DoomPolicy().to(device).eval()
    with torch.inference_mode():
        logits, value, _ = model(encoded)
    assert logits.device.type == device
    assert value.device.type == device
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_convenience_methods_and_factory():
    model = DoomPolicy().eval()
    obs = ObservationEncoder()(make_observation(1))
    with torch.inference_mode():
        logits_fwd, val_fwd, hidden_fwd = model(obs)
        logits_step, val_step, hidden_step = model.step(obs)
    assert torch.equal(logits_fwd, logits_step)
    assert torch.equal(val_fwd, val_step)
    assert hidden_fwd is None and hidden_step is None
    assert model.init_hidden() is None

    from submission.model import build_policy

    built = build_policy()
    assert isinstance(built, DoomPolicy)


def test_policy_accepts_raw_observation_and_matches_encoded_path():
    observation = make_observation(4)
    encoder = ObservationEncoder()
    encoded = encoder(observation)
    model = DoomPolicy().eval()
    with torch.inference_mode():
        raw_logits, raw_value, _ = model(observation)
        encoded_logits, encoded_value, _ = model(encoded)
    assert torch.allclose(raw_logits, encoded_logits)
    assert torch.allclose(raw_value, encoded_value)

