"""Tests for doom_spatial_gru_v3, mapgen, enhanced rewards, and vectorized recurrent PPO."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.optim as optim

from doom_env import make_env, spaces as doom_spaces
from submission.agent import DoomAgent
from submission.checkpoint import SPATIAL_GRU_ARCHITECTURE_ID, load_policy_state
from submission.model import DoomPolicy, build_policy
from submission.observation import ObservationEncoder
from training.checkpoint import load_checkpoint, save_checkpoint
from training.check_submission import check_export
from training.envs import SubprocVecEnv, default_curriculum_maps, default_heldout_maps
from training.export_checkpoint import export_checkpoint
from training.mapgen import build_curriculum_specs, write_udmf_wad
from training.ppo import compute_gae, ppo_update
from training.rewards import RewardConfig, RewardShaper
from training.rollout import VecRolloutCollector


def _obs(entity_count: int = 2) -> dict:
    features = np.zeros((doom_spaces.MAX_ENTITIES, len(doom_spaces.ENTITY_FEATURES)), dtype=np.float32)
    entity_type = np.zeros((doom_spaces.MAX_ENTITIES,), dtype=np.int64)
    mask = np.zeros((doom_spaces.MAX_ENTITIES,), dtype=np.int8)
    if entity_count:
        features[:entity_count, 0] = np.linspace(0.1, 2.0, entity_count, dtype=np.float32)
        features[:entity_count, 2] = 1.0
        entity_type[:entity_count] = 3
        mask[:entity_count] = 1
    return {
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.zeros((120, 160), dtype=np.uint8),
        "vitals": np.array([100, 0, 2, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "rays": {
            "distance": np.linspace(0.1, 1.0, doom_spaces.NUM_RAYS, dtype=np.float32),
            "hit_type": np.arange(doom_spaces.NUM_RAYS, dtype=np.int64) % len(doom_spaces.RAY_HIT_TYPES),
        },
        "visible_entities": {"features": features, "entity_type": entity_type, "mask": mask},
    }


@pytest.mark.parametrize("entity_count", [0, 1, doom_spaces.MAX_ENTITIES])
def test_v3_shapes_and_sequence_parity(entity_count):
    torch.manual_seed(0)
    policy = DoomPolicy(architecture_id=SPATIAL_GRU_ARCHITECTURE_ID).eval()
    encoder = ObservationEncoder()
    B, T = 2, 5
    dones = torch.zeros(B, T)
    dones[0, 2] = 1.0  # Mid-sequence episode boundary for batch 0

    step_obs_list = []
    for _ in range(T):
        batch = {
            "vitals": np.stack([_obs(entity_count)["vitals"] for _ in range(B)]),
            "rays": {
                "distance": np.stack([_obs(entity_count)["rays"]["distance"] for _ in range(B)]),
                "hit_type": np.stack([_obs(entity_count)["rays"]["hit_type"] for _ in range(B)]),
            },
            "visible_entities": {
                "features": np.stack([_obs(entity_count)["visible_entities"]["features"] for _ in range(B)]),
                "entity_type": np.stack([_obs(entity_count)["visible_entities"]["entity_type"] for _ in range(B)]),
                "mask": np.stack([_obs(entity_count)["visible_entities"]["mask"] for _ in range(B)]),
            },
        }
        enc = encoder(batch)
        enc["prev_action"] = torch.zeros(B, 9)
        step_obs_list.append(enc)

    h = policy.init_hidden(batch_size=B, device="cpu")
    step_logits, step_vals = [], []
    with torch.inference_mode():
        for t in range(T):
            l_t, v_t, h = policy.step(step_obs_list[t], h)
            step_logits.append(l_t)
            step_vals.append(v_t)
            h = h * (1.0 - dones[:, t]).view(1, B, 1)

        seq_obs = {
            k: torch.stack([step_obs_list[t][k] for t in range(T)], dim=1)
            for k in step_obs_list[0]
        }
        seq_logits, seq_vals, seq_h = policy.forward_sequence(seq_obs, dones, policy.init_hidden(B, "cpu"))

    assert torch.allclose(torch.stack(step_logits, dim=1), seq_logits, atol=1e-5)
    assert torch.allclose(torch.stack(step_vals, dim=1), seq_vals, atol=1e-5)
    assert torch.allclose(h, seq_h, atol=1e-5)


def test_v3_export_preflight_and_agent_reset(tmp_path):
    model = DoomPolicy(architecture_id=SPATIAL_GRU_ARCHITECTURE_ID)
    ckpt = save_checkpoint(tmp_path / "train.pt", model, config={"deterministic_eval": False})
    exported = export_checkpoint(ckpt, tmp_path / "export" / "checkpoint.pt", deterministic_eval=False)
    res = check_export(exported.parent, mode="stochastic", source_checkpoint=ckpt)
    assert res["architecture_id"] == SPATIAL_GRU_ARCHITECTURE_ID

    agent = DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cpu", weights_dir=exported.parent)
    assert agent.model is not None and agent.model.is_recurrent
    obs = _obs(2)
    agent.reset()
    torch.manual_seed(7)
    a1 = agent.act(obs)
    agent.act(obs)
    agent.reset()
    torch.manual_seed(7)
    a2 = agent.act(obs)
    assert np.array_equal(a1, a2)


def test_reward_shaper_damage_door_and_key_reset():
    shaper = RewardShaper(RewardConfig(damage_dealt_coef=0.03, door_open_reward=1.5, locked_door_open_reward=3.0))
    s0 = {
        "x": 96.0, "y": 96.0, "angle": 0.0, "kills": 0, "items": 0,
        "damage_taken": 0.0, "damage_dealt": 0.0, "keys": [0, 0, 0, 0, 0, 0],
        "door_lock": np.array([-1, 0, 129]), "sector_ceiling": np.array([128.0, 0.0, 0.0]),
    }
    shaper.reset(s0)
    s1 = dict(s0, damage_dealt=20.0, sector_ceiling=np.array([128.0, 40.0, 0.0]))
    _, comp1 = shaper.step(s0, s1, np.zeros(9, dtype=np.int8))
    assert comp1["damage_dealt"] == pytest.approx(0.6)
    assert comp1["door_open"] == pytest.approx(1.5)

    # Same open door does not pay again
    s2 = dict(s1, keys=[0, 0, 1, 0, 0, 0], sector_ceiling=np.array([128.0, 120.0, 48.0]))
    _, comp2 = shaper.step(s1, s2, np.zeros(9, dtype=np.int8))
    assert comp2["door_open"] == pytest.approx(3.0)
    assert comp2["key"] == pytest.approx(5.0)
    assert len(shaper.visited_cells) == 1  # Reset on key pickup then current cell added


def test_mapgen_disjoint_and_vec_recurrent_ppo_smoke(tmp_path):
    train_specs, heldout_specs = build_curriculum_specs()
    assert not ({s.name for s in train_specs} & {s.name for s in heldout_specs})
    wad_path = write_udmf_wad(tmp_path / "sample.wad", heldout_specs[2].grid, player_angle=heldout_specs[2].player_angle)
    env = make_env(wad_path, "MAP01", episode_ticks=120, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()

    vec = SubprocVecEnv([(str(wad_path), "MAP01")], num_envs=2, episode_ticks=80, seed=1)
    try:
        collector = VecRolloutCollector(vec, device="cpu")
        policy = DoomPolicy(hidden_size=32, fusion_size=64, architecture_id=SPATIAL_GRU_ARCHITECTURE_ID)
        opt = optim.Adam(policy.parameters(), lr=3e-4)
        batch = collector.collect(policy, num_steps=16)
        assert batch["rewards"].shape == (16, 2)
        assert batch["init_hiddens"].shape == (16, 2, 64)
        adv, ret = compute_gae(
            batch["rewards"], batch["values"], batch["dones"], batch["last_value"],
            truncations=batch["truncations"], bootstrap_values=batch["bootstrap_values"],
        )
        metrics = ppo_update(policy, opt, {
            "obs": batch["obs"],
            "actions": batch["actions"],
            "old_log_probs": batch["log_probs"],
            "returns": ret,
            "advantages": adv,
            "old_values": batch["values"],
            "dones": batch["dones"],
            "init_hiddens": batch["init_hiddens"],
        }, {"seq_len": 8, "minibatch_size": 16, "update_epochs": 2})
        assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["value_loss"])
    finally:
        vec.close()


def test_random_mapgen_and_cycler_cleanup(tmp_path):
    from training.envs import MapCyclerEnv
    from training.mapgen import generate_random_map

    rng = np.random.default_rng(123)
    for tier in (1, 2, 3, 4):
        spec = generate_random_map(rng, tier=tier)
        assert spec.tier == tier
        wad = write_udmf_wad(tmp_path / f"rand_t{tier}.wad", spec.grid, player_angle=spec.player_angle)
        env = make_env(wad, "MAP01", episode_ticks=80, render_buffers=False)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        env.close()

    cycler = MapCyclerEnv(
        [("maps/training/tier1_projectile.wad", "MAP01")],
        episode_ticks=80,
        seed=5,
        random_map_prob=1.0,
    )
    obs, _ = cycler.reset()
    assert cycler.observation_space.contains(obs)
    tmp_wad = cycler._tmp_wad
    assert tmp_wad is not None and tmp_wad.exists()
    cycler.close()
    assert not tmp_wad.exists()


def test_reward_shaper_door_reopen_on_key_and_stagnation():
    shaper = RewardShaper(RewardConfig(stagnation_grace_steps=3, stagnation_penalty=0.1, locked_door_open_reward=6.0))
    s0 = {
        "x": 96.0, "y": 96.0, "angle": 0.0, "kills": 0, "items": 0,
        "damage_taken": 0.0, "damage_dealt": 0.0, "keys": [0, 0, 0, 0, 0, 0],
        "door_lock": np.array([-1, 0, 129]), "sector_ceiling": np.array([128.0, 0.0, 0.0]),
    }
    shaper.reset(s0)
    # Open normal door once
    s1 = dict(s0, sector_ceiling=np.array([128.0, 40.0, 0.0]))
    _, c1 = shaper.step(s0, s1, np.zeros(9, dtype=np.int8))
    assert c1["door_open"] == pytest.approx(1.5)
    # Door closes again before key pickup -> cannot be farmed
    s2 = dict(s1, sector_ceiling=np.array([128.0, 0.0, 0.0]))
    _, c2 = shaper.step(s1, s2, np.zeros(9, dtype=np.int8))
    assert c2["door_open"] == 0.0
    # Pick up red key while normal door is closed -> normal door is eligible once more
    s3 = dict(s2, keys=[0, 0, 1, 0, 0, 0])
    _, c3 = shaper.step(s2, s3, np.zeros(9, dtype=np.int8))
    assert c3["key"] == pytest.approx(5.0)
    s4 = dict(s3, sector_ceiling=np.array([128.0, 40.0, 0.0]))
    _, c4 = shaper.step(s3, s4, np.zeros(9, dtype=np.int8))
    assert c4["door_open"] == pytest.approx(1.5)
    # Stagnation triggers after 3 idle steps without progress
    for idx in range(4):
        _, cs = shaper.step(s4, s4, np.zeros(9, dtype=np.int8))
    assert cs["stagnation"] == pytest.approx(-0.1)
