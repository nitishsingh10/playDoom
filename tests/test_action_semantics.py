"""Verify native edge actions without altering the official wrapper."""

import numpy as np
import pytest

from doom_env import make_env

WAD = "maps/training/tier2_door.wad"
NOOP = np.zeros(9, np.int8)


def test_weapon_cycle_is_edge_triggered_and_changes_can_be_delayed():
    traces = {}
    for mode in ("held", "pulsed"):
        env = make_env(WAD, episode_ticks=240, render_buffers=False)
        try:
            observation, _ = env.reset(seed=0)
            previous = observation["vitals"][2]
            changes = []
            for step in range(50):
                action = NOOP.copy()
                action[8] = 1 if mode == "held" else int(step % 12 == 0)
                observation, *_ = env.step(action)
                current = observation["vitals"][2]
                if current != previous:
                    changes.append((step, int(current)))
                previous = current
            traces[mode] = changes
        finally:
            env.close()
    assert len(traces["held"]) == 1
    assert len(traces["pulsed"]) > 1
    # Later changes follow the press after the current weapon animation finishes.
    assert any(step % 12 != 0 for step, _ in traces["pulsed"][1:])


def test_repressing_use_reverses_an_opening_door_but_holding_does_not():
    traces = {}
    for mode in ("held", "repress"):
        env = make_env(WAD, episode_ticks=200, render_buffers=False)
        try:
            env.reset(seed=0)
            forward = NOOP.copy()
            forward[0] = 1
            for _ in range(8):
                env.step(forward)
            door_sector = next(iter(env.mapinfo.door_sectors))
            ceilings = []
            for step in range(4):
                action = NOOP.copy()
                action[7] = 1 if mode == "held" else int(step != 1)
                env.step(action)
                ceilings.append(env.privileged_state()["sector_ceiling"][door_sector])
            traces[mode] = ceilings
        finally:
            env.close()
    assert np.all(np.diff(traces["held"]) > 0)
    assert traces["repress"][2] < traces["repress"][1]
    assert traces["repress"][3] == pytest.approx(0)


def test_terminal_weapon_metrics_are_refreshed_when_observation_is_stale():
    env = make_env(WAD, episode_ticks=4, render_buffers=False)
    try:
        original, _ = env.reset(seed=0)
        action = NOOP.copy()
        action[8] = 1
        observation, _, _, truncated, _ = env.step(action)
        state = env.privileged_state()
        assert truncated
        assert observation["vitals"][2] == original["vitals"][2] == 2
        assert state["selected_weapon"] == 1
        assert state["selected_weapon_ammo"] == 0
    finally:
        env.close()
