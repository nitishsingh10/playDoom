"""Record gameplay MP4 videos using the promoted submission DoomAgent."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from doom_env import make_env, spaces as doom_spaces
from submission.agent import DoomAgent

SHOWCASE_SPECS = [
    ("maps/two_rooms.wad", "MAP01", "two_rooms (Unseen Test Map)"),
    ("maps/heldout/heldout_t1_ambush.wad", "MAP01", "Tier 1: Combat & Ambush"),
    ("maps/heldout/heldout_t3_blue_maze.wad", "MAP01", "Tier 3: Keycard + Backtrack"),
    ("maps/heldout/heldout_t4_gauntlet_a.wad", "MAP01", "Tier 4: Full Gauntlet A"),
    ("maps/heldout/heldout_t4_gauntlet_b.wad", "MAP01", "Tier 4: Full Gauntlet B"),
]


def _find_best_run(agent: DoomAgent, wad: str, map_name: str, candidate_seeds: list[int], episode_ticks: int):
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
    best = None
    try:
        for seed in candidate_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            agent.reset()
            frames = []
            done = False
            while not done:
                action = agent.act(obs)
                p = env.privileged_state()
                frames.append((obs["rgb"].copy(), action.copy(), dict(p)))
                obs, _, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
            final_p = env.privileged_state()
            frames.append((obs["rgb"].copy(), np.zeros(9, dtype=np.int8), dict(final_p)))
            score = (
                int(final_p["exit_reached"]) * 10000
                + int(not final_p["dead"]) * 1000
                + int(final_p["kills"]) * 200
                + int(sum(final_p["keys"])) * 150
                - int(final_p["damage_taken"])
                - len(frames) * 0.1
            )
            if best is None or score > best[0]:
                best = (score, seed, frames, final_p)
    finally:
        env.close()
    return best


def _render_frame(rgb: np.ndarray, label: str, step_idx: int, action: np.ndarray, priv: dict) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    up = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
    hud = np.zeros((64, 640, 3), dtype=np.uint8)
    hud[:] = (24, 24, 28)

    hp = int(round(priv.get("health", 0)))
    kills = int(priv.get("kills", 0))
    keys = int(sum(priv.get("keys", [])))
    ammo = int(round(priv.get("selected_weapon_ammo", 0)))
    btns = [doom_spaces.BUTTONS[i].replace("MOVE_", "M_").replace("TURN_", "T_").replace("SELECT_NEXT_WEAPON", "WPN")
            for i in range(9) if action[i]]
    act_str = "+".join(btns) if btns else "IDLE"

    line1 = f"{label} | Step {step_idx:03d} | HP:{hp} Ammo:{ammo} Kills:{kills} Keys:{keys}"
    line2 = f"Act: {act_str[:58]}"
    cv2.putText(hud, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(hud, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 220, 140), 1, cv2.LINE_AA)

    if priv.get("exit_reached"):
        cv2.rectangle(up, (160, 195), (480, 255), (20, 110, 20), -1)
        cv2.putText(up, "EXIT REACHED!", (195, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    elif priv.get("dead"):
        cv2.rectangle(up, (190, 195), (450, 255), (20, 20, 130), -1)
        cv2.putText(up, "PLAYER DIED", (210, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([up, hud])


def _title_card(title: str, subtitle: str) -> np.ndarray:
    card = np.zeros((544, 640, 3), dtype=np.uint8)
    card[:] = (18, 20, 26)
    cv2.putText(card, title, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, subtitle, (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 210, 255), 1, cv2.LINE_AA)
    return card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs/demo_videos"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--episode-ticks", type=int, default=2100)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agent = DoomAgent(doom_spaces.observation_space(), doom_spaces.action_space(), "cuda")
    seeds = list(range(100, 112))

    showcase_path = args.out_dir / "playdoom_showcase.mp4"
    writer = cv2.VideoWriter(str(showcase_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 544))
    assert writer.isOpened(), f"Failed to open VideoWriter at {showcase_path}"

    try:
        for wad, map_name, label in SHOWCASE_SPECS:
            _, best_seed, frames, final_p = _find_best_run(agent, wad, map_name, seeds, args.episode_ticks)
            stem = Path(wad).stem
            clip_path = args.out_dir / f"{stem}_seed{best_seed}.mp4"
            clip_writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 544))
            subtitle = (
                f"Seed {best_seed} | Exit={final_p['exit_reached']} | "
                f"Kills={final_p['kills']} | Keys={int(sum(final_p['keys']))} | Dmg={final_p['damage_taken']:.0f}"
            )
            card = _title_card(label, subtitle)
            for _ in range(int(args.fps * 1.5)):
                writer.write(card)
                clip_writer.write(card)

            for idx, (rgb, action, priv) in enumerate(frames):
                fr = _render_frame(rgb, stem, idx, action, priv)
                writer.write(fr)
                clip_writer.write(fr)
            # Hold final frame 1.5s
            for _ in range(int(args.fps * 1.5)):
                writer.write(fr)
                clip_writer.write(fr)
            clip_writer.release()
            print(f"Recorded {clip_path} ({len(frames)} steps) -> {subtitle}")
    finally:
        writer.release()

    print(f"Showcase video saved: {showcase_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
