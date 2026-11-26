"""Print the emitted doom_env observation schema and a short value-range sample."""

from __future__ import annotations

import argparse
import json

import numpy as np

from doom_env import make_env, spaces as doom_spaces


def inspect(wad: str, map_name: str = "MAP01", steps: int = 8, seed: int = 0) -> dict:
    env = make_env(wad, map_name, episode_ticks=max(steps * doom_spaces.FRAME_SKIP, 4), render_buffers=False)
    try:
        observation, _ = env.reset(seed=seed)
        samples = [observation]
        action = np.zeros(len(doom_spaces.BUTTONS), dtype=np.int8)
        for _ in range(steps):
            observation, _, terminated, truncated, _ = env.step(action)
            samples.append(observation)
            if terminated or truncated:
                break

        schema = {}
        for key, value in samples[0].items():
            if isinstance(value, dict):
                schema[key] = {
                    subkey: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "min": float(np.min(array)) if array.size else None,
                        "max": float(np.max(array)) if array.size else None,
                    }
                    for subkey, array in value.items()
                }
            else:
                schema[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(np.min(value)) if value.size else None,
                    "max": float(np.max(value)) if value.size else None,
                }
        return {
            "map": f"{wad}:{map_name}",
            "sample_count": len(samples),
            "observation": schema,
            "entity_counts": [int(item["visible_entities"]["mask"].sum()) for item in samples],
            "action_shape": list(env.action_space.shape),
            "action_dtype": str(action.dtype),
            "action_buttons": list(doom_spaces.BUTTONS),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad", default="maps/training/tier2_door.wad")
    parser.add_argument("--map", default="MAP01")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(inspect(args.wad, args.map, args.steps, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
