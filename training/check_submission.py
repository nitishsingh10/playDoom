"""Read-only weights/config/inference preflight. No gameplay or training.

This checks packaging and action parity, not gameplay quality. --require-committed
also checks that the exported files match the current Git commit, as opposed to
existing only on the VM. Nothing is exported, copied or promoted by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from doom_env import spaces
from submission.actions import StructuredActionDistribution
from submission.agent import DoomAgent
from submission.checkpoint import extract_state_dict
from training.evaluate import load_policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_committed(paths):
    for path in paths:
        try:
            root = Path(subprocess.check_output(
                ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                text=True, stderr=subprocess.PIPE,
            ).strip()).resolve()
            relative = path.resolve().relative_to(root).as_posix()
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"], stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, ValueError) as error:
            raise ValueError(f"File is absent from the current Git commit: {path}") from error
        if hashlib.sha256(committed).hexdigest() != digest(path):
            raise ValueError(f"File differs from the current Git commit: {path}")


def probe_observation(enemy=False):
    observation = {
        "rgb": np.zeros((spaces.SCREEN_HEIGHT, spaces.SCREEN_WIDTH, 3), dtype=np.uint8),
        "depth": np.zeros((spaces.SCREEN_HEIGHT, spaces.SCREEN_WIDTH), dtype=np.uint8),
        "vitals": np.zeros(len(spaces.VITALS_FIELDS), dtype=np.float32),
        "rays": {"distance": np.ones(spaces.NUM_RAYS, dtype=np.float32),
                 "hit_type": np.ones(spaces.NUM_RAYS, dtype=np.int64)},
        "visible_entities": {
            "features": np.zeros((spaces.MAX_ENTITIES, len(spaces.ENTITY_FEATURES)), dtype=np.float32),
            "entity_type": np.zeros(spaces.MAX_ENTITIES, dtype=np.int64),
            "mask": np.zeros(spaces.MAX_ENTITIES, dtype=np.int8),
        },
    }
    observation["vitals"][:5] = [100, 0, 2, 50, 50]
    if enemy:
        observation["visible_entities"]["features"][0, :3] = [0.25, 0, 1]
        observation["visible_entities"]["entity_type"][0] = spaces.ENTITY_TYPES.index("enemy_projectile")
        observation["visible_entities"]["mask"][0] = 1
        observation["rays"]["hit_type"][0] = spaces.RAY_HIT_TYPES.index("enemy")
    return observation


def check_export(weights_dir, *, mode, source_checkpoint=None, require_committed=False):
    weights_dir = Path(weights_dir).resolve()
    weights, config = weights_dir / "checkpoint.pt", weights_dir / "config.json"
    for path in (weights, config):
        if not path.is_file():
            raise FileNotFoundError(f"Missing exported file: {path}. DoomAgent must not use its fallback.")
    paths = [weights, config]
    if source_checkpoint is not None:
        paths.append(Path(source_checkpoint).resolve(strict=True))
    before = {str(p): digest(p) for p in paths}
    if require_committed:
        check_committed([weights, config])
    expected_deterministic = mode == "deterministic"
    if mode not in ("stochastic", "deterministic"):
        raise ValueError("mode must be stochastic or deterministic")
    if json.loads(config.read_text()).get("deterministic_eval") is not expected_deterministic:
        raise ValueError(f"Export config does not explicitly select {mode} inference")
    with torch.random.fork_rng():
        agent = DoomAgent(spaces.observation_space(), spaces.action_space(), "cpu", weights_dir=weights_dir)
        if agent.model is None:
            raise ValueError("DoomAgent loaded no model; fallback is not a trained submission")
        exported = extract_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        actual = agent.model.state_dict()
        if not all(torch.isfinite(v).all() for v in actual.values()):
            raise ValueError("Export contains nonfinite model parameters")
        if set(actual) != set(exported) or not all(torch.equal(actual[k], v) for k, v in exported.items()):
            raise ValueError("DoomAgent parameters do not exactly match the export")
        reference = load_policy(source_checkpoint, "cpu") if source_checkpoint is not None else agent.model
        if source_checkpoint is not None and (
            set(actual) != set(reference.state_dict()) or not all(
                torch.equal(v, reference.state_dict()[k]) for k, v in actual.items())
        ):
            raise ValueError("Export parameters do not match the selected training checkpoint")
        for enemy in (False, True):
            observation = probe_observation(enemy)
            with torch.inference_mode():
                logits, _, _ = reference(observation)
            for seed in (0, 17, 109):
                agent.reset()
                torch.manual_seed(seed)
                distribution = StructuredActionDistribution(logits)
                expected = distribution.mode() if expected_deterministic else distribution.sample()
                torch.manual_seed(seed)
                chosen = agent.act(observation)
                if not np.array_equal(chosen, expected[0].cpu().numpy()):
                    raise ValueError("DoomAgent action does not match the selected checkpoint/distribution")
    if any(digest(Path(p)) != h for p, h in before.items()):
        raise RuntimeError("Inspected files changed during preflight")
    return {"model_loaded": True, "architecture_id": agent.config.architecture_id,
            "inference_mode": mode, "exact_export_parameters": True,
            "exact_source_parameters": True if source_checkpoint is not None else None,
            "synthetic_action_parity_checks": 6, "files_unchanged": True,
            "current_commit_files_verified": require_committed, "sha256": before,
            "gameplay_quality_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("--mode", choices=("stochastic", "deterministic"), required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--require-committed", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_export(args.weights_dir, mode=args.mode, source_checkpoint=args.source_checkpoint,
                              require_committed=args.require_committed)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"SUBMISSION PREFLIGHT FAILED: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
