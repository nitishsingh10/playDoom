# Training

Training uses `uv run python -m training.train` for a real ViZDoom rollout/PPO run. The compact
spatial policy and its preprocessing are shared with inference through `submission.model` and
`submission.observation`.

Use `uv run python -m training.inspect_obs --wad maps/training/tier2_door.wad` to audit the
official observation schema. Use `uv run python -m training.evaluate --checkpoint PATH --map
maps/two_rooms.wad:MAP01` for a local held-out comparison. The starter test map is a smoke-test
map, not a hidden evaluation map.

Run logs and checkpoints default to `training/runs/structured_v1/` and are ignored by Git.
`training.export_checkpoint` writes a versioned tensor-only checkpoint and matching config to
`training/runs/structured_v1/export/`; it refuses to replace files in `submission/weights/`.
Use `python -m training.validate_checkpoint training/runs/structured_v1/export` to validate
an export in a temporary submission copy while preserving the baseline. Promotion is a separate
selection step after gameplay comparisons, with a backup of any existing checkpoint and config.
Training-only privileged state is read only in `training.rewards`/metrics and never enters the
policy observation or inference path.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.
