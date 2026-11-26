# Training Maps

These small PWADs are synthetic Doom-format training layouts generated for this project and
committed here so training does not depend on test-only files or external downloads. Their source
layouts are also used by `tests/test_env.py`; no third-party map assets are included.

| File | Tier / focus | Source layout |
| --- | --- | --- |
| `tier1_projectile.wad` | Combat and survival; imp projectile exposure | `########## / #P......i# / ##########` |
| `tier2_door.wad` | Door use and exit traversal | `########## / #P..D...E# / ##########` |
| `tier3_red_key.wad` | Red key, locked red door, exit progression | `############ / #P.r..R...E# / ############` |

The PWADs were copied from the checked-in synthetic test fixtures whose layouts are specified in
`tests/test_env.py`. They are training aids, not evaluation maps. Use multiple seeds and keep
`maps/two_rooms.wad` or other maps out of the training set when using them for held-out checks.

Example:

```bash
uv run python -m training.train --wad maps/training/tier2_door.wad --map MAP01 \
  --updates 3 --rollout-steps 16 --device cpu --save-dir /tmp/doom-checkpoints \
  --log-dir /tmp/doom-logs
```

The included maps are intentionally small fixtures, not a substitute for a broader varied
curriculum or held-out evaluation set. Add licensed maps or procedural variants before final
submission and record their source and license here.
