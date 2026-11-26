# Training Maps & Curriculum Suite

**Owned by Member 4 (Environment, rollouts, maps, and reward shaping)**

## 1. Curriculum Tiers

To prevent catastrophic forgetting and guide policy learning from basic locomotion to complex long-horizon key puzzles, training maps are organized into 4 progressive curriculum stages matching the official evaluation tiers:

### Tier 1: Locomotion, Survival & Basic Combat
- **Focus:** Navigating open spaces, dodging projectile hazards, engaging enemies (zombieman, shotgun guy, imp).
- **Layouts:** Rectangular and L-shaped arenas with obstacle columns for cover.
- **Goal:** Learn to move forward, circle-strafe, and aim without getting trapped against walls.

### Tier 2: Multi-Room Navigation & Door Traversal
- **Focus:** Exploring partitioned spaces, identifying doorways, pressing `USE` without bouncing doors.
- **Layouts:** 2–4 connected rooms via standard manual doors (`D`).
- **Goal:** Develop spatial exploration habits, moving room-to-room rather than loitering in the spawn room.

### Tier 3: Keycards & Locked Door Progression
- **Focus:** Long-horizon dependency: Locating keycard (`K`), remembering locked door location (`L`), backtracking and unlocking door to reach exit (`E`).
- **Layouts:** Branched corridors where keycard is placed at the end of branch A and locked door at branch B.
- **Goal:** Temporal credit assignment across room transitions.

### Tier 4: Full Gauntlet
- **Focus:** Multi-room labyrinth combining hostile encounters, multiple keys (Blue/Yellow/Red), hazardous sectors, and exit traversal.
- **Layouts:** Complex multi-sector layouts with mixed enemy types and elevation changes.

---

## 2. Map Sourcing & Licensing

- **Base IWAD:** `freedoom2.wad` (bundled with `vizdoom==1.3.3`, under BSD-3-Clause license).
- **Included Starter PWAD:** `maps/two_rooms.wad` (starter template map for verification).
- **Procedural WAD Generation:**
  Using `doom_env.wad` and ASCII layout compiler:
  ```python
  from fixtures import fixture_wad
  wad_path = fixture_wad("##########\n#P..D.K.E#\n##########")
  ```
  Procedural map synthesis guarantees non-overlapping train and evaluation layouts, preventing any layout memorization.

---

## 3. Held-Out Local Validation Set

To strictly guard against overfitting:
- **Training Set:** Generated procedural variants and training PWADs in `maps/training/`.
- **Held-Out Validation Set:** Held-out fixed seed configurations and maps used exclusively by Member 4 / Member 1 for checkpoint benchmarking:
  - Never exposed during gradient updates or PPO rollouts.
  - Used by `evaluate_policy` to diagnose failure modes (stuck, door bounce, ammo exhaustion, combat death).
