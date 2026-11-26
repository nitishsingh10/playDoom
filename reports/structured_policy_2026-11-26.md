# Structured PPO action policy — 26 November 2026

The first milestone is implemented: the learned policy uses a structured distribution that cannot select opposing directions, while the environment and submission still receive `MultiBinary(9)` actions. Sampling, replay log probabilities, entropy and inference share the same implementation. No targeting rules, action overrides, observation redesign or reward changes were introduced. The smoke checkpoint is a wiring experiment, not a submission candidate.

The original trained checkpoint and config were subsequently supplied from the user's Downloads folder and copied byte-for-byte into `training/runs/legacy_baseline/`. Their hashes and the structured candidate's hashes are recorded in `reports/evaluation/paired_2026-11-26/manifest.json`. The baseline has the expected nine-head actor, hidden size 32 and fusion size 128. Its config uses stochastic evaluation, threshold 0.5, and no RGB/depth. The supplied checkpoint/config files and submission weights were left unchanged. The controlled comparison below replaces the earlier missing-baseline limitation; the historical pilot summary was not used as a substitute.

## Data flow and root causes

`doom_env.spaces` fixes the nine external button indices. `DoomEnv.step()` passes those buttons unchanged to ViZDoom for four tics. Public observations contain rays, entities, vitals, RGB and depth. `ObservationEncoder` uses only rays/entities/vitals, with identical normalization for rollout and inference. Entity classes 1–3 identify enemies; masked padding is excluded. Entity angles retain sine/cosine bearing, and distance is reported divided by 1024 before encoder normalization. Vitals retain selected weapon, selected ammo and all four reserve-ammo counts.

`DoomPolicy` encodes rays/entities/vitals, pools them, fuses them, and emits actor logits plus a value. `RolloutCollector._step_policy()` samples an action, stores the nine buttons and their behavior log probability, and steps the engine. Privileged state goes only to rewards/metrics. `train_loop()` computes GAE, then `ppo_update()` forwards stored observations and evaluates stored buttons under the current policy. `DoomAgent._infer_action()` and `training.evaluate` now use the same distribution as collection and PPO replay.

The old actor emitted nine independent Bernoulli logits. Its support included forward+backward, strafe-left+right and turn-left+right. This was a representation error: neither improved weights nor a probability threshold could exclude those combinations. A second Bernoulli fallback in rollout collection could silently restore that behavior if imports failed; it has been removed. The old agent and evaluator also separately applied sigmoid/thresholding, so both needed updating.

Additional combat findings, kept separate from this change:

- The ray encoder has no ray-angle/index feature and applies mean/max pooling over all rays. Its summary is invariant to permuting ray indices, so the ray branch loses front/left/right/back information. Entity bearings remain available, so this is a limitation rather than proof of the sole cause of combat failure. A 90-degree ray-index permutation changed logits only by floating-point roundoff in a local check.
- The reward has no damage-dealt, aiming or engagement signal. A kill gives +2 shaping and +0.1 raw reward; receiving 100 damage gives −2; death gives −5 shaping and −1 raw reward. Each new cell gives +0.5 plus at most +0.05 movement reward. Four new cells are therefore roughly one kill. Combat credit is sparse, especially before the agent learns to aim, and exploration can compete with combat.
- The living cost is −0.001 per decision, or −0.525 over 525 decisions. PPO normalizes minibatch advantages; value targets still retain reward scale. No reward coefficients were changed. The structured maximum entropy is `3*ln(3)+3*ln(2)=5.3753` rather than `9*ln(2)=6.2383`; the existing entropy coefficient remains 0.01.
- `USE` and weapon selection are native button-down actions. Held frames are not repeated presses. Engine tests show that re-pressing USE can reverse an opening door; weapon changes can be delayed by animations. Zero selected-weapon ammo can mean the fist after cycling from the initially loaded pistol, not an ammo-observation bug. The agent's edge tracker records actions and does not rewrite them.
- Terminal observations reuse the previous public observation because ViZDoom has no final state buffer. Refreshed privileged engine variables are needed for exact final health, damage, kills and weapon changes. Timeout value bootstrapping still uses the last available public observation; it cuts propagation across resets but remains an approximation of the unavailable final state.

## Action distribution and compatibility

| Internal logits | Distribution and choices | External buttons |
| --- | --- | --- |
| 0:3 | Categorical: none / forward / backward | 0, 1 |
| 3:6 | Categorical: none / strafe left / strafe right | 2, 3 |
| 6:9 | Categorical: none / turn left / turn right | 4, 5 |
| 9 | Bernoulli attack | 6 |
| 10 | Bernoulli use | 7 |
| 11 | Bernoulli select next weapon | 8 |

The categorical distributions normalize their logits with softmax. Independent longitudinal/lateral decisions preserve diagonal movement. There is no post-sampling conflict filter. Stored pairs reconstruct choices exactly: `00 -> none`, `10 -> first`, `01 -> second`; invalid/conflicting stored actions fail clearly. Joint log probability and entropy sum the three categorical terms and three Bernoulli terms. Deterministic inference selects each categorical argmax and Bernoulli mode; stochastic inference samples those same factors. Categorical ties choose the first maximal choice (none when all three logits are equal), and Bernoulli ties choose on, consistently across callers. The legacy `action_threshold` applies only to explicit historical evaluation, not to structured mode selection.

New checkpoints use `architecture_id="doom_structured_v1"`, `checkpoint_version=1`, and `model_config` with network widths. Legacy raw nine-head weights are recognized as `doom_bernoulli_v1`, never converted. Normal training/submission loading rejects them clearly. The evaluator requires `--legacy-baseline` to use the original architecture and distribution. The submission agent reads tensor-only exports with `weights_only=True` and needs no training imports. Resumable checkpoints receive export guidance instead of an unsafe inference load.

Default training and exports now go under `training/runs/structured_v1/`. Export writes matching `config.json`, refuses baseline destinations and source overwrite, and preserves network widths. `training.validate_checkpoint` copies submission code plus the selected export into a temporary directory and invokes the existing validator. Baseline files are never replaced for validation. A small GPU-resume correction restores RNG byte tensors on CPU even when checkpoint tensors were mapped to CUDA.

## Files changed

| Files | Purpose |
| --- | --- |
| `submission/actions.py` (new), `training/ppo.py` | Shared structured sampling, mode, log probability and entropy; PPO replay integration |
| `submission/model.py`, `submission/schema.py`, `submission/agent.py` | Twelve-head actor, explicit architecture/config, matching inference, one PyTorch CPU thread, valid missing-weight fallback |
| `training/rollout.py`, `training/train.py` | Remove inconsistent sampler fallback; keep nine-button storage; separate experiment defaults |
| `submission/checkpoint.py` (new), `training/checkpoint.py`, `training/export_checkpoint.py` | Architecture/version checks, width metadata, safe export and RNG restoration |
| `training/evaluate.py`, `env/doom_env/wrapper.py` | Exact action/combat diagnostics; add weapon/ammo fields only to privileged metrics, leaving public observation/action contracts unchanged |
| `training/validate_checkpoint.py` (new) | Validate selected exports without baseline promotion |
| `tests/test_actions.py`, `tests/test_agent.py`, `tests/test_checkpoint.py`, `tests/test_evaluate.py`, `tests/test_action_semantics.py` (new) | Distribution, inference, compatibility, diagnostic isolation and native engine edge tests |
| `tests/test_model.py`, `tests/test_ppo.py`, `tests/test_rollout.py`, `tests/test_validate.py` | Updated logits, PPO gradients/ratios, rollout replay and export tests |
| `tests/test_remote.py`, `tests/test_validate.py` | Test portability: send hostile socket payloads concurrently to avoid macOS buffer deadlock; skip Linux-only affinity/RSS test off Linux |
| `README.md`, `training/README.md`, this report and evaluation JSONs | Current architecture, safe workflow, evidence and reproducible commands |

## Verification and measured gameplay

Environment: macOS ARM CPU, Python 3.14.6, PyTorch 2.15.0, ViZDoom 1.3.3, Gymnasium 1.4.0, NumPy 2.5.4, pytest 9.2.0. No NVIDIA L4 was available here. The initial full-suite run encountered pre-existing macOS Unix-socket path limits and a socket-buffer test deadlock. Short `/tmp` test paths and the test portability changes above resolve them; production protocol behavior was unchanged.

Tests cover binary `[batch,9]` output; all movement/rotation constraints; diagonal support; arbitrary leading batch shapes; all 27 deterministic direction combinations; finite probabilities/entropy; all 216 valid actions' normalized probabilities and independently calculated entropy; sample/replay agreement and unchanged-policy zero KL; gradients through all 12 logits; optimizer updates; actual rollout replay; versioned save/load/export; explicit legacy rejection; both inference modes; and engine USE/weapon semantics. Final result: **166 passed, 1 skipped in 41.68 seconds**, recorded in `training/runs/structured_v1/pytest.txt`. The skipped CPU-affinity/RSS test requires Linux.

The smoke run used seed 17, two PPO updates, 32 rollout decisions per update, minibatches of 16, two optimization epochs and a 120-tic episode cap. It completed 64 decisions in 1.00 seconds. Update entropy was 5.364/5.362, value loss approximately 0.0003/0.0005 and KL rounded to 0.0000. It exercised real ViZDoom observations, GAE, optimizer updates and checkpoint/export, not combat convergence. Export: `training/runs/structured_v1/export/checkpoint.pt` with matching `config.json`.

The exported checkpoint passed `validate.py` through `training.validate_checkpoint` on all three requested maps at a 2100-tic cap. Initialization took about 1.0 second. Per-step p50 was about 0.8 ms and maximum about 2 ms in the final validator run, below the approximately 40 ms decision budget. Local warnings: no Git metadata in the temporary staging copy; CPU affinity unavailable on macOS; peak RAM unavailable without Linux `/proc`. CUDA/VRAM and Docker read-only/network sandbox enforcement were not verified on this host. `DoomAgent` performs no network calls or repository writes. Passing the validator does not establish useful gameplay.

Gameplay used MAP01, seeds 0–4 and 2100 tics on each of the three requested maps, separately for deterministic and stochastic inference. Exact evidence is in `reports/evaluation/structured_v1_smoke_2026-11-26_deterministic.json` and `reports/evaluation/structured_v1_smoke_2026-11-26_stochastic.json`.

| Map and mode | Episodes | Mean decisions | Kills / exits | Mean damage | Mean final health | Attack / visible steps | End reason |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| tier1_projectile, deterministic | 5 | 91.4 | 0 / 0 | 105.6 | −5.6 | 0 / 169 | 5 deaths |
| tier1_projectile, stochastic | 5 | 89.8 | 0 / 0 | 112.8 | −12.8 | 114 / 245 | 5 deaths |
| tier2_door, deterministic | 5 | 525 | 0 / 0 | 0 | 100 | 0 / 0 | 5 timeouts |
| tier2_door, stochastic | 5 | 525 | 0 / 0 | 0 | 100 | 0 / 0 | 5 timeouts |
| tier3_red_key, deterministic | 5 | 525 | 0 / 0 | 0 | 100 | 0 / 0 | 5 timeouts |
| tier3_red_key, stochastic | 5 | 525 | 0 / 0 | 0 | 100 | 0 / 0 | 5 timeouts |

Each opposing pair had exactly **0 conflicts (0%)** across all 5707 deterministic decisions and all 5699 stochastic decisions. For these episodes every visible enemy was also inside the diagnostic range of 1024 units, so useful-range attack counts equal the visible counts above. This range is a distance proxy, not a weapon-specific reach, aiming or damage guarantee. Full JSONs also contain activation counts, button-down edges, actual weapon changes, zero-ammo observations and reward-component totals.

## Paired comparison with the supplied baseline

Both policies were rerun on the same three MAP01 maps, seeds 0–4, CPU and 2100-tic cap in both modes: 60 episodes total. Stochastic evaluation is primary because it matches the supplied baseline config. Deterministic evaluation is a secondary controlled condition. All rerun structured results exactly matched the previously saved smoke results. Baseline weights/config passed byte/hash preservation checks. Independent fidelity checks against the original `git HEAD` model and inference implementation found identical preprocessing/logits on 48 observations and matching actions in 768 seeded deterministic/stochastic checks.

| Stochastic result | Supplied baseline | Structured smoke |
| --- | ---: | ---: |
| Exits, all maps | 10 / 15 (66.7%) | 0 / 15 |
| tier1_projectile exits | 0 / 5 | 0 / 5 |
| tier2_door exits | 5 / 5 | 0 / 5 |
| tier3_red_key exits | 5 / 5 | 0 / 5 |
| Total kills | 0 | 0 |
| Projectile-map deaths | 5 / 5 | 5 / 5 |
| Projectile-map mean decisions | 76.2 | 89.8 |
| Projectile-map mean damage received | 106.8 | 112.8 |
| Projectile-map mean final health | −6.8 | −12.8 |
| Projectile-map attack while enemy visible | 114 / 326 (35.0%) | 114 / 245 (46.5%) |
| Door-map mean decisions | 115.2 | 525 (timeout) |
| Red-key-map mean decisions | 153.4 | 525 (timeout) |

| Stochastic opposing pair | Baseline count / decisions (%) | Structured count / decisions (%) |
| --- | --- | --- |
| Forward/backward | 218 / 1724 (12.65%) | 0 / 5699 (0%) |
| Strafe left/right | 170 / 1724 (9.86%) | 0 / 5699 (0%) |
| Turn left/right | 510 / 1724 (29.58%) | 0 / 5699 (0%) |

The pair counts above are measured simultaneously pressed pairs, not estimates from marginal activations. On the projectile map specifically, baseline conflicts were forward/backward 83/381 (21.78%), strafe left/right 73/381 (19.16%) and turn left/right 137/381 (35.96%). All visible enemies in these runs also satisfied the 1024-unit useful-range proxy, so attack-within-range counts equal attack-visible counts. Neither measure establishes a hit. The baseline made 66 actual weapon changes on the projectile map versus 83 for the smoke policy; USE counts were 246 held decisions / 86 button-down edges versus 212 / 111. A lower activation count therefore did not mean fewer presses.

Under deterministic evaluation both policies had zero kills/exits, died on all projectile episodes and timed out on the door/key maps. The legacy policy selected both turn buttons on every projectile decision (291/291), with no visible-enemy attack; the structured policy had zero conflicts and also no visible-enemy attack. Deterministic failure is not representative of the supplied stochastic evaluation setting.

The trained baseline currently navigates the door/key maps better than the 64-decision smoke policy. Do not promote the smoke export. The action-representation defect is fixed, but combat improvement has not been demonstrated. The baseline export contains no training-step, optimizer, RNG, training-map or commit metadata, so its training budget cannot be recovered. These are matched evaluation conditions, not a training-budget-matched ablation; five seeds per map/mode support descriptive findings only. Different distributions generate different action draws and trajectories even with the same random seed.

Full per-episode evidence and metric summaries are in `reports/evaluation/paired_2026-11-26/{baseline_stochastic,baseline_deterministic,structured_stochastic,structured_deterministic,comparison}.json`. The next experiment is to train the structured policy with a comparable documented training budget before assessing gameplay improvement. Investigate directional ray features, combat credit and temporal/edge-action memory as separate ablations. Damage-dealt instrumentation would be training/diagnostic information only, never policy input or an inference rule. No long training run was launched.

## Reproduction

Run from the repository root. The following commands reproduce the local setup when PyTorch is already installed, as it was on this host. On a fresh system, install the PyTorch build appropriate to its CPU/CUDA runtime as well.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install gymnasium==1.4.0 vizdoom==1.3.3 pytest==9.2.0 -e ./env
TMPDIR=/tmp .venv/bin/python -m pytest -q --basetemp=/tmp/doom-structured-tests

.venv/bin/python -m training.train \
  --updates 2 --rollout-steps 32 --minibatch-size 16 --epochs 2 \
  --device cpu --episode-ticks 120 --seed 17 \
  --save-dir training/runs/structured_v1/smoke/checkpoints \
  --log-dir training/runs/structured_v1/smoke/logs

.venv/bin/python -m training.export_checkpoint \
  training/runs/structured_v1/smoke/checkpoints/checkpoint_latest.pt \
  training/runs/structured_v1/export/checkpoint.pt

TMPDIR=/tmp .venv/bin/python -m training.validate_checkpoint \
  training/runs/structured_v1/export \
  --map maps/training/tier1_projectile.wad \
  --map maps/training/tier2_door.wad \
  --map maps/training/tier3_red_key.wad \
  > training/runs/structured_v1/validator.txt 2>&1

for mode in deterministic stochastic; do
  extra=()
  if [ "$mode" = stochastic ]; then extra=(--stochastic); fi
  .venv/bin/python -m training.evaluate \
    --checkpoint training/runs/structured_v1/export/checkpoint.pt \
    --map maps/training/tier1_projectile.wad \
    --map maps/training/tier2_door.wad \
    --map maps/training/tier3_red_key.wad \
    --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
    --device cpu --episode-ticks 2100 "${extra[@]}" \
    --output "training/runs/structured_v1/evaluation_${mode}.json"
done
```

The supplied files were preserved under `training/runs/legacy_baseline/`. To reproduce that copy, use the commands below; `-n` preserves existing files. The following loop reproduces all four paired evaluation files, with identical map/seed/cap arguments. `--stochastic` matches the supplied baseline config; the deterministic condition uses its threshold of 0.5. Do not put old weights into the structured agent's default weights directory.

```bash
mkdir -p training/runs/legacy_baseline reports/evaluation/paired_2026-11-26
cp -n "$HOME/Downloads/checkpoint.pt" training/runs/legacy_baseline/checkpoint.pt
cp -n "$HOME/Downloads/config.json" training/runs/legacy_baseline/config.json

for policy in baseline structured; do
  legacy=()
  checkpoint=training/runs/structured_v1/export/checkpoint.pt
  if [ "$policy" = baseline ]; then
    legacy=(--legacy-baseline --action-threshold 0.5)
    checkpoint=training/runs/legacy_baseline/checkpoint.pt
  fi
  for mode in stochastic deterministic; do
    extra=()
    if [ "$mode" = stochastic ]; then extra=(--stochastic); fi
    .venv/bin/python -m training.evaluate \
      --checkpoint "$checkpoint" "${legacy[@]}" "${extra[@]}" \
      --map maps/training/tier1_projectile.wad \
      --map maps/training/tier2_door.wad \
      --map maps/training/tier3_red_key.wad \
      --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
      --device cpu --episode-ticks 2100 \
      --output "reports/evaluation/paired_2026-11-26/${policy}_${mode}.json"
  done
done
```
