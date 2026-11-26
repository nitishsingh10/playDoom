# Team Implementation Status

Last reviewed: 2026-11-26

## Repository sync

- Base: `main` / `origin/main` at `68828dd`; fetched and verified current. All four member feature branch tips are ancestors of `origin/main`.
- Current work is on local branch `codex/complete-doom-workflow`, branched from synced `main`.
- No commits or pushes have been made. No merge conflicts were present.
- PDFs are excluded by `.gitignore` (`*.pdf`); no PDF is tracked. The brief draft is local and ignored.

## Member completion review

| Member | Status | Evidence / remaining work |
| --- | --- | --- |
| 1 - integration and final agent | Code complete; final artifact pending | Agent loads from `submission.model`, keeps inference imports inside `submission/`, and returns valid actions. The validator passed with a temporary checkpoint. `submission/weights/` still has no selected trained checkpoint, so the current default is the random fallback. |
| 2 - PPO and training | Implemented and real-smoke verified | PPO/GAE/checkpoint code and a working CLI are present. Nested observations work through shared preprocessing; truncation bootstraps from the episode's final observation. One CPU PPO update completed while cycling maps. Longer training and checkpoint selection on held-out maps remain. |
| 3 - spatial model | Implemented and integrated | Encoder/model shape and device tests pass. `DoomPolicy` accepts raw official observations and encoded tensors, using the same preprocessing in training and inference. The collector-to-policy-to-PPO path passed a one-update run. The model remains feed-forward with no cross-room memory. |
| 4 - environment, rollouts, rewards, metrics, maps | Implemented; meaningful evaluation pending | Three synthetic training PWADs, seeded map cycling, observation inspection, metrics, and checkpoint evaluation tools are present. Reward shaping pays movement only on first entry into a cell and counts edge-triggered USE spam. Pilot logs showed living/door penalties dominating sparse exploration, so defaults were tuned down/up respectively. A broader held-out set and report on a trained candidate remain. The official wrapper does not expose a door-open counter. |

## Verification

- Full applicable local test suite: **72 passed, 2 deselected**. The Linux-only affinity test calls `os.sched_getaffinity`, unavailable on macOS. The existing remote protocol `deep_json` test is excluded because it stalled in a prior run.
- Real CPU training: 320 PPO updates and 20,480 total rollout decisions while cycling the synthetic maps; resumable checkpoints and metrics were written under `/tmp`.
- `training.inspect_obs` ran against the red-key map; `training.export_checkpoint` produced tensor-only weights; `training.evaluate` completed one short local test-map episode.
- Extended evaluation: `maps/two_rooms.wad:MAP01`, seeds 0-4, 0/5 exits, 1.0 survival rate, 0 kills, 0 damage, and 6 visited 64-unit cells per episode at the default threshold. The 15 episodes across the three synthetic training maps also reached 0 exits. A threshold sweep from 0.25 to 0.95 did not improve exit rate. This checkpoint is not a candidate model. See [evaluation report](../reports/evaluation/pilot_2026-11-26.json).
- Validator with temporary exported weights: **passed** on the earlier 64-update pilot. Warnings were macOS CPU affinity/RAM measurement limitations and expected local uncommitted/untracked temporary files. The temporary weights were removed.
- Local architecture measurement: 34,090 parameters; CPU inference mean 0.206 ms and p95 0.225 ms over 200 calls after warm-up. CUDA was unavailable.
- The two-page project-brief draft was refreshed with the 320-update evaluation and rendered for visual review.

## Remaining project deliverables

1. Improve combat and exit behavior, then train for a useful duration across broader maps/seeds; the 20,480-step run did not exit on the held-out proxy.
2. Select/export the best checkpoint into `submission/weights/`, track it with Git LFS, and rerun the validator on that candidate.
3. Replace the local project-brief draft with the final two-page brief after the team confirms its USP and held-out results.
4. Resolve the source-document conflict about the starter `Dockerfile`: the PS says not to submit a Dockerfile, while the implementation packet says preserve starter files and asks the team to check with organizers.
5. Record external map sources/licenses if the team expands beyond the bundled synthetic maps.
