# The 99 unscoreable RxR-CE val-unseen episodes (2026-11-13)

User: "98 of the 3,669 RxR-CE val-unseen of the best 71.9 R2R is failed, right? Can you help to fix those eps and eval
and update the latest RxR result for me?" Then: "you can stop the training a little bit and run these eval for me and
update the latest result about RxR".

## What failed

Every full RxR evaluation (3,669 English guide episodes) had 99 episodes with `failure_reason` "Required simulator
metrics unavailable" (GRPO run 1: 98 of them, plus episode 4052 of the same set failing earlier with an invalid model
response 'MOVE', and episode 3105 of another scene with the same invalid response). The 99 IDs are identical in every run
(`rxr99_episode_ids.json`, sha256 275f026f...); the paper (X_suppl, Evaluation) counted them as failures for every model.

All 99 are in scene `oLBMNvg9in8` (345 RxR episodes there; the other 246 were scored). The policy ran each of them to the
500-action cap; `distance_to_goal` was infinite at every step, so SR = SPL = 0 and NE / OS were undefined (the old NE and
OS averages covered 3,570 episodes only).

## Cause

The scene's navmesh (`scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.navmesh`, sha256 686e5c40..., no stored settings, read
as the default agent 1.5 m / 0.1 m) has 12 islands: one of 143.7 m^2 and 11 small ones (1.0 to 10.7 m^2) on the basement
(y = -2.2), mezzanine (y = 3.0 to 3.8) and attic (y = 5.8 to 6.2) levels. Each of the 99 episodes has its start or goal on
a small island and its reference path crosses islands (150 crossings, 18 island pairs), so the goal is unreachable.

Our evaluator uses the R2R agent (height 1.5 m, radius 0.1 m; cameras at 1.25 m) for RxR as well. VLN-CE's RxR config
(`habitat_extensions/config/rxr_vlnce_english_task.yaml`) uses HEIGHT 0.88, RADIUS 0.18, and habitat-sim rebuilds the
navmesh whenever the agent's height or radius differs from the loaded navmesh's settings
(`habitat_sim/simulator.py::_config_pathfinder`). So RxR-CE was generated on the navmesh rebuilt for the 0.88 m agent.

Checks (CPU, habitat-sim 0.2.4, `scratchpad/rxrfix/*.py`):

| navmesh of oLBMNvg9in8 | islands | 99 episodes: goal reachable | GT-action replay reaches goal (99 / other 246) | GT replay = recorded GT positions (other 246) |
|---|---|---|---|---|
| file as loaded (1.5 m / 0.1 m) | 12 | 0 | 0 / 234 | 54 within 5 cm (median dev 0.13 m) |
| rebuilt 1.5 m / 0.1 m | 12 | 0 | (same as the file) | |
| rebuilt 0.88 m / 0.18 m (RxR-CE agent) | 4 | 99 | 99 / 246 | 246 within 5 cm (median dev 0.000 m) |
| rebuilt 0.88 m / 0.10 m | 5 | 99 | | |

The agent height decides it (low clearances on the stairs to the basement and attic).

## Fix

Re-evaluate exactly the 99 episodes with `main_agent.height: 0.88, radius: 0.18` (habitat-lab config
`docs/evaluation_setup/vln_rxr15_rxrce_agent.yaml` in the frozen trees `svln/code/rxrce_agent_eval_20261113`, local and
gpu2; launcher `ops/local/eval_rxr_rxrce_agent.sh`). Habitat then rebuilds the scene's navmesh for the RxR-CE agent; nothing
else changes (cameras stay at 1.25 m: the first RGB frames of tested episodes are pixel-identical under both configs;
15-degree turns, 0.25 m steps, 500-action cap, greedy decoding, seed 42). Both trees' `src/qwen_vl` (79 files) are
byte-identical to the code of the original RxR evaluations. The other 3,570 episodes keep their original records.
`rxr_fix_merge.py` replaces the 99 records and recomputes the summary with the evaluator's `summarize()`.

Runs (all done 2026-11-13): locally CoRe run 1 and GRPO run 1 (the 99), Stage 2 (`sfr`), `rewind_sfr_s125` and
`rewind_path_sfr_mix1k_s125` (the 52 of the 99 that lie in sets A + B, `rxr52_episode_ids.json`), all on GPU 1; on gpu2
CoRe run 2 (`rewind_sfr_mix8k_aplus` s125, the 99) on GPU 3 while the ablation `abl_unverified_sfr_r2r1k` was paused at
its step-40 resumable save (22:30-22:58, resumed). The user decided not to re-evaluate the gpu2 checkpoints 250-750.
`rxr_fix_merge.py` replaces the records (`replace`) or combines a new 3,570-episode evaluation with the 99 (`combine`,
for the seeds without RxR results, queued in `rxr_fix_final_evals.sh`); outputs `<model>_<set>_fixed_val_unseen/merged/`.
Paired tests: `rxr_fixed_compare.py` (same statistics as compare_evals.py); report numbers: `rxr_fixed_report_tables.py`,
`rxr_report_update.py`; paper: `rxr_paper_update.py`.

Incidents: (1) 22:27 the first 52-episode run, started next to the CoRe seed-1414 trainer, caused a host OOM kill of
trainer rank 1 (resumed from step_0005, update 5 lost); the trainer was then paused for the remaining runs
(`seed1414_rxr_pause.sh`) and resumed at 22:53. (2) gpu2 GPU 3 also holds another user's 51 GB job, so only 4 of 5
evaluation workers fitted; the first s125 attempt was stopped and moved to `*_aborted_w5_20261113`, rerun with 4.

Note for the paper: the 3,570 other episodes still use the R2R-agent navmesh. On it the GT actions of the scored episodes
of this scene drift by a median 0.13 m from the recorded positions; a fully RxR-CE-consistent evaluation would rebuild the
navmesh for the 0.88 m / 0.18 m agent in every scene (a full re-evaluation; not done).

## Results

On the 99 episodes alone (RxR-CE agent navmesh): CoRe run 1 93 / 99 successes, CoRe run 2 92, GRPO run 1 91; on the 52:
Stage 2 46 / 52, path-aware rewind 50, rewind_sfr_s125 48. No episode is unscoreable any more; GRPO run 1 keeps one
policy failure outside the 99 (episode 3105, invalid response).

| model | set | SR before -> after | SPL before -> after | nDTW before -> after | OS before -> after |
|---|---|---|---|---|---|
| CoRe run 1 `rewind_sfr_mix1k_s125` | full | 71.08 -> 73.62 | 57.50 -> 59.56 | 69.51 -> 70.68 | 78.06 -> 80.68 |
| CoRe run 2 `rewind_sfr_mix8k_aplus_s125` (gpu2) | full | 70.59 -> 73.10 | 57.44 -> 59.49 | 69.56 -> 70.47 | 77.41 -> 80.05 |
| GRPO run 1 `grpo_sfr_mix8k_aplus_s125` | full | 69.96 -> 72.44 | 57.67 -> 59.64 | 69.39 -> 70.28 | 77.41 -> 80.02 |
| Stage 2 `sfr` | 2k | 63.55 -> 65.85 | 52.15 -> 54.03 | 64.48 -> 65.16 | 71.85 -> 74.35 |
| `rewind_sfr_s125` (R2R-only prompts) | 2k | 70.70 -> 73.10 | 56.91 -> 58.80 | 69.83 -> 70.55 | 78.35 -> 80.90 |
| `rewind_path_sfr_mix1k_s125` | 2k | 69.00 -> 71.50 | 55.92 -> 57.80 | 68.83 -> 69.62 | 76.20 -> 78.80 |

CoRe runs 1 + 2: 73.36 +- 0.37 SR (was 70.84 +- 0.35). CoRe run 1 vs GRPO run 1 on the full set: +1.17 SR, McNemar
p 0.16, 95% CI [-0.44, 2.78] (was +1.12, p 0.18). Every set (A, B, C, 2k, full) of every model: `rxr_fixed_report_tables.md`.
