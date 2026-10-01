# R2R `val_unseen` results: SimpleMemVLN FullContext-B and Window8-B

Report written 2026-10-01. The four closed-loop Habitat evaluations finished on
2026-09-30 (Asia/Ho_Chi_Minh). **Every result below is final for all 1,839
R2R-VLNCE `val_unseen` episodes**, with zero recorded inference failures.

## Main results

| Checkpoint | Training snapshot | Successes | SR ↑ | SPL ↑ | Navigation error ↓ | Oracle success ↑ | 500-step forced STOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| FullContext-B epoch 1 | Completed epoch 1 | **789 / 1,839** | **42.90%** | **39.30%** | **6.60 m** | 52.53% | 147 |
| FullContext-B epoch 2 | Completed epoch 2 | 750 / 1,839 | 40.78% | 36.38% | 6.81 m | 53.67% | 186 |
| FullContext-B epoch 3 | Completed epoch 3 | 761 / 1,839 | 41.38% | 37.50% | 6.75 m | 54.92% | 187 |
| Window8-B `epoch-1/` | **Intermediate optimizer step 1353** | 756 / 1,839 | 41.11% | 36.86% | 6.71 m | 54.21% | 208 |

**FullContext epoch 1 is the best of these four checkpoints** by both SR and
SPL. Its SR is 2.12 percentage points above epoch 2 and 1.52 points above
epoch 3. Window8's SR is 1.79 points and SPL 2.44 points below FullContext
epoch 1. The Window8 checkpoint is independently initialized and trained with
its own memory policy; its published `epoch-1/` weights were saved at step 1353
and do **not** represent a completed first epoch. The comparison therefore does
not isolate the effect of an eight-step window at equal training progress.

On the matched episode IDs, FullContext epoch 1 and Window8 both succeeded on
494 episodes; only FullContext succeeded on 295, and only Window8 on 262. They
both failed on 788. This explains the 33-success difference without implying
that one policy dominates the other on every route.

## Evaluation protocol

- Dataset: R2R-VLNCE v1-3 preprocessed `val_unseen`, the same 1,839 episode IDs
  in every run. Habitat-Lab/Habitat-Sim protocol version 0.2.4, seed 42.
- RGB observation: 640 × 480, horizontal field of view 79°. Actions: forward
  0.25 m, turn 15°, or STOP. Success distance: 3 m. Episode cap: 500 actions.
- The model observes the current RGB frame before selecting each action. It
  greedily generates a normal text action with a 16-token budget and receives
  its **own generated action history** on later steps. There is no teacher
  action history, oracle STOP, or action-trie restriction. Each episode begins
  with fresh memory state.
- STOP is normally chosen by the model. A final STOP is forced at the 500-step
  cap so Habitat can compute STOP-dependent metrics consistently. Model-predicted
  STOP occurred in 1,692 / 1,653 / 1,652 / 1,631 episodes for FullContext
  epochs 1–3 / Window8, respectively; the complementary forced STOP counts
  appear in the table. All four runs recorded zero invalid-response or other
  inference failures.
- SR and SPL are Habitat's final episode metrics, averaged over all 1,839
  scheduled episodes. Navigation error is final distance to goal. Oracle
  success means the agent was within 3 m of the goal at **any visited point**;
  it is not the final success rate. nDTW was unavailable in this Habitat
  configuration and is not reported.

## Integrity and provenance

Each output directory contains `summary.json`, `episodes.jsonl`,
`evaluation_contract.json`, and `runtime.json`. I checked that all four
summaries have `complete: true`, `episodes: 1839`, `scheduled_episodes: 1839`,
and `failures: 0`. Each journal has exactly 1,839 records with unique episode
IDs matching the contract, and its success count reproduces the reported SR.
The contracts share dataset SHA256
`1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`
and Habitat configuration SHA256
`dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f`.
The local output directories are:

- `artifacts/evaluation/fullcontext-b/epoch-{1,2,3}`
- `artifacts/evaluation/window8-b/epoch-1`

Checkpoints came from
[FullContext-B](https://huggingface.co/anhdao69/SimpleMemVLN-R2R-FullContext-B)
at immutable revision `73d866c31e398596a67cd8be6f7dbfd25ab9241d` and
[Window8-B](https://huggingface.co/anhdao69/SimpleMemVLN-R2R-Window8-B)
at `cd17ac0d434f3df908ca1401aa11e86e1a125d40`. Published checkpoint
SHA256 entries were verified locally. Both use the pinned Qwen3.5-4B base.
The evaluation source and environment are described in
`reports/R2R_EVALUATION_2026-09-29.md`.

## Speed and historical context

The separately measured, **isolated** 64-action inference medians were
91.88 / 94.39 / 96.50 ms per action for FullContext epochs 1–3 and
84.57 ms for Window8 epoch 1. Window8 remained near 84 ms per action through
a 500-action repeated-frame replay while FullContext grew with retained KV.
These controlled repeated-frame timings are detailed in
`reports/R2R_INFERENCE_SPEED_2026-09-30.md`; they are not Habitat rollout
latency or navigation scores. The simultaneous four-worker Habitat run shares
one GPU and its per-action latency is affected by contention and route length.

For orientation, earlier local StageVLN-v2 full-split summaries recorded v0
Uniform8 at 43.88% SR / 39.12% SPL and v1 SlidingWindow4 at 35.62% SR /
30.18% SPL, each over 1,839 `val_unseen` episodes. Those policies have
different checkpoints, prompts/history implementations, and evaluation code;
this report does not treat their score gaps as controlled architecture effects.

The training collector's capture-before-action timing was not independently
reconstructed from its original source or replay. The evaluation loop itself
uses observation before action. No official test-set result is claimed here.
