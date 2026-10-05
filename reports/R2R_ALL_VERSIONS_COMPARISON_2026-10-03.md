# R2R `val_unseen`: all evaluated StageVLN and SimpleMemVLN versions

Updated 2026-10-05 from the local StageVLN-v2 and SimpleMemVLN evaluation journals, checkpoint metadata, training reports, and per-action timing files. The completed-results table contains only full-split R2R evaluations; a separately timestamped section records the ongoing Window8 candidate epoch-2 run. It does not use training loss, one-episode smoke runs, or partial RxR runs as navigation scores.

## What is comparable

All completed R2R quality rows below were checked against the same set of **1,839 R2R-VLNCE `val_unseen` episode IDs**. The evaluation dataset gzip has SHA256 `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`; the Habitat config has SHA256 `dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f`. The Stage path and Janus path hold byte-identical dataset files; the scene directory used by Stage resolves to the Janus scene directory. Both evaluators use Habitat 0.2.4, 640×480 RGB at 79° HFOV, a 0.25 m forward action, 15° turns, a 3 m success radius, seed 42, and a 500-action cap with a forced final STOP when needed. SR and SPL are final Habitat episode metrics. Oracle success is proximity at **any** visited point; navigation error is final distance to goal.

This is a **checkpoint comparison**, not a controlled architectural ablation. Training data, schedules, history representation, prompts, output policy, and model runtimes differ. Stage executes an invalid generated action as STOP; SimpleMem records a failed episode as zero SR/SPL. Stage recorded zero invalid outputs in these full runs; the joint SimpleMem text epoch-1 run had one failed episode. No official R2R test-set result is claimed.

## Version and training dataset

| Family / version | Decision-time history and output | Training data and checkpoint status |
|---|---|---|
| Stage v0 Uniform4 | Four uniformly selected past RGB frames plus current; fresh Qwen text generation each action; no recurrent memory | R2R only, 10,819 episodes / 631,244 labeled states; one-epoch export |
| Stage v0 Uniform8 | Eight uniformly selected past frames plus current; fresh text generation; no recurrent memory | Same R2R corpus; independently trained one-epoch export |
| Stage v1 Sliding4 | Four immediately preceding frames plus current; fresh text generation; no recurrent memory | Same R2R corpus; independently trained one-epoch export |
| Stage v2 Memory64/R0, epochs 1–2 | Learned 64-slot recurrent memory; current frame only in reader; fresh text generation | Same R2R corpus; two successive exports from a five-epoch training schedule |
| Stage v3 Memory64/R4 | Learned 64-slot memory plus four recent frames and current; fresh text generation | Same R2R corpus; one-epoch export |
| Stage v3 memory disabled | Same v3 weights and Recent4 reader, with recurrent memory disabled **only at inference** | Same v3 checkpoint; inference ablation, not separately trained |
| SimpleMem `streaming` FullContext-B, epochs 1–3 | Persistent append-only full-context KV/GDN state; greedy canonical text action and generated action feedback | R2R only, 10,819 episodes / 631,244 actions; successive snapshots at steps 1,353 / 2,706 / 4,059 of one three-epoch schedule |
| SimpleMem `streaming` Window8-B, epoch 1 | Instruction prefix plus latest eight complete observation/feedback groups in full-attention KV; persistent GDN state; text actions | Same R2R-only corpus, independently initialized; completed epoch 1 at step 1,353 of a three-epoch schedule; later checkpoints were unavailable for this evaluation |
| SimpleMem `streaming` joint FullContext-B, epochs 1–2 | FullContext streaming with text actions | Joint R2R (10,819) + English RxR_15deg (19,996) = 30,815 training episodes / 3,128,624 actions; successive snapshots at steps 3,852 / 7,704 of one two-epoch schedule |
| SimpleMem `streaming_logits` joint candidate, epochs 1–2 | FullContext streaming; score only pretrained LM rows A/B/C/D, choose argmax, append fixed candidate-token feedback; no generated action text | Same joint R2R + English RxR_15deg corpus; successive snapshots at steps 3,852 / 7,704 of one two-epoch schedule |
| SimpleMem joint Window8-B text, epoch 1 | Eight complete observation/action groups retained in full-attention KV; persistent GDN; generated text actions | Same joint corpus; completed first epoch at step 3,852 of a two-epoch schedule; R2R evaluation complete |
| SimpleMem `streaming_logits` joint Window8 candidate, epochs 1–2 | Four-way candidate logits; prefix plus eight observation/feedback groups in full-attention KV; persistent GDN; selected candidate token fed back | Same joint corpus; epoch 1 at update 3,852 and epoch 2 at 7,704 of a two-epoch schedule; epoch-1 R2R complete, epoch 2 running |
| SimpleMem `streaming_logits` joint FullContext candidate NoHistory, epoch 1 | Full visual context and persistent GDN; four-way candidate logits; no action-content feedback, only fixed assistant closure | Same joint corpus; completed **one-epoch** cosine schedule, update 3,852/3,852; 116 warmup updates, versus 232 for the two-epoch candidate models |

The candidate labels map A/B/C/D to forward/left/right/STOP. Stage's explicit frame histories re-encode selected frames for each decision; SimpleMem streams each new observation into persistent state. The SimpleMem Window8 policy evicts old **full-attention KV groups**, while GDN state and logical positions continue across the episode. These implementations are therefore different even when two rows mention “8” or “4.”

The older Window8 narrative called step 1,353 “intermediate.” Its saved `trainer_state.json` records `epoch: 1.0, global_step: 1353`: it is the **completed first epoch**, while still being an intermediate checkpoint in the planned three-epoch schedule. No Window8 epoch-2/3 result is inferred from that file.

### Exact inference contracts for the new setups

The three newly evaluated candidate checkpoints use the pinned Qwen3.5-4B base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, frozen vision encoder during training, trainable language backbone/tied LM rows, four-way cross entropy without class weighting, backbone LR 5e-6, weight decay 0.01, and training seed 429. Global episode batch is eight. Evaluation uses the checkpoint metadata without overriding its memory or feedback policy.

- **Window8 text:** autoregressively generates canonical action text; feeds the generated response back. Window8 counts the current observation among the eight retained groups.
- **Window8 candidate, epochs 1 and 2:** chooses argmax over A/B/C/D (`32/33/34/35`) and appends `[selected_id, 248046, 198]`. No action text is generated. Old observation/feedback groups are evicted together from full-attention KV; the instruction prefix stays.
- **FullContext candidate with action history:** same candidate decision/feedback contract, but all observation/feedback groups remain in full-attention KV.
- **FullContext candidate NoHistory:** chooses the same four-way action, then appends only `[248046, 198]` for every class. Neither canonical action text nor the selected candidate token is appended. **Visual history is retained.** This is not a stateless or current-frame-only policy.

All evaluated SimpleMem variants preserve advancing logical/MRoPE positions and FP32 GDN recurrent state across observations, resetting at episode boundaries. Model weights and primary computation use BF16, while recurrent state and rotary calculations preserve FP32. STOP is a model decision; the evaluator forces STOP only at the 500-action cap and records that separately.

## Completed R2R quality

All rows are full-split results over 1,839 episodes. Success counts were recomputed from the journals; every journal has 1,839 distinct IDs matching the shared set.

| Policy | Successes | SR ↑ | SPL ↑ | NE ↓ | Oracle success ↑ | Invalid / failed |
|---|---:|---:|---:|---:|---:|---:|
| Stage v0 Uniform4 | 717 | 38.99% | 35.43% | 7.12 m | 51.28% | 0 invalid |
| Stage v0 Uniform8 | 807 | 43.88% | 39.12% | 6.44 m | 53.40% | 0 invalid |
| Stage v1 Sliding4 | 655 | 35.62% | 30.18% | 7.40 m | 49.86% | 0 invalid |
| Stage v2 Memory64/R0 epoch 1 | 280 | 15.23% | 13.95% | 8.63 m | 22.51% | 0 invalid |
| Stage v2 Memory64/R0 epoch 2 | 271 | 14.74% | 12.96% | 9.50 m | 26.43% | 0 invalid |
| Stage v3 Memory64/R4 | 595 | 32.35% | 27.83% | 7.63 m | 47.15% | 0 invalid |
| Stage v3, memory disabled | 594 | 32.30% | 28.06% | 7.52 m | 47.47% | 0 invalid |
| SimpleMem R2R-only FullContext-B epoch 1 | 789 | 42.90% | 39.30% | 6.60 m | 52.53% | 0 failed |
| SimpleMem R2R-only FullContext-B epoch 2 | 750 | 40.78% | 36.38% | 6.81 m | 53.67% | 0 failed |
| SimpleMem R2R-only FullContext-B epoch 3 | 761 | 41.38% | 37.50% | 6.75 m | 54.92% | 0 failed |
| SimpleMem R2R-only Window8-B epoch 1 | 756 | 41.11% | 36.86% | 6.71 m | 54.21% | 0 failed |
| SimpleMem joint FullContext-B text epoch 1 | 891 | 48.45% | 44.81% | 6.09 m | 56.12% | 1 failed |
| **SimpleMem joint FullContext-B text epoch 2** | **916** | **49.81%** | **45.40%** | **5.99 m** | **60.58%** | 0 failed |
| SimpleMem joint FullContext candidate epoch 1 | 874 | 47.53% | 43.45% | 6.25 m | 56.17% | 0 failed |
| SimpleMem joint FullContext candidate epoch 2 | 882 | 47.96% | 44.01% | 6.26 m | 58.56% | 0 failed |
| SimpleMem joint Window8-B text epoch 1 | 907 | 49.32% | 45.56% | 6.04 m | 59.87% | 0 failed |
| SimpleMem joint Window8 candidate epoch 1 | 862 | 46.87% | 42.96% | 6.48 m | 54.59% | 0 failed |
| SimpleMem joint FullContext candidate NoHistory epoch 1 | 899 | 48.89% | 45.30% | 6.00 m | 57.53% | 0 failed |

The highest observed SR in this set is joint FullContext-B text epoch 2 at **49.81%**. Candidate-logit epoch 2 improves on its own epoch 1 by eight successes (**0.44 percentage points**) but remains **1.85 points below** joint text epoch 2. This is a checkpoint comparison across independently trained policies, not an isolated effect of the output format. Stage v3 with memory disabled differs from enabled v3 by only one success; that single inference toggle does not support a claim that memory improved this trained policy. The R2R-only FullContext epoch-1 snapshot is the best of its three R2R-only epochs; training longer did not improve this validation SR.

### What the completed comparisons support

- **NoHistory versus FullContext candidate with action history:** 899 versus 874 successes at epoch 1 (**+1.36 percentage points SR**, +1.85 points SPL), and 899 versus 882 at epoch 2 (**+0.92 points SR**, +1.29 points SPL). Deltas are computed from unrounded metrics; subtracting displayed rounded SR values can give 0.93 instead of 0.92. NoHistory has the higher measured full-split SR, but this does not isolate a causal benefit of removing action tokens: its one-epoch schedule ends at update 3,852, while the with-history epoch-1 checkpoint is halfway through a two-epoch schedule. There are no repeated training seeds or significance claims here.
- **Joint Window8 text versus joint FullContext text, epoch 1:** 907 versus 891 successes, **+0.87 points SR**. Window8 text is nine successes below the best joint FullContext text epoch 2 (**−0.49 points SR**), while its SPL is slightly higher: 45.56% versus 45.40%.
- **Joint Window8 candidate versus FullContext candidate, epoch 1:** 862 versus 874 successes, **−0.65 points SR**. Window8 candidate versus Window8 text is **−2.45 points SR** (862 versus 907), across independently trained output policies.

| Completed joint policy | Model-predicted STOP episodes | Forced STOP at cap | Mean actions/episode |
|---|---:|---:|---:|
| FullContext candidate epoch 1 | 1676 | 163 | 102.51 |
| FullContext candidate epoch 2 | 1687 | 152 | 99.18 |
| SimpleMem joint Window8-B text epoch 1 | 1627 | 212 | 115.39 |
| SimpleMem joint Window8 candidate epoch 1 | 1654 | 185 | 107.21 |
| SimpleMem joint FullContext candidate NoHistory epoch 1 | 1708 | 131 | 92.09 |

These rows have zero evaluation failures. Forced STOP is part of the shared protocol, not a model-predicted STOP or an evaluation failure.

## Controlled inference speed

Each row below uses one model process at a time on the same otherwise idle 96 GiB NVIDIA Blackwell GPU (compute capability 12.0), BF16, Torch 2.10.0+cu129, four CPU threads and the expandable CUDA allocator. The input is the **same real 640×480 Habitat RGB frame**, SHA256 `e69cf147c19881da0e08cc69fe0204dc5de0f13fce8579eae5ecd78850558e27`, and the same instruction: “Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait in the entrance to the other room.” Each short probe resets once, receives 16 untimed warmup observations, then receives **64 timed repeated-frame observations** while accumulating its **own** history. CUDA is synchronized around each full decision. The measured boundary includes preprocessing, vision, memory/history work, language computation and action decoding or selection. It excludes loading, reset, Habitat simulation and cross-process IPC. No gold action history is supplied. All measured decisions were valid.

The October 5 refresh paused the only active R2R rollout at a committed episode boundary and verified that no model compute processes remained before benchmarking. Each probe ran in its own process, one model at a time; the rollout was then resumed from its journal. The three new candidate rows pool **three independent 16-warmup + 64-measured-action runs** (192 timed actions per row); their pooled median, mean and p95 use all three runs, without selecting the fastest repetition. The refreshed Window8 text and FullContext candidate controls each use one new 64-action run. Other rows retain the earlier isolated measurements, whose per-action medians and means were recomputed from the saved records on October 5. Thus this is a complete coverage table, not a claim that every older checkpoint was rerun today.

| Policy | Median ms/action ↓ | Mean | p95 | Generated tokens/action |
|---|---:|---:|---:|---:|
| Stage v0 Uniform4 | 236.81 | 238.22 | 247.99 | 3 |
| Stage v0 Uniform8 | 351.79 | 352.95 | 366.41 | 3 |
| Stage v1 Sliding4 | 237.98 | 238.04 | 238.50 | 3 |
| Stage v2 Memory64/R0 epoch 1 | 145.18 | 145.17 | 145.55 | 3 |
| Stage v2 Memory64/R0 epoch 2 | 145.24 | 145.27 | 145.69 | 3 |
| Stage v3 Memory64/R4 | 242.05 | 242.06 | 242.58 | 3 |
| Stage v3, memory disabled | 236.75 | 236.80 | 237.27 | 3 |
| SimpleMem R2R-only FullContext-B epoch 1 | 91.88 | 92.11 | 95.25 | 3 |
| SimpleMem R2R-only FullContext-B epoch 2 | 94.39 | 94.12 | 97.29 | 3 |
| SimpleMem R2R-only FullContext-B epoch 3 | 96.50 | 95.38 | 101.44 | 3 |
| SimpleMem R2R-only Window8-B epoch 1 | 84.57 | 84.67 | 85.28 | 3 |
| SimpleMem joint FullContext-B text epoch 1 | 95.55 | 95.44 | 98.38 | 3 |
| SimpleMem joint FullContext-B text epoch 2 | 94.69 | 94.58 | 97.18 | 3 |
| SimpleMem joint FullContext candidate epoch 1 | 63.57 | 63.45 | 65.20 | 0 |
| SimpleMem joint FullContext candidate epoch 2 | 63.14 | 63.02 | 64.90 | 0 |
| SimpleMem joint Window8-B text epoch 1 | 84.31 | 85.72 | 91.21 | 3 |
| SimpleMem joint Window8 candidate epoch 1 | 57.09 | 57.15 | 57.52 | 0 |
| SimpleMem joint Window8 candidate epoch 2 (R2R partial) | 57.63 | 58.99 | 61.18 | 0 |
| SimpleMem joint FullContext candidate NoHistory epoch 1 | 62.75 | 62.79 | 67.36 | 0 |

The matched **joint** epoch-1 candidate is 1.50× faster at the median than joint epoch-1 FullContext text (63.57 versus 95.55 ms/action). Its four-row logit projection alone averages about 0.046 ms, but the image encoder, language append and deterministic feedback append still run. This is whole-policy serving latency, not an isolated causal measurement of the output head. Stage uses Transformers 5.3.0 and fresh reader generation; SimpleMem uses Transformers 5.11.0 and persistent streaming. Their per-action figures describe these implementations on this machine and should not be labeled architecture-only speedups.

FullContext latency rises with history length. In a separate 500-measured-action repeated-frame replay after the same 16 warmups:

| Policy | Median ms/action | p95 | Last 100-action bucket median |
|---|---:|---:|---:|
| SimpleMem R2R-only FullContext-B epoch 1 | 120.44 | 157.04 | 152.81 |
| SimpleMem R2R-only Window8-B epoch 1 | 84.36 | 90.25 | 84.62 |
| SimpleMem joint FullContext-B text epoch 1 | 120.47 | 156.07 | 152.53 |
| SimpleMem joint FullContext candidate epoch 1 | 83.53 | 108.98 | 105.74 |
| SimpleMem joint FullContext candidate epoch 2 | 86.08 | 108.71 | 107.23 |
| SimpleMem joint Window8-B text epoch 1 | 85.00 | 91.65 | 85.09 |
| SimpleMem joint Window8 candidate epoch 1 | 57.44 | 60.52 | 57.27 |
| SimpleMem joint Window8 candidate epoch 2 (R2R partial) | 57.54 | 60.73 | 57.68 |
| SimpleMem joint FullContext candidate NoHistory epoch 1 | 84.63 | 111.20 | 108.18 |

The historical Window8 and FullContext candidate rows are different checkpoints and training policies. FullContext candidate KV grows; Window8 bounds full-attention KV, explaining its flatter long-horizon timing. The joint Window8 text checkpoint retained at most 2,618 KV tokens in these timing runs; FullContext candidate epoch 2 reached 165,218 at 516 total steps. Three clean 64-action repetitions for joint Window8 text had medians **90.79, 88.29, and 84.22 ms**; the refreshed short-run table uses the new 84.31 ms control measurement. This historical spread matters when comparing small speed differences. These repeated-frame runs are latency probes, **not navigation rollouts or SR measurements**. Stage variants do not have a matched 500-action probe here.

### Repeat spread and recorded rollout latency

Independent short-run medians, in ms/action:

| Newly measured policy | Repetition 1 / 2 / 3 |
|---|---:|
| SimpleMem joint Window8 candidate epoch 1 | 56.95 / 57.09 / 57.31 |
| SimpleMem joint Window8 candidate epoch 2 (R2R partial) | 60.80 / 57.27 / 60.25 |
| SimpleMem joint FullContext candidate NoHistory epoch 1 | 62.16 / 62.15 / 63.40 |

With the refreshed controls, Window8 candidate epoch 1 is **1.48×** faster than Window8 text on this short probe. NoHistory's short-run median is close to the FullContext candidate controls; these few repeats do not establish a general speed advantage. On the 500-action probe, FullContext NoHistory latency grows with KV history, while Window8 remains approximately flat. Checkpoint quality and inference speed remain separate measurements.

Completed-rollout model latencies were recorded under changing GPU contention and actual, variable-length trajectories. They are provided for operational context only and **must not be used as controlled speed comparisons**:

| Completed policy | Model p50 / mean / p95 ms/action | Model + IPC p50 ms/action | Peak retained KV tokens |
|---|---:|---:|---:|
| Joint Window8 text epoch 1 | 203.58 / 184.79 / 291.41 | 209.72 | 2,718 |
| Joint Window8 candidate epoch 1 | 178.93 / 179.44 / 234.93 | 184.98 | 2,758 |
| Joint FullContext NoHistory epoch 1 | 200.00 / 205.01 / 297.62 | 206.17 | 159,648 |

October 5 raw measurements and exact commands are in `artifacts/inference_speed/refresh-20261005/` (`manifest.json`, 15 measurement JSONs and their logs). Each new candidate has `*-64-r{1,2,3}.json` and `*-500-r1.json`; the controls have `*-64-r1.json`. All 15 probes completed with valid decisions. The 500-action measurements each use a separate reset and 16 warmup observations; they are not pooled with the short runs.

## Unfinished or unmeasured variants

- Joint text-policy RxR_15deg evaluations are **partial**: epoch 1 stopped at 803/3,669 (43.21% partial SR), and epoch 2 stopped at 278/3,669 (40.29% partial SR, one failure). These are excluded from the R2R quality table and are not final RxR scores.
- Joint Window8 text epoch 1, Window8 candidate epoch 1, and FullContext NoHistory epoch 1 are now **complete** and included above. Window8 candidate epoch 2 remains partial; see the timestamped snapshot below.
- The new `streaming_logits_dual` branch implements an additional associative step lane alongside native GDN. Its 2026-10-04 implementation report describes joint adaptation from the NoHistory parent, numerical gates, and server timing probes. No full R2R journal or matched Blackwell policy benchmark for that trained variant is available in this local evaluation set; no SR or comparable speed is assigned to it.
- Candidate controls (`canonical_action_text` feedback, frozen LM rows, copied linear head, square-root/effective-number class weights) are implemented configurations, not completed full-training and Habitat-evaluation rows. No SR or speed is imputed to them.

## Ongoing R2R evaluation snapshot

At **2026-10-05 07:13:07 Asia/Ho_Chi_Minh**, Window8 candidate epoch 2 had **920/1,839 episodes**, **427 successes**, **46.41% partial SR**, **41.34% partial SPL**, and **0 evaluation failures**. The rollout was paused at 889 completed episodes for isolated speed measurements, then resumed with the identical checkpoint, source and runtime contracts; the journal has advanced beyond that point. This row is not a final score and is excluded from completed-model rankings.

## Evidence and limits

Candidate FullContext epoch 2 came from immutable Hub revision `a283921eb72713cee1a591febe8e1db4f5c0337b`; all eight epoch-2 SHA256 manifest entries passed. Its completed journal has 1,839 unique episode IDs, the same order, source hashes, dataset and Habitat configuration as candidate epoch 1; only checkpoint path and weight hash differ. The full Stage summaries and journals are under `StageVLN-v2/evaluation/r2r_v*/`; SimpleMem summaries, journals, contracts and runtime metadata are under `SimpleMemVLN/artifacts/evaluation/`. The controlled per-action JSON files are under `SimpleMemVLN/artifacts/inference_speed/`. The earlier Stage v0 Uniform4/v2/v3 timings were measured with their exported policy sessions and the same 16+64 timing boundary; clean repeat files for Uniform8 and Sliding4 were used after an overlapping-GPU run was discarded. SimpleMem timings use `scripts/vln/benchmark_inference.py`. The raw evaluation and timing artifacts are local and are **not** included in this Git repository.

Earlier SimpleMem evaluation, speed, and training reports are preserved under `reports/archive/` on the `streaming` and `streaming_logits` branches. The StageVLN-v2 `v2_r2r_evaluation.md` and checkpoint training metadata provide the Stage provenance. Those older reports describe their own observation dates; the completed rows above were checked from the current local artifacts.

### New checkpoint and execution provenance

| Setup | Hugging Face repository under `anhdao69/` | Immutable download revision | Evaluated checkpoint / local journal |
|---|---|---|---|
| Joint Window8 text | `SimpleMemVLN-R2R-RxR15deg-Window8-B` | `3c17fdd7519742d4510fb1e2f6c3151a2bb9ae67` | `epoch-1`; `window8-rxr15-text-r2r/epoch-1` |
| Joint Window8 candidate | `SimpleMemVLN-R2R-RxR15deg-Window8-CandidateLogits` | `04d3b2264a532a8ad7f778a3e23391fa12f6c98f` | `epoch-1`; `window8-candidate-r2r/epoch-1` |
| Joint Window8 candidate | `SimpleMemVLN-R2R-RxR15deg-Window8-CandidateLogits` | `4e1b922fb7d243a01170d963d4b675a0483bab72` | `epoch-2`; `window8-candidate-r2r/epoch-2` |
| Joint FullContext candidate NoHistory | `SimpleMemVLN-R2R-RxR15deg-FullContext-CandidateLogits-NoHistory` | `40de30d7febd215849f6bf56679eb1a11a47a073` | `epoch-1`; `fullcontext-candidate-nohistory-r2r/epoch-1` |

All downloaded manifest entries were verified before evaluation. The three newly completed journals contain the same 1,839 unique official episode IDs in the same deterministic order, with matching dataset, Habitat configuration, base assets, seed, action cap, failure policy and lossless transport. The new candidate evaluations use source commit `9786b15a9889712caee8bf9ab16a523a3b5b546d`, based on `streaming_logits` commit `f194e238aa9cab5d8271791a62630cdb9376d7cf`; the latest NoHistory serializer was retained. Its validation suite passed 115 tests with four GPU/training-specific tests skipped, followed by real-checkpoint Habitat smoke tests. Candidate epoch-1/2 Window8 evaluation contracts match except for checkpoint path and weight hash.

Weight SHA256 values:

- Joint Window8 text epoch 1: `e5c81d168326d59e6b48321fbd73da713b280a92dc2fcd0865e15aa8382f281e`.
- Joint Window8 candidate epoch 1: `4f481c576ae0253679820bec257238cda75c72a1dfcb25c9c0bab880ed3f6605`.
- Joint Window8 candidate epoch 2: `fb5aec690ddb6419ad5656efa2c36bdefdb933c086683280dc0c91712fe894d1`.
- Joint FullContext candidate NoHistory epoch 1: `6d492d012a872193adefb002057ea4c4e42a28301ffae1d1177c200e6a8b136d`.
