# R2R `val_unseen`: all evaluated StageVLN and SimpleMemVLN versions

Verified 2026-10-03 from the local StageVLN-v2 and SimpleMemVLN evaluation journals, checkpoint metadata, training reports, and per-action timing files. This report consolidates **completed** R2R results. It does not use training loss, one-episode smoke runs, or partial RxR runs as navigation scores.

## What is comparable

All 14 policies below were checked against the same set of **1,839 R2R-VLNCE `val_unseen` episode IDs**. The evaluation dataset gzip has SHA256 `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`; the Habitat config has SHA256 `dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f`. The Stage path and Janus path hold byte-identical dataset files; the scene directory used by Stage resolves to the Janus scene directory. Both evaluators use Habitat 0.2.4, 640×480 RGB at 79° HFOV, a 0.25 m forward action, 15° turns, a 3 m success radius, seed 42, and a 500-action cap with a forced final STOP when needed. SR and SPL are final Habitat episode metrics. Oracle success is proximity at **any** visited point; navigation error is final distance to goal.

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
| SimpleMem `streaming_logits` joint candidate, epoch 1 | FullContext streaming; score only pretrained LM rows A/B/C/D, choose argmax, append fixed candidate-token feedback; no generated action text | Same joint R2R + English RxR_15deg corpus; epoch-1 snapshot at step 3,852 of a two-epoch schedule |

The candidate labels map A/B/C/D to forward/left/right/STOP. Stage's explicit frame histories re-encode selected frames for each decision; SimpleMem streams each new observation into persistent state. The SimpleMem Window8 policy evicts old **full-attention KV groups**, while GDN state and logical positions continue across the episode. These implementations are therefore different even when two rows mention “8” or “4.”

The older Window8 narrative called step 1,353 “intermediate.” Its saved `trainer_state.json` records `epoch: 1.0, global_step: 1353`: it is the **completed first epoch**, while still being an intermediate checkpoint in the planned three-epoch schedule. No Window8 epoch-2/3 result is inferred from that file.

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

The highest observed SR in this set is joint FullContext-B text epoch 2 at **49.81%**. Candidate-logit epoch 1 is **2.28 percentage points below** it, and **0.92 points below** the joint text epoch-1 snapshot. Its faster action selection therefore has a measured quality cost at these checkpoints. Stage v3 with memory disabled differs from enabled v3 by only one success; that single inference toggle does not support a claim that memory improved this trained policy. The R2R-only FullContext epoch-1 snapshot is the best of its three R2R-only epochs; training longer did not improve this validation SR.

## Controlled inference speed

Each row below uses one model process at a time on the same otherwise idle 96 GiB NVIDIA Blackwell GPU (compute capability 12.0), BF16, Torch 2.10.0+cu129, four CPU threads and the expandable CUDA allocator. The input is the **same real 640×480 Habitat RGB frame**, SHA256 `e69cf147c19881da0e08cc69fe0204dc5de0f13fce8579eae5ecd78850558e27`, and the same instruction: “Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait in the entrance to the other room.” Each policy resets once, receives 16 untimed warmup observations, then receives **64 timed repeated-frame observations** while accumulating its **own** history. CUDA is synchronized around each full decision. The measured boundary includes preprocessing, vision, memory/history work, language computation and action decoding or selection. It excludes loading, reset, Habitat simulation and cross-process IPC. No gold action history is supplied. All measured decisions were valid.

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
| **SimpleMem joint FullContext candidate epoch 1** | **62.70** | **63.86** | **68.63** | **0** |

The matched **joint** epoch-1 candidate is 1.52× faster at the median than joint epoch-1 text (62.70 versus 95.55 ms/action). Its four-row logit projection alone averages about 0.046 ms, but the image encoder, language append and deterministic feedback append still run. This is whole-policy serving latency, not an isolated causal measurement of the output head. Stage uses Transformers 5.3.0 and fresh reader generation; SimpleMem uses Transformers 5.11.0 and persistent streaming. Their per-action figures describe these implementations on this machine and should not be labeled architecture-only speedups.

FullContext latency rises with history length. In a separate 500-measured-action repeated-frame replay after the same 16 warmups:

| Policy | Median ms/action | p95 | Last 100-action bucket median |
|---|---:|---:|---:|
| SimpleMem R2R-only FullContext-B epoch 1 | 120.44 | 157.04 | 152.81 |
| SimpleMem R2R-only Window8-B epoch 1 | 84.36 | 90.25 | 84.62 |
| SimpleMem joint FullContext-B text epoch 1 | 120.47 | 156.07 | 152.53 |
| SimpleMem joint FullContext candidate epoch 1 | 83.53 | 108.98 | 105.74 |

The Window8 and candidate rows are different checkpoints and training policies. The candidate remains FullContext and its KV grows; Window8 bounds full-attention KV, explaining its flatter long-horizon timing. These repeated-frame runs are latency probes, **not navigation rollouts or SR measurements**. Stage variants do not have a matched 500-action probe here.

## Unfinished or unmeasured variants

- Joint text-policy RxR_15deg evaluations are **partial**: epoch 1 stopped at 803/3,669 (43.21% partial SR), and epoch 2 stopped at 278/3,669 (40.29% partial SR, one failure). These are excluded from the R2R quality table and are not final RxR scores.
- Joint Window8 text training and joint Window8 candidate training have submission or startup-gate reports, but no completed R2R `val_unseen` checkpoint evaluation is available here. The Window8 candidate job is dependent on the FullContext candidate job; queued training is not a result.
- Candidate controls (`canonical_action_text` feedback, frozen LM rows, copied linear head, square-root/effective-number class weights) are implemented configurations, not completed full-training and Habitat-evaluation rows. No SR or speed is imputed to them.

## Evidence and limits

The full Stage summaries and journals are under `StageVLN-v2/evaluation/r2r_v*/`; SimpleMem summaries, journals, contracts and runtime metadata are under `SimpleMemVLN/artifacts/evaluation/`. The controlled per-action JSON files are under `SimpleMemVLN/artifacts/inference_speed/`. The Stage v0 Uniform4/v2/v3 timings were newly measured with their exported policy sessions and the same 16+64 timing boundary; clean repeat files for Uniform8 and Sliding4 were used after an overlapping-GPU run was discarded. SimpleMem timings use `scripts/vln/benchmark_inference.py`. The raw evaluation and timing artifacts are local and are **not** included in this Git repository.

Further recipe and provenance details are in the SimpleMem `streaming` / `streaming_logits` reports `R2R_VAL_UNSEEN_RESULTS_2026-10-01.md`, `R2R_INFERENCE_SPEED_2026-09-30.md`, and `CANDIDATE_R2R_EVALUATION_2026-10-03.md`, and the StageVLN-v2 report `v2_r2r_evaluation.md` plus checkpoint training metadata. Those older reports describe their own observation dates; the completed rows above were checked from the current local artifacts.
