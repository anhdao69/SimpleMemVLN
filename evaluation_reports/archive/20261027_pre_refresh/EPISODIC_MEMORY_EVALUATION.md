# Episodic-memory campaign — R2R evaluation report

## L2/L3 latency and SR update — 2026-10-25

| Model | Median ms | p95 ms | Mean ms | Actions/s | R2R successes / completed | SR | SPL | Navigation status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| L2 | 109.716 | 115.079 | 110.094 | 9.083 | 678 / 1384 | 48.99% | 45.42% | Interrupted; partial / 1,839 |
| L3 | 153.127 | 168.287 | 155.638 | 6.425 | 593 / 1199 | 49.46% | 46.41% | Interrupted; partial / 1,839 |

Measured 2026-10-25T10:00:28.687170+07:00 on the same RTX PRO6000 Blackwell Max-Q, driver580.178.04 and 300W limit. This is a **separate post-reboot session** using the unchanged original benchmark worker: batch1, the same 80 preloaded RGB frames/instruction, three trials of16 warm-up +64 timed actions (192 samples/model), CUDA-synchronized observe wall time; simulator, transport and disk reads excluded. No detected foreign GPU process. Native BF16 backbone and embedded learned rankers retained (feature preparation FP32, ranker FP64, priorities FP32). Warm-up/initialization excluded; this is not a cold-start benchmark or a proof of cross-session statistical superiority.

Navigation results were independently recomputed from saved journals with contract-order and source/config identity checks. The workstation rebooted at **2026-10-25 08:35:33 +07:00**; both evaluation processes are absent. These SRs are **partial, interrupted results**, not completed evaluations. L2 and L3 cover different-sized ordered prefixes, so their SRs are not a matched full-panel comparison. No navigation evaluation was restarted by this latency measurement.

[Raw benchmark evidence](/data14t/anhdh35/latency_l_20261025/manifest.json) · [Verification](evidence/L_LATENCY_SR_20261025_VERIFICATION.json) · [All 35 configurations: latency and SR CSV](evidence/ALL_LATENCY_SR_20261025.csv).


## Current verified results — 2026-10-24T21:21:47.211620+07:00

### S campaign and matched baselines

| Model | Attempted / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status |
|---|---:|---:|---:|---:|---:|---|
| M5_E1 R4/B1200 | 1,839 | 950 | 51.658510 | 48.401034 | 0 | FINAL — navigation validated |
| M5_E1 served B0 | 1,839 | 919 | 49.972811 | 45.991456 | 0 | FINAL — navigation validated |
| R4B0 trained | 1,839 | 896 | 48.722132 | 44.964877 | 0 | FINAL — navigation validated |
| S1 BF16 | 1,839 | 921 | 50.081566 | 46.539055 | 0 | FINAL — navigation validated |
| S1 FP32 | 1,839 | 917 | 49.864057 | 46.432850 | 0 | FINAL — navigation validated |
| S2 | 1,839 | 926 | 50.353453 | 46.896355 | 0 | FINAL — navigation validated |
| S3 | 1,839 | 928 | 50.462208 | 47.152476 | 0 | FINAL — navigation validated |
| S4 | 1,839 | 917 | 49.864057 | 46.671319 | 0 | FINAL — navigation validated |
| S5 | 1,839 | 901 | 48.994018 | 45.378002 | 0 | FINAL — navigation validated |
| S6 | 1,839 | 929 | 50.516585 | 47.013465 | 0 | FINAL — unqualified diagnostic |
| S7 | 1,839 | 907 | 49.320283 | 45.529253 | 1 | ALL ATTEMPTED — final validation FAILED |
| S8N | 1,839 | 964 | 52.419793 | 48.679311 | 0 | FINAL — navigation validated |
| S8W | 1,839 | 948 | 51.549755 | 48.690482 | 0 | FINAL — navigation validated |
| S9 | 1,839 | 932 | 50.679717 | 47.439929 | 0 | FINAL — navigation validated |

S7 has one invalid-response failure (`MOVE`, episode672), so its observed full-panel
SR/SPL are failure-adjusted and final navigation validation failed. S6 remains an
unqualified diagnostic. S1 BF16, S4, S5 and M5_E1 B1200 retain numerical-replay
failures despite passing navigation validation. All other listed S-campaign
navigation results have zero execution failures and stored final validation passes.

[Complete S-campaign report](S_CAMPAIGN_R2R_EVALUATION.md) · [Evidence snapshot](evidence/ALL_RESULTS_20261024.json).

### Historical episodic checkpoints — refreshed from saved journals

| Model | Attempted / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status |
|---|---:|---:|---:|---:|---:|---|
| V0 | 1,839 | 852 | 46.329527 | 41.756802 | 0 | COMPLETE — recomputed |
| M1 | 1,839 | 799 | 43.447526 | 40.204753 | 0 | COMPLETE — recomputed |
| M2 | 1,839 | 906 | 49.265905 | 45.877272 | 0 | COMPLETE — recomputed |
| M2c | 1,839 | 848 | 46.112017 | 43.365987 | 0 | COMPLETE — recomputed |
| M5 | 1,839 | 938 | 51.005982 | 47.438938 | 0 | COMPLETE — recomputed |
| M5c | 1,839 | 877 | 47.688961 | 43.944998 | 0 | COMPLETE — recomputed |
| Window8-B-E1 | 1,839 | 907 | 49.320283 | 45.561094 | 0 | COMPLETE — recomputed |
| M5 E1 R2/B1800 seed431 | 1,839 | 944 | 51.332246 | 47.525482 | 0 | FINAL — navigation validated |

These are distinct checkpoint/source identities from the matched S-campaign
baselines. M1 and M5c are now complete (799 and877 successes); historical partial
tables below are retained as dated evidence. “COMPLETE — recomputed” means this
refresh verified full unique coverage, recorded action metadata and recomputed
metrics against saved summaries; it does not assert an unavailable final-validator
file or new numerical qualification. No reruns were launched.

## Archived reports and provenance

The sections below preserve earlier reports. Any older running/partial counts or
“next update” instructions are superseded by the current tables above. Existing
qualification caveats and checkpoint identities still apply.

## Final M5 E1 R2/B1800 seed431 — 2026-10-21

Full R2R VLN-CE val_unseen evaluation completed on 2026-10-21. Final validation
recomputed metrics from the complete journal and verified unique episode IDs,
contract order, checkpoint hashes and action metadata.

| Model | Successes / episodes | SR (%) | SPL (%) | NE (m) | Oracle success (%) | State |
|---|---:|---:|---:|---:|---:|---|
| M5 E1 R2/B1800 seed431 | 944 / 1,839 | 51.332246 | 47.525482 | 5.726252 | 58.945079 | FINAL |

Zero execution failures and zero invalid responses. Mean episode length:
104.2447 actions; mean trajectory length: 9.6024 m; forced STOPs: 175.
Peak KV occupancy: 2,610 rows against the 4,616 structural guard;
peak allocated GPU memory: 8.6931 GiB.

This is the separately trained **seed431, R2/B1800, one-epoch-schedule** model,
not the historical P2-M5 seed429 R4/B1200 export. It retains two recent groups
including current and 1,800 episodic visual tokens per attention layer,
committed_mean_l2_v1 scoring, text action history and the saved serving contract.
The inactive selector seed is 429. Training completed 3,852 updates with 116 warmup
updates. No teacher, auxiliary heads, PCA or step lanes are enabled.

Checkpoint: `anhdao69/Streaming_VGGT/M5_E1_R2_B1800_seed431/epoch-1`.
Pinned HF revision: `e1d614ab3eb5f2737dbe27de69906b05bea372e3`.
Weights SHA256: `5ed04b21391ce8a92c070cdd53ab4e3b0be56b30952302f7c999daadd766c76c`.
Source branch: `c2_vggt_omega_g3_m5_layers`, frozen commit
`fde1cbef213e27320920400a826addea8d5e114a`; src/config trees match training revision
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`.

Qualification: all 18 published artifact hashes verified, 724 finite backbone
tensors, strict GPU load, 59 targeted tests, native scorer check, 64/128-step
replays, reset/retry checks and 520-step bounded stream passed. The 128-step replay
had one raw next-token disagreement out of 383 but passed the unchanged
atol0.75 + rtol0.03 and RMS<=0.15 criteria; passing does not imply bitwise or greedy
trajectory equivalence. Full evaluation used Habitat-Lab/Sim0.2.4, seed42,
500-action cap, 3m success radius, 640x480 RGB, 79-degree HFOV, 0.25m forward
steps and 15-degree turns. Final journal validation passed.

Run root: `/data14t/anhdh35/m5_e1_r2_b1800_seed431_epoch1/evaluation/`.
Results: `results/full/M5_R2_B1800_s431/{episodes.jsonl,summary.json}`.
Validation: `full_validation.json`; qualification: `audit/M5/passed.json`.
This additional result does not replace historical checkpoint identities or
establish a causal memory-budget/seed advantage.

Previous seven-model R2R val_unseen refresh:  all seven episodic/reference rows below have completed the full 1,839 episodes. **M5 has the highest SR (51.005982%) among these runs.** M1 and M5 c are no longer partial. The result table and status statements in the archived 16 October snapshot later in this document are superseded by this refresh.

## Previous verified refresh — 2026-10-20 03:26 UTC

All listed summaries report 1,839/1,839 episodes, zero execution failures and zero invalid responses. M1 and M5 c now have complete evaluation journals and final metric checks; the earlier partial-prefix rows below are preserved only as historical evidence.

### Current results

| Version | Evaluated / scheduled | Successes | SR (%) | SPL (%) | State |
|---|---:|---:|---:|---:|---|
| V0 | 1,839 / 1,839 | 852 | 46.329527 | 41.756802 | FINAL |
| M1 | 1,839 / 1,839 | 799 | 43.447526 | 40.204753 | FINAL |
| M2 | 1,839 / 1,839 | 906 | 49.265905 | 45.877272 | FINAL |
| M2 c | 1,839 / 1,839 | 848 | 46.112017 | 43.365987 | FINAL |
| M5 | 1,839 / 1,839 | 938 | 51.005982 | 47.438938 | FINAL |
| M5 c | 1,839 / 1,839 | 877 | 47.688961 | 43.944998 | FINAL |
| Window8-B | 1,839 / 1,839 | 907 | 49.320283 | 45.561094 | FINAL |

### Current navigation and memory measurements

| Version | NE (m) | Oracle success (%) | Mean steps | Forced STOPs | Peak KV rows | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 6.4879 | 59.3801 | 126.32 | 240 | 2718 | 8.668 |
| M1 | 6.7313 | 53.9424 | 117.23 | 222 | 3036 | 8.679 |
| M2 | 5.7585 | 56.5525 | 102.24 | 176 | 3018 | 8.705 |
| M2 c | 6.0642 | 52.9636 | 91.58 | 139 | 3018 | 8.678 |
| M5 | 5.8706 | 57.8032 | 100.03 | 159 | 2646 | 8.693 |
| M5 c | 6.2144 | 56.4981 | 102.76 | 166 | 2646 | 8.669 |
| Window8-B | 6.0358 | 59.8695 | 115.39 | 212 | 2718 | 8.668 |

The current completed M5−M5 c aggregate difference is **+3.317021 percentage points SR** and **+3.493940 percentage points SPL**; M2−M2 c remains +3.153888 pp SR and +2.511285 pp SPL. These are descriptive comparisons of the evaluated checkpoints, not statistical significance claims. The full runs make the controls available for the planned paired uncertainty analysis; they do not replace that analysis. No confidence interval, multi-seed robustness result, or final pre-registered gate conclusion is asserted here.

The full M1 result is **43.447526% SR / 40.204753% SPL**; M5 c is **47.688961% / 43.944998%**. M5 remains **51.005982% / 47.438938%**. Checkpoint artifacts and campaign-specific provenance remain at `/data14t/anhdh35/simplememvln_m1_eval`, `/data14t/anhdh35/simplememvln_m5c_eval`, and `/data14t/anhdh35/simplememvln_g3_m5_eval` respectively.

> **Historical detail note:** The material after the current refresh is the original 2026-10-16 snapshot, retained as an audit trail. Any result counts, ranks, or “running” labels in that archived material describe that date only; use the refresh above for current scores and status.


V0, M2, M2 c and M5 are complete. M1 and M5 c are running. M5 currently has the highest completed SR among the episodic arms; the unfinished controls prevent a final conclusion about the benefit of committed-write selection.

## Historical snapshot and scope (2026-10-16)

Snapshot collected through **2026-10-16 02:45:04 UTC / 2026-10-16 09:45:04 Asia/Ho_Chi_Minh**. Results were read sequentially and frozen at the counts below; running jobs may have advanced since collection. This Markdown file is a dated report, not a live dashboard. SR/SPL use percentages; changes use percentage points (pp).

All results here are **R2R VLN-CE val_unseen**, with **1,839 scheduled episodes**. Every published checkpoint evaluated here is epoch 1. Training used joint R2R/RxR data, but **no full RxR navigation evaluation result is available in this report**. RxR training/qualification fixtures are not RxR benchmark scores.

Completed and partial runs are explicitly separated. The partial denominator is the number of evaluated episodes shown, not 1,839; unevaluated episodes have not been assigned zero scores. Episode order groups scenes, so a partial prefix is not a random representative sample. Do not rank partial SR against final full-split SR.

## Shared evaluation protocol

| Item | Verified setting |
|---|---|
| Protocol | `R2R-VLNCE-0.2.4`, split `val_unseen` |
| Simulator | Habitat-Lab 0.2.4 / Habitat-Sim 0.2.4, Python 3.10.20 |
| Policy environment | Python 3.12.13, PyTorch 2.10.0+cu129, Transformers 5.11.0 |
| Attention stack | FlashAttention 2.8.3+cu.12.9.torch.2.10, FLA/fla-core 0.5.2, Triton 3.6.0 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB class |
| Base model | Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Observations | RGB 640×480, HFOV 79°; policy receives current RGB and instruction/history |
| Actions | Canonical text MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP; Habitat IDs 1,2,3,0 |
| Motion | Forward 0.25 m, turns 15° |
| Success | Habitat navigation success with STOP within the configured 3 m goal radius |
| Evaluation seed / cap | 42 / 500 actions, cap-ending forced STOP recorded separately |
| Dataset | `/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz` |
| Dataset SHA-256 | `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3` |
| Habitat config SHA-256 | `dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f` |
| Scenes | `/storage/anhdh35/JanusVLN/data/scene_datasets_full`; 11 evaluation scene assets checked |
| Failure policy | `zero_sr_spl_keep_partial_metrics`: failed episodes remain in SR/SPL denominator |

Frozen ordered episode IDs, dataset/config hashes, base-asset hashes, scenes, seed, action cap, protocol and failure policy match across all 11 runs in the reporting snapshot. The evaluator/stream modules were kept fixed across campaigns; model-specific architecture and memory contracts remain checkpoint-specific. Evaluation sources are isolated snapshots and were not switched under running processes.

**Definitions:** SR is the mean binary navigation success; SPL is mean success weighted by shortest-path versus executed-path efficiency. NE is final distance to goal. Oracle success records whether the trajectory ever reached the goal region; it does not mean the model stopped correctly. Forced STOP is an episode hitting the action cap, not an execution exception. nDTW is unavailable (`null`, zero coverage), not zero. NE/oracle coverage equals the evaluated count in these snapshots.

All listed runs have **0 execution failures and 0 invalid responses** at the snapshot. Navigation failures (SR=0) are distinct from execution failures. Timing is affected by concurrent GPU workloads; neither model latency nor wall time here establishes isolated speed superiority.

## Report verification and historical validator flags

The report independently reread the episode journals, checked unique episode IDs and exact ordered membership, and recomputed all saved summary fields. Integral fields and structure matched exactly; floating values matched with relative/absolute tolerance **1e-12**. This tolerance only handles aggregation/JSON rounding and does not alter navigation decisions or numerical replay gates.

Older V0/M2 c orchestration (`campaign_status.json`), M2 (`M2-status.json`) and G1 (`status.json`) retain failed final-validator flags from exact floating-point equality assertions. They are not evidence of missing episodes: all four journals contain 1,839 unique completed episodes and zero execution failures. This report freshly verified their metrics. Across all snapshot runs, SPL recomputation differences were at most 7.22e-16 in fraction units and mean-path differences at most 2.85e-14 m. Historical status files were preserved. Newer coordinators use the 1e-12 final-metric check.

The newer G3, M5, Window8-B and A0 coordinators report complete after their final contract/journal checks. A1, M5 c and M1 are still running at this snapshot. No partial run is certified as a final completed evaluation.

## Results: every evaluated episodic arm and native reference

| Version (epoch 1) | Completed / scheduled | Successful | SR (%) | SPL (%) | Status |
|---|---:|---:|---:|---:|---|
| V0 | 1,839 / 1,839 | 852 | 46.329527 | 41.756802 | FINAL |
| M1 | 31 / 1,839 | 13 | 41.935484 | 40.360377 | RUNNING — partial |
| M2 | 1,839 / 1,839 | 906 | 49.265905 | 45.877272 | FINAL |
| M2 c | 1,839 / 1,839 | 848 | 46.112017 | 43.365987 | FINAL |
| M5 | 1,839 / 1,839 | 938 | 51.005982 | 47.438938 | FINAL |
| M5 c | 1,224 / 1,839 | 583 | 47.630719 | 44.105767 | RUNNING — partial |
| Window8-B | 1,839 / 1,839 | 907 | 49.320283 | 45.561094 | FINAL |

The companion [VGGT report](VGGT_EVALUATION.md) records G1/G3/A0/A1. Window8-B is an independently trained historical reference evaluated here with the V0 source checkout; it is not a second V0 checkpoint.

### Additional navigation and memory measurements

| Version | NE (m) | Oracle success (%) | Mean steps | Forced STOPs | Peak retained KV rows | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 6.4879 | 59.3801 | 126.32 | 240 | 2718 | 8.668 |
| M1 | 7.1088 | 70.9677 | 74.81 | 0 | 2966 | 8.661 |
| M2 | 5.7585 | 56.5525 | 102.24 | 176 | 3018 | 8.705 |
| M2 c | 6.0642 | 52.9636 | 91.58 | 139 | 3018 | 8.678 |
| M5 | 5.8706 | 57.8032 | 100.03 | 159 | 2646 | 8.693 |
| M5 c | 6.5788 | 57.6797 | 107.75 | 125 | 2600 | 8.651 |
| Window8-B | 6.0358 | 59.8695 | 115.39 | 212 | 2718 | 8.668 |

Observed peak KV is not the configured structural guard. Partial runs may reach a higher peak later. Forced STOP counts remain in all reported metrics and are not execution failures.

## Arm definitions and the question each comparison asks

| Arm | Recent groups R (includes current) | Old visual tokens B per attention layer | Selector | KV guard | Question |
|---|---:|---:|---|---:|---|
| V0 | 8 | 0 | None, recent FIFO | 5,120 | Matched episodic-code baseline |
| M1 | 9 | 0 | None, recent FIFO | 5,504 | Does one extra recent complete group explain the gain? |
| M2 | 8 | 300 | `committed_mean_l2_v1` | 5,420 | Does importance-selected old context help? |
| M2 c | 8 | 300 | `random_hash_v1`, seed 429 | 5,420 | Is importance scoring better than generic old-context access? |
| M5 | 4 | 1,200 | `committed_mean_l2_v1` | 4,784 | Does redistributing recent versus older context help? |
| M5 c | 4 | 1,200 | `random_hash_v1`, seed 429 | 4,784 | Does scoring matter at M5's redistributed budget? |
| Window8-B | 8 | 0 | Legacy native Window8 | 4,096 | Historical independently trained checkpoint reference |

The episodic mode is `episodic_window_v1`; all six arms use canonical text actions and `vln_append_only_chat_v1`. There is no step lane, teacher, auxiliary loss, new model parameter or per-source cap in these arms. The bank stores native full-attention K/V rows, not GDN recurrent snapshots. All six episodic arms are independently trained from the original pretrained model recipe; M5 is not initialized from trained M2, and M5 c is not M5 weights with a runtime selector toggle.

### Visibility and cache behavior

Each of eight full-attention layers (3,7,11,15,19,23,27,31) sees the instruction prefix, its selected old visual rows, and R recent complete observation/action/framing groups including the current group. Within the current group, attention remains causal. Before observation t, source group t−R expires from recent history and its 300 visual rows become eligible. Older text/framing/action tokens are not bank candidates.

Physical order is prefix | selected old rows sorted by original token ID | recent groups. Keys keep their original rotary positions; eviction does not reset logical positions or re-encode history. No full-history K/V backup is retained. Native GDN recurrent and convolution states still process every original token and are not evicted by attention-window pruning. Reset clears session state; retries return the committed result without advancing the cache.

The structural guard is max_prefix2048 + R×max_group384 + B. M1 adds one complete group, totaling more than 300 tokens once text/framing is counted. M1 versus M2 is a near-budget comparison, not exact total KV equality. M5/M5 c have 4×300+1200=2400 visual rows at full occupancy, matching V0's visual count, but their total KV and runtime differ. Actual peak occupancy is reported above.

### Committed-write versus random controls

M2/M5 compute a detached FP32 committed-write score from each full-attention layer's immediately preceding GDN (pairs 2→3, 6→7, …, 30→31). The score is the mean over heads of the norm of the gated write residual, using the native normalized key and decayed prior state. The qualified helper is `fp32_torch_wy_block64_v1`, with shared prefix/observation/response segmentation. Priorities are fixed at write time. Running top-B selection breaks ties by original token ID; rejected old rows are discarded permanently.

M2 c/M5 c instead use the uint 64 prefix of SHA-256(version | selector_seed | episode_uid | layer | original_logical_token_id). Priorities are activation independent and reproducible for the same identity. No committed-write scorers are created. **The evaluator UID contains the absolute scene path**: moving scenes to a different path can change random selections even with seed 429. Preserve the recorded episode UIDs and scenes path when reproducing these controls; matching the integer episode ID alone is insufficient.

M1/V0 have B0 and no selector computation. M1 must remain R9/B0 in the generalized episodic runtime; converting it to legacy Window8 would change the model being evaluated.

## Training provenance and schedule caveats

| Arm | Published attempt/profile | World size / GAS | Epoch-1 update / configured total |
|---|---|---|---|
| V0 | P1-V0-s429-H2-a2 | 2 / 4 | 3,852 / 7,704 |
| M1 | P1-M1-s429-H2-a2 | 2 / 4 | 3,852 / 7,704 |
| M2 | P1-M2-s429-H2-a1 | 2 / 4 | 3,852 / 7,704 |
| M2 c | P1-M2 c-s429-H4-a2 | 4 / 2 | 3,852 / 7,704 |
| M5 | P2-M5-s429-H4-a2 | 4 / 2 | 3,852 / 7,704 |
| M5 c | P2-M5 c-s429-H4-a2 | 4 / 2 | 3,852 / 7,704 |

These are one-epoch exposures from a configured **two-epoch** LR schedule, warmup 232, seed 429, nominal global batch 8, navigation loss and full episode BPTT. They are not completed one-epoch cosine schedules like G1/G3/A0/A1. H2/H4 have equal nominal batch but different rank assignment/reduction order and execution profiles; do not claim bitwise matched training.

V0 and M2 have identical recorded training-manifest byte hashes and ordered 30,815 episode IDs. V0 versus M2 c has matching ordered IDs but different cross-server path-dependent manifest bytes; ID equality alone does not verify identical image bytes/labels. Generalizing cross-server equivalence beyond recorded checks would be unjustified.

M1's published README records updates 1–3142 on newton and a full-state checkpoint 3142 continuation on Bridges-2 for 3143–3852, with the same source/config/order. Its published resolved YAML and resume-contract navigation match the export exactly. Full optimizer recovery shards are not included in this M1 upload. This is provenance, not a local rerun of training recovery.

M5 c's published event log records interruption near 3251, continuation from complete checkpoint 3250 on the same H4 attempt, then epoch 1 completion. Use the published H4-a2 weights, not the older queued H4-a1 labels in historical reports.

M5/M5 c training was launched under an explicit user override of the planned Phase2 evidence gate. Their existence does not mean the pre-registered gate was satisfied. A final gate decision and scientific claim require complete controls and the planned uncertainty analysis.

## Descriptive paired comparisons

All pairs below use the intersection of completed episode IDs and verify matching UIDs/instructions. For unfinished M1/M5 c, the completed comparator is restricted to that exact prefix. The table is descriptive; no scene-bootstrap confidence interval, significance claim or confirmation-seed result is provided.

| A minus B | Shared episodes | A SR (%) | B SR (%) | ΔSR (pp) | ΔSPL (pp) | A-only / B-only successes | Scope |
|---|---:|---:|---:|---:|---:|---:|---|
| M2 − V0 | 1,839 | 49.2659 | 46.3295 | +2.9364 | +4.1205 | 300 / 246 | Full split |
| M2 c − V0 | 1,839 | 46.1120 | 46.3295 | -0.2175 | +1.6092 | 303 / 307 | Full split |
| M2 − M2 c | 1,839 | 49.2659 | 46.1120 | +3.1539 | +2.5113 | 319 / 261 | Full split |
| M5 − V0 | 1,839 | 51.0060 | 46.3295 | +4.6765 | +5.6821 | 339 / 253 | Full split |
| M5 − M2 | 1,839 | 51.0060 | 49.2659 | +1.7401 | +1.5617 | 293 / 261 | Full split |
| M1 − V0 | 31 | 41.9355 | 25.8065 | +16.1290 | +17.1025 | 7 / 2 | Partial shared prefix only |
| M5 c − M5 | 1,224 | 47.6307 | 51.7974 | -4.1667 | -3.9971 | 158 / 209 | Partial shared prefix only |
| Window8-B − V0 | 1,839 | 49.3203 | 46.3295 | +2.9908 | +3.8043 | 289 / 234 | Full split |

The complete M2−M2 c difference is compatible with a scoring advantage on these checkpoints, but it does not alone establish causality across different training topologies. M5's gain over V0 cannot distinguish budget redistribution from scoring; M5 c is the required matched-allocation control and is not finished. M1's small completed prefix cannot settle whether extra recent context explains M2. No second-seed confirmation or final pre-registered gate outcome is claimed.

## Local qualification and numerical limits

| Arm | Learned-checkpoint replay / selection evidence | Long-stream evidence |
|---|---|---|
| V0 | Actual smoke-history replay: 1/221 raw token flips; max drift 0.375, RMS about 0.01793; reset/retry pass | 520-step bounded-state gate passed |
| M2 c | 65-step smoke-history replay: zero raw token flips and zero selected-ID mismatches | 520-step bounded-state gate passed |
| M2 | 69 steps/206 tokens: raw argmax 100%; max drift 0.375, RMS 0.01949; mean/min selected overlap 98.9973%/96.6667% | 520 steps peak 2,948/5,420; 300 old rows/layer |
| M5 | 64/128 steps: max drift 0.53125/0.75, RMS 0.03434/0.03428; mean overlap 99.2635%/99.0379%, min 98.0833%/97.4167% | 520 steps peak 2,546/4,784; 1,200 old rows/layer |
| M5 c | 64/128 steps: 2/191 and 4/383 raw token flips; RMS 0.025351/0.025327; all 1,536 selected-ID comparisons exact | 520 steps peak 2,546/4,784; 1,200 old rows/layer |
| M1 | 64/128 steps: 2/191 and 2/383 raw token flips; RMS 0.019724/0.019313; exact prefix+R9 IDs every layer | 520 steps peak 2,936/5,504; zero old rows/scorers |
| Window8-B | 64/128 steps: 0/191 and 1/383 flips; RMS 0.020077/0.019563; exact retry/reset | 520 steps peak 2,618/4,096 |

Every run passed strict checkpoint loading, finite backbone tensor checks, real Habitat smoke execution and its recorded replay/storage gates before full evaluation. Each episodic serving state contains 724 finite native backbone tensors. M5/M5 c/M1 use the exact published training commit; V0/M2/M2 c use later c2_ablations code whose 47 model files match their published training source. M5's local targeted suite passed 69 tests; M5 c/M1 each passed 62. Initial V0/M2 c local checks recorded 45 passes and one Hopper-only training-dispatch assertion failure on Blackwell, plus 20 forward tests passed; that environment-specific failure was not silently counted as a pass.

Inherited BF16 replay gates are atol 0.75, rtol 0.03 and RMS≤0.15. Fixed forced canonical histories expose comparable logits, not unconstrained policy equivalence. Raw argmax flips near ties are preserved. Storage tests cycle real Habitat RGB and force actions; they are not navigation scores. A historical 96-step base-model B300 replay failure remains in the original campaign report and is not erased by passing learned-weight fixtures.

M2/M5 priorities depend on hidden activations, so BF16 train/serve drift can change selected IDs. M2 helper/native-FLA state error was 0.199%; M5 was 0.1936%, under the 1% gate. Max relative source-score differences were about 5.17% for M2 and 11.07%/20.76% for M5's replay lengths. These diagnostic differences are real limits of equivalence, not reasons to silently change selectors. Random M2 c/M5 c selections must match exactly for fixed identities; M1/V0 have no selections. No replay tolerance or checkpoint memory contract was relaxed.

### V0 versus Window8-B: code and training distinction

The current Window8-B rerun uses the exact V0 evaluation source checkout (`7cb2d3c`) but loads Window8-B's own weights and native Window8 configuration. V0 uses episodic R8/B0. Sharing source files does not make weights or configuration identical.

The historical Window8 training attention implementation accumulated overlapping K/V gradients in BF16. V0's episodic attention uses FP32 accumulation and batched attention. These can change optimization trajectories; they do not prove a complete explanation of the SR gap. The separate native batched-window implementation must not be broadly described as FP32 accumulation. Backward-pass differences are not directly exercised by fixed-checkpoint inference.

The older Window8-B epoch 1 run also reported 49.320283%SR/45.561094%SPL. The new completed rerun reaches the same aggregates; this alone does not certify every trajectory or implementation detail identical. It is one checkpoint evaluated twice, not two independent training seeds. The portable source bundle previously copied for inspection is `/home/anhdh35/V0_Window8B_evaluation` (and its `.tar.gz`), with the training-difference note.

## Missing or not-yet-final versions

M1 and M5 c results above are provisional. M3/visual-group selection, seed 430 V0/winner confirmations, and other conditional refinements have no navigation result in this reporting snapshot. Do not label unrun planned variants as zero-SR results. The separate cross-cluster M1 mentioned in historical scheduling reports is not another evaluated checkpoint here; the single published H2-a2 identity below defines this report's M1.

### Observed latency under shared GPU load

These values summarize the evaluated episodes in this snapshot, including partial runs. They are contention-affected measurements, not controlled speed comparisons. Model timing excludes transport and Habitat; Habitat step timing includes simulation and rendering.

| Version | Model mean (ms/action) | Model p50 | Model p95 | Transport + model mean | Habitat step mean |
|---|---:|---:|---:|---:|---:|
| V0 | 224.70 | 229.61 | 238.74 | 230.79 | 4.35 |
| M1 | 429.81 | 363.70 | 857.61 | 436.85 | 8.18 |
| M2 | 239.61 | 233.87 | 270.00 | 245.63 | 5.43 |
| M2c | 258.12 | 234.40 | 244.36 | 264.27 | 4.59 |
| M5 | 212.19 | 162.35 | 287.63 | 218.16 | 4.93 |
| M5c | 316.93 | 226.88 | 743.70 | 323.70 | 7.63 |
| Window8-B | 178.47 | 172.80 | 227.79 | 184.47 | 4.66 |

## Checkpoint and code identities

### V0

- Published export: [Streaming_episodic/P1-V0-s429/P1-V0-s429-H2-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/670c1a80b423ddbabd6d56895e35debd90410eab/P1-V0-s429/P1-V0-s429-H2-a2/export-e1).
- Pinned Hub revision: `670c1a80b423ddbabd6d56895e35debd90410eab`.
- Weight SHA-256: `65a7d8b2855510768339a8ef09a1f9ccc866e0e2625e5e6ff459f9131f05333a`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `e789a13f8b6ea3abc29086b82205a19024a6e7b5`.
- Local export: `/data14t/anhdh35/simplememvln_episodic_eval/checkpoints/P1-V0-s429/P1-V0-s429-H2-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_episodic_eval`.

### M1

- Published export: [Streaming_episodic/P1-M1-s429/P1-M1-s429-H2-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/707a971fde141ddbb122bfd5c3cac8ccb0c76546/P1-M1-s429/P1-M1-s429-H2-a2/export-e1).
- Pinned Hub revision: `707a971fde141ddbb122bfd5c3cac8ccb0c76546`.
- Weight SHA-256: `3c9064b310235f2e1762e4fcf92d62bb1663c38b4774e8a1a6827a2b3a501eb6`.
- Evaluation source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Published training source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Local export: `/data14t/anhdh35/simplememvln_m1_eval/checkpoints/P1-M1-s429/P1-M1-s429-H2-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_m1_eval`.

### M2

- Published export: [Streaming_episodic/P1-M2-s429/P1-M2-s429-H2-a1/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/9440b4d790f653333c80eeb9801af05d5edac378/P1-M2-s429/P1-M2-s429-H2-a1/export-e1).
- Pinned Hub revision: `9440b4d790f653333c80eeb9801af05d5edac378`.
- Weight SHA-256: `2cfb26a6213f0a10f19908d1a6343f70a260eb9c00005e3c8a7cd1577ba43919`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `e789a13f8b6ea3abc29086b82205a19024a6e7b5`.
- Local export: `/data14t/anhdh35/simplememvln_episodic_eval/checkpoints/P1-M2-s429/P1-M2-s429-H2-a1/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_episodic_eval`.

### M2 c

- Published export: [Streaming_episodic/P1-M2 c-s429/P1-M2 c-s429-H4-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/670c1a80b423ddbabd6d56895e35debd90410eab/P1-M2c-s429/P1-M2c-s429-H4-a2/export-e1).
- Pinned Hub revision: `670c1a80b423ddbabd6d56895e35debd90410eab`.
- Weight SHA-256: `cea5891796b1ab4b416ff240a66966e4ce6f2abe8d17bdd58942ff6ca55e87c8`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Local export: `/data14t/anhdh35/simplememvln_episodic_eval/checkpoints/P1-M2c-s429/P1-M2c-s429-H4-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_episodic_eval`.

### M5

- Published export: [Streaming_episodic/P2-M5-s429/P2-M5-s429-H4-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/08e47e5fa2c856ebbbbcddc0e973b7a73ab3f258/P2-M5-s429/P2-M5-s429-H4-a2/export-e1).
- Pinned Hub revision: `08e47e5fa2c856ebbbbcddc0e973b7a73ab3f258`.
- Weight SHA-256: `66ddb5553793d7ae275a7f2531dc098f81e185e2426402e3709be07086280bad`.
- Evaluation source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Published training source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Local export: `/data14t/anhdh35/simplememvln_g3_m5_eval/checkpoints/M5/P2-M5-s429/P2-M5-s429-H4-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_g3_m5_eval`.

### M5 c

- Published export: [Streaming_episodic/P2-M5 c-s429/P2-M5 c-s429-H4-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/6026dfdda962b99dc7ae4793b424809f765ff226/P2-M5c-s429/P2-M5c-s429-H4-a2/export-e1).
- Pinned Hub revision: `6026dfdda962b99dc7ae4793b424809f765ff226`.
- Weight SHA-256: `94f0f7aaaa991a4b99b367e6d547bb17a3d2755654c621ccfe207bef1ff895f5`.
- Evaluation source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Published training source: `a340e131b63c9ab7846690acdb369599462076c4`.
- Local export: `/data14t/anhdh35/simplememvln_m5c_eval/checkpoints/P2-M5c-s429/P2-M5c-s429-H4-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_m5c_eval`.

### Window8-B

- Published export: [SimpleMemVLN-R2R-RxR15 deg-Window8-B/epoch-1](https://huggingface.co/anhdao69/SimpleMemVLN-R2R-RxR15deg-Window8-B/tree/0a111c781477440d45afef600f3b79c340416d9f/epoch-1).
- Pinned Hub revision: `0a111c781477440d45afef600f3b79c340416d9f`.
- Weight SHA-256: `e5c81d168326d59e6b48321fbd73da713b280a92dc2fcd0865e15aa8382f281e`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `33a2735ae4e2fff125e4e231693abc70af137f8e`.
- Local export: `/data14t/anhdh35/simplememvln_window8_b_eval/checkpoints/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_window8_b_eval`.

## Evidence and reproduction

| Version | Source/runtime evidence root on evaluation server | Result directory |
|---|---|---|
| V0 | `/data14t/anhdh35/simplememvln_episodic_eval` | `results/full/V0` |
| M1 | `/data14t/anhdh35/simplememvln_m1_eval` | `results/full/M1` |
| M2 | `/data14t/anhdh35/simplememvln_episodic_eval` | `results/full/M2` |
| M2 c | `/data14t/anhdh35/simplememvln_episodic_eval` | `results/full/M2c` |
| M5 | `/data14t/anhdh35/simplememvln_g3_m5_eval` | `results/full/M5` |
| M5 c | `/data14t/anhdh35/simplememvln_m5c_eval` | `results/full/M5c` |
| Window8-B | `/data14t/anhdh35/simplememvln_window8_b_eval` | `results/full/Window8-B-E1` |

Each result directory contains `summary.json`, `episodes.jsonl`, `runtime.json`, and `evaluation_contract.json`. Root download manifests, source verification/preflight, tensor inventories, test logs and `audit/` document qualification. M5 shares a root with G3 and uses `audit/M5`; M1/M5 c use arm-specific audit subdirectories. V0/M2 c share `campaign_status.json`; M2/M5/M1/M5 c have arm-specific status files. Preserve older failed-validator evidence as explained above.

To reproduce, use the exact source/checkpoint/config/base and simulator versions, the frozen episode list and exact scene-path/UID convention. Root `run_eval.py` scripts launch separate model and Habitat environments; older V0/M2 scripts have different CLIs, so inspect before invoking. Use a fresh result directory and do not overwrite journals or launch duplicate jobs against running output paths. Compare full final SR/SPL and actual memory/cost; preserve the configured R/B/selector without substituting another arm.

Implementation references at immutable commits:

- [Episodic plan, including M1 and Phase2 controls](https://github.com/anhdao69/SimpleMemVLN/blob/a340e131b63c9ab7846690acdb369599462076c4/plans/episodic_mem.md)
- [Core implementation](https://github.com/anhdao69/SimpleMemVLN/blob/a340e131b63c9ab7846690acdb369599462076c4/reports/episodic_idea/02_implementation.md)
- [Ablation qualification and submission history](https://github.com/anhdao69/SimpleMemVLN/blob/7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f/reports/episodic_idea/09_ablation_runs.md)

## Next update

Wait for M1 and M5 c to reach 1,839 episodes and pass final validation. Replace their partial rows and shared-prefix comparisons with full results, then perform the intended paired uncertainty/gate analysis before asserting selector superiority. No automatic report refresh is configured. Copies on other branches are documentation only; use the pinned runtime above for inference.
