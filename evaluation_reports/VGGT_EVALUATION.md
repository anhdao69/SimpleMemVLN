# VGGT campaign — R2R evaluation report

**October 19 final update:** all six local evaluations below completed 1,839 R2R val_unseen episodes with zero execution failures. **G3 has the highest local SR, 51.984774%; the new plain-M5 control M5_E1 has SR 51.658510% and the highest local SPL, 48.401034%.** The qualified G3_M5 variants score below both references on SR and SPL. L8 remains an unqualified diagnostic because its serving replay failed.

## Completed local evaluations verified October 19

These are the completed October 17–19 local evaluations, independently rechecked
from their original journals for this update. They are distinct from the older
campaign snapshot below. Percentages use the full 1,839-episode denominator.

| Checkpoint | Successful / evaluated | SR (%) | SPL (%) | Serving qualification | Evaluation status |
|---|---:|---:|---:|---|---|
| G3 | 956 / 1,839 | 51.984774 | 48.095941 | Passed | Final |
| **M5_E1, true one-epoch schedule** | **950 / 1,839** | **51.658510** | **48.401034** | **Passed** | **Final** |
| Older M5, epoch 1 of two-epoch schedule | 938 / 1,839 | 51.005982 | 47.438938 | Passed | Final |
| G3_M5, layer 24 | 917 / 1,839 | 49.864057 | 46.140869 | Passed | Final |
| G3_M5_L8_L16_L24 | 916 / 1,839 | 49.809679 | 46.265741 | Passed | Final |
| G3_M5_L8 | 898 / 1,839 | 48.830886 | 46.037255 | **Failed 128-step replay** | **Final diagnostic only** |

Every row has zero execution failures and zero invalid responses. All worker
exit codes are 0. For each run, the merged journal exactly matches its original
shards and frozen ordered list of 1,839 unique episode IDs. Every merged and
shard summary field was freshly recomputed with that run's frozen metric
function; values matched at relative/absolute tolerance 1e-12. Contracts and
runtimes agree within each run. Final validation passed for all six, but this
validates dataset coverage and aggregation, **not L8's failed serving replay**.

All six use the same local Python 3.9.25/Habitat 0.2.4 simulator setup,
dataset/config hashes, scene paths, pinned base assets, seed 42, 500-action cap
and failure policy. M5_E1 pairings freshly verified matching IDs, episode UIDs,
scene paths and instructions against every other local run. G3 uses native
Window8: current plus seven previous groups in full-attention KV, with the
instruction prefix retained. It does not use full-history attention KV; native
GDN recurrence persists across the episode. All M5-family rows use R4/B1200
committed-write episodic memory. L8/L16/L24 denote zero-based training layers,
not attention-window sizes.

The new M5_E1 control uses the true 3,852-update schedule with 116 warmup
updates, matching the one-epoch schedule of the G3_M5 variants. Older M5 is
an epoch-1 checkpoint of a 7,704-update schedule with 232 warmup updates.
M5_E1 exceeds older M5 by 12 successes (+0.652529 pp SR, +0.962096 pp SPL).
Relative to G3, M5_E1 has six fewer successes (-0.326264 pp SR) and
+0.305093 pp SPL. These are single-seed descriptive results, not a significance
claim. See the [M5_E1 final report and paired comparisons](EPISODIC_MEMORY_EVALUATION.md#m5_e1-new-matched-one-epoch-control-2026-10-19).

| Local checkpoint | Final summary | Frozen evaluation source |
|---|---|---|
| G3 | [g3_epoch1_20261018/full](../evaluation/g3_epoch1_20261018/full/summary.json) | `c2_vggt_omega_g3` at `9d658c245b2b9d14232e8db0b235bf92e857a43d` |
| Older M5 | [m5_epoch1_20261018/full](../evaluation/m5_epoch1_20261018/full/summary.json) | `c2_ablations` at `a340e131b63c9ab7846690acdb369599462076c4` |
| G3_M5 | [g3_m5_epoch1_20261017/full](../evaluation/g3_m5_epoch1_20261017/full/summary.json) | `c2_vggt_omega_g3_m5` at `295e85e6081be2a8d8d9d786f2ba47ca22e0667b` |
| G3_M5_L8 | [g3_m5_l8_epoch1_20261018/full_unqualified](../evaluation/g3_m5_l8_epoch1_20261018/full_unqualified/summary.json) | `c2_vggt_omega_g3_m5_layers` at `f4d1c2f216e68a960a5a691d21cbf1f08ea4a901` |
| G3_M5_L8_L16_L24 | [g3_m5_l8_l16_l24_epoch1_20261018/full](../evaluation/g3_m5_l8_l16_l24_epoch1_20261018/full/summary.json) | `c2_vggt_omega_g3_m5_layers` at `5bdf5eed274709f0349a9d6c1572b623a0a2f066` |
| M5_E1 | [m5_e1_epoch1_20261019/full](../evaluation/m5_e1_epoch1_20261019/full/summary.json) | `c2_vggt_omega_g3_m5_layers` at `c5d5422b4408e4701437543f94e51ceac2b47b16` |

The [fresh evidence record](evidence/LOCAL_R2R_FINAL_20261019.json) preserves
all six full summaries, immutable Hub revisions, weight SHA-256 values,
source identities, qualification status, journal hashes, paired comparisons
and the M5_E1 resume-prefix checks. Reproduce the checks with
`python evaluation_reports/evidence/verify_local_r2r_20261019.py` from the
repository root. Evaluation outputs and frozen source archives are unchanged.

## Historical snapshot and scope

**G3_M5 update: 2026-10-18 Asia/Ho_Chi_Minh.** The full run completed at **2026-10-17 17:33:43 UTC / 2026-10-18 00:33:43 Asia/Ho_Chi_Minh**. Its journals and summaries were independently rechecked for this update.

The original campaign tables below retain the snapshot collected through **2026-10-16 02:45:04 UTC / 2026-10-16 09:45:04 Asia/Ho_Chi_Minh**, plus the October 18 G3_M5 update. They are historical records; the completed local reruns and later checkpoints are reported separately above. A1, M1 and M5 c were partial at the older snapshot and their current status was not checked here. This Markdown file is a dated report, not a live dashboard. SR/SPL use percentages; changes use percentage points (pp).

All results here are **R2R VLN-CE val_unseen**, with **1,839 scheduled episodes**. Every published checkpoint evaluated here is epoch 1. Training used joint R2R/RxR data, but **no full RxR navigation evaluation result is available in this report**. RxR training/qualification fixtures are not RxR benchmark scores.

Completed and partial runs are explicitly separated. The partial denominator is the number of evaluated episodes shown, not 1,839; unevaluated episodes have not been assigned zero scores. Episode order groups scenes, so a partial prefix is not a random representative sample. Do not rank partial SR against final full-split SR.

## Shared evaluation protocol

| Item | Verified setting |
|---|---|
| Protocol | `R2R-VLNCE-0.2.4`, split `val_unseen` |
| Simulator | Habitat-Lab 0.2.4 / Habitat-Sim 0.2.4; historical Python 3.10.20, G3_M5 Python 3.9.25 |
| Policy environment | Python 3.12.13, PyTorch 2.10.0+cu129, Transformers 5.11.0 |
| Attention stack | FlashAttention 2.8.3+cu.12.9.torch.2.10, FLA/fla-core 0.5.2, Triton 3.6.0 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB class |
| Base model | Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Observations | RGB 640×480, HFOV 79°; policy receives current RGB and instruction/history |
| Actions | Canonical text MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP; Habitat IDs 1,2,3,0 |
| Motion | Forward 0.25 m, turns 15° |
| Success | Habitat navigation success with STOP within the configured 3 m goal radius |
| Evaluation seed / cap | 42 / 500 actions, cap-ending forced STOP recorded separately |
| Historical dataset path | `/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz` |
| Dataset SHA-256 | `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3` |
| Habitat config SHA-256 | `dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f` |
| Historical scenes path | `/storage/anhdh35/JanusVLN/data/scene_datasets_full`; 11 evaluation scene assets checked |
| Failure policy | `zero_sr_spl_keep_partial_metrics`: failed episodes remain in SR/SPL denominator |

The original report verified matching ordered episode IDs, dataset/config hashes, base-asset hashes, scenes, seed, action cap, protocol and failure policy across its 11 historical runs. That evidence is retained here; those older journals were not available locally for a fresh comparison with G3_M5. Evaluation sources are isolated snapshots and were not switched under running processes.

G3_M5 used the same dataset/config SHA-256 values, registered 1,839 episode IDs, base revision, seed, action cap and failure policy. Its dataset is `/media/vmo-perception/disk_2/anhdh35/JanusVLN/data/datasets/r2r/{split}/{split}.json.gz`, and scenes are under `/media/vmo-perception/disk_2/anhdh35/JanusVLN/data/scene_datasets`. All 11 scenes passed render/metric checks; the 22 local GLB/navmesh hashes are preserved in the evidence file. Cross-host scene-byte equality and episode UID equality with the historical runs were not established.

The G3_M5 simulator environment uses Python 3.9.25, NumPy 1.26.4, Hydra 1.3.2 and OmegaConf 2.3.0; the historical environment recorded Python 3.10.20, Hydra 1.3.4 and OmegaConf 2.3.1. Habitat-Lab source is `1639e1ae732ba1e84199a1a04b79c7243c3f8586`; Habitat-Sim is `f179b584bcd713c5a2a998132211e2cae881d6d1`. The resolved configuration and both package inventories are preserved. These environment differences limit strict cross-run equivalence.

**Definitions:** SR is the mean binary navigation success; SPL is mean success weighted by shortest-path versus executed-path efficiency. NE is final distance to goal. Oracle success records whether the trajectory ever reached the goal region; it does not mean the model stopped correctly. Forced STOP is an episode hitting the action cap, not an execution exception. nDTW is unavailable (`null`, zero coverage), not zero. NE/oracle coverage equals the evaluated count in these snapshots.

All listed runs have **0 execution failures and 0 invalid responses** at the snapshot. Navigation failures (SR=0) are distinct from execution failures. Timing is affected by concurrent GPU workloads; neither model latency nor wall time here establishes isolated speed superiority.

## Report verification and historical validator flags

The original report independently reread the historical episode journals, checked unique episode IDs and exact ordered membership, and recomputed all saved summary fields. Integral fields and structure matched exactly; floating values matched with relative/absolute tolerance **1e-12**. This tolerance only handles aggregation/JSON rounding and does not alter navigation decisions or numerical replay gates.

Older V0/M2 c orchestration (`campaign_status.json`), M2 (`M2-status.json`) and G1 (`status.json`) retain failed final-validator flags from exact floating-point equality assertions. They are not evidence of missing episodes: all four journals contain 1,839 unique completed episodes and zero execution failures. The original report verified their metrics at the historical snapshot. Across the historical snapshot runs, SPL recomputation differences were at most 7.22e-16 in fraction units and mean-path differences at most 2.85e-14 m. Historical status files were preserved. Newer coordinators use the 1e-12 final-metric check.

The historical G3, M5, Window8-B and A0 coordinators reported complete after their final contract/journal checks. A1, M5 c and M1 were still running at that snapshot. No partial run is certified as a final completed evaluation.

For this update, G3_M5's merged journal was checked against the four original shard journals (459/460/460/460 episodes). All 1,839 IDs are unique, have exact scheduled order/membership and match the historical registered ID set. Every merged and shard summary field was recomputed with the unchanged source evaluator and matched at tolerance 1e-12; shard contracts and runtimes agree. All four workers exited 0, `complete=true`, and final validation passed. The frozen source hashes also reverified. The [machine-readable evidence record](evidence/G3_M5_R2R_EPOCH1_20261018.json) retains this fresh verification separately from historical claims.

## Historical results with the October 18 G3_M5 update

| Version (epoch 1) | Completed / scheduled | Successful | SR (%) | SPL (%) | Status |
|---|---:|---:|---:|---:|---|
| G1 | 1,839 / 1,839 | 914 | 49.700924 | 46.072125 | FINAL |
| G3 | 1,839 / 1,839 | 956 | 51.984774 | 48.095941 | FINAL |
| **G3_M5** | **1,839 / 1,839** | **917** | **49.864057** | **46.140869** | **FINAL** |
| A0 | 1,839 / 1,839 | 910 | 49.483415 | 45.394182 | FINAL |
| A1 | 1,356 / 1,839 | 680 | 50.147493 | 46.306672 | Partial at Oct 16 snapshot |
| Window8-B | 1,839 / 1,839 | 907 | 49.320283 | 45.561094 | FINAL |
| V0 | 1,839 / 1,839 | 852 | 46.329527 | 41.756802 | FINAL |

Window8-B and V0 are reference runs, not VGGT-supervised versions. Their own training histories differ from the one-epoch VGGT schedule. The companion [episodic memory report](EPISODIC_MEMORY_EVALUATION.md) records all episodic arms.

### Additional navigation and memory measurements

| Version | NE (m) | Oracle success (%) | Mean steps | Forced STOPs | Peak retained KV rows | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 6.0579 | 55.8999 | 83.84 | 94 | 2718 | 8.668 |
| G3 | 5.7935 | 58.7819 | 94.51 | 133 | 2718 | 8.670 |
| G3_M5 | 5.8037 | 56.6612 | 91.78 | 130 | 2646 | 8.673 |
| A0 | 5.9695 | 55.3018 | 93.42 | 129 | 2718 | 8.670 |
| A1 | 6.4297 | 57.6696 | 100.47 | 114 | 2683 | 8.656 |
| Window8-B | 6.0358 | 59.8695 | 115.39 | 212 | 2718 | 8.668 |
| V0 | 6.4879 | 59.3801 | 126.32 | 240 | 2718 | 8.668 |

Peak retained KV is the observed run maximum, not the configured upper bound. G1/G3/A0/A1 use Window8 and a 4,096-row guard. G3_M5 uses four recent groups plus 1,200 old visual rows per full-attention layer, with a 4,784-row guard. V0 instead uses the generalized episodic R8/B0 path with a 5,120-row structural guard.

G3_M5 mean trajectory length is 9.4223 m, with complete NE, trajectory-length and oracle-success coverage. Its 1,709 predicted STOPs and 130 cap-ending forced STOPs account for all 1,839 final STOPs. Forced stops remain in the scored denominator; nDTW has zero coverage and is unavailable.

## What each version actually implements

| Version | Training supervision | Serving architecture | Teacher at inference? |
|---|---|---|---|
| G1 | Current visual residual alignment to online Omega descriptors, learned MLP projector | Native Window8; auxiliary projector removed | No |
| G3 | Delayed visual-region prediction from native GDN layer-24 state through read-only probes | Native Window8; probes, query/decoder heads removed | No |
| G3_M5 | G3 delayed native-state supervision during M5 episodic training | M5 R4/B1200 committed-write selection; probes, query/decoder heads removed | No |
| A0 | Navigation-only architecture control; no teacher loss | Window8 **plus trained GeoAddress Q/K adapters** | No |
| A1 | Direct native Q/K relation KL against online Omega teacher similarities | Same retained GeoAddress architecture as A0 | No |

### G1: current representation alignment

The implemented teacher is frozen **online FP32 VGGT-Omega-1B-512**, not an offline PCA feature bank. Independent single-view images are processed as `[B,1,3,448,592]`; layer-23 descriptors exclude 17 special tokens, pool a 28×37 patch grid to 15×20, and retain all 2,048 feature dimensions. G1 uses non-affine LN2048 plus 0.5×fixed 2D sin/cos positional targets.

The student target is the 300 image-token residual outputs after decoder index 23. Its training-only projector is affine LN2560 → Linear2560→4096 → GELU → Linear4096→2048 (18,885,632 parameters). Up to 16 stratified frames per episode are supervised with FP32 cosine loss, globally normalized by frame count; auxiliary weight ramps to 0.2 over 771 updates. The projector is absent from serving. No PCA, offline target cache, episodic KV bank or new serving lane is involved.

### G3: delayed information in native recurrent state

G3 also uses frozen online single-view Omega features, without PCA or added positional descriptors. Targets are four quadrant means plus one area-weighted global mean of the normalized 2,048-wide visual descriptors. Up to eight delayed source/read pairs use lags 8–32.

At native GDN index 24, a training-only query MLP uses detached normalized current native queries, a region embedding and lag encoding. Five read-only probes per pair use zero value/update/decay contributions and are removed before native output projection. A decoder predicts the 2,048-wide target from the read state. Auxiliary cosine loss uses a global pair denominator and weight ramp to 0.1 over 771 updates. Query/decoder/probes do not become inference tokens, KV rows or persistent serving modules. Serving is native Window8.

### G3_M5: delayed native-state supervision with M5 memory

G3_M5 combines the G3 training objective above with unchanged M5 episodic attention: four recent complete groups including the current observation and 1,200 older visual K/V rows per full-attention layer. Selection uses `committed_mean_l2_v1`, selector seed 429 and no per-source cap. Native FP32 GDN recurrent state persists through the full episode; attention eviction does not reset it. Serving uses canonical appended text actions and `vln_append_only_chat_v1`, with no teacher, auxiliary heads, probes, PCA or extra lane. The model starts from the original Qwen base, not trained G3 or trained M5 weights.

### A0 / A1: persistent GeoAddress adapters

Both arms retain four trained bias-free matrices at decoder 24: separate Q and K residual paths 2560→8→256, totaling **45,056 parameters**. They modify raw Q/K heads 0 and 1 (head width 128), only for image tokens, after causal convolution/SiLU and before native head expansion/L2 normalization. Native values, gates, convolution, text-token Q/K and other raw heads are unchanged directly by the adapter operation; subsequent hidden states can still differ. The adapters are part of the learned serving model and must not be stripped.

The explicit image-mask contract is `vln_append_only_chat_geoaddress_v1`. Initial prefill, cached multi-token append and single-token decode all use the adapters. Native recurrent/conv state remains in use; there is no extra episodic bank or lane. Seed 429 initializes down projections; up projections begin at zero, but the evaluated checkpoints contain nonzero learned matrices.

A0 is navigation-only (auxiliary weight 0). A1 independently starts from the same pretrained base, not A0's trained weights. Its frozen online Omega teacher produces unpositioned 2,048-wide descriptors. Each selected pair forms 32-query × 300-source similarity distributions. The student uses its actual 128-wide native Q/K at two heads, temperature 0.1 and native normalization; **feature widths need not match because the distributions have matching shape**. KL(teacher || student) averages queries/heads and normalizes globally by valid pair count. Earlier student keys retain gradients; teacher targets are detached. Maximum weight is 0.1 after a 771-update ramp. No PCA or dimension-conversion MLP is used in A1. Teacher and relation loss are training-only.

The late A1 identity repair excludes unused positional-grid bytes while retaining identities of used teacher weights/source/preprocessing; it does not change the serving equations. Descriptor similarity is not certified physical correspondence.

## Training exposure and interpretation

G1/G3/A0/A1 completed 3,852/3,852 updates of a true one-epoch schedule, warmup 116, seed 429, nominal batch 8 and joint R2R + English RxR15 data (30,815 episodes). Their published training topology is four GPUs/GAS2. Frozen vision/merger and full-episode BPTT are retained. Vision microbatch 8 and attention batch 32 are training execution settings; **the serving window is 8, not 32**.

G3_M5 also completed a true joint one-epoch schedule: 3,852 updates with 116 warmup updates. Its serving memory is R4/B1200 as specified above.

Episodic V0/M1/M2/M2 c/M5/M5 c are epoch-1 exports at 3,852/7,704 with warmup 232. Window8-B also comes from a configured two-epoch schedule. Consequently G1/G3/G3_M5/A0/A1 versus those references is a practical checkpoint comparison, not a clean auxiliary-only experiment. A1 versus A0 is the more directly matched teacher-effect comparison, but A1 was unfinished at the historical snapshot and single-seed outcomes do not establish statistical robustness.

### G3_M5 versus completed historical results

| G3_M5 minus reference | Shared registered split size | Δsuccesses | ΔSR (pp) | ΔSPL (pp) |
|---|---:|---:|---:|---:|
| G3 | 1,839 | -39 | -2.1207 | -1.9551 |
| M5 | 1,839 | -21 | -1.1419 | -1.2981 |

These are aggregate differences against the recorded final reference scores; SPL differences use the references' published six-decimal percentages. No per-episode pairing or significance estimate was recomputed for these comparisons. The G3_M5 result is lower on both metrics than either individual reference. G3_M5 versus M5 also changes the learning-rate schedule (3,852 versus 7,704 total planned updates, warmup 116 versus 232); together with the environment differences above, this prevents attributing the difference solely to G3 supervision. A single seed does not establish a general interaction between G3 and M5.

### Historical descriptive paired comparisons

The next table compares identical episode IDs, UIDs and instructions. For A1, the comparator is restricted to A1's completed prefix; these values are not the comparator's full-run result. No confidence interval or significance claim is made.

| A minus B | Shared episodes | A SR (%) | B SR (%) | ΔSR (pp) | ΔSPL (pp) | A-only / B-only successes | Scope |
|---|---:|---:|---:|---:|---:|---:|---|
| G3 − G1 | 1,839 | 51.9848 | 49.7009 | +2.2838 | +2.0238 | 286 / 244 | Full split |
| G1 − Window8-B | 1,839 | 49.7009 | 49.3203 | +0.3806 | +0.5110 | 281 / 274 | Full split |
| G3 − Window8-B | 1,839 | 51.9848 | 49.3203 | +2.6645 | +2.5348 | 298 / 249 | Full split |
| A0 − Window8-B | 1,839 | 49.4834 | 49.3203 | +0.1631 | -0.1669 | 303 / 300 | Full split |
| A1 − A0 | 1,356 | 50.1475 | 48.6726 | +1.4749 | +1.7893 | 225 / 205 | Partial shared prefix only |

G3's completed advantage over G1 is a descriptive comparison of different auxiliary designs. A0 is not an unchanged native baseline because it retains adapters. Partial A1 results cannot yet establish the full-split A1−A0 effect.

## Historical qualification and G3_M5 limitations

| Arm | Targeted local tests | Trained replay evidence | 520-step storage check |
|---|---|---|---|
| G1 | 33 passed; 1 training-only Omega preprocessing test deselected | 64/128 steps: RMS 0.030369/0.028647; max drift 0.328125; raw token flips 1/191 and 2/383 | Peak 2,618 / guard 4,096 |
| G3 | 24 passed | 64/128: RMS about 0.03275/0.03092; max drift 0.5; one raw token flip in each | Peak 2,618 / 4,096 |
| G3_M5 | 81 passed, 25 warnings | 64/128: RMS 0.031916/0.030446; max drift 0.5625/0.6875; raw token flips 1/191 and 3/383 | Peak 2,546 / 4,784; all eight banks end at 1,200 rows |
| A0 | 25 passed | 64/128: RMS 0.039986/0.037494; max drift 0.75; flips 0/191 and 2/383 | Peak 2,618 / 4,096 |
| A1 | 33 passed | 64/128: RMS 0.031206/0.029061; max drift 0.5; flips 0/191 and 1/383 | Peak 2,618 / 4,096 |

The original four exports passed strict loading, finite tensor checks, two real Habitat smoke episodes, reset/retry checks and recurrent-state/cache invariants before full launch. G1/G3 serving has 724 native tensors; A0/A1 has 728 including all four adapters. Source-manifest matches: G1 53 files, G3 57, A0/A1 63 each. Actual trained A0/A1 instrumentation verified image Q/K modifications, unchanged unselected heads/text rows at the adapter, and all cache modes.

G1/G3 training-wrapper reconstruction uses published native and auxiliary weights; A0/A1 reconstruction uses published native+adapter weights and training metadata. Their evaluation logits match the serving export bitwise on the checked fixture. Original cluster recovery artifacts were not independently available locally; this is not a claim of repeating the publisher's complete recovery-byte comparison.

The original four-arm replay uses real rendered Habitat RGB and fixed forced canonical histories, checking every response token including EOS. The 64/128 fixtures contain 103 distinct images across the longer fixture and are not expert-quality navigation evidence. Inherited BF16 thresholds are atol 0.75, rtol 0.03, RMS ≤0.15. Raw argmax differences are explicitly retained: a tolerance pass is not bitwise or universally free-running action equivalence. The 520-step cyclic-image test establishes bounded storage on its fixture, not navigation SR. No thresholds were relaxed for these weights. The initial A0 audit monitoring hook was corrected for keyword hidden-state arguments; production model source was unchanged.

G3_M5 independently passed strict loading/serializer checks, finite checks on 724 native tensors, auxiliary-head absence, two real Habitat smoke episodes, reset/retry checks and 64/128-step replay using real rendered observations with fixed canonical histories. Replay selected-token overlap averaged 99.0748%/98.8612% (minimum 97.3333%/97.5000%). The same BF16 thresholds above were retained; raw token flips and differing bank selection remain visible. These checks do not establish identical free-running behavior. Its 520-step storage fixture is separate from the scored navigation run. The local G3_M5 audit did not reconstruct the training wrapper or repeat distributed training; the implementation report's publication qualification remains historical evidence.

Remote H100 training/recovery/teacher qualification described in implementation reports is historical evidence; the local inference checks did not repeat the complete distributed training campaign. Checks cannot guarantee the absence of every possible bug.

## Versions without a navigation result in this report

R1 has an implementation/training report on `v2_vggt_omega_r1`, but no R1 local navigation evaluation was launched or recorded in this campaign snapshot. R0 and other variants absent from the results tables likewise have no result in this report. Do not substitute training losses, qualification success or older unrelated dual-lane runs for missing benchmark results. The historical VGGT-family rows cover G1, G3, G3_M5, A0 and A1; the later local results also include G3_M5_L8 and G3_M5_L8_L16_L24.

### Observed latency under shared GPU load

These values summarize the evaluated episodes in this snapshot, including partial runs. They are contention-affected measurements, not controlled speed comparisons. Model timing excludes transport and Habitat; Habitat step timing includes simulation and rendering.

| Version | Model mean (ms/action) | Model p50 | Model p95 | Transport + model mean | Habitat step mean |
|---|---:|---:|---:|---:|---:|
| G1 | 92.53 | 86.01 | 143.56 | 98.95 | 2.22 |
| G3 | 185.31 | 179.03 | 226.68 | 191.35 | 4.13 |
| G3_M5 | 330.24 | 350.45 | 357.46 | 336.81 | 5.57 |
| A0 | 167.26 | 144.22 | 478.96 | 173.45 | 4.39 |
| A1 | 311.12 | 226.82 | 726.21 | 317.92 | 7.33 |
| Window8-B | 178.47 | 172.80 | 227.79 | 184.47 | 4.66 |
| V0 | 224.70 | 229.61 | 238.74 | 230.79 | 4.35 |

G3_M5 ran four independent workers sharing GPU 0, each using the unchanged branch evaluator and a disjoint contiguous episode panel. Coordinator wall time was 4 h 19 min 17 s (2026-10-17 13:14:26–17:33:43 UTC). The merged timing distributions are computed over all recorded actions; this is concurrent rollout timing, not isolated policy latency.

## Checkpoint and code identities

### G1

- Published export: [Streaming_VGGT/G1/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/0580a3ad633bfe53176fc96c377f88f4c0386456/G1/epoch-1).
- Pinned Hub revision: `0580a3ad633bfe53176fc96c377f88f4c0386456`.
- Weight SHA-256: `a620ab17b1c0cbec6f7c5f95d67177a4b078ae53e83ea08c6696c5c4797a63ae`.
- Evaluation source: `67b206b8a2e338e436e340c6c41dd146d4b2eb0e`.
- Published training source: `86d68b2eff9db7c5cffdd86f0d98ca98f7e1c93a`.
- Local export: `/data14t/anhdh35/simplememvln_g1_eval/checkpoints/G1/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_g1_eval`.

### G3

- Published export: [Streaming_VGGT/G3/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/4911455eb3c8ff02038c9eccc56ee0b9733aa24d/G3/epoch-1).
- Pinned Hub revision: `4911455eb3c8ff02038c9eccc56ee0b9733aa24d`.
- Weight SHA-256: `a71be6b5291b6acddbeb389318befaded5818c5b1438d5d44cf351c64248333b`.
- Evaluation source: `9d658c245b2b9d14232e8db0b235bf92e857a43d`.
- Published training source: `22593831c80b742ee77b478ce7cb76fb44bc328b`.
- Local export: `/data14t/anhdh35/simplememvln_g3_m5_eval/checkpoints/G3/G3/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_g3_m5_eval`.

### G3_M5

- Published export: [Streaming_VGGT/G3_M5/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/249464698837e6d71e275ed769fd40fa90d41d6e/G3_M5/epoch-1).
- Pinned Hub revision: `249464698837e6d71e275ed769fd40fa90d41d6e`; all 18 downloaded files passed checksum/size verification.
- Weight SHA-256: `4e2d2095325fa779dcdece278f52213013d6889e72705ff860ab2eba1b13194f`.
- Navigation configuration SHA-256: `f07953c882bf8fa6b8bc5bc546f5ec714acb2d976c13dacf31cf44acc081a44b`.
- Evaluation branch/source: `c2_vggt_omega_g3_m5` / `295e85e6081be2a8d8d9d786f2ba47ca22e0667b`.
- Qualified training source: `9031d75d2aa8c3cbd1d6a1bb394f2de5a4a1abc5`; executable trees `src`, `scripts`, `configs`, `train`, `tests` match the evaluation source exactly.
- Local evidence root: `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/evaluation/g3_m5_epoch1_20261017`.
- Local export: `checkpoints/G3_M5/epoch-1` beneath that root; the frozen evaluation checkout is `source/`.

### A0

- Published export: [Streaming_VGGT/A0/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/ec3942d782c4dc27e996a3b70e29d93e5eaa492a/A0/epoch-1).
- Pinned Hub revision: `ec3942d782c4dc27e996a3b70e29d93e5eaa492a`.
- Weight SHA-256: `5e63563ddb891c43e6a02073125ddad8892f0a85c77a5b1d0b6637150b6f08e6`.
- Evaluation source: `0090a07f72fabe6ac2d1d48c949b3ea9f09bdd9b`.
- Published training source: `0c22a7bdac465aedd5d1413fa08d9eedc12206c1`.
- Local export: `/data14t/anhdh35/simplememvln_a0_eval/checkpoints/A0/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_a0_eval`.

### A1

- Published export: [Streaming_VGGT/A1/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/37f877aceed4dfd5b22fca9508150ff693b169b2/A1/epoch-1).
- Pinned Hub revision: `37f877aceed4dfd5b22fca9508150ff693b169b2`.
- Weight SHA-256: `0d9078ffd00e39d60655866912b849e0d9949407b9fa35424545134b00337bea`.
- Evaluation source: `4f772d3bac877322e2cb58e127c17f87e202f17c`.
- Published training source: `6e717787313b2218cd2fd40b059f2942fee1820f`.
- Local export: `/data14t/anhdh35/simplememvln_a1_eval/checkpoints/A1/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_a1_eval`.

### Window8-B

- Published export: [SimpleMemVLN-R2R-RxR15 deg-Window8-B/epoch-1](https://huggingface.co/anhdao69/SimpleMemVLN-R2R-RxR15deg-Window8-B/tree/0a111c781477440d45afef600f3b79c340416d9f/epoch-1).
- Pinned Hub revision: `0a111c781477440d45afef600f3b79c340416d9f`.
- Weight SHA-256: `e5c81d168326d59e6b48321fbd73da713b280a92dc2fcd0865e15aa8382f281e`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `33a2735ae4e2fff125e4e231693abc70af137f8e`.
- Local export: `/data14t/anhdh35/simplememvln_window8_b_eval/checkpoints/epoch-1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_window8_b_eval`.

### V0

- Published export: [Streaming_episodic/P1-V0-s429/P1-V0-s429-H2-a2/export-e1](https://huggingface.co/anhdao69/Streaming_episodic/tree/670c1a80b423ddbabd6d56895e35debd90410eab/P1-V0-s429/P1-V0-s429-H2-a2/export-e1).
- Pinned Hub revision: `670c1a80b423ddbabd6d56895e35debd90410eab`.
- Weight SHA-256: `65a7d8b2855510768339a8ef09a1f9ccc866e0e2625e5e6ff459f9131f05333a`.
- Evaluation source: `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f`.
- Published training source: `e789a13f8b6ea3abc29086b82205a19024a6e7b5`.
- Local export: `/data14t/anhdh35/simplememvln_episodic_eval/checkpoints/P1-V0-s429/P1-V0-s429-H2-a2/export-e1`.
- Local evidence root: `/data14t/anhdh35/simplememvln_episodic_eval`.

## Evidence and reproduction

| Version | Source/runtime evidence root on evaluation server | Result directory |
|---|---|---|
| G1 | `/data14t/anhdh35/simplememvln_g1_eval` | `results/full/G1` |
| G3 | `/data14t/anhdh35/simplememvln_g3_m5_eval` | `results/full/G3` |
| G3_M5 | `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/evaluation/g3_m5_epoch1_20261017` | `full` |
| A0 | `/data14t/anhdh35/simplememvln_a0_eval` | `results/full/A0` |
| A1 | `/data14t/anhdh35/simplememvln_a1_eval` | `results/full/A1` |
| Window8-B | `/data14t/anhdh35/simplememvln_window8_b_eval` | `results/full/Window8-B-E1` |
| V0 | `/data14t/anhdh35/simplememvln_episodic_eval` | `results/full/V0` |

At each result directory, inspect `summary.json`, `episodes.jsonl`, `runtime.json` and `evaluation_contract.json`. For the historical runs, root `audit/`, `preflight.json`, tensor inventories, download manifests, test logs and `audit_notes.txt` (V0 uses `campaign_notes.txt`) retain qualification evidence. G3 shares its root with M5 and has arm-specific audit subdirectories. A0/A1 use `A0-status.json`/`A1-status.json`; G1 uses `status.json`; G3 uses `G3-status.json`.

For G3_M5, `full/` contains the merged journal, summary, contract, runtime and passing `validation.json`; `full/shard-00` through `shard-03` preserve the original evaluator outputs. Root `status.json`, `download_verified.json`, `source_verification.json`, `serving_audit.json`, `simulator_preflight.json`, `scene_assets.json`, `targeted_tests.log` and `review_record.json` retain qualification and provenance. The [repository evidence record](evidence/G3_M5_R2R_EPOCH1_20261018.json) includes the final summary/contract/runtime, audit records, source identities and artifact SHA-256 values, including the merged journal hash; the large journals and weights remain at the local evidence root.

G3_M5's root `run_eval.sh` launches the separate policy and simulator environments; `evaluate_full.py` partitions and validates the split. The frozen selection is `episode_ids.json`, with `panel-00.json` through `panel-03.json` for the workers. Inspect the saved scripts and resolved Habitat config before reproducing, and select a fresh output root: the coordinator contains fixed evidence-root paths. Do not overwrite this completed run.

For the historical runs' reproduction, use the immutable source and exact checkpoint above, the saved navigation configuration, pinned base/runtime, dataset/scene assets and frozen `r2r_episode_ids.json`. The existing root `run_eval.py` launches the model environment and Habitat environment separately. Inspect each script's CLI before use; G1 and older V0 scripts differ from the newer arm/mode launchers. Write a new output directory for a rerun instead of overwriting evidence. Never strip A0/A1 adapters or install an online teacher into their serving path.

Implementation references at immutable commits:

- [G1 report](https://github.com/anhdao69/SimpleMemVLN/blob/67b206b8a2e338e436e340c6c41dd146d4b2eb0e/reports/vggt_five/G1/implementation_report.md)
- [G3 report](https://github.com/anhdao69/SimpleMemVLN/blob/9d658c245b2b9d14232e8db0b235bf92e857a43d/reports/vggt_five/G3/implementation_report.md)
- [G3_M5 implementation and publication report](https://github.com/anhdao69/SimpleMemVLN/blob/295e85e6081be2a8d8d9d786f2ba47ca22e0667b/reports/vggt_five/G3_M5/implementation_report.md)
- [A0/A1 implementation ledger](https://github.com/anhdao69/SimpleMemVLN/blob/4f772d3bac877322e2cb58e127c17f87e202f17c/reports/vggt_five/A0_A1/IMPLEMENTATION_2026-10-14.md)
- [A1 completed training/publication report](https://github.com/anhdao69/SimpleMemVLN/blob/4f772d3bac877322e2cb58e127c17f87e202f17c/reports/vggt_five/A0_A1/A1_CONTINUATION_2026-10-14.md)


## G3_M5_L8 epoch 1: unqualified diagnostic evaluation (2026-10-18)

The requested `G3_M5_L8/epoch-1` export is downloaded and verified at
`evaluation/g3_m5_l8_epoch1_20261018`. The correct branch is
`c2_vggt_omega_g3_m5_layers`, using frozen source
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`; the fetched branch tip
`7ededbf19f534582a7af1c446d6904ee0b76078e` has identical executable sources.
Hub revision is `792a0c959d8e6be8aee20e9cfa560c4f4b5a6b9b`, and weight SHA-256 is
`4081cb1fd573be20a5d656846bd6044a0b79fe956e1a8164e1d2ce9fa677657e`.
All 19 Hub file digests, all 18 publisher manifest entries, and all 241
published source-seal entries verified. Strict loading found 724 finite native
tensors; 109 targeted tests passed without skips. L8 identifies training
supervision at zero-based decoder layer 8. Serving still uses M5 R4/B1200
with committed-write scoring, exactly matching the original G3_M5 navigation
metadata; no teacher or training head is installed during evaluation.

**Serving qualification failed and remains failed.** The 64-step replay passed,
but two independent 128-step attempts reproduced 20 out-of-tolerance logits
among 95,106,560 values using the original `atol=0.75`, `rtol=0.03` budget.
The diagnostic RMS was 0.0392229483 (within the separate 0.15 RMS limit), and
maximum absolute difference was 0.9375. The outliers were at response positions
202/203 (observation 67); argmax agreed at those positions. Two other response
positions had raw argmax changes near ties. The cause is unresolved. These
observations do not establish offline/online equivalence or waive the failure.
See root `serving_audit.json`, `replay_diagnostic.json`, and saved full logits
`diagnostic_128_logits.pt`. Original tests, thresholds, model, and serving
sources are unchanged.

The independent two-episode Habitat smoke completed with zero execution
failures and zero successes; this is not a full-split SR estimate. The full
1,839-episode R2R VLN-CE `val_unseen` run launched at 2026-10-18 03:06:36
Asia/Ho_Chi_Minh (2026-10-17 20:06:36 UTC) with two workers as an explicitly
**unqualified diagnostic evaluation of the published streaming policy**, with
seed 42, cap 500, and the same dataset, scene assets, simulator, and metric
rules as the previous local run. `evaluate_unqualified.py` requires reset,
retry, 520-step storage, provenance, and smoke checks before launch. The
original `evaluate_full.py` still refuses this checkpoint because its replay
qualification failed. `diagnostic_run_decision.json` records this distinction.
The diagnostic coordinator labels its status, summary, final contract, and
aggregation validation with `serving_qualification_passed=false`; any final
aggregation pass refers only to dataset coverage and metric recomputation.
Exact evolved-session reset, retry, and greedy reset checks passed. The
520-step storage test passed with peak resident KV 2,546 (guard 4,784), and
all eight episodic banks held exactly 1,200 rows. Both coordinator cleanup
tests passed. Outputs are preserved in `full_unqualified/`.

**Final diagnostic result:** 898 successes / 1,839 episodes, **SR 48.830886%,
SPL 46.037255%**, zero execution failures and zero invalid responses.
The two workers exited 0; final coverage and aggregation validation passed at
**2026-10-18 07:12:54 UTC / 14:12:54 Asia/Ho_Chi_Minh**. Serving qualification
remains failed. This report's fresh journal recomputation confirms the saved
metrics without changing that status. Shared-GPU timings are not isolated
model latency measurements.

## G3_M5_L8_L16_L24 epoch 1: latest-code evaluation (2026-10-18)

The requested triple-layer checkpoint was evaluated from the then freshly
fetched `c2_vggt_omega_g3_m5_layers` tip
`5bdf5eed274709f0349a9d6c1572b623a0a2f066`. Its executable source, configurations,
tests and dependency lockfiles are identical to published training source
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`; later commits contain reports and
operations evidence. An isolated source archive preserves the existing working
copy and prior evaluation outputs.

Checkpoint: `anhdao69/Streaming_VGGT/G3_M5_L8_L16_L24/epoch-1`, immutable Hub
revision `36c478841d489366ccf91c06033fea8921e9d0b1`, expected weight SHA-256
`c206e32e4bbe884ae6ac5dbcd1a46ac7589d9218b7bc8f27a70e29a821b295ce`.
The completed training ledger contains every update 1–3,852. This arm supervises
zero-based native layers [8,16,24], averages their auxiliary losses, and retains
M5 R4/B1200 serving. Its tokenizer, processor, architecture and navigation
metadata match the L8 export byte-for-byte. The publisher removed 27 tensors
from the three training-only heads; no teacher or readout head is used during
navigation. Published final-export validation was CPU-only, so the earlier
qualification-checkpoint GPU evidence is not treated as a local learned-weight
replay pass.

Local evidence root: `evaluation/g3_m5_l8_l16_l24_epoch1_20261018`.
All 253 saved source hashes, 241 publisher source-seal hashes, 14 pinned base
files and 22 scene assets reverified. All 109 targeted tests passed with no
skips after supplying the pinned processor path; the initial log with two
missing-path skips is retained. All 11 Habitat scenes passed fresh render and
navmesh checks; the 128-frame replay fixture and all 1,839 episode IDs exactly
match the prior local runs. Four-worker coordinator cleanup passed both child
failure and interruption tests.

All 19 Hub files and all 18 publisher-manifest entries verified, including the
complete weight SHA-256 above. Local strict GPU loading found 724 finite native
tensors. Learned-checkpoint qualification **passed** with unchanged tolerances:
64/128-step replay RMS 0.034710/0.032162, maximum absolute drift 0.375/0.5;
one raw argmax change was recorded at each length. The committed-scorer/native
FLA relative error was 0.002095 (limit 0.01). Exact retry, evolved-session reset,
and greedy reset checks passed. The 520-step test held peak resident KV at
2,546 (guard 4,784), with all eight episodic banks ending at 1,200 rows.
The two-episode Habitat smoke completed with zero execution failures (one
success); its SR is only a smoke statistic.

The full evaluation launched at **2026-10-18 20:58:17 Asia/Ho_Chi_Minh**
(13:58:17 UTC), with four disjoint panels covering all 1,839 episodes. It uses
R2R VLN-CE `val_unseen`, seed 42, a 500-action cap, and the unchanged
simulator/metric protocol. Root `status.json` and `full/summary.json` now record
completion; `full/shard-00` through `shard-03` preserve original episode journals.

**Final result:** 916 successes / 1,839 episodes, **SR 49.809679%, SPL 46.265741%**,
zero execution failures and zero invalid responses. All four workers exited 0.
Final validation passed at **2026-10-18 18:27:21 UTC / October 19 01:27:21
Asia/Ho_Chi_Minh**, including exact full coverage, common contracts/runtimes
and recomputed metrics. This update independently rechecked those results.
Both serving qualification and final aggregation passed for this checkpoint;
the separate L8 checkpoint's failed replay remains recorded under its own run.

## Next update

All six local R2R evaluations in the October 19 table are complete; L8 remains an unqualified diagnostic. Refresh historical A1 only when a verified 1,839-episode final journal is available, then replace its dated partial row and shared-prefix comparison with full results. Preserve the checkpoint/source identity and the dated history. No automated refresh is configured by this report. Publishing this file to multiple branches is documentation distribution, not evidence that every branch can load every listed checkpoint.
