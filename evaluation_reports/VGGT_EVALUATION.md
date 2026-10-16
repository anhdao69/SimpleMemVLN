# VGGT campaign — R2R evaluation report

G3 has the highest completed SR among the evaluated versions in these two campaigns. G1, G3 and A0 are complete; A1 is still running. This is a navigation result report with implementation and provenance checks, not a claim that training success guarantees improved navigation.

## Snapshot and scope

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

## Results: VGGT versions and reference checkpoints

| Version (epoch 1) | Completed / scheduled | Successful | SR (%) | SPL (%) | Status |
|---|---:|---:|---:|---:|---|
| G1 | 1,839 / 1,839 | 914 | 49.700924 | 46.072125 | FINAL |
| G3 | 1,839 / 1,839 | 956 | 51.984774 | 48.095941 | FINAL |
| A0 | 1,839 / 1,839 | 910 | 49.483415 | 45.394182 | FINAL |
| A1 | 1,356 / 1,839 | 680 | 50.147493 | 46.306672 | RUNNING — partial |
| Window8-B | 1,839 / 1,839 | 907 | 49.320283 | 45.561094 | FINAL |
| V0 | 1,839 / 1,839 | 852 | 46.329527 | 41.756802 | FINAL |

Window8-B and V0 are reference runs, not VGGT-supervised versions. Their own training histories differ from the one-epoch VGGT schedule. The companion [episodic memory report](EPISODIC_MEMORY_EVALUATION.md) records all episodic arms.

### Additional navigation and memory measurements

| Version | NE (m) | Oracle success (%) | Mean steps | Forced STOPs | Peak retained KV rows | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 6.0579 | 55.8999 | 83.84 | 94 | 2718 | 8.668 |
| G3 | 5.7935 | 58.7819 | 94.51 | 133 | 2718 | 8.670 |
| A0 | 5.9695 | 55.3018 | 93.42 | 129 | 2718 | 8.670 |
| A1 | 6.4297 | 57.6696 | 100.47 | 114 | 2683 | 8.656 |
| Window8-B | 6.0358 | 59.8695 | 115.39 | 212 | 2718 | 8.668 |
| V0 | 6.4879 | 59.3801 | 126.32 | 240 | 2718 | 8.668 |

Peak retained KV is the observed run maximum, not the configured upper bound. All four VGGT-arm serving models use Window8 and a 4,096-row guard. V0 instead uses the generalized episodic R8/B0 path with a 5,120-row structural guard.

## What each version actually implements

| Version | Training supervision | Serving architecture | Teacher at inference? |
|---|---|---|---|
| G1 | Current visual residual alignment to online Omega descriptors, learned MLP projector | Native Window8; auxiliary projector removed | No |
| G3 | Delayed visual-region prediction from native GDN layer-24 state through read-only probes | Native Window8; probes, query/decoder heads removed | No |
| A0 | Navigation-only architecture control; no teacher loss | Window8 **plus trained GeoAddress Q/K adapters** | No |
| A1 | Direct native Q/K relation KL against online Omega teacher similarities | Same retained GeoAddress architecture as A0 | No |

### G1: current representation alignment

The implemented teacher is frozen **online FP32 VGGT-Omega-1B-512**, not an offline PCA feature bank. Independent single-view images are processed as `[B,1,3,448,592]`; layer-23 descriptors exclude 17 special tokens, pool a 28×37 patch grid to 15×20, and retain all 2,048 feature dimensions. G1 uses non-affine LN2048 plus 0.5×fixed 2D sin/cos positional targets.

The student target is the 300 image-token residual outputs after decoder index 23. Its training-only projector is affine LN2560 → Linear2560→4096 → GELU → Linear4096→2048 (18,885,632 parameters). Up to 16 stratified frames per episode are supervised with FP32 cosine loss, globally normalized by frame count; auxiliary weight ramps to 0.2 over 771 updates. The projector is absent from serving. No PCA, offline target cache, episodic KV bank or new serving lane is involved.

### G3: delayed information in native recurrent state

G3 also uses frozen online single-view Omega features, without PCA or added positional descriptors. Targets are four quadrant means plus one area-weighted global mean of the normalized 2,048-wide visual descriptors. Up to eight delayed source/read pairs use lags 8–32.

At native GDN index 24, a training-only query MLP uses detached normalized current native queries, a region embedding and lag encoding. Five read-only probes per pair use zero value/update/decay contributions and are removed before native output projection. A decoder predicts the 2,048-wide target from the read state. Auxiliary cosine loss uses a global pair denominator and weight ramp to 0.1 over 771 updates. Query/decoder/probes do not become inference tokens, KV rows or persistent serving modules. Serving is native Window8.

### A0 / A1: persistent GeoAddress adapters

Both arms retain four trained bias-free matrices at decoder 24: separate Q and K residual paths 2560→8→256, totaling **45,056 parameters**. They modify raw Q/K heads 0 and 1 (head width 128), only for image tokens, after causal convolution/SiLU and before native head expansion/L2 normalization. Native values, gates, convolution, text-token Q/K and other raw heads are unchanged directly by the adapter operation; subsequent hidden states can still differ. The adapters are part of the learned serving model and must not be stripped.

The explicit image-mask contract is `vln_append_only_chat_geoaddress_v1`. Initial prefill, cached multi-token append and single-token decode all use the adapters. Native recurrent/conv state remains in use; there is no extra episodic bank or lane. Seed 429 initializes down projections; up projections begin at zero, but the evaluated checkpoints contain nonzero learned matrices.

A0 is navigation-only (auxiliary weight 0). A1 independently starts from the same pretrained base, not A0's trained weights. Its frozen online Omega teacher produces unpositioned 2,048-wide descriptors. Each selected pair forms 32-query × 300-source similarity distributions. The student uses its actual 128-wide native Q/K at two heads, temperature 0.1 and native normalization; **feature widths need not match because the distributions have matching shape**. KL(teacher || student) averages queries/heads and normalizes globally by valid pair count. Earlier student keys retain gradients; teacher targets are detached. Maximum weight is 0.1 after a 771-update ramp. No PCA or dimension-conversion MLP is used in A1. Teacher and relation loss are training-only.

The late A1 identity repair excludes unused positional-grid bytes while retaining identities of used teacher weights/source/preprocessing; it does not change the serving equations. Descriptor similarity is not certified physical correspondence.

## Training exposure and interpretation

G1/G3/A0/A1 completed 3,852/3,852 updates of a true one-epoch schedule, warmup 116, seed 429, nominal batch 8 and joint R2R + English RxR15 data (30,815 episodes). Their published training topology is four GPUs/GAS2. Frozen vision/merger and full-episode BPTT are retained. Vision microbatch 8 and attention batch 32 are training execution settings; **the serving window is 8, not 32**.

Episodic V0/M1/M2/M2 c/M5/M5 c are epoch-1 exports at 3,852/7,704 with warmup 232. Window8-B also comes from a configured two-epoch schedule. Consequently G1/G3/A0/A1 versus those references is a practical checkpoint comparison, not a clean auxiliary-only experiment. A1 versus A0 is the more directly matched teacher-effect comparison, but A1 is unfinished and single-seed outcomes do not establish statistical robustness.

### Descriptive paired comparisons

The next table compares identical episode IDs, UIDs and instructions. For A1, the comparator is restricted to A1's completed prefix; these values are not the comparator's full-run result. No confidence interval or significance claim is made.

| A minus B | Shared episodes | A SR (%) | B SR (%) | ΔSR (pp) | ΔSPL (pp) | A-only / B-only successes | Scope |
|---|---:|---:|---:|---:|---:|---:|---|
| G3 − G1 | 1,839 | 51.9848 | 49.7009 | +2.2838 | +2.0238 | 286 / 244 | Full split |
| G1 − Window8-B | 1,839 | 49.7009 | 49.3203 | +0.3806 | +0.5110 | 281 / 274 | Full split |
| G3 − Window8-B | 1,839 | 51.9848 | 49.3203 | +2.6645 | +2.5348 | 298 / 249 | Full split |
| A0 − Window8-B | 1,839 | 49.4834 | 49.3203 | +0.1631 | -0.1669 | 303 / 300 | Full split |
| A1 − A0 | 1,356 | 50.1475 | 48.6726 | +1.4749 | +1.7893 | 225 / 205 | Partial shared prefix only |

G3's completed advantage over G1 is a descriptive comparison of different auxiliary designs. A0 is not an unchanged native baseline because it retains adapters. Partial A1 results cannot yet establish the full-split A1−A0 effect.

## Qualification and limitations

| Arm | Targeted local tests | Trained replay evidence | 520-step storage check |
|---|---|---|---|
| G1 | 33 passed; 1 training-only Omega preprocessing test deselected | 64/128 steps: RMS 0.030369/0.028647; max drift 0.328125; raw token flips 1/191 and 2/383 | Peak 2,618 / guard 4,096 |
| G3 | 24 passed | 64/128: RMS about 0.03275/0.03092; max drift 0.5; one raw token flip in each | Peak 2,618 / 4,096 |
| A0 | 25 passed | 64/128: RMS 0.039986/0.037494; max drift 0.75; flips 0/191 and 2/383 | Peak 2,618 / 4,096 |
| A1 | 33 passed | 64/128: RMS 0.031206/0.029061; max drift 0.5; flips 0/191 and 1/383 | Peak 2,618 / 4,096 |

All four exports passed strict loading, finite tensor checks, two real Habitat smoke episodes, reset/retry checks and recurrent-state/cache invariants before full launch. G1/G3 serving has 724 native tensors; A0/A1 has 728 including all four adapters. Source-manifest matches: G1 53 files, G3 57, A0/A1 63 each. Actual trained A0/A1 instrumentation verified image Q/K modifications, unchanged unselected heads/text rows at the adapter, and all cache modes.

G1/G3 training-wrapper reconstruction uses published native and auxiliary weights; A0/A1 reconstruction uses published native+adapter weights and training metadata. Their evaluation logits match the serving export bitwise on the checked fixture. Original cluster recovery artifacts were not independently available locally; this is not a claim of repeating the publisher's complete recovery-byte comparison.

Replay uses real rendered Habitat RGB and fixed forced canonical histories, checking every response token including EOS. The 64/128 fixtures contain 103 distinct images across the longer fixture and are not expert-quality navigation evidence. Inherited BF16 thresholds are atol 0.75, rtol 0.03, RMS ≤0.15. Raw argmax differences are explicitly retained: a tolerance pass is not bitwise or universally free-running action equivalence. The 520-step cyclic-image test establishes bounded storage on its fixture, not navigation SR. No thresholds were relaxed for these weights. The initial A0 audit monitoring hook was corrected for keyword hidden-state arguments; production model source was unchanged.

Remote H100 training/recovery/teacher qualification described in implementation reports is historical evidence; the local inference checks did not repeat the complete distributed training campaign. Checks cannot guarantee the absence of every possible bug.

## Versions without a navigation result in this report

R1 now has an implementation/training report on `v2_vggt_omega_r1`, but no R1 local navigation evaluation was launched or recorded in this campaign snapshot. R0 and any other planned VGGT variants likewise have no result in this report. Do not substitute training losses, qualification success or older unrelated dual-lane runs for missing benchmark results. G1, G3, A0 and A1 are the evaluated VGGT-family versions covered here.

### Observed latency under shared GPU load

These values summarize the evaluated episodes in this snapshot, including partial runs. They are contention-affected measurements, not controlled speed comparisons. Model timing excludes transport and Habitat; Habitat step timing includes simulation and rendering.

| Version | Model mean (ms/action) | Model p50 | Model p95 | Transport + model mean | Habitat step mean |
|---|---:|---:|---:|---:|---:|
| G1 | 92.53 | 86.01 | 143.56 | 98.95 | 2.22 |
| G3 | 185.31 | 179.03 | 226.68 | 191.35 | 4.13 |
| A0 | 167.26 | 144.22 | 478.96 | 173.45 | 4.39 |
| A1 | 311.12 | 226.82 | 726.21 | 317.92 | 7.33 |
| Window8-B | 178.47 | 172.80 | 227.79 | 184.47 | 4.66 |
| V0 | 224.70 | 229.61 | 238.74 | 230.79 | 4.35 |

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
| A0 | `/data14t/anhdh35/simplememvln_a0_eval` | `results/full/A0` |
| A1 | `/data14t/anhdh35/simplememvln_a1_eval` | `results/full/A1` |
| Window8-B | `/data14t/anhdh35/simplememvln_window8_b_eval` | `results/full/Window8-B-E1` |
| V0 | `/data14t/anhdh35/simplememvln_episodic_eval` | `results/full/V0` |

At each result directory, inspect `summary.json`, `episodes.jsonl`, `runtime.json` and `evaluation_contract.json`. Root `audit/`, `preflight.json`, tensor inventories, download manifests, test logs and `audit_notes.txt` (V0 uses `campaign_notes.txt`) retain qualification evidence. G3 shares its root with M5 and has arm-specific audit subdirectories. A0/A1 use `A0-status.json`/`A1-status.json`; G1 uses `status.json`; G3 uses `G3-status.json`.

For reproduction, use the immutable source and exact checkpoint above, the saved navigation configuration, pinned base/runtime, dataset/scene assets and frozen `r2r_episode_ids.json`. The existing root `run_eval.py` launches the model environment and Habitat environment separately. Inspect each script's CLI before use; G1 and older V0 scripts differ from the newer arm/mode launchers. Write a new output directory for a rerun instead of overwriting evidence. Never strip A0/A1 adapters or install an online teacher into their serving path.

Implementation references at immutable commits:

- [G1 report](https://github.com/anhdao69/SimpleMemVLN/blob/67b206b8a2e338e436e340c6c41dd146d4b2eb0e/reports/vggt_five/G1/implementation_report.md)
- [G3 report](https://github.com/anhdao69/SimpleMemVLN/blob/9d658c245b2b9d14232e8db0b235bf92e857a43d/reports/vggt_five/G3/implementation_report.md)
- [A0/A1 implementation ledger](https://github.com/anhdao69/SimpleMemVLN/blob/4f772d3bac877322e2cb58e127c17f87e202f17c/reports/vggt_five/A0_A1/IMPLEMENTATION_2026-10-14.md)
- [A1 completed training/publication report](https://github.com/anhdao69/SimpleMemVLN/blob/4f772d3bac877322e2cb58e127c17f87e202f17c/reports/vggt_five/A0_A1/A1_CONTINUATION_2026-10-14.md)

## Next update

Wait for all 1,839 A1 episodes and final validation, then replace its partial row and shared-prefix comparison with full results. Preserve the checkpoint/source identity and the dated history. No automated refresh is configured by this report. Publishing this file to multiple branches is documentation distribution, not evidence that every branch can load every listed checkpoint.
