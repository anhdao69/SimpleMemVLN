# M5 GDN S campaign — R2R evaluation results

Verified snapshot: **2026-10-22T20:40:50.657074+07:00** (Asia/Ho_Chi_Minh). This report covers the
available S1, S2, S3 and S5 trained epoch-1/seed429 checkpoints and the separate
S1 FP32 serving experiment. It does not claim evaluations of unpublished or
unevaluated arms. Four runs are final; S5 is a fixed partial snapshot.

## Navigation results

| Model | Completed / 1,839 | Successes | SR (%) | SPL (%) | Status |
|---|---:|---:|---:|---:|---|
| S1 BF16 | 1,839 | 921 | 50.081566 | 46.539055 | FINAL |
| S1 FP32 | 1,839 | 917 | 49.864057 | 46.432850 | FINAL |
| S2 BF16 | 1,839 | 926 | 50.353453 | 46.896355 | FINAL |
| S3 BF16 | 1,839 | 928 | 50.462208 | 47.152476 | FINAL |
| S5 BF16 | 1,118 | 561 | 50.178891 | 46.343560 | PARTIAL — RUNNING |

All five snapshots have zero recorded execution failures and zero invalid
responses. Final navigation validation does **not** mean numerical replay parity
passed: S1 BF16 and S5 BF16 have disclosed failures below. S5 has
721 episodes remaining at this snapshot.
Its current SR must not be ranked against full-split SR without matching IDs.

| Model | NE (m) | Oracle success (%) | Trajectory length (m) | Mean actions | Forced STOPs | Peak KV rows | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 BF16 | 5.774322 | 57.531267 | 9.653281 | 86.9576 | 108 | 2646 | 8.6909 |
| S1 FP32 | 5.921898 | 56.443719 | 9.523072 | 89.2474 | 122 | 2646 | 17.3715 |
| S2 BF16 | 5.797960 | 56.552474 | 9.309996 | 94.9337 | 145 | 2646 | 8.6781 |
| S3 BF16 | 5.809722 | 57.150625 | 9.351083 | 92.9277 | 138 | 2646 | 8.6781 |
| S5 BF16 | 6.342745 | 57.334526 | 9.745493 | 90.8828 | 74 | 2600 | 8.6775 |

S5's second-table measurements are also partial. nDTW is unavailable in these
journals (zero coverage), not zero performance. Latency measurements were taken
under varying concurrent GPU workloads and are not controlled speed benchmarks.

## Matched comparisons and interpretation

S3 has the highest observed final SR among these completed S-arm runs:928/1839,
seven successes more than S1 BF16 and two more than S2. These small descriptive
differences alone do not establish a statistically reliable advantage.

S1 FP32 versus S1 BF16 uses the same checkpoint and full episode IDs:
917 versus921 successes, a difference of -0.217510 percentage points. SPL is
46.432850% versus46.539055%. Paired outcomes:784 both succeed,137 BF16-only
successes,133 FP32-only successes,785 both fail. Similar aggregate results do
not imply identical trajectories or per-episode behavior.

The FP32 experiment also changes text attention to exact reference attention
and vision attention to SDPA, disables TF32, and uses Triton IEEE FP32 products.
Thus the comparison measures this serving configuration as a whole, not dtype
alone. FP32 improved replay agreement on the saved fixtures, but did not improve
aggregate SR in this completed run. This is not proof that numerical differences
are harmless in general or that all implementation logic is correct.

All five models on the same completed episode intersection:

| Model | Matched episodes | Successes | Matched SR (%) |
|---|---:|---:|---:|
| S1 BF16 | 1118 | 580 | 51.878354 |
| S1 FP32 | 1118 | 567 | 50.715564 |
| S2 BF16 | 1118 | 584 | 52.236136 |
| S3 BF16 | 1118 | 568 | 50.805009 |
| S5 BF16 | 1118 | 561 | 50.178891 |

The intersection is an ordered completed subset, not a random sample. S5's
full-split conclusion remains pending. Historical P2-M5 and the scientific
S0=M5_E1 checkpoint are distinct identities; this refresh does not silently
substitute one for the other or report new S0 navigation measurements.

## Model and evaluation contract

All S arms here were trained separately with seed429 for one joint R2R + English
RxR15deg epoch (30815 unique episodes,3852 optimizer updates,116 warmup updates).
They use episodic_window_gdn_sweep_v1, four recent groups including current,
1200 old native visual K/V rows per full-attention layer, original positions,
birth-fixed priorities, resident-plus-newly-eligible selection, original-ID ties,
and no resurrection. Native GDN state remains active for every arm.

| Arm | Priority, mean over heads | Scorer backend |
|---|---|---|
| S1 | Unscaled pre-write residual norm ‖v − Sbarᵀk‖ | fp32_torch_unscaled_residual_block64_v1 |
| S2 | β‖v‖ | fp32_native_prepare_only_v1 |
| S3 | ‖v‖ | fp32_native_prepare_only_v1 |
| S5 | Cosine prediction error | fp32_torch_unscaled_residual_block64_v1 |

S5 uses 1−clip(dot(prediction,value)/(norm(prediction)*norm(value)+1e-6),−1,1),
with zero score for value norm<=1e-6. The prediction uses the decayed prior state.
S5 averages heads; S4's minimum-over-heads rule is not used. No teacher,
calibration artifact, learned scorer or auxiliary head is required for these arms.
BF16 runs retain FP32 score arithmetic as prescribed by their training contracts.

Protocol:1839 R2R VLN-CE val_unseen episodes; Habitat-Lab/Sim0.2.4; seed42;
500-action cap;3m success radius;640x480 RGB;79deg HFOV;0.25m forward;15deg turns;
greedy canonical text actions and action history. This report contains R2R
navigation results; joint RxR15 training does not make these RxR evaluations.
Dataset SHA256:1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3.
Habitat config SHA256:dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f.

## Numerical qualification and limitations

Unchanged BF16 replay tolerance: elementwise atol0.75 + rtol0.03, RMS<=0.15.
A maximum error above0.75 can pass because the relative term also applies.
Forced response histories matching is not independent greedy action agreement.
The current workstation fixtures and saved previous-M5 fixture are distinct.

| Model | Fixture | Max logit error | RMS | Top-token agreement (%) | Mean selection overlap (%) | Numerical gate |
|---|---|---:|---:|---:|---:|---|
| S1 BF16 | current64 | 1.156250 | 0.058749 | 98.9529 | 98.3917 | FAIL |
| S1 BF16 | current128 | 1.500000 | 0.056971 | 98.6945 | 97.7609 | FAIL |
| S1 BF16 | previous-M5-128 | 1.062500 | 0.046826 | 99.2167 | 97.7476 | PASS |
| S2 BF16 | current64 | 0.718750 | 0.038884 | 99.4764 | 99.2474 | PASS |
| S2 BF16 | current128 | 0.937500 | 0.039204 | 99.2167 | 99.0150 | PASS |
| S2 BF16 | previous-M5-128 | 0.750000 | 0.036911 | 99.7389 | 98.9595 | PASS |
| S3 BF16 | current64 | 0.875000 | 0.045260 | 97.9058 | 98.6597 | PASS |
| S3 BF16 | current128 | 0.875000 | 0.046616 | 98.6945 | 98.0911 | PASS |
| S3 BF16 | previous-M5-128 | 0.765625 | 0.042803 | 100.0000 | 98.0994 | PASS |
| S5 BF16 | current64 | 0.562500 | 0.041939 | 99.4764 | 98.6179 | PASS |
| S5 BF16 | current128 | 0.750000 | 0.043282 | 99.4778 | 98.0808 | PASS |
| S5 BF16 | previous-M5-128 | 0.875000 | 0.043358 | 100.0000 | 98.1618 | FAIL |
| S1 FP32 (saved diagnostic) | 64 | 0.140746 | 0.003975 | 100.0000 | 99.9561 | PASS |
| S1 FP32 (saved diagnostic) | 128 | 0.143094 | 0.004092 | 100.0000 | 99.9283 | PASS |
| S1 FP32 (saved diagnostic) | previous_m5_128 | 0.079043 | 0.002248 | 100.0000 | 99.9552 | PASS |

S1 BF16 and S5 BF16 continued under the user's existing authorization with
numerical failures explicitly retained in evaluation_decision.json and
campaign_disclosed.py. No precision/model/tolerance was changed to pass a gate.
S2/S3 passed all three saved BF16 fixtures; this is not bit-identical parity.
All BF16 arms passed strict load, scorer checks, bitwise reset, and520-step
bounded-cache tests (2546/4784 rows;1200 episodic rows at all8 attention layers).
S5's129-token actual-input score comparison with an independent FP64 recurrence
had maximum error4.006865e-7. That verifies the checked scorer inputs, not the
entire policy on every possible input.

S1 FP32 uses the separate saved precision diagnosis under
/data14t/anhdh35/m5_s1_replay_diagnosis_20261021/precision/:
three replay fixtures passed with100% top-token agreement and maximum errors
approximately0.08–0.14. This evidence is not a fresh rerun during this report update.
Attention validation and runtime dtype checks are saved in its evaluation root.

Source tests:191 selected tests passed for S1/S3. S2 reused the identical-source
result with a recorded reference. S5 ran147 memory_sweep tests and separately
referenced the broader identical-source191-test result. Checkpoint hashes,
finite tensors and checkpoint-specific GPU audits are separate evidence.

## Provenance and reproducibility

Branch:m5_gdn_s_sweep; frozen evaluation source fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6.
Model/config trees match training source dabf4ad085d20512640fe0ba3bda72ad4724dc76.
The FP32 experiment's operational overrides are separately retained; the frozen
BF16 model source was not rewritten. Checkpoint repo:anhdao69/M5_S.

| Model | HF pinned revision | Weight SHA256 |
|---|---|---|
| S1 BF16 | `0a481ea1d3b84907981d03c028c543aa2d8317e8` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` |
| S1 FP32 | `0a481ea1d3b84907981d03c028c543aa2d8317e8` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` |
| S2 BF16 | `54988913b2fc85de3dbac1c8df8398dabb4b916b` | `88dba8325a014868ca0633634ff0725dbb05b255e36b3d8fb9c78acefeab3044` |
| S3 BF16 | `a910d9429eb2e9d4bf7271f68d02c99abf4c4a4e` | `172b541a79a42d493c5acdb9c8c5478f8f540669577893cbf7163f94b0b0e9d1` |
| S5 BF16 | `2f4883da8a96a41b341a55114aeab531449bdee7` | `6d4c8e68101ec4dca67b6a8a771108ebdf2877a68fc5abd7dd10955162426f6b` |

| Model | Evaluation root | Run report |
|---|---|---|
| S1 BF16 | `/data14t/anhdh35/m5_s1_epoch1/evaluation` | `/data14t/anhdh35/m5_s1_epoch1/evaluation/REPORT.md` |
| S1 FP32 | `/data14t/anhdh35/m5_s1_fp32_epoch1/evaluation` | `/data14t/anhdh35/m5_s1_fp32_epoch1/evaluation/REPORT.md` |
| S2 BF16 | `/data14t/anhdh35/m5_s2_epoch1/evaluation` | `/data14t/anhdh35/m5_s2_epoch1/evaluation/REPORT.md` |
| S3 BF16 | `/data14t/anhdh35/m5_s3_epoch1/evaluation` | `/data14t/anhdh35/m5_s3_epoch1/evaluation/REPORT.md` |
| S5 BF16 | `/data14t/anhdh35/m5_s5_epoch1/evaluation` | `/data14t/anhdh35/m5_s5_epoch1/evaluation/REPORT.md` |

Each root retains run_eval.py, verify_result.py, models.json (or its experiment
reference), results/full/<arm>/episodes.jsonl, summary.json, runtime.json and
evaluation_contract.json. BF16 audits live under audit/<arm>/; FP32 retains
server_fp32.py and fp32_patch.py. Final validators were rerun for all four
completed runs during this refresh. S5 metrics were recomputed from one captured
journal prefix and its IDs/actions checked; no final validation is claimed.

Machine:VMO-ANHDH35-D. Model Python:/storage/anhdh35/SimpleMemVLN/.venv/bin/python.
Habitat Python:/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python.
Base model:/storage/anhdh35/SimpleMemVLN/artifacts/checkpoints/qwen-base.
Dataset:/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed.
Scenes:/storage/anhdh35/JanusVLN/data/scene_datasets_full.

[Machine-readable evidence snapshot](evidence/S_CAMPAIGN_SNAPSHOT_20261022T204050+0700.json) records exact counts, recomputed
metrics, model/runtime/contract metadata, qualification outcomes, paired results,
and SHA256 identities of the journal snapshot and supporting files. For S5 the
journal hash covers only the recorded prefix length; its live file continues to
grow. This is a timestamped report, not a live dashboard.
