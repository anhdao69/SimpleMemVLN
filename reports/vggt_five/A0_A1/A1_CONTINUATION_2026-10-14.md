# A1 continuation after A0 — 2026-10-14

Current status: A1 completed all3,852 updates (epoch1.0) successfully on2026-10-15 at13:19:30UTC /20:19:30Vietnam time. SupervisorPASS, exit0, zero GPU/host OOMs. The verified serving model is published at [anhdao69/Streaming_VGGT/A1/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/main/A1/epoch-1), Hub revision `37f877aceed4dfd5b22fca9508150ff693b169b2`. Earlier progress sections below are historical; the final section records completion and publication.

## Scientific setup retained

A1 independently initializes the pinned original Qwen3.5-4B base, seed429. It does not initialize from A0, G3, or another trained policy. One joint R2R + English RxR15 epoch contains30,815 episodes, global batch8 (one episode per GPU, four GPUs, two accumulation microsteps),3,852 optimizer updates. LR warmup116 and one-epoch cosine schedule match A0; backbone peak5e-6, GeoAddress adapter peak5e-5. Auxiliary weight ramps to0.1 over771 updates independently of LR warmup.

The shared A0/A1 runtime is image-token-only rank8 native Q/K adapters at decoder24, rawheads0/1,45,056 parameters, zero-initialized up projections. Native Window8, text action/history policy, full episode BPTT and serving adapter export remain intact.

A1 uses frozen resident FP32 online VGGT-Omega-1B-512, independent single-view images, no preprocessing feature cache, no PCA, no dimension converter MLP, no added positional descriptor. The user previously explicitly selected direct native Q/K relation supervision. The two feature widths need not match: teacher2048-wide and native128-wide features both form32x300 similarity distributions. Up to8 delayed source/read pairs per episode use lag8–32. KL(teacher || student) averages32 queries and2 heads, then uses the global pair denominator. Teacher targets are detached; earlier student keys remain differentiable. Empty-support episodes remain legal.

CPUAdam/ZeRO2, pinned activation arena, four CPU math threads, eight loader workers, teacher batch8, vision batch8 and Window8 attention batch32 retain the prior optimized configuration. No truncated BPTT or schedule changes were introduced.

## Continuation fixes

1. Qualification seals now explicitly name qualified arms and require the complete gate inventory. A passing A0-only record cannot authorize A1. Failed, missing, duplicate or source-mismatched evidence is rejected. The production launcher supplies its arm when checking the seal.
2. Qualification and queue CLIs support `--arms A1`. The continuation uses the existing completed A0 run as predecessor and does not retrain A0.
3. The queue waits for successful supervisor finalization, a validated complete checkpoint3852, and epoch completion. It also waits until the earlier GPU Slurm step has exited. The interactive shell step is the only ordinary step permitted during idle checks.
4. A fresh independent review reproduced a restart race: an orphan A1 could complete while a restarted watcher waited for idle, leading to a duplicate invocation. The launch path now rechecks A1 completion after waiting. A regression first reproduced the unwanted launch and then passed; an incomplete-run case verifies legitimate resume still launches.
5. Existing queue and launcher file locks prevent concurrent controllers/launches for the same run. Interrupted production uses the same run directory and full-state automatic resume. A qualification failure blocks production and preserves failed evidence; it does not silently skip a gate or retry indefinitely.

## Validation evidence

- Targeted scientific, runtime, export/finalization, QC and queue tests:39 passed after the review fix.
- Full local suite after review fix:369 passed,35 skipped,4 environment failures (before the fix:367 passed). The failures require cluster `squeue`, DeepSpeedCPUAdam, the VGGT-Omega package and `VLN_MODEL_PATH`, respectively. These are not counted as passes; a full server suite is mandatory before production.
- Production launcher shell syntax passed.
- Independent fresh-context review found the restart race above and no other concrete inspected launch/scientific blocker. GPU memory, distributed execution and navigation quality were not certified by that review.

## Historical pending H100 gates and scheduling

Worker-2 allocation4925 has four H100s,60 CPUs and768GiB RAM. Existing A0 step4925.41 remains untouched. Worker-2's other GPUs were allocated when this continuation was prepared; tests must wait for A0.

Sequence: successful A0 checkpoint/export and step exit -> A1-only H100 qualification -> A1 production only on a source-bound PASS seal. This adds qualification time between A0 and A1. It avoids interrupting A0 or using another worker.

Required gates: initial full regression; online teacher QC; mixed episodes; mixed empty support; longest RxR and R2R; offload threshold and mixed extremes; uninterrupted versus fresh-process checkpoint resume; trained serving export/replay on both datasets; representative R2R/RxR timings; distributed pair normalization; final full regression. The longest RxR fixture has627 frames and199,735 serialized tokens. Qualification requires allocated GPU memory <=72GiB; profile reserved-memory and host guards remain active. Production host working-set guard is691.2GiB and free-disk guard220GiB.

The login-side watcher survives an SSH/VPN disconnect and uses only worker-2. It does not reserve another allocation. If allocation4925 itself is cancelled, the watcher blocks; preserved full checkpoints permit explicit restart in a suitable worker-2 allocation. There is no claim that a cancelled allocation automatically reappears.

No reliable full-epoch A1 ETA is available until the representative worker-2 measurements complete. Earlier short worker-1 mixed smoke timing is historical and is not substituted for final qualification.

## Decisions and limits

- Keep A0 running and wait for its GPUs; cost is a qualification gap before A1.
- Permit deployment of the fail-closed watcher with local environment failures, but require server tests to pass; cost is possible blocking during qualification.
- Treat hardware correctness and memory capacity as pending measurements; no100% correctness or improvement claim.
- Verify remote process state and artifact hashes separately from the read-only reviewer.
- Leave downstream navigation performance to evaluation; successful training alone does not establish improvement.

## Historical deployment before the identity repair

- Committed runtime source: `70455f07dd5dc4f65f808212e744e7745625b759` on `v2_vggt_omega_a1`.
- Archive SHA256: `bf4216ed40fc829cef4e0e0378c7e8225e46ba6db9ec47f35be99a3b6e51db60`.
- Campaign: `/mnt/data/vmo-ai-task/anhdh35/a1-omega-worker2-20261014-continuation`.
- Immutable source: campaign `source-70455f0`; all175 runtime file hashes matched the local source after extraction.
- Server CPU queue suite:21 passed in4.82s. This is deployment verification, not H100 qualification.
- Detached login watcher PID3361209; launch metadata in `watcher_launch.json`, state in `queue_state.json`, console log in `watcher.log`. It holds `queue.lock`.
- Verified initial state: `waiting`, stage `predecessor`, arms `[A1]`, allocation4925, worker-2. No A1 GPU step launched while A0 was active.
- Predecessor: `/mnt/data/vmo-ai-task/anhdh35/a01-omega-worker2-20261014-retry/a0_joint_e1`.
- Future A1 output: campaign `a1_joint_e1/train`. Qualification evidence: campaign `qualification`; production cannot start without `qualification_pass.json` for A1.

A1 hardware gates remain pending. Preserve this distinction when inspecting the queue later; a running watcher is not a passing GPU test or a running training process.

Verification on a new SSH connection at2026-10-14 19:25:24UTC: watcher3361209 remained alive with parentPID1; queue state continued refreshing as waiting; Slurm contained only shell4925.0 and A0 step4925.41. A0 advanced to2904/3852 with zero recorded CUDA/host OOMs. Captured queue and launch metadata are in `evidence/a1-continuation/`.

## Teacher-identity repair after the first qualification attempt

A0 subsequently finished all3,852 updates with supervisorPASS and zero new OOMs. A1 automatically entered worker-2 step4925.44. The initial server suite passed408tests and teacher QC passed; the first mixed training smoke then rejected the configured A1 teacher identity before update1. The queue correctly blocked production.

Diagnostics found a concrete portability defect in the identity: it included the parent G1 positional grid's exact floating-point bytes, although A1 replaces that grid with literal zero before producing any target. The same source generated position hashes `dfee8e96...` on worker-2 and `4faa61e...` on the login node. Two instrumented training reruns did not reproduce the original mismatch, so the exact field differing in the original rank remains unrecorded; this report does not claim that the intermittent original failure was conclusively traced to that field.

The repair removes only unused `position` and `position_sha256` fields from A1's persisted teacher identity. It retains weight, external-source, preprocessing-code, base-adapter, A1-adapter, target, layer, dimensions and pooling checks. The unchanged parent constructor still validates its complete locally computed identity. A1 still replaces the parent grid with literal zero; descriptor computation and direct-Q/K relation loss are unchanged. G1 and G3 code/contracts are unchanged. Production A1 config records the new A1 source hash.

Regressions first reproduced the false rejection when only unused positional bytes changed (2failed/1passed), then passed after repair. They also verify that changing teacher weights is still rejected and that strict parent validation plus zero-position initialization remain intact. A fresh immutable snapshot and full worker-2 qualification are required before any production start; previous failed evidence is preserved.


## Repaired qualification and production — verified 2026-10-14 23:29:24 UTC

Runtime commit `6e717787313b2218cd2fd40b059f2942fee1820f` is pushed to GitHub branch `v2_vggt_omega_a1`. The immutable server source is `/mnt/data/vmo-ai-task/anhdh35/a1-omega-worker2-20261014-repair/source-6e71778`; archive SHA256 is `8443b5f1c474b01e320856e54e31868436b71def6b3bdd5c3aab329786e07d87`. All175 runtime-file hashes matched the deployed source. The repaired A1 identity also validates on the login CPU. G1/G3 and completed A0 sources and artifacts were not modified.

The detached watcher3370976 completed qualification in step4925.49 and launched production step4925.50 at approximately23:24:34UTC. Source-bound `check_qualification(..., arm='A1')` was independently rerun successfully after launch. Its seal contains all16 required gates with returncode0:

- Initial full regression:411 passed in48.30s; final full regression:411 passed in46.51s.
- Online teacher QC; mixed episodes and mixed empty support.
- Longest RxR and R2R; offload threshold; mixed extreme lengths.
- Uninterrupted recovery reference, fresh-process resume and numerical comparison.
- Trained serving export/replay on both datasets.
- Representative R2R and RxR timings; distributed pair normalization.

The longest RxR fixture (627frames,199,735tokens) completed three updates with peak allocated61.729GiB and peak reserved76.463GiB. The mixed extreme test completed three updates with peak allocated61.729GiB and reserved69.002GiB. No new CUDA/host OOM was recorded in the qualification runs. These are measured cases, not a guarantee against every future resource failure. The production RAM working-set guard remains691.2GiB within the768GiB allocation, with CPUAdam/ZeRO2 and pinned activation offload enabled. Total cgroup usage can approach its limit because it includes reclaimable clean filesystem cache; the working-set metric and actual OOM events are recorded separately.

Fresh-process recovery passed numerical comparison: whole-model relative L2 difference9.43e-9, adapter relative L2 difference5.23e-8, maximum absolute parameter difference1.40e-7, and matching compared losses0.416747719. This is numerical agreement, not bitwise equality. All native Q/K up projections changed during the continued update, and the exported serving model retained its45,056 adapter parameters.

| Representative four-H100 fixture | Steady seconds per optimizer update | Episode exposures |
|---|---:|---:|
| R2R | 9.73 | 64 |
| English RxR15 | 19.31 | 64 |

These timings exclude initialization and the first interval. They use short repeated representative fixtures; they are not measured full-epoch durations. Production has variable episode lengths, global batch8, and periodic full-state checkpoint I/O. No new speed change was introduced after qualification.

### Production configuration and recovery proof

The actual resolved configuration and emitted training contract confirm original Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,30,815 joint episodes, one epoch,3,852 updates,116 LR warmup updates, microbatch1 per rank and accumulation2 across four ranks. All four ranks emitted `SCHEDULE_GATE_PASS (3852, 116)`. Window8, canonical text action history, full episode BPTT, backboneLR5e-6 and GeoAddress adapterLR5e-5 are retained. The run starts independently from base; it does not adapt A0/G3 weights.

The actual teacher configuration names `VGGT-Omega-1B-512`, frozen resident FP32 online evaluation and `direct-native-qk-cosine-v1`. It uses the Omega source and weights under `third_party/vggt-omega` and `models/vggt-omega/vggt_omega_1b_512.pt`. There is no PCA, pre-extracted feature bank or dimension-conversion MLP. Teacher2048-dimensional and student128-dimensional features form relations over the same image patches; matching those relation distributions does not require matching feature widths. This preserves the user's selected direct native Q/K objective.

At23:29:24UTC, all four production profiles showed update12/3852 with zero CUDA OOMs. Observed production peaks through that point were47.762GiB allocated and52.029GiB reserved. All recorded action and auxiliary losses were finite. Each rank had nonempty supervised pairs with positive auxiliary weight. For example, rank0 update12 had8 pairs, global update denominator64, auxiliary loss sum3.982134 and ramp weight0.00155642. Thus the online teacher loss is being computed and included in training.

`validate_complete(train/checkpoint-10)` passed: step10, world size4,22 nonempty files with the recorded sizes, including model, distributed optimizer/scheduler, RNG states and resume contract. The total checkpoint size is70,588,068,573bytes. Training advanced beyond the save. The launcher keeps full-state automatic resume in the same run directory, saves first at10 and every100 updates thereafter, and preserves epoch-boundary recovery. If the allocation is cancelled, restart still requires a suitable worker-2 allocation; this is not an automatic replacement allocation.

Production directory: `/mnt/data/vmo-ai-task/anhdh35/a1-omega-worker2-20261014-repair/a1_joint_e1/train`. Watcher state and supervisor logs remain in the parent campaign. The initial failed campaign is preserved separately. Detailed captured source seal, all experiment results, recovery comparison, resolved config, four-rank progress, auxiliary records and checkpoint inventory are in [verified_production.json](evidence/a1-repair/verified_production.json).

The final independent review found no Critical/Important findings and independently passed the three identity regressions. The full epoch and navigation evaluation remain pending; successful qualification and early training do not establish downstream navigation quality.


## Completed epoch and verified Hugging Face publication — 2026-10-15

A1 finished3852/3852 updates, epoch1.0, and the queue recorded complete at13:19:30UTC (20:19:30Vietnam,09:19:30New York). The supervisor returnedPASS/exit0 after50,095.94seconds, approximately13h55m including startup/finalization. Mean update wall time was12.9686seconds. The action-weighted epoch loss was0.124885; this is distinct from the combined navigation-plus-auxiliary Trainer loss. No navigation-quality evaluation is implied.

All four ranks completed the run. No new host OOM or OOM-kill events occurred, and recorded CUDA OOM counts were zero. Peak allocated GPU memory was61.729GiB, peak reserved72.057GiB, and sampled peak RAM working set475.655GiB within the768GiB allocation. Clean filesystem cache is recorded separately from working memory. The final complete recovery checkpoint3852 retains22 files, including distributed optimizer/scheduler and per-rank RNG state, on the cluster.

Publication validated all728 final model tensors: each is finite, has the same shape/dtype and is bitwise equal to the corresponding tensor in complete checkpoint3852. The four GeoAddress Q/K adapter tensors have the expected shapes and45,056 parameters, with nonzero trained up projections. The serving weight file was hardlinked unchanged; only research configuration was removed from serving navigation metadata. Native Window8, text action history and GeoAddress runtime adapters remain required. Frozen online Omega was training-only. No PCA, pre-extracted feature bank or dimension-conversion MLP was introduced.

Model folder: [A1/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/main/A1/epoch-1).
Immutable published revision: `37f877aceed4dfd5b22fca9508150ff693b169b2`.

The atomic Hub commit adds the A1 model/provenance and updates README/checksums while preserving existing G1/G3/A0 files. All23 files totaling10,371,469,392bytes were verified at that exact revision: large files by Hub LFS SHA256 and byte size, small files by downloaded-content SHA256 and byte size. Authentication used the existing cluster credential without embedding credentials in scripts or reports. The upload ran as a detached login-node CPU process; it did not start another GPU job or cancel allocation4925.

The model requires the GeoAddress-capable SimpleMemVLN runtime, training source `6e717787313b2218cd2fd40b059f2942fee1820f`, and original pinned Qwen3.5-4B base. VGGT-Omega is not required for serving. The published folder contains serving weights and provenance; full optimizer/RNG recovery checkpoints remain in `/mnt/data/vmo-ai-task/anhdh35/a1-omega-worker2-20261014-repair/a1_joint_e1/train`.

Evidence: [completed run and verified upload](evidence/a1-publication/verified_completion_upload.json); [publication script](evidence/a1-publication/publish_a1_epoch1_20261015.py). The script's existing-folder assertion intentionally prevents silently overwriting A1 if rerun. Remote publication evidence is under `a1_joint_e1/publication-a1-e1/UPLOAD_VERIFIED.json`.
