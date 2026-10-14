# A1 continuation after A0 — 2026-10-14

Status after deployment: implementation and CPU review complete; persistent watcher3361209 is waiting for A0. Worker-2 H100 qualification is pending A0 completion. This report supersedes the earlier A1 deferral. No production A1 success or memory qualification is claimed here.

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

## Pending H100 gates and scheduling

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

## Verified deployment

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
