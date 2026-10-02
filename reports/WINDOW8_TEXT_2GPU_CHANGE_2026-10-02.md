# Window8 text-policy two-GPU replacement — startup smoke gated

## Update-50 reporter failure and correction

Job 4625 passed the startup memory test (72.941 GiB reserved) and completed
50 production updates, then failed in the reporting callback. The original
reporter required exactly four profile rows per update despite the new two-rank
layout. Both rank files contain all 50 updates; this was not missing data or OOM.
No recovery checkpoint exists because the first save was scheduled at update 100.

The callback now passes `TrainingArguments.world_size` explicitly. The reporter
checks the exact expected rank files, one profile per rank/update, and exposure
coverage for each rank/update. Missing ranks and duplicate updates remain errors.
Regression tests exercise actual update-50 and epoch callbacks at 1, 2, and 4
ranks. No model, optimizer, loss, or schedule settings change.

Validation: 87 local tests passed, 10 GPU/tokenizer-dependent tests skipped.
The patched callback was replayed successfully against the actual failed
Window8 run and completed four-rank FullContext logs (first 50 and epoch 2).
Window8 first-50 report: action-weighted loss 0.3137337, last-update loss
0.2095579, mean 41.742 s/update, peak reserved 59.666 GiB.
Remote validation artifacts and corrected modules are retained under
`outputs/reporting-fix-20261002-xP6bW8/`. No jobs were changed or resubmitted
during this fix. Existing immutable job sources were preserved; a new submission
must package these corrected modules, not reuse the old cached source/launcher.

## Approved replacement submission

User approved running the mandatory smoke test inside the 768 GiB allocation
and cancelling interactive 4620. **New job 4625** replaces both pending 4595 and
held draft 4624. Both superseded pending jobs were cancelled; 4620 was explicitly
cancelled after 4625 was verified and released. Jobs 4593 and 4623 were untouched.

Slurm's cached batch-script SHA256 matched the submitted launcher. Source/data
checksums passed, and runtime checksums cover the CPU optimizer config, gate
helper, stress manifest, and launcher. Production cannot start unless a fresh
three-update longest-episode run exits successfully on both ranks, peak reserved
is <=78 GiB, and per-rank raw losses plus logged loss/gradient norms are finite.
Production starts a separate process from pretrained weights, not smoke weights.
Slurm stdout: `slurm-4625.log` in the campaign below. Fresh profiles are stored as
`profile-startup-4625-*`; full training logs are in `train/train.log`.

Local suite: **73 passed, 10 GPU/tokenizer-dependent skips**. Bash syntax passed.
Fresh review identified zero-based exposure numbering and the need for raw
per-rank finite-loss checks; both are covered by passing regression tests.
The original 240 GiB stress run failed; the 768 GiB startup test has not yet been
claimed to pass. Cancelling an allocation is not reversible; it does not delete
the retained logs or checkpoints.

## Historical preparation and evidence

User explicitly selected the existing text-output policy, not candidate logits.
Comparison baseline is FullContext text job 4593, not candidate job 4623.

Original pending job: **4595**, still unchanged. Replacement: **4624**, submitted
**held**, pinned to worker-2, dependent on **afterany:4620**. Interactive allocation
4620 has not been cancelled. Do not release replacement or cancel original until
the two-GPU gates below pass. A dependency permits scheduling after cancellation;
it does not grant scheduling priority or guarantee immediate start.

Campaign:
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/window8_text_2gpu_20261002-bu1eVU`.
Source is a checksum-verified copy of the original pending job's immutable source
at `outputs/window8_recovery_20261001/source`, not the candidate-logit branch.

Intended recipe is unchanged: joint R2R/RxR_15deg, pretrained Qwen3.5-4B, qwen_text,
Window8/GDN, two epochs, LR 5e-6, 7,704 steps, 232 warmup steps, global batch eight.
Parallel layout changes from 4 ranks × GAS2 to **2 ranks × GAS4**, still one
complete episode/microbatch. Resource request: 24 CPUs, 768 GiB RAM, 72 hours;
four loader workers/rank and OMP2. Retain both epochs and two rolling recovery
points; recovery interval 100 updates. No other training jobs are changed.

## Validation so far

Original four-GPU worst-case peak reserved: 76.943 GiB.
New two-GPU/GAS4 test repeated the same 627-observation episode in all eight
global slots (199,735 tokens/episode). Update 1 succeeded at 77.002 GiB reserved,
loss 0.349, 132.5 s/update. **Update 2 failed with CUDA OOM** after optimizer
states existed: requested 3.43 GiB with only ~1.3 GiB free. This proves simply
reducing GPU count is unsafe; the old four-GPU memory gate cannot be reused.
The failure did not cancel allocation 4620.

Proposed memory-only adjustment: ZeRO-2 CPU optimizer offload with the existing
PyTorch AdamW implementation (`zero_force_ds_cpu_optimizer: false`). Model,
objective, LR groups and training budget remain unchanged. This is slower and
not a throughput-matched or bitwise-identical hardware comparison.

An initial offload reference test without that flag triggered Accelerate's
automatic DeepSpeedCPUAdam replacement, which failed its CUDA 12.8/12.9 build
check. The new flag is supported in the installed Accelerate implementation
and avoids replacing the existing optimizer; no version guard is disabled.
The corrected config passed the two-rank distributed text-loss reference test
(three updates, max weight error 0) and episode-sampler checkpoint resume test
(resumed updates 2–3, max error 0).

On the subsequent SSH retry, the full-model test ran as step **4620.27** with
CPU optimizer offload. It failed before the first complete optimizer update:
Slurm recorded **OUT_OF_MEMORY**, exit `0:125`, with an explicit `oom_kill`
event; rank 1 received SIGKILL. This is a **host-memory limit failure**, not a
reported CUDA OOM. Allocation 4620 has only **240 GiB RAM**, whereas replacement
4624 requests **768 GiB**. The test cannot certify either the GPU peak or the
production RAM requirement. Logs are retained in `profile-worst-2gpu-cpu.log`
and `profile-worst-2gpu-cpu/` in the campaign directory.

Latest verified queue: 4624 held, original 4595 pending, interactive 4620 running.
No production job was released/cancelled. Need a larger-memory validation
allocation or an explicitly approved fail-closed startup test under the 768 GiB
replacement allocation. All eight worker-2 GPUs are currently allocated.

## Historical continuation plan (superseded by approved startup gate)

1. Inspect both jobs/allocation before doing anything.
2. Obtain a sufficiently large host-memory allocation without changing or
   cancelling interactive 4620 without permission.
3. Preserve the tested CPU-offload config and passing reference/resume logs.
4. Re-run the three-update worst-case test with `--deepspeed` pointing to the
   corrected config, in a **new** profile output directory. Require all ranks
   finish updates 1–3 without OOM/nonfinite loss, peak reserved <=78 GiB.
5. Publish two-gpu-PASS.json (passed, updates, peak_reserved_gib, world_size=2,
   accumulation=4, global_batch=8, output_mode=qwen_text). Update runtime.sha256
   to the tested config. Verify all checksums and held job 4624's batch script.
6. Only then cancel job 4595 if it is still pending, verify cancellation and
   release 4624. Leave 4620 for the user to cancel. Record final state/results.

Local Bash syntax and suite passed (64 passed, 10 tokenizer/GPU skips). Launcher
draft: `train/slurm/window8_text_2gpu.slurm`. This document does not claim the
replacement is ready to train.
