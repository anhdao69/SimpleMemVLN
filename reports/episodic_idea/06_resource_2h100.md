# 06 — H2 runtime profile (2×H100, GAS4, ZeRO-2 CPU optimizer offload)

Profile: `configs/episodic/profile_h2.yaml` + `configs/deepspeed_zero2_cpu_optimizer.json`
(`zero_force_ds_cpu_optimizer: false` keeps the inherited torch AdamW; no optimizer
replacement). Same staging/offload/chunking settings as H4. Global batch 8 = 2 ranks × GAS4.
This is **not** numerically identical to H4 and is never resumed from/into H4.

Qualification status: **not qualifiable in the interactive allocation**; full suite
**pending in-job** (job 898756, 512 GiB).

## Interactive evidence (128 GiB, cgroup v1)

| Attempt | Guard | Outcome |
|---|---:|---|
| Debug smoke while the real-model replay shared the cgroup (dev source) | 80 GiB | Guard SIGTERM at 85.8 GiB (V0) / 94.5 GiB (M2) working set before update 1 completed. Not an OOM; FAIL retained. |
| Smoke, chunked optimizer groups, nothing else running | 80 GiB | Guard SIGTERM at 83.8 / 80.3 GiB during the first optimizer step (Adam moments allocated lazily). FAIL retained. |
| **Labelled path smoke**, guard raised to 110 GiB (0.86 × limit) — *not* a qualification | 110 GiB | **PASS**: 3 updates on both ranks, finite losses, exact exposures, GPU reserved 23.2 GiB, per-rank RSS 48.2 GB, sampled working set 102.1 GiB, no OOM kill. Update time 13.3 / 7.9 / 7.9 s on tiny episodes ⇒ the CPU AdamW step costs ≈6.5 s per update. |

Conclusion: the H2 CPU-optimizer path runs, but its optimizer state alone (≈34 GB per rank:
FP32 master + two moments + FP32 gradient partition for half the parameters) makes the
128 GiB allocation unsuitable for any honest H2 gate. This is a blocker of the interactive
allocation, not of the profile.

## Production request and forecast

Per rank at the longest episode: ≈48 GB (optimizer/offload buffers) + 32.7 GB arena +
≤9.2 GB staging (4 × 627 frames) ≈ 90 GB; 2 ranks ≈ 180 GB plus checkpoint transient →
512 GiB request, guard 460.8 GiB. Update time ≈ 2× H4 compute (GAS4) + ≈6.5 s CPU Adam ⇒
≈28–32 s ⇒ ≈30–35 h for epoch 1 (estimate; replaced by the in-job speed gate).
Historical: an H2 GPU-optimizer attempt passed update 1 at 77.0 GiB and OOMed at update 2;
a 240 GiB CPU-offload attempt was OOM-killed; therefore no GPU-optimizer H2 variant is used.
