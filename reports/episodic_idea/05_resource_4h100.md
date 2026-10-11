# 05 — H4 runtime profile (4×H100, GAS2, ZeRO-2 GPU optimizer)

Profile (final, snapshot `e28bd01`): `configs/episodic/profile_h4.yaml` + `deepspeed.json`.
One complete episode per rank per microstep, global batch 8, `optimizer_group_chunks: 1`
(D9), `activation_offload_backend: pinned_arena_v1` (threshold 65,536 tokens; fork-safe,
installed at train begin, D12), `stage_pixels_bf16`, **2 persistent loader workers per
rank, prefetch 2** (D13), no DataLoader pinning, `malloc_trim` per update,
`window_attention_batch_steps: 8` (D10), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Earlier interactive passes below used earlier source revisions (recorded per gate).

Qualification status: **partially GPU-qualified in the interactive allocation**
(gates that fit in 128 GiB). The longest-RxR, above-offload, mixed-tail (with the
longest), stressed checkpoint and representative-speed gates are **pending in-job**
(job 898834, attempt `P1-V0-s429-H4-a3`, 512 GiB) — they run on that allocation before
it may claim the run lease.

## Interactive allocation 894903 (128 GiB cgroup v1, guard 80 GiB = limit − 48)

### Development-source pass (worktree, commits 154601c→a7e5ae5)

Gates were launched sequentially; the first four ran before `optimizer_group_chunks`
existed, the rest after (resolved configs in each gate directory record which).
Every rank finished every update, finite losses/gradient norms, exact exposures
(3 updates × 8 slots), no new OOM-kill events. GPU = max reserved over ranks/updates.

| Gate | Pass | GPU reserved GiB | Host working set GiB (sampled) | Host anon GiB | New oom_kill | Losses (updates 1–3) | Update s (rank 0) |
|---|---|---:|---:|---:|---:|---|---|
| V0 smoke (4 shortest R2R + 4 shortest RxR) | ✅ | 36.58 | 71.3* | 64.6* | 0 | .2871 .2871 .2850 | 7.9 1.2 1.2 |
| M2 smoke | ✅ | 36.58 | 68.2* | 61.5* | 0 | .2797 .2797 .2788 | 4.9 1.4 1.4 |
| V0 longest R2R ×8 (58,290 tok) | ✅ | 50.40 | 59.5* | 52.8* | 0 | .4031 .4031 .4035 | 37.2 14.7 14.6 |
| M2 longest R2R ×8 | ✅ | 50.40 | 50.4* | 43.7* | 0 | .4035 .4035 .4038 | 20.2 16.6 16.5 |
| V0 just below offload ×8 (65,526) | ✅ | 56.01 | 31.9 | 25.2 | 0 | .4077 .4077 .4054 | 39.0 16.6 16.5 |
| M2 just below offload ×8 | ✅ | 56.01 | 30.9 | 24.1 | 0 | .4025 .4025 .4048 | 21.2 18.7 18.7 |
| V0 checkpoint save (smoke, 2 updates, save each) | ✅ | 32.59 | 52.2 | 45.4 | 0 | .2871 .2871 | 3.2 2.4 |
| V0 fresh-process resume → update 3 | ✅ | 32.38 | 47.6 | 46.7 | 0 | (.2871 .2871) .2822 | 19.4 |
| M2 checkpoint save | ✅ | 32.59 | 48.7 | 47.2 | 0 | .2797 .2797 | 10.3 1.8 |
| M2 fresh-process resume → update 3 | ✅ | 32.38 | 47.3 | 46.5 | 0 | (.2797 .2797) .2800 | 8.3 |

\* Before `optimizer_group_chunks`: a ~2 s host spike during DeepSpeed ZeRO-2 init
(each param group is copied and flattened on CPU) dominated the peak: 64.6 GiB anon for
4 ranks. With 16 identical-hyperparameter chunks the same phase peaks at ~22 GiB and the
just-below-offload gate's whole-run anon peak is 25.2 GiB. Update 1 has LR 0 (warmup),
so updates 1–2 share a loss by construction.

Per-rank steady RSS during training is 2.7–4.0 GB (no arena needed below 65,536 tokens).
Recovery checkpoint (4 ranks): **66 GB** (BF16 weights `pytorch_model.bin`, ZeRO-2 FP32
master + Adam shards in `global_step*/`, scheduler, 4 RNG states, trainer state,
navigation/resume contracts); the save transient is ≈ +11 GB host per rank. The resume
run reloaded the DeepSpeed shards and the trainer state in a fresh process and trained
the next update on the same topology. Uninterrupted-vs-resumed loss comparison on the
same fixture/code is recorded by the in-job gate (BF16 training is not bitwise
deterministic; matching losses are not claimed as identity).

Timing, 58K-token R2R (all ranks equal length): V0 forward 2.37 s + backward/step 4.96 s
per episode; M2 forward 3.22 s (+0.85 s: score side pass at 8 layers, plan build, wider
windows) with an unchanged backward → 16.6 vs 14.7 s per update (+13%). Rough corpus
forecast (mean episode ≈101.5 obs ≈31.8K tokens; offload episodes slower): H4 ≈ 11–13 s
per update ⇒ ≈13–16 h for epoch 1 (3,852 updates) **before** measurement; the in-job
speed gate (20 ordinary-sampler updates) replaces this estimate.

### Authoritative interactive pass from the campaign snapshot `a7e5ae5`

See `quals/interactive_894903/H4_snapshot_a7e5ae5/H4/H4_suite.json` (results appended to
07 when complete).

## Host-RAM budget for the production request (512 GiB)

Per rank at the longest episode: ~3–4 GB process + 32.7 GB pinned arena (exact:
32 layers × 199,735 × 2560 × 2 B) + ≤4.6 GB BF16 staging (2 × 627 frames) ≈ 41 GB;
4 ranks ≈ 165 GB, plus ≈44 GB checkpoint-save transient and reclaimable page cache.
Guard = min(0.9 × 512, 512 − 48) = 460.8 GiB. The in-job gates measure the real peak.
Historical comparison: 417.9 GiB guarded working set on the same longest fixture with
the inherited loader/pinning/offload settings.

## Additional measurements (interactive, this session)

**Optimizer group chunking (D9)** — same snapshot `a7e5ae5`, longest R2R ×8, V0, 3 updates:

| `optimizer_group_chunks` | Peak allocated GiB (upd 1 / 2–3) | Peak reserved GiB | Host anon peak GiB |
|---:|---|---:|---:|
| 1 | 38.19 / 46.02 | 50.30 | 39.9 |
| 16 | 42.75 / 50.58 | 52.51 | 26.2 |

**Longest RxR episode, single GPU, forward+backward, no optimizer (D10)** — real
`82sE5b5pLXE:31149:22206`, 199,735 tokens, full BF16 gradients (7.83 GiB) materialized,
arena offload of all 32 checkpoint inputs (host anon 47 GiB):

| Arm | attention batch | Peak allocated GiB | Peak reserved GiB | fwd+bwd s |
|---|---:|---:|---:|---:|
| V0 | 8 | 66.47 | 76.51 | 76.8 (incl. JIT warm-up) |
| M2 | 8 | 66.47 | 76.51–76.53 | 30.2–31.8 |
| M2 | 4 | 66.47 | 76.53 | 30.5 |
| M2 | 16 | 66.47 | 76.53 | 29.9 |

A ZeRO-2 H4 rank instead holds a ¼ gradient partition (−≈5.8 GiB) plus FP32 master and
Adam partitions (+≈12.6 GiB): ≈73 GiB allocated expected at the longest episode, so the
78 GiB reserved gate is tight; it is decided by the in-job longest-RxR gate (historical
inherited recipe: 76.94 GiB reserved). H2 keeps optimizer state on CPU and has headroom.

**Loader workers (D13)** — ordinary sampler, 512-episode random subsets with all
episodes < 60K tokens, V0, 20 updates, wall per update over updates 2–20:

| Workers/rank | Data | Wall s/update | Median gap s | fwd/bwd per microstep s | Host working set GiB |
|---:|---|---:|---:|---|---:|
| 0 | subset A, cold Lustre | **31.88** | 7.30 | 1.14 / 3.23 | 64.6 |
| 1 | subset A, warm (re-read; confounded) | 7.24 | 0.11 | 1.05 / 2.35 | 59.1 |
| **2** | **subset B (disjoint), cold** | **7.29** | **0.11** | 1.05 / 2.30 | 68.8 |

Two workers make cold-read training ≈4.4× faster in wall time on this subset. Forecast
for the full corpus (longer and offloaded episodes, recovery saves every 250 updates):
**≈10–13 h for H4 epoch 1** (estimate only; the in-job speed gate measures it).
