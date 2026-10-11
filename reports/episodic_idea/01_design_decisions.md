# 01 — Design decisions and deviations log

Each entry: observation → alternatives → choice → science or engineering → tests → cost.
The plan (`plans/episodic_mem.md`) is not edited to hide any of these.

## D1. Training length: one epoch on the two-epoch schedule
*Observation.* The plan's §3.4 table says "Epochs 1" but specifies 7,704 expected
updates / 232 warmup, which equals two epochs of 30,815 episodes at global batch 8
(`ceil(30815/8)=3852` per epoch); §3.4 text says E1 is "a mid-schedule snapshot of the
same two-epoch trajectory" and forbids shortening the LR schedule and stretching it
later. The user asked to train epoch 1 only and to keep training state so epoch 2 can
continue.
*Choice.* Declared schedule = 2 epochs, 7,704 updates, 232 warmup, cosine to 0.1×
(`ScheduleGuard` checks this at start). Every Phase-1 run stops at the epoch-1 boundary
(`--stop-after-epochs 1`, update 3,852) after the epoch save, exports `final/`, and
publishes the complete epoch-1 checkpoint (DeepSpeed shards, scheduler, RNG, sampler
state) so epoch 2 resumes exactly with `--resume_from_checkpoint` on the **same
profile/world size**. Science: unchanged recipe; E1 is diagnostic in the plan, and
results must be labelled "E1 (mid-schedule)". Matches the historical native Window8
E1 protocol (49.32% R2R SR).

## D2. Score-helper placement (plan §5.3/§7.1)
*Alternatives.* (a) project-owned mixer that prepares q/k/v/g/β once for both the
differentiable FLA call and the scorer; (b) leave the native decoder layer untouched and
recompute the preparation in a no-grad side pass from the layer's input.
*Choice.* (b). The native layer (and its checkpointing/offload and the TileLang
backward) is bitwise untouched; the side pass runs before the checkpointed call, so
recomputation can never produce new scores, and no global/hook state is used. Cost:
one extra no-grad `in_proj_qkv/b/a` + conv per scoring layer (measured 49 ms for 30.6K
tokens per layer on H100, including the scan). Engineering only.

## D3. Score backend and segmentation (versioned)
`score_backend = fp32_torch_wy_block64_v1`: exact WY block form in FP32, blocks ≤64,
lower-triangle exponents masked before `exp`, no division, FP32 L2 key norm (FLA formula,
eps 1e-6), native sigmoid β and softplus gate, GVA mapping value-head h → key-head
h//2. `score_segmentation = prefix_obs_response_block64_v1`: segments are prefix,
each observation block, and each group's complete response (ordinary tokens + EOS +
separator); serving buffers single response tokens and scans them at the closing
append, so training and serving use identical block boundaries. The native GDN
serving segmentation is unchanged. The helper's state is a separate FP32 state (not
the policy state); in serving it is part of the bounded session metadata (8 × 2 MiB).

## D4. Attention backward accumulates K/V gradients in FP32
The inherited batched backward accumulated overlapping K/V gradients in BF16. With the
prefix read by every window (and episodic rows read by up to hundreds of later windows)
BF16 summation error grows roughly with √(#windows)·2⁻⁹. The new episodic backend
accumulates in `promote(dtype, fp32)` buffers (FP64 in reference tests), deterministic
per-window scatter (unique IDs per window). Applies to all Phase-1 arms equally
(`episodic_fa2_varlen_gather_fp32acc_v1`); forward is bitwise identical to the legacy
batched path for V0. Cost: +2×T×4×256×4 B transient per attention backward (≈1.6 GB at
T=200K).

## D5. Host-RAM engineering (no objective/data change)
Findings: (i) PyTorch's caching host allocator rounds every pinned request up to a power
of two (`CachingHostAllocator.h`: `PowerOf2Ceil`) and never frees cached blocks; both
`save_on_cpu(pin_memory=True)` and DataLoader pinning go through it. (ii) The inherited
recipe used 4 loader workers/rank (prefetch 2) and pinned FP32 pixel batches (4.6 GB per
627-frame episode). Changes, all versioned in the profile configs:
* `activation_offload_backend: pinned_arena_v1` — one `cudaHostRegister`ed arena per rank,
  sized exactly from the audited maximum (32 layers × 199,735 × 2560 × 2 B ≈ 32.7 GB),
  bump allocation, rewound only when every saved input was consumed, fails loudly when
  exhausted. Same saved values, same gradients.
* `stage_pixels_bf16: true` — FP32→BF16 round-to-nearest-even on CPU is bitwise identical
  to the cast DeepSpeed/Trainer performs on transfer; halves staging.
* `dataloader_num_workers: 0`, `dataloader_pin_memory: false`, no persistent workers,
  `malloc_trim(0)` after each update; per-update RSS/cgroup/arena logs.
Rejected: pageable offload as default (slower synchronous copies); FP8/compressed offload
(changes numerics); disk offload (unaccounted); dropping/shortening episodes (forbidden).

## D6. Visual-only candidates, uncapped, per-layer
As specified: only the 300 image rows of an expired group are candidates; per-layer
selection from its own GDN l−1 scores; `per_source_cap: null` (a cap is rejected by the
contract in v1); M3 (`visual_group`) is rejected in v1 (Phase-2 only). Random control:
first 8 bytes (big-endian) of SHA-256(`random_hash_v1|seed|episode_uid|layer|token`).

## D7. Qualification location
128 GiB interactive memory cannot hold the longest-RxR gates (and H2's CPU-optimizer
state alone exceeds the 80 GiB guard there). Decision: run every gate that fits in the
interactive allocation sequentially, and make the full profile suite (V0 + M2: smoke,
longest R2R, offload boundary, longest RxR, mixed tail, stressed checkpoint
save/reload, representative speed) a **mandatory in-job gate** of every race candidate
on its own high-RAM allocation, before it may claim the lease. Matched ablations run a
startup longest-episode gate for their own arm. No PASS is ever reused across
allocations.

## D8. Train/serve committed-score skew (recorded, not fixed)
Real-model replay (03/04) shows online and offline committed scores differ by up to
~22–27% (relative to the group maximum) because the committed delta is a small residual
v−Sᵀk that amplifies BF16 hidden-state differences between full-sequence and incremental
forwards. Exact per-step 300-row sets therefore often differ (M2c random: 0 mismatches).
Decision: keep `committed_mean_l2_v1` unchanged for Phase 1 (plan §5.4 forbids silently
relaxing/replacing the score); report overlap/margins; a rank-bucket/quantized score is
a separately versioned Phase-2 stabilization candidate only.

## D9. One optimizer group per hyperparameter set (chunking measured and reverted)
*Observation.* `optimizer_group_chunks: 16` removed the ZeRO-2 init host spike (H4 anon
peak 64.6 → 25.2 GiB) but, measured on the same snapshot, fixture (longest R2R ×8) and
profile, raised peak **allocated GPU** memory 46.02 → 50.58 GiB (+4.56 GiB, reserved
50.30 → 52.51). (A first comparison was invalid: the runtime profile is merged last and
overrode a policy-overlay `chunks: 1`; it is retained as `chunks1_invalid_overlay_overridden`.)
*Choice.* GPU headroom at the 199,735-token episode is the binding constraint (historical
76.94 GiB reserved vs the 78 GiB gate), while the ~16 GB/rank init host transient is
affordable at 512 GiB. Both profiles use `optimizer_group_chunks: 1` (inherited grouping).
Engineering only.

## D10. Attention batch steps = 8 (qualified)
Single-GPU forward+backward of the real longest episode (no optimizer; full BF16 grads),
M2: batch 4 / 8 / 16 → identical peak allocated 66.47 GiB (reserved 76.51–76.53), total
30.5 / 31.8 / 29.9 s (within noise). Attention scratch is not the peak contributor at
T≈200K; keep the plan's conservative 8 for every arm.

## D11. Superseding never-started submissions
Jobs 898755/898756 (attempts `P1-V0-s429-{H4,H2}-a1`, source `a7e5ae5` with 16 chunks)
were still PENDING when D9 was measured. They were cancelled by exact job ID after a live
identity check (user, job name, campaign comment, state), recorded in the ledger as
`superseded_before_start`, and replaced by attempts `a2` (jobs 898758/898759, source
`469b107`). No allocation, training state or checkpoint existed for them. The campaign
manifest keeps the source history (`resnapshot`); every attempt verifies the source hashes
recorded at its own submission.
