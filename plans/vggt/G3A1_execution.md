# G3 + A1: native-state reads with GeoAddress relation addressing (2026-10-16)

## Specification

Branch `c2_vggt_omega_g3_a1`, started from `origin/v2_vggt_omega_a1` 2aa2c26. That branch already
contains G3's complete implementation (2259383, 9d658c2) plus A0/A1. The new run starts from
the original Qwen3.5-4B base `851bf6e8…`, not from any trained G3/A1 checkpoint.

Keep G3 exactly:
- delayed native-state reads at GDN decoder 24 through read-only probes (zero write and decay,
  unit key);
- 5 regional unpositioned LN2048 online-Omega targets per pair;
- query MLP 4160→512→4096 on the detached anchor query, decoder 4096→1024→2048;
- head LR 1e-4, λ_max 0.1.

Keep A1 exactly:
- rank-8 image-token GeoAddress residuals on raw GDN24 heads 0/1 (45,056 parameters,
  serving-persistent), adapter LR 5e-5;
- KL(teacher‖student) over 32×300 relation distributions on the adapted native Q/K,
  temperature 0.1, λ_max 0.1.

Both share G3's deterministic pair sampler (≤8 pairs, lags 8–32). One frozen FP32 online
VGGT-Omega-1B-512 call runs on the union of source and read frames (≤16). The losses are:
L = nav_sum/actions + λ(u)·(G3_sum + A1_sum)/global_pairs, with λ(u)=0.1·min(1,u/771). This is
exactly each objective at its specified weight, because the two share the denominator and the
ramp.

Schedule: one joint R2R + English RxR15 epoch, a true one-epoch decay (3,852 updates, warmup 116,
cosine to 0.1), backbone LR 5e-6. Topology: global batch 8 (4 GPUs × GAS 2), full BPTT, frozen
vision/merger, native Window8 with canonical text history.

Serving is the A1 architecture: the GeoAddress adapters are kept, and the G3 head and teacher
are removed.

## Combination decisions

1. **One fused layer24 forward** (`models/vggt_g3a1_forward.py`). The operations run in native
   order:
   1. projections and convolution;
   2. GeoAddress residual on the raw heads;
   3. A1 relation KL on the adapted raw heads 0/1;
   4. head repetition;
   5. G3 probes;
   6. one FLA call;
   7. restore the real rows before norm, out_proj and MLP.

   It returns both sums separately for logging. The single-arm paths are untouched.
2. **G3 query context is the post-adapter native query at the anchor.** That is the query the
   deployed model uses. Anchors are cue/text rows, where the image-only adapter adds no direct
   residual. G3 probes read a state written with adapted image keys, so the G3 loss also trains
   the adapters. That interaction is intended and is verified by test.
3. **Teacher identity uses A1's portable unpositioned base** (no unused position bytes). It binds
   G3's unchanged regions code and A1's relation code.
4. **Empty support uses the existing productions' paths unchanged.** That means G3
   `empty_loss` and A1's empty relation. Both completed full epochs on worker-2.

## Newton environment (this cluster)

- **Teacher:** the gated weights `facebook/VGGT-Omega vggt_omega_1b_512.pt`, SHA-256
  `c02da418…`, were downloaded with the user's access. The source is
  `facebookresearch/vggt-omega` `main` 48b23c8. The pinned commit ad02778c is no longer in the
  public history, but every commit since 299be1f has the identical teacher source-tree hash
  `b0e8d9d7…`. The A1 and G3 identities computed here equal the worker-2 configs exactly.
- **CUDA:** isolated CUDA 12.9.1 build components, from the repo's installer with only the venv
  path changed.
- **Compiler:** GCC 12 is wrapped with `-mno-avx512fp16`. EL8 binutils 2.30 cannot assemble the
  AVX512-FP16 `vmovw` that `-march=native` emits on this CPU. The CPUAdam-vs-torch-AdamW test
  passes with the wrapper.
- **Allocation:** 4 × H100, only **4 CPUs**, 512G RAM. The host guard is 460.8 GiB.

## Progress ledger

- Implementation 5f56a45; runtime selection and launcher 877815e (production source).
- Tests: 6 new CPU contract tests and 2 new H100 fused-layer tests pass. The full suite gave 416
  passed and 2 environment failures (CUDA 12.6 mismatch, Omega source not on path). Those 2 pass
  under the newton environment (26 passed in the env-dependent set). One skip remains: the G3
  real-Omega test hard-codes a worker-2 fixture.
- 4-GPU gates (all complete-episode, verified per rank, update and exposure; zero OOM):
  - smoke with mixed empty support;
  - longest RxR at 67.89/76.55 GiB GPU and 274 GiB host;
  - recovery reference and fresh-process resume, with the comparison PASS;
  - export with exact source/export logits;
  - speed, workers 1 vs 2.
- Export replay: R2R streaming-vs-offline exceeded atol 0.75 (max 1.0, 34/12.4M elements). The
  control on the same model with zeroed adapters gives max 1.008 (28 elements), so this is the
  native BF16 path on this cluster, not G3A1. RxR passes. The threshold was not relaxed; this is
  recorded as an environment limitation of the replay audit.
- Production launched 2026-10-16 14:12 EDT in interactive job 926610 from the immutable snapshot
  `production/source` (877815e). The checkpoint-stop is set for 45 min before the job ends.
