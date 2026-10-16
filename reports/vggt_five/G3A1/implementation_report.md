# G3 + A1: native-state reads with GeoAddress relation addressing (newton, 4 × H100)

**Status (2026-10-16 14:25 EDT): implementation and qualification complete; one-epoch production
training is running** in interactive job 926610 (evc104), from the immutable source 877815e.
Checkpoint-10 has been validated as complete (22 files, 70.76 GB). No navigation result exists
yet; successful training gates do not imply navigation improvement.

## What G3A1 is

G3 (best evaluated VGGT arm, R2R val_unseen SR 51.98) supervises the native GDN decoder-24
recurrent state with training-only read probes. A1 (partial SR 50.15; +1.47 pp over A0 on the
shared prefix) adds serving-persistent rank-8 GeoAddress Q/K adapters to the same GDN decoder 24.
It trains them with a relation KL against online Omega similarities. G3A1 runs both at once:

| | G3 | A1 | **G3A1** |
|---|---|---|---|
| Serving architecture | native Window8 | Window8 + GeoAddress (45,056 params) | **= A1** |
| Training-only modules | state query + decoder (10.53 M) | none | **G3 head** |
| Objectives at decoder 24 | delayed regional state reads | 32×300 relation KL | **both** |
| λ_max / ramp | 0.1 / 771 | 0.1 / 771 | **0.1 each / shared 771** |
| LRs (backbone / head / adapter) | 5e-6 / 1e-4 / — | 5e-6 / — / 5e-5 | **5e-6 / 1e-4 / 5e-5** |
| Pairs | ≤8, lags 8–32 | the same sampler | **identical pair list** |
| Teacher frames | sources (≤8) | sources ∪ reads (≤16) | **sources ∪ reads, one call** |
| Schedule | 3,852 updates, warmup 116, cosine to 0.1 | same | **same (true 1-epoch decay)** |

Loss: `nav_sum/global_actions + λ(u)·(G3_sum + A1_sum)/global_pairs`. Both objectives share the
pair denominator and ramp, so this is exactly each at its specified weight. The two sums are
logged separately (`aux_state_loss_sum`, `aux_relation_loss_sum`).

## Implementation (branch `c2_vggt_omega_g3_a1`, from `v2_vggt_omega_a1` 2aa2c26)

- **`models/vggt_g3a1_forward.py`** is the single fused uncached layer-24 forward, mirroring
  `geoaddress_gdn` and `native_state_gdn` in native order: projections and convolution → raw split
  → **GeoAddress residual** → **A1 relation KL on the adapted raw heads 0/1** → head repetition →
  **G3 probe insertion** → one pinned FLA chunk call → real rows restored before native gated
  norm, out_proj, residual and MLP. It returns `[G3_sum, A1_sum]`.
- **G3 query context** is the detached post-adapter native query at the anchor, i.e. the
  deployed query. Anchors are cue/text rows, where the image-only adapter has no direct
  residual. Probes read state written with adapted keys, so the G3 loss also trains the adapters.
- **`research_forward.py`** routes decoder 24 to the fused path only when both the state head and
  the image mask are present, and asserts that GeoAddress sits at layer 24. G3, A0, A1 and G1
  routing is unchanged.
- **`nav_model.py`** creates the G3 head for G3A1. It validates shapes, image-token positions and
  per-pair causality (source cells < read cells < read anchor), and builds the relation closure
  on the relation targets.
- **`episode_dataset.py` / `geoaddress_sampling.py`** produce the A1 metadata unchanged plus G3
  anchors/lags on the same ordered pair list. A cross-check confirms the G3 and A1 layouts agree.
- **`research/g3a1_teacher.py`** provides `OnlineG3A1Teacher` with the portable unpositioned
  identity, and `g3a1_targets`. The latter returns region means of the source descriptors and
  the A1 teacher relations.
- **`train_episode.py`** selects the teacher, builds both targets and logs both sums.
  **`vggt_g1.parse_g1`** adds a strict G3A1 contract: GeoAddress required, adapter LR 5e-5,
  direct-native-Q/K relation, λ 0.1, head LR 1e-4, layer 24, ramp 771.
- **`scripts/vln/export_vggt_g1.py`** handles G3A1 export: it requires the exact four adapter
  tensors and the G3 head, drops the head, and rejects any other training head.
- **Configs:** `configs/vggt/g3a1_newton_h4.yaml` (runtime + science) and
  `g3a1_joint_one_epoch.yaml` (schedule). **Launcher and environment:**
  `train/launch_g3a1_newton_joint_e1.sh`, `train/g3a1_newton_env.sh`. **Profiler:**
  `scripts/vln/profile_vggt_g3a1.py`.

## Tests

| Test | Result |
|---|---|
| `test_vggt_g3a1.py` (CPU, 6) | config contract; dataset G3A1 fields **identical** to the G3-only and A1-only outputs for the same episode; empty support; targets equal the single-arm constructions; identity; export rules |
| `test_vggt_g3a1_cuda.py` (H100 FLA, 2) | zero adapter: fused output and G3 loss **bitwise =** G3 layer, A1 loss **bitwise =** A1 layer; no pairs: output **bitwise =** A1 layer; active adapter: A1 loss bitwise exact, real rows vs A1 0.039%, probe reads vs FP64 recurrence 0.40% (gate 2%); checkpoint on/off gradients bitwise equal; no gradient after the last anchor; G3 loss alone trains the adapters |
| Full suite (GPU) | 416 passed; 2 environment failures, which pass under the newton env (26 passed in the env-dependent set incl. CPUAdam vs torch AdamW); 1 skip (G3 real-Omega test hard-codes a worker-2 fixture) |

## Four-H100 qualification (complete episodes, all ranks/updates/exposures verified, 0 OOM)

| Gate | Result | GPU alloc / reserved GiB | Host GiB |
|---|---|---:|---:|
| smoke, mixed empty support (24 exposures, 96 pairs, 12 empty) | PASS, 6.29 s/update | 20.99 / 28.56 | 110 |
| **longest RxR**, 8 × 627 steps, 199,735 tokens, 192 pairs | PASS, 97.3 s/update | **67.89 / 76.55** (ceiling 78) | 274 (guard 460.8) |
| recovery reference (save every update) | PASS, ~115 s per full save | 20.99 / 28.56 | 139 |
| fresh-process resume 2→3 | PASS | 20.99 / 28.56 | 173 |
| recovery comparison | whole-model rel-L2 1.15e-8; G3 head **bitwise equal**; adapters rel-L2 4.35e-8; identical update-3 loss 0.2849681377; head biases and 4,172 adapter elements moved | | |
| speed, workers 1/rank | 31.37 s/update | 39.23 / 48.85 | 164 |
| **speed, workers 2/rank (selected)** | **30.46 s/update** | 39.23 / 48.85 | 166 |
| serving export | exact source/export logits; adapters kept, G3 head removed | | |

**Recorded failures:**
- **Transient startup error.** The first recovery-reference start died with an NCCL `Cuda failure
  999` before any update. It started seconds after the previous 4-GPU gate exited, and the kernel
  log shows no new Xid. A clean retry passed. The failed attempt is kept as evidence.
- **Export replay tolerance.** The R2R 17-step streaming-vs-offline replay exceeded the unchanged
  atol 0.75 (max 1.0, 34 of 12.4 M logits). RMS was 0.068 (limit 0.15) and argmax agreement 98%.
  RxR passed (0.566). A control on the same exported model with the adapter up-projections zeroed
  gives the same R2R excess (1.008, 28 elements). So this comes from this cluster's native BF16
  offline-vs-streaming numerics, which the episodic campaign also recorded here (0.75–0.86), not
  from G3A1. The threshold was not relaxed. Serving parity on the evaluation hardware should be
  re-audited before evaluation.

## Newton environment notes

- **Teacher:** gated `facebook/VGGT-Omega` `vggt_omega_1b_512.pt` (SHA-256 `c02da418…`). The
  source is `facebookresearch/vggt-omega` `main` 48b23c8. The pinned ad02778c is absent from the
  public history, but the teacher source-tree hash `b0e8d9d7…` matches every commit from 299be1f.
  The G3 and A1 identities computed here equal the worker-2 configs exactly. Assets live in
  `/groups/yshang/an221229/vggt_omega_assets`.
- **CPUAdam toolchain:** isolated CUDA 12.9.1 build components, and GCC 12 wrapped with
  `-mno-avx512fp16`. EL8 binutils 2.30 cannot assemble `vmovw`; the CPUAdam numerical test
  passes.
- **Allocation:** 926610 has 4 × H100, only **4 CPUs** and 512G RAM. The selected runtime is 1
  OMP thread and 2 persistent loader workers per rank. Everything else matches the qualified
  G3/A1 runtime: CPUAdam ZeRO-2, pinned arena, vision/teacher batch 8, attention batch 32.

## Production

- Source: immutable read-only snapshot `/groups/yshang/an221229/checkpoints/g3a1-newton-20261016/production/source`
  (commit 877815e, `source.commit`, `source_manifest.sha256` with 323 files).
- Run: `/groups/yshang/an221229/checkpoints/g3a1-newton-20261016/production/g3a1_joint_e1/train`.
  All 4 ranks printed `SCHEDULE_GATE_PASS (3852, 116)`.
- Training contract: 30,815 episodes, 3,128,624 actions, world 4, GAS 2, global batch 8, tokens
  700–199,735. Recovery saves at 10 and then every 100 updates, keeping 2 rolling plus the epoch
  checkpoint.
- **Expected duration:** about 19 s/update on corpus-average episodes (length-scaled from the
  speed fixture), i.e. roughly **20–22 h** for the epoch including checkpoints on this 4-CPU node.
  Worker-2 with 60 CPUs took about 14 h. Job 926610 ends at 18:01:53 EDT. `deadline_stop.sh`
  requests an orderly checkpoint and stop at 17:16:53. Expect several hundred updates in this
  allocation.

### Resume (same 4-GPU topology, same snapshot and run directory)

```bash
# inside a new 4 x H100 allocation (512G or more RAM; set HOST_GUARD_GIB=0.9*RAM if larger):
RUN=/groups/yshang/an221229/checkpoints/g3a1-newton-20261016/production
cd $RUN/source && OMP_NUM_THREADS=1 bash train/launch_g3a1_newton_joint_e1.sh $RUN/g3a1_joint_e1
```

The launcher resumes from the newest complete checkpoint (`--resume_from_checkpoint auto`) with
optimizer, scheduler, RNG, sampler, G3 head and adapters intact. A launcher lock prevents duplicate
starts. CPUAdam is built per node into `vggt_omega_assets/torch_ext_<host>` on first use. After a
deadline stop, archive the handled `REQUEST_CHECKPOINT_AND_STOP` flag; the job also archives it
automatically.
