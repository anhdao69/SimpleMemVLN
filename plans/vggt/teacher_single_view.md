> **R1 update (2026-10-15):** `R1_lane_state.md` and `R1_omega_execution.md` supersede historical R1 cache/PCA/dimension/schedule defaults: online frozen FP32 Omega, full2048 regional targets and training-only student MLP, exact registered four-lane R0 architecture, one joint R2R + English RxR15 epoch. R0 E1 used a two-epoch LR schedule; report this confound.

> **G3 online update (2026-10-13):** `G3_native_state.md` and `G3_omega_execution.md` replace this document's offline/PCA target requirements for G3. It now uses frozen online single-view Omega, full2048-dimensional regional targets, no added positional term, and four-H100 qualification. Other unimplemented arms retain their original specifications.

> **G1 update (2026-10-13):** User requests ONLINE VGGT-Omega, no pre-extraction/PCA, the SpatialForcing projector, and4H100 tests. For G1, `G1_current_alignment.md` supersedes conflicting offline/PCA/two-GPU sections below. Other arms are not implemented by this task.

# Shared single-view VGGT targets and sampling

**Spec:** `vggt_five_runs.md`. This pipeline serves G1, G3, A1 and R1. A0 does not read teacher targets. It creates derived feature caches only, not new navigation trajectories or action labels.

## 1. Source and provenance contract

Use `facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce`. Download official `facebook/VGGT-1B` once in a separate teacher environment if dependencies conflict with the locked student environment. Record actual immutable model revision, every loaded weight hash, source hash and package versions in `teacher_manifest.json`. The weight hash is intentionally not invented in this plan; extraction refuses to proceed without recording it.

Set teacher `.eval()`, `requires_grad_(False)`, use inference_mode. BF16 compute is allowed; pool, normalize, PCA projection, cosine and relation distributions use FP32. Only the aggregator is needed; do not run camera/depth/track heads. No teacher weights are resident in the policy-training job.

Every call must have `S=1`. Independent views may be batched in B. The official image loader returns an S-first stack, so explicitly add/rearrange the B/S axes: do not turn a list of independent views into one multi-view scene. Log asserted shape `[B,1,3,H,W]` in extraction metadata.

## 2. Image and patch transformation

Input RGB bytes must match the dataset frame. For width640/height480, the inspected official crop preprocessing gives width518/height392, input `[0,1]`, and VGGT normalizes internally. Output spatial patch grid is height28/width37, patch14. Validate these values against the pinned code at execution.

Call aggregator -> `(outputs, patch_start_idx)`. Select `outputs[23]`, assert not None, shape `[B,1,1041,2048]` for 1036 patches plus five special tokens under the pinned image shape. Slice `[..., patch_start_idx:, :]`, not a hardcoded raw tensor flatten. Assert `patch_start_idx==5` for this version, but use the returned index in indexing.

Reshape patch tokens row-major to `[B,2048,28,37]`. `adaptive_avg_pool2d(...,(15,20))` creates a resampled teacher grid, not an exact one-to-one original-pixel correspondence. Noninteger pooling regions overlap; this is accepted for this feature target, but must be documented. Validate student ordering using an actual processor/merger synthetic-pattern fixture, not just matching token counts. A later quadrant split is nonoverlapping in the resulting 15x20 grid.

For each pooled token f: `f_ln = layer_norm(f,[2048],weight=None,bias=None,eps=1e-6)` in FP32.

## 3. One shared compression transform

Use dimension **256**, not a dimension selected separately for each arm. Fit a shared PCA on 100,000 pooled/layer-normalized tokens from up to 2,000 distinct TRAIN physical trajectories. For fewer available unique trajectories, use all and report the number. Do not fit on validation scenes or use instructions to condition the teacher.

Suggested bounded-memory fitter: stream feature blocks into count, FP64 first moment and second moment/covariance statistics; eigendecompose the 2048x2048 covariance once; store mean `[2048]` and the top256 orthonormal columns `[2048,256]`. A properly versioned approximate PCA is also an engineering substitution only if frozen before every extraction/run. Record fitting identities/order/seed and retained variance, but do not claim >=90% unless measured.

For each frame:

`Z = (f_ln - mean) @ components`, shape `[300,256]`.

Do NOT whiten or L2-normalize tokens before region averaging. Store unnormalized PCA tokens in FP16; targets are cast FP32 for loss. The same transform is used by every run. Per-teacher PCA comparisons are not part of these five runs.

### 3.1 Current alignment positional term, G1 only

Keep the attached current-alignment design's positional augmentation, while making its magnitude explicit. Form fixed grid sin/cos vectors E0 of width256 from normalized row/column centers; 64 frequencies per axis with sin/cos, frequencies `10000**(-arange(64)/64)`, arguments `2*pi*coordinate*frequency`. Scale E0 to global RMS1 over all 300x256 values. Compute a scalar `target_rms` from training PCA tokens only. G1 target is `Z + 0.5*target_rms*E0`. No image-specific RMS or validation tuning. Store E0 hash and scalar. Other arms use Z without the added positional term.

This positional term is part of G1's chosen method. Therefore comparing G1 to G3 is a method screen, not a single-factor delay ablation.

## 4. Frame cache identity and storage

With S=1 and deterministic preprocessing, use key:

`SHA256(image_bytes_hash || teacher_weights_hash || code_hash || preprocessing_version || teacher_layer || pooling_version || PCA_hash)`.

Store source physical-trajectory and episode-to-frame maps separately. Deduplicate identical RGB only for teacher computation. Do NOT deduplicate navigation training episodes or action labels: multiple instructions over one physical path remain distinct supervised episodes.

Dense frame record: Z `[300,256]` FP16 = 153,600 bytes before metadata. Region record: four quadrants plus global `[5,256]` = 2,560 bytes. G3/R1 can load only region records. A1 needs dense source and read-frame tokens; G1 needs dense sampled current-frame tokens.

Fixed base sample at 30,815*16 frames would be ~75.73 decimal GB before dedup. A1's union of up to 24 frames/episode is ~113.60 decimal GB before dedup. Region-only 16-frame samples are ~1.26 decimal GB. These are arithmetic ceilings for the stated counts, not measured extracted size or GPU time.

Use bounded FP16 binary shards (for example at most4096 frames/shard), an immutable index with exact offsets/shapes, atomic temporary-file rename, per-shard hashes and completion manifest. No millions of tiny per-token files; no loading an entire shard into every DataLoader worker. Memory-map and copy only requested records. Missing required entries are errors, not zero features. A finite teacher output with no valid pair is distinct from a missing cache entry.

## 5. Fixed sampling and delayed pair plan

### 5.1 Frame sample

Per physical trajectory of length T, K=min(16,T). Split [0,T) into K nonempty integer segments. In each segment choose one integer using a stable SHA256-based draw of `(trajectory_key,segment_id,cache_seed)`. Sort chosen unique frame indices. The sample is fixed across epochs/arms; it is NOT claimed equivalent to online resampling across multiple epochs.

### 5.2 Temporal pairs

Enumerate `(tau,t)` from sampled source frames with `8 <= t-tau <= 32` and `t<T`. Assign candidate pair to one of lag bins [8,15], [16,23], [24,32]. Rank candidates by stable hash of physical trajectory, tau,t and pair seed. Select up to3/3/2 per bin, then fill unused quota from remaining candidates by hash up to eight total. No pair repeats. Read times may coincide; multiple source queries at the same anchor are sorted deterministically by `(tau,region)`.

Use the identical pair list for G3, R1 and A1. Repeated instructions on a path receive identical teacher targets and pair choices but different causal student context, as intended. t and tau never appear as student text or target-image input.

### 5.3 Cache union

G1 uses base sampled frames. G3/R1 use source regional targets. A1 additionally needs dense teacher targets for its current/read frames. Extract `base_frames union selected_read_frames`, still S=1 for every view. No uncached read frame is approximated by the nearest cached source.

### 5.4 Regions

Given `[15,20,D]`, use:

- p0: rows0:7, cols0:10;
- p1: rows0:7, cols10:20;
- p2: rows7:15, cols0:10;
- p3: rows7:15, cols10:20;
- p4: all15x20.

Average unnormalized PCA tokens, then use cosine normalization at loss time. These are coarse feature regions, not object segments or metric cells. Global means are weighted by actual cell counts, not an unweighted average of four unequal-area quadrants.

## 6. GeoAddress-SV relation target

For each selected `(tau,t)`, sample 32 unique current patch IDs: partition the 15x20 grid into 4x8 nonempty rectangles using integer floor boundaries, then hash-select one cell per rectangle from `(trajectory_key,tau,t,rectangle_id,seed)`. Keep the rectangle order fixed. This is deterministic spatial coverage, not a learned selector. Teacher descriptors for both images were computed independently.

Let `a_i=normalize(Z_t[i])` and `b_j=normalize(Z_tau[j])` in FP32, j=0..299. Form:

`P_T(i,j)=softmax_j(a_i dot b_j / 0.1)`.

No added positional term, hard same-place labels, mutual-nearest threshold, visibility assumptions or cross-scene negatives. This is a **soft feature-relation target**. Finite zero-norm descriptor rows are an extraction/QC error for this target, not fabricated uniform physical correspondences.

QC reports row entropy, max probability, similarity margins, same-frame/identical-frame behavior and sensitivity to spatial permutations. High entropy is not inherently invalid, but if all rows are numerically uniform A1 has no selective address-learning signal: report/block launch rather than secretly switching to a multi-view teacher. Qualitative patch visualizations must not be labeled ground-truth correspondences.

## 7. Interfaces proposed

`build_aux_frame_plan(trajectory_key: str, T: int, seed: int) -> FramePlan`

`fit_teacher_projection(stream: Iterable[Tensor], out: Path) -> ProjectionManifest`

`extract_single_view_targets(frame_paths: list[Path], manifest: TeacherManifest) -> list[FrameTarget]`

`load_targets(plan: EpisodeAuxPlan, cache: TeacherCache, kind: str) -> AuxTargets`

`soft_relation_targets(current: Tensor[32,256], source: Tensor[300,256], temperature: float=0.1) -> Tensor[32,300]`

Create `scripts/vln/extract_vggt_single_view.py`, `src/qwen_vl/data/vggt_targets.py`, `src/qwen_vl/research/vggt_sampling.py`. Target extraction is a separate stage; no simulator or teacher process in policy workers.

## 8. Acceptance tasks

- [ ] Verify B/S axis isolation by permuting other independent batch images and checking a frame's target within declared tolerance.
- [ ] Verify shape, returned patch offset, row-major ordering and 15x20 transform on synthetic spatial patterns.
- [ ] Fit PCA from training identities only; check shapes, finite reconstruction error and hash-stable reuse.
- [ ] Test short T=1,7,8,9; no invalid delayed pair, no duplicate frame, no fake target.
- [ ] Test cache-key change for weights/preprocess/PCA and unchanged key for identical single-view RGB.
- [ ] Verify G3/R1 regional targets are bitwise identical before dtype conversion.
- [ ] Profile 1,000 frames and project cost with uncertainty; do not assert “a few GPU-hours” in advance.
- [ ] Audit cache disk plus at least two complete recovery checkpoints and export headroom before training.
