> **G1 update (2026-10-13):** User requests ONLINE VGGT-Omega, no pre-extraction/PCA, the SpatialForcing projector, and4H100 tests. For G1, `G1_current_alignment.md` supersedes conflicting offline/PCA/two-GPU sections below. Other arms are not implemented by this task.

# Common implementation for exactly five VGGT runs

> **For agentic workers:** use an available execution/planning workflow task-by-task, with independent verification at each gate. This is a plan, not an already implemented API.

**Goal:** enable functional training-only auxiliary outputs and optional native Q/K adapters while preserving the existing streaming policy contracts.
**Architecture:** one narrow research training loop over the original decoder instances; selected layers return small auxiliary loss sums in addition to native hidden outputs. Serving uses the ordinary native/lane path except for installed runtime Q/K adapters in A0/A1.
**Tech stack:** pinned project PyTorch/Transformers/FLA/FlashAttention/DeepSpeed environment; no dependency upgrade as part of this work.
**Spec:** `vggt_five_runs.md` and `teacher_single_view.md`.

## Global constraints

Exactly G1/G3/A0/A1/R1 are new policy runs. Existing V0/R0 are read-only. Canonical text, Window8, original token IDs/MM positions, full BPTT and all dataset episodes remain. Teacher targets are detached and do not enter the student stream. Native recurrent state is FP32; model weights/main compute BF16. Additions must preserve native parameter names/tied storage. Metadata is immutable per forward, not a module-global current-episode value.

## Review focus

1. A read anchor accidentally shifted after the current action gives a label-conditioned auxiliary task.
2. Probe rows entering conv, residual, positional ledger or attention modify the deployed policy stream.
3. An auxiliary hook run twice under checkpointing produces duplicate or stale loss/graphs.
4. Empty delayed-target episodes or uneven rank counts break gradient participation/normalization.
5. Auxiliary save/export can silently drop runtime Q/K adapters or accidentally retain training-only queries.

## 1. File map

New files (proposed):

| File | Responsibility |
|---|---|
| `research/vggt_five_contract.py` | allowed run IDs, config, immutable auxiliary metadata and export/runtime distinctions |
| `research/vggt_sampling.py` | pure frame/pair sampling, target support counts |
| `data/vggt_targets.py` | bounded cache loading and provenance |
| `models/vggt_aux_heads.py` | alignment projector, context-conditioned query heads and state-read decoders |
| `models/gdn_research.py` | version-pinned native preparation, no-write query insertion, selected-head address residuals |
| `models/research_forward.py` | functional training decoder loop returning hidden and small loss sums |
| `train/vggt_five_loss.py` | global unit counts, ramp and summed auxiliary normalization |
| `scripts/vln/check_vggt_five.py` | native-off, parity, gradients and trained-export checks |
| `scripts/vln/profile_vggt_five.py` | real resource/recovery qualification |
| `scripts/vln/run_vggt_five.py` | allowlisted five-run launcher; baselines not launchable |

Existing files to modify narrowly: `models/nav_model.py`, `models/parallel_step_lane.py`, `models/install_step_lane.py` only as needed for pure read access, `data/episode_serializer.py`, `data/episode_dataset.py`, `train/trainer.py`, `train/train_episode.py`, `train/vln_runtime.py`, `contracts.py`. All are relative to `src/qwen_vl/` except scripts. Do not refactor unrelated candidate/episodic/RL paths.

## 2. Typed per-forward contracts

Use frozen dataclasses whose tensor members are never mutated during forward/backward. Tensor indices live on the execution device only for the active microbatch.

```python
@dataclass(frozen=True)
class EpisodeAuxPlan:
    episode_uid: str
    source_steps: tuple[int, ...]          # <=16 base samples
    visual_positions: torch.Tensor       # [T,300], original sequence positions
    pairs: tuple[tuple[int,int], ...]      # (tau,t), <=8
    anchors: torch.Tensor                # [P], original pre-action indices
    relation_query_cells: torch.Tensor   # [P,32]
    frame_units: int
    pair_units: int

@dataclass(frozen=True)
class AuxTargets:
    alignment: torch.Tensor | None       # [F,300,256], detached
    recall: torch.Tensor | None          # [P,5,256], detached
    relation: torch.Tensor | None        # [P,32,300], detached FP32
    teacher_hash: str
    sampling_hash: str

@dataclass(frozen=True)
class ResearchOutput:
    last_hidden_state: torch.Tensor      # [1,L,2560]
    aux_sums: torch.Tensor               # [3], FP32: alignment, recall, relation
```

Response-target positions in the existing serializer include all canonical response tokens. Compute each action's anchor as its minimum response target position minus one, not `read_positions` borrowed from another output mode. Check each anchor is before every supervised token of that action and in the expected observation/cue span. Group end remains after EOS/separator. `visual_positions` must contain actual expanded image-token positions, not approximate offsets `step*300`.

DataLoader returns CPU tensors/plain containers. Reconstruct frozen metadata objects only at active-microbatch preparation, with an explicit `move_aux_to_device` that returns new objects; do not assume the upstream Trainer recursively transfers arbitrary dataclasses. Ensure column filtering/collation retains the declared auxiliary fields. Plan support counts are known on CPU before forward. Relation target construction is done in the loader/preparation stage, not using current student logits. Training max sequence length counts original serialized tokens; GDN probe rows are a separate temporary-kernel scratch length.

## 3. Research training loop, no shared mutable hooks

Implement:

`research_text_forward(language_model, embeddings, position_ids, step_plan, plan, targets, spec) -> ResearchOutput`.

Use the pinned native text forward to obtain positional embeddings/masks and exact final norm semantics. Copy the minimal layer iteration only; do not create a second transformer with copied parameters. `Qwen3_5DecoderLayer.forward` returns a Tensor and performs input_norm -> mixer -> residual -> post_norm -> MLP -> residual.

For each decoder:

`run_research_layer(layer, hidden, immutable_context) -> (new_hidden, aux_sums_3)`.

Call `layer.forward` inside the explicit checkpoint wrapper rather than triggering its checkpointing `__call__` a second time. For unmodified layers, run the original native/lane function exactly. For selected GDN24, use a narrow version-checked preparation/kernel/postprocessing helper; selected G1 captures only its sampled visual rows after decoder23's full residual/MLP, then computes its loss **inside the checkpointed function** so the large feature tap is not retained externally.

The closure binds the current layer, metadata and detached targets by value (avoid Python loop-variable late binding). Return auxiliary scalar tensors from the checkpointed function; accumulate them outside. Never use `module.latest_aux`, persistent lists of differentiable outputs, or callbacks that append again during recomputation.

For baseline parity diagnostic (not a new run), the research loop with all additions off must match the native loop for outputs and qualified gradients. A0/A1 must use the same loop/adapter path; G3/R1 probe-off validation is mandatory.

### 3.1 Offload integration

Reuse `checkpoint_with_cpu_offload`/`offload_context` from the inspected branch, with non-reentrant checkpointing. Prefer passing only the main hidden tensor as the explicit tensor input to each checkpoint function; bind immutable detached small targets/position metadata in its closure, and account for their lifetime on GPU. If additional tensor inputs are saved/offloaded, expand `arena_bytes_for` or reserve a separately validated margin. Do not assume the old “one hidden per layer” bound still holds for new arguments.

No `autograd.grad` diagnostic calls during an active bounded-arena training forward unless repeated unpack/release semantics have been verified. Collect gradient-ratio diagnostics in an isolated disposable process with the corresponding qualified backend. No `retain_graph=True` on full long episodes.

## 4. Exact native preparation and no-write rows

Implement a version-pinned helper:

`prepare_native_gdn(module, x, cache=None) -> PreparedGDN`

where PreparedGDN contains Q/K/V after causal conv+SiLU, raw/repeated head maps, effective g and beta, z, and native cache-update metadata if serving. Preserve the native single-token update vs multi-token continuation and FP32 g/rotary behavior. For training, process convolution over the original entire sequence BEFORE inserting auxiliary rows. Never split/reinitialize convolution just because an auxiliary anchor exists.

At every requested anchor insert query rows after the original anchor token. Each row has learned q, finite fixed unit k, zero v, effective beta=0 and effective log-decay g=0. Insert **after Q/K head expansion** so per-value-head learned queries have the correct shape. Pass `use_qk_l2norm_in_kernel=True` and original query scale exactly once. Do not use `sigmoid(0)` as a zero beta or a raw zero preactivation as a zero effective decay.

Maintain exact original-to-extended index maps. Run the approved differentiable chunk kernel. Split output into real-token raw reads and probe reads. Remove probes before reshape against native z, RMS-gating, output projection, residual and later layers. Original tokens, step spans, MRoPE positions and labels are unchanged. No probes in inference or native cache cursors.

For recurrence convention `S:[B,H,Dk,Dv]`:

`S_bar = exp(g)*S; delta=beta*(v-S_bar.T@k); S_new=S_bar+k@delta.T; read=S_new.T@(q/sqrt(Dk))`.

For a probe, S_new=S and read is the desired associative query. In real arithmetic nonprobe outputs/state and nav gradients are unchanged. Chunk boundaries can shift BF16 numerics: qualify them, do not claim bitwise production equality.

### 4.1 G3 vs R1 integration

G3 inserts rows only into the native GDN24 kernel. R1 inserts rows only into the lane R24 kernel; its native kernel receives no probe rows. For R1, get detached sampled native q24 from the same native preparation (do not rerun the whole native mixer). The installed native+lane wrapper must not execute the lane twice: selected-layer research dispatch explicitly computes original native output once, lane output once, then the unchanged decoder residual/MLP.

Other installed lanes run through their ordinary function. Do not replace the existing serving sidecar or copy a live episode state between layers.

## 5. Global auxiliary loss reduction

Before an optimizer accumulation window, compute global counts using the same ranks/GAS window as existing action normalization:

`N_nav=sum(actions), N_align=sum(sampled_frames), N_recall=sum(valid_pairs), N_relation=sum(valid_pairs)`.

Implement `UpdateAuxCounts` alongside, not instead of, the existing `num_items_in_batch` action count. Minimal integration: `_get_num_items_in_batch` computes action count as before and adds per-microbatch immutable global aux counts to batch dictionaries. Verify the actual Trainer API with tests; do not pass an unexpected dictionary where upstream expects the scalar count.

For rank r/microbatch b:

`loss = world_size * [nav_sum/N_nav + lambda_align*align_sum/N_align + lambda_recall*recall_sum/N_recall + lambda_relation*relation_sum/N_relation]`.

The trainer must pop `update_aux_counts` and any bookkeeping-only fields before `model(**inputs)`; only the new declared `aux_plan`/`aux_targets` arguments enter the model forward. Terms with global zero support evaluate to a differentiable zero. All local numerator sums use their internal per-frame/per-pair averages; no extra multiplication by300 or5. DDP/DeepSpeed averaging yields the desired global means; existing code already prevents an extra GAS division. Keep that contract.

An episode with no delayed pair still trains all its navigation actions. Add `zero_participation()` for every registered auxiliary parameter when local support=0, using a zero-valued differentiable expression; test DDP ranks with zero support and globally empty windows. No fake teacher targets, padding actions or dropped episodes.

Preserve output `loss_sum` as navigation numerator for existing reporters. Add `aux_loss_sums`, `aux_counts`, `aux_weights`, and separately named `total_loss` logging. Do not overwrite nav loss with the mixed objective and compare its numerical curve against V0.

## 6. Optimizer, RNG and initialization

Create all modules before DeepSpeed/DDP wrapping. Use RNG forks/local generators so constructing auxiliary heads does not perturb native model/lane initialization or dataset order. A0/A1 adapter weights must be bitwise identical at start; R1 fresh lane weights must match the registered R0 initialization procedure.

Extend existing exact-once parameter-group checks with group precedence: geoaddress runtime adapters; auxiliary heads; lane; classifier/merger; native. No parameter may appear twice. Teacher parameters must not appear anywhere. Native tied weights remain tied. Exclude biases/norms/scalar gates from weight decay following existing policy.

## 7. Save, resume and export

Training recovery stores native weights, runtime adapter/lane weights, all auxiliary heads, optimizer/scheduler, RNG/sampler state, global progress and complete research/cache/sampling hashes. Recovery requires the same run ID, topology, recipe and target identities. A1 never resumes from A0 as a “parent”. R1 never resumes from trained R0 as extra exposure in this campaign.

Serving export strips ONLY the explicitly registered training-only `research_aux.*` parameter namespace. Preserve A0/A1 `.geoaddress.*` and R1 `.step_lane.*` parameters. Save separate `research_training.json` provenance, while runtime `navigation.json` contains only behavior needed for serving. Strict loader reconstructs runtime adapters/lanes then strict-loads; no broad `strict=False` or ignore-missing native weights.

Export comparison is trained model in aux-disabled serving mode vs reloaded export on the SAME histories. G3's probe-augmented training forward can have small numerical differences from probe-free serving; test separately, do not conflate the two parity claims.

## 8. Task checklist and acceptance

### Task C0: contract and baseline binding
- [ ] Test allowed new IDs are exactly G1/G3/A0/A1/R1; V0/R0 submission refused.
- [ ] Test registry mismatch catches lane count/clock/initialization/exposure differences.
- [ ] Implement config parser and metadata comparison; run regression suite; commit.

### Task C1: metadata/cache
- [ ] Tests for T=1/8/9/long, actual image token positions, source/read causality, deterministic repeat and hash failures.
- [ ] Implement plan/cache interfaces; no teacher loading in workers; commit.

### Task C2: auxiliary modules and functional forward
- [ ] Native-off logits/loss/gradients, no duplicate MLP/residual, same tensor names/tied storage.
- [ ] Add one-layer pure auxiliary outputs and checkpoint/offload integration; commit.

### Task C3: GDN probes
- [ ] Independent FP64 reference with Dk!=Dv, state/read/native-gradient equality, no future gradient for an isolated anchor loss.
- [ ] Pinned FLA forward/backward at actual32/4-head layouts; TF32-disabled strict reference plus separately declared BF16 tolerance.
- [ ] Production nav-only augmented-vs-unaugmented gradient comparison; checkpoint on/off and native-output deletion tests; commit.

### Task C4: global losses/optimizer
- [ ] Two-rank unequal frame/pair counts and partial accumulation; globally empty aux count; teacher grad absent; all new parameters assigned once.
- [ ] Tiny optimizer reference and save-resume parity; commit.

### Task C5: serving/export
- [ ] Aux heads absent in export, Q/K adapters retained, R0 lanes preserved, teacher not loaded at inference.
- [ ] Per-run trained export fixed-history and free-running response-token replay, reset/retry/eviction; commit.

### Task C6: resource and production
- [ ] Complete gates in resources_and_evaluation.md with exact source/manifest/teacher hashes.
- [ ] Publish code/tests/failed evidence and launch only five allowlisted arms.
