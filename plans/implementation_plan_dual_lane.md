# Parallel step-clocked lane implementation and execution plan

## Current resolved scope (2026-10-04)

User-approved final run: adapt the completed FullContext candidate-logits no-history wrapper for **one joint R2R + RxR pass**, using four existing H100s in interactive allocation4652. Preserve other training jobs and their source/environment. Both FullContext and Window8 code paths are required, but no FullContext checkpoint is silently converted to Window8.

Implementation commits: foundation `951cb04`; integration `cca4808`. Baseline `f194e23`. The source commit used for the immutable production snapshot and final gate outcomes are recorded in the implementation report.

Implemented areas:
- Tasks1–4: immutable config/roles, token-identical serializer metadata, native FLA lane and separate inference cache.
- Tasks5–6: per-instance native mixer adapter, explicit checkpoint kwargs, prefix/observation/feedback append roles, session reset/retry/cache accounting and unchanged native Window8 eviction.
- Task7: explicit `--init-policy-checkpoint`, protected parent validation, exact inherited object retention, lane optimizer groups, strict serving reconstruction and existing strict DeepSpeed resume. Existing-output rejection occurs before initialization/provenance writes.
- Task8: bounded parameter summaries and opt-in observed gate/state/write/output diagnostics plus fixed-history interventions. Closed-loop study and semantic retention measurements remain research work, not implementation success claims.
- Task9: unit/kernel/tokenizer tests and real-model replay; four-rank training/reload/resume and worst-case memory gates must finish before production launch. Consult the report for actual PASS/FAIL results, not these source checkboxes.
- Task10: the requested single joint adaptation run only. Matched native continuation, calibrated-token control, multi-seed closed-loop SR/SPL and any DSR extension are future separately scheduled experiments.

Schedule:30815 episodes, global batch8 (4×1×GAS2),3852 optimizer updates,116warmup; native LR5e-6, lane LR1e-4, navigation loss unchanged. `configs/vln_dual_full_no_history_joint.yaml` is the final scientific recipe. `configs/vln_dual_window8_no_history_joint.yaml` supports independently initialized or matching-parent Window8 experiments. Smoke tests use explicit short manifests and separate output directories, never counted as full data coverage.

Release checklist to resolve in the report:
- [x] Read local plans, verify SSH/allocation and protect jobs4643/4649.
- [x] Reconcile latest origin; isolate branch rather than pull dirty server root.
- [x] Baseline105H100 tests and completed compatible parent reload.
- [x] CPU and real-H100 lane/kernel/tokenizer tests; exact zero output and later-writer gradient tests.
- [x] FullContext and Window8 R2R/RxR fixed-history numerical gates.
- [x] Independent review and regression fixes for output-provenance preservation.
- [x] Four-rank STOP-only optimizer smoke.
- [x] Final post-review suite (210 H100 tests, zero skips), exact interleaved sessions and observed diagnostics on R2R/RxR.
- [x] Four-rank mixed data optimizer smoke and trained export reload.
- [x] Exact distributed optimizer/master/RNG restoration and same next batches/scheduler; numerically bounded continuation and strict resumed-export reloads (see the report’s retained failed initial logit threshold).
- [x] Two-update longest R2R FullContext, longest RxR FullContext/Window8, and FullContext just-below-offload-threshold resource gates.
- [x] Three-update mixed-tail staging gate, all ranks and ordered exposure records verified.
- [x] Commit verified source `3c8f580`; launch immutable joint run in step `4652.29` (live progress and publication timestamp are recorded in the report).
- [x] Final detailed report and resolved plans, including live launch verification and numerical limitations.
- Publication procedure: push `streaming_logits_dual` and verify its remote HEAD equals the documentation commit; preserve both isolated worktrees and the active immutable source.

The checkboxes below preserve the supplied acceptance inventory, including future research experiments. The release checklist above and measured report are the live execution status; unchecked items in the preserved inventory are not a claim that implemented functionality is missing.

## Original task specification and acceptance details

Parallel Step-Clocked Lane Implementation Plan
For agentic workers: Use superpowers:executing-plans, or an available subagent-driven workflow, to implement one task at a time. This is a specification for new work, not a patch already installed in SimpleMemVLN. All new commands in this file are unavailable until their owning tasks pass.

Goal: add four small, step-written associative memory lanes to the working streaming policy without changing data, tokens, action semantics or native parameter names.
Architecture: attach a step_lane child to the existing GDN instances at decoder layers16/20/24/28. The lane reads all tokens but writes only the last existing token in each completed observation/feedback group. In training its recurrence is functional within a whole episode; at inference each session owns a separate FP32 sidecar cache.
Tech stack: the checked repository stack: Python3.12 in production, Torch2.10, Transformers5.11.0, the pinned FLA/FlashAttention/DeepSpeed environment. Verify installed versions/hashes from the actual run; do not upgrade dependencies as part of this change. The bundled CPU reference was run in a different, explicitly recorded local environment.
Spec: [dual_lane.md](dual_lane.md). Acceptance requirements are the explicit test/gate checklists in this plan; the originally referenced separate design/acceptance files were not supplied.

Global constraints:
No pose, summary token, new RGB file, new target, action chunk, DSR loss, or native-head repurposing.
New namespace: model.step_lane in config; linear_attn.step_lane.* in weights.
Four selected GDN layers [16,20,24,28]; two key/four value heads, Dk=Dv128.
Same parent checkpoint, action mode, feedback, attention policy, vision preprocessing and additional exposure in matched arms.
No cached-state training; no truncation or detach inside the full-episode lane scan.
No change to native cache eviction, logical/multimodal positions, original navigation loss, or global action denominator.
Serving cache is session-owned; no episode state registered as a parameter or model buffer.
Disabled/absent feature does not alter old metadata, old state-dict keys, outputs or test paths.

Review focus
1. End-of-step is defined by serialized structure, not token ID; candidate no-history still closes a turn and must not regain gold actions.
2. Lane metadata must survive native layer checkpoint recomputation without mutable module/global state.
3. The post-decision write must not influence the current target; full future-target perturbation tests must use causal prefixes correctly.
4. Warm-start must copy actual trained wrapper tensors and tied storage, not only a generic base model or reinitialized copied head.
5. A one-step episode can have zero lane gradient; all parameters must nevertheless participate under the current DDP/ZeRO configuration.
File ownership map
File	Status	Responsibility
src/qwen_vl/research/lane_contract.py	New	Spec dataclass, validation, role enum, namespace identity
src/qwen_vl/data/lane_metadata.py	New	Immutable roles derived from prefix/observation/group spans
src/qwen_vl/models/parallel_step_lane.py	New	Projections, masks, pure recurrence adapter, output
src/qwen_vl/models/install_step_lane.py	New	Targeted native forward composition; state-key stability
src/qwen_vl/stream/lane_cache.py	New	Per-session state and append cursors
src/qwen_vl/train/lane_initialization.py	New	Explicit weights-only initialization and manifest
src/qwen_vl/train/lane_reporting.py	New	Detached, bounded summaries at optimizer boundaries
src/qwen_vl/data/episode_serializer.py	Modify	Emit roles for enabled models without token changes
src/qwen_vl/models/nav_model.py	Modify	Pass explicit lane metadata; install during construction/load
src/qwen_vl/train/vln_runtime.py	Modify	Reconstruct lane before strict checkpoint load
src/qwen_vl/contracts.py	Modify	Validate lane compatibility; preserve disabled contracts
src/qwen_vl/stream/session.py	Modify	Sidecar reset and explicit per-append roles
src/qwen_vl/train/trainer.py	Modify	Exact new parameter grouping; unchanged loss normalization
src/qwen_vl/train/train_episode.py	Modify	New initialization flag, lane LR, provenance and reporting
scripts/vln/check_step_lane.py	New	Real tokenizer/model/kernel/parity acceptance gate
scripts/vln/diagnose_step_lane.py	New	Fixed-history probes, read/reset interventions and reports
tests/vln/test_step_lane_*.py	New	Unit/integration tests assigned below


Do not implement every possible future API. In particular, classification support, pose, direct-state auxiliary losses and new checkpoint formats are outside v1.
Task 0 — freeze the executable baseline and identify the parent
Files: create reports/STEP_LANE_PREFLIGHT_<date>.json and immutable experiment manifest; do not alter model code.
Consumes: actual checkout, environment, parent checkpoint path, generic Qwen base path, selected training manifest.
Produces: parent_identity.json with source SHA/dirty status, all checkpoint weight-file hashes, navigation metadata hash, base revision, runtime/kernel versions, dataset hash, output/feedback/window contract, and declared episode budget.
- [ ] Run git rev-parse HEAD; git status --short; explicitly review any changes after the inspected pin.
- [ ] Inspect actual navigation.json and all weight shards; distinguish pytorch_model.bin wrapper weights from a generic HF backbone. Do not use an index-file hash as a weight-content hash.
- [ ] Run baseline unit tests using the repository's actual interpreter and record skips separately. Existing tests are not assumed to pass just because earlier reports did.
- [ ] Load the parent with existing load_checkpoint; run one complete expert-observation replay and save action/logit and native-cache reference signatures.
- [ ] Record whether the parent is Window8/FullContext and whether candidate feedback is none, candidate-token or canonical text. Do not silently change these in any arm.
- [ ] Confirm the target source tree does not contain an incompatible unpublished EgoLane implementation. If it does, port its math/tests only after a source review; do not install a second wrapper on top.
- [ ] Write parent_config.yaml directly from the saved navigation.json["config"] and empty_output.yaml as {}. These are the CLI base/output inputs for adaptation, so protected parent fields are not accidentally replaced by current defaults. Validate exact round-trip before adding a training/lane overlay.
- [ ] Commit the baseline experiment manifest without secrets, model weights or dataset images.
Task 1 — pure config and role contract
Files: new research/lane_contract.py, data/lane_metadata.py; modify root contracts.py; test test_step_lane_contract.py, test_step_lane_metadata.py.
Interfaces:
@dataclass(frozen=True)
class StepLaneSpec:
    version: str
    layers: tuple[int, ...]
    num_key_heads: int
    num_value_heads: int
    key_dim: int
    value_dim: int
    clock_mode: str  # step_end | token_rate_calibrated
    prefix_mode: str  # read_only_zero
    half_lives_steps: tuple[float, ...]
    beta_init: float
    token_rate_reference: float | None
    projection_init_std: float
    eps: float

parse_step_lane_spec(config: dict, text_config) -> StepLaneSpec | None
build_lane_roles(prefix_length: int,
                 observation_spans: tuple[tuple[int,int], ...],
                 complete_spans: tuple[tuple[int,int], ...],
                 total_tokens: int) -> torch.Tensor  # uint8[1,L]
enabled absent/false returns None and emits no new old-model metadata. With enabled=true, validate version='parallel_step_lane_v1', selected layer types, dimensions, sorted unique layers, finite positive half-lives, 0<beta_init<1, mode and prefix rule. Reject classification for v1; candidate none is allowed but preserved explicitly. token_rate_reference is required only for the calibrated control.
- [ ] Write failing tests: duplicate/wrong layer, invalid modes, nonfinite half-life, unsupported classification, all three candidate feedback formats, and legacy config unchanged.
- [ ] Write role tests with two groups of unequal feedback length: prefix all0; observation all1; feedback2; each final group token3. A repeated newline ID must not become a writer elsewhere.
- [ ] Test gap/overlap/truncated spans and an empty post-decision closing block raise. Require writer index greater than its action decision index.
- [ ] Implement only metadata/config construction. Check there is no tensor floating conversion requirement; roles remain integer under DeepSpeed input preparation.
- [ ] Run tests and commit independently.
Task 2 — serialize the exact same tokens plus metadata
Files: modify data/episode_serializer.py; test test_step_lane_serializer.py.
Consumes: StepLaneSpec, build_lane_roles.
Produces: existing episode dict plus step_lane_roles only when lane enabled. Original step_plan spans remain complete observation/feedback groups.
- [ ] Save the observation end before appending feedback inside encode_episode; collect observation spans as local metadata.
- [ ] After completing all existing blocks, build roles and add step_lane_roles to the result. Do not add summary embeddings/placeholders, change prompt/cue bytes, shorten feedback or change token IDs.
- [ ] For text B, independently derive the decision position from the first response target minus1. For candidate, use read_positions. Verify writers are strictly after those positions.
- [ ] Compare enabled-vs-disabled serialization: exact input_ids, mm_token_type_ids, pixel_values, image_grid_thw, all action targets, positions and step_plan. The only difference is additional metadata/config.
- [ ] Candidate no-history: changing every action label must not change its serialized token stream or roles; labels/loss targets may change. Feedback history mode must change only the expected feedback suffix, never earlier observations.
- [ ] Terminal STOP and a one-step episode still have exactly one STEP_END. Run real-tokenizer tests before committing.
Task 3 — reference math and lane module
Files: new models/parallel_step_lane.py; tests test_step_lane_math.py, test_step_lane_module.py.
Interfaces:
class ParallelStepLane(nn.Module):
    def __init__(self, hidden_size: int, spec: StepLaneSpec, *, device, dtype): ...
    def forward(self, x: Tensor, roles: Tensor,
                initial_state: Tensor | None = None, *,
                return_final_state: bool = False) -> tuple[Tensor, Tensor | None]: ...
# x [B,L,2560], roles [B,L], output [B,L,2560], state [B,4,128,128]
Use the complete equations, biases, init and dtype policy from design section4. A small sequential reference is for tests only. Production uses the installed FLA functions, and fails rather than silently choosing the slow reference.
- [ ] Write unequal-dimension tests (Dk=3,Dv=5) for update/read orientation, zero gates and read-after-write behavior.
- [ ] Test mask removes both decay and correction on nonwriters, while query-dependent readout may vary. Test init half-lives after actual parameter casting within an explicit rounding tolerance.
- [ ] Implement the projections with exactly 5,263,500 parameters per production lane; verify the count. No learned initial state, output bias or lane convolution.
- [ ] Test zero-initialized out_proj gives zero branch output on identical x and finite states. Test all intermediate results are finite; never rely on 0 * NaN to preserve baseline.
- [ ] Test initial out_proj gradient is nonzero on a multi-step fixture; upstream lane gradients can be zero on the first backward. After one update opens the projection, confirm qkv/gate gradients exist for relevant earlier writers.
- [ ] Test first decision before any writer has zero lane contribution even with nonzero out_proj; one-step STOP-only sequence has valid zero lane gradients.
- [ ] Test full sequence versus 1-token and irregular-block execution for outputs/final state and full gradients with no detaches. Do not assume arbitrary chunk sizes imply TBPTT.
- [ ] On GPU, compare native FLA chunk/recurrent calls with the same prepared q/k/v/g/beta to an FP32 reference, including exact beta0/g0. Commit only after reference unit tests pass; production release waits for Task9.
Task 4 — sidecar state and append accounting
Files: new stream/lane_cache.py; test test_step_lane_cache.py.
Interfaces:
class StepLaneCache:
    def __init__(self, spec: StepLaneSpec, episode_uid: str, *, device): ...
    def initial_for(self, layer_idx: int, logical_start: int) -> Tensor: ...
    def commit(self, layer_idx: int, logical_start: int,
               appended_tokens: int, final_state: Tensor) -> None: ...
    def assert_complete_append(self, logical_end: int) -> None: ...
    def reset(self, episode_uid: str) -> None: ...
Own one FP32 tensor and processed-token count per selected layer. Data are not module buffers/parameters and never enter state_dict. New cache zeroes all states/counters; initial_for verifies the incoming offset and returns a state for an inference-only call. commit checks dimensions, device, finiteness, sequential append start, no gradients and atomically replaces that layer's state. Preserve actual token counts, including prefix and nonwriter reads.
- [ ] Test disjoint caches, layer indexing, wrong shape/dtype/nonfinite state, negative/skipped offsets, and repeated commit rejection.
- [ ] Test zero-write append advances token cursor but not mathematical state value. Mutation protection must cover any kernel aliasing of initial/final tensors.
- [ ] Test complete-append check catches a selected layer not executed and returns only after every selected layer reaches the same logical end.
- [ ] Test reset clears all states. Native evict_kv must not reference this object or change it.
- [ ] Reject cache commits when gradients are enabled. Keep functional state tensors as a separate test interface; do not repurpose this inference cache for BPTT.
- [ ] Run tests and commit.
Task 5 — install the parallel lane without renaming native parameters
Files: new models/install_step_lane.py; modify models/nav_model.py, train/vln_runtime.py; tests test_step_lane_install.py, test_step_lane_checkpointing.py.
Interfaces:
install_step_lanes(model: SimpleMemVLNForNavigation,
                   spec: StepLaneSpec, *, init_seed: int) -> None
# Hidden/forward kwargs added:
step_lane_roles: Tensor | None
step_lane_cache: StepLaneCache | None
step_lane_logical_start: int | None
A targeted bound method on selected native linear_attn instances consumes lane kwargs, calls the untouched original native forward, computes lane output from the same normalized hidden_states, commits sidecar state only in inference, and returns the sum. Record installation spec and original callable without duplicate module registration.
- [ ] Write tests that existing parameter names/object IDs and native kernel attributes are unchanged. Only .step_lane.* keys may be added. Same-spec reinstall is a no-op; incompatible existing wrappers fail.
- [ ] Patch SimpleMemVLNForNavigation.hidden and forward to accept/pass role metadata. Active lane missing roles must raise, including online calls.
- [ ] Forward lane kwargs through the language model and native decoder. Explicitly remove private lane kwargs before original full-attention backends; retain step_plan, stream_append, positions and original kwargs. Use per-instance passthrough, not global backend mutation.
- [ ] For gradient-enabled offline forward, require sidecar=None and native training use_cache=False. Each lane scans the entire sequence from zero. Do not capture a serving object in a closure.
- [ ] Test ordinary vs non-reentrant checkpointed forward/backward on the actual patched decoder path. An outer mutable diagnostic/schedule variable changed after forward must not change recomputed lane behavior.
- [ ] Test installation inside an RNG fork preserves outside CPU/CUDA RNG states. New initialized parameters follow native BF16 device/dtype; recurrence and gate arithmetic follow design FP32 rules.
- [ ] Compare unmodified policy with installed-zero-output policy on the exact same serialized inputs: original logits, loss and native parameter gradients. New parameters may have gradients; do not require the trained result to remain identical after the optimizer step.
- [ ] Commit. Keep existing SimpleMemVLN._apply rotary-preservation behavior unchanged.
Task 6 — update streaming session at real append boundaries
Files: modify stream/session.py; tests test_step_lane_session.py.
Consumes: serializer protocol, StepLaneCache, installed adapter.
Produces: unchanged action_result schema plus optional declared lane state-byte diagnostics, never new default target inputs.
- [ ] reset: create a new sidecar before prefix append; prefix roles all0. Preserve original native reset/rope logic.
- [ ] _append(block, *, step_lane_roles=None): capture logical_start before PositionLedger advances; pass explicit roles/cache/start to model.hidden; assert native dtypes and sidecar completion afterward. Disabled path remains unchanged.
- [ ] Observation append: roles allOBSERVATION, no writer at the pre-action cue.
- [ ] Candidate feedback: allFEEDBACK except final tokenSTEP_END; this applies to candidate-token, canonical-text and fixed none closure. The latter must not insert class IDs into inputs.
- [ ] Text ordinary token append: FEEDBACK only. EOS+separator append: final token only STEP_END. Preserve original response validation and fail-closed invalidation.
- [ ] Test writer trace equals offline roles for the same feedback history. Teacher-forced parity cannot use gold offline versus independently generated online history after divergence.
- [ ] Test Window8 at the eviction boundary t=8/9/16, native KV length, sidecar persistence, logical positions, repeated same-step requests and same-image new-step writes.
- [ ] Test independent episodes interleaved on the same model have disjoint R. No module hidden state is allowed.
- [ ] No next-step writer deferral, planning cache, observation skipping or new environment call. Commit.
Task 7 — exact warm-start, optimizer and recovery
Files: new train/lane_initialization.py; modify train/train_episode.py, train/trainer.py, train/vln_runtime.py; tests test_step_lane_initialization.py, test_step_lane_optimizer.py, test_step_lane_recovery.py.
Interfaces:
initialize_step_lane_from_policy(parent_checkpoint: str, base_model_path: str,
                                 target_config: dict, *, lane_seed: int)
    -> tuple[SimpleMemVLNForNavigation, EpisodeSerializer, dict]
# New CLI: --init-policy-checkpoint PATH (mutually exclusive with resume)
- [ ] Implement by strictly loading the parent wrapper first, comparing protected fields, updating only declared training/lane config and installing new lanes. Reuse the loaded backbone/classifier objects so a trained copied head is not reconstructed. Rebuild serializer metadata from the unchanged processor and target config.
- [ ] Forbid parent-to-child changes of output mode, serializer/feedback, visibility, vision settings, base revision or native head mode under this warm-start flag. Training budget/LR and the new lane config may change explicitly.
- [ ] Preserve all inherited state tensors exactly; missing/unexpected inherited keys are errors. If porting through a state-dict loader, missing keys must be exactly expected fresh lane keys, not merely tolerated by strict=False.
- [ ] Add Lane LR1e-4 grouping before generic classifier/backbone classification. Assign decay/no-decay exactly once; frozen vision stays out. Assert union/uniqueness of parameter IDs and respect tied LM embeddings.
- [ ] Leave compute_loss, global action count, GAS scaling and class weighting unchanged. Rerun existing distributed normalization tests, then one actual two-rank update with lanes.
- [ ] Save lane spec inside navigation config; inherited serializer name stays the same because tokens are unchanged. Strict resume compares full config including lane; strict serving construction creates the same lane before loading weights.
- [ ] Record parent all-weight hashes and init report in a separate provenance artifact. Do not mutate parent checkpoint files.
- [ ] Reject simultaneous init/resume. Resume restores optimizer slots, scheduler, data order, RNG and weights, but not an inference episode cache. Full-episode training checkpoints occur at optimizer boundaries.
- [ ] Test save/reload of zero and nonzero lane weights; normal trained export retains lanes. Removing a lane is an inference intervention or a new model, not an auxiliary-head export.
- [ ] Commit with round-trip equality evidence.
Task 8 — bounded diagnostics and experiments
Files: new train/lane_reporting.py, scripts/vln/diagnose_step_lane.py; tests test_step_lane_reporting.py.
Outputs: JSON reports separated into algebraic stats, teacher-forced replay, closed-loop results and resource timings.
- [ ] Log per-layer/head writer beta/alpha, write magnitude, state norm, lane-output/native-output RMS ratio and out_proj norm. Detach/reduce sampled observations; do not retain autograd graphs or copy all states at every token.
- [ ] Label initial/current decay-only half-life as such. If probing conditional erasure, state normalized-key assumptions and distinguish it from semantic recall or full network Jacobians.
- [ ] First-action lane output should be zero under empty prefix state; verify repeated writer counts and interval identity on replay.
- [ ] Inference interventions: lane output disabled, lane reset at group boundaries, and lane freeze after an observed boundary. Declare what changes and what remains: old lane influence may already exist in native GDN/KV, so these are not pure isolation of every lane-mediated pathway.
- [ ] Evaluate exactly matched episode keys and compute win/loss pairs, SPL, STOP/timeout and path-length strata. Do not use a scene-ordered partial prefix as the sole development panel.
- [ ] Diagnostics off must leave outputs identical and timing free of hooks. Commit.
Task 9 — real execution gates, not just toy tests
Files: new scripts/vln/check_step_lane.py; schema tests. Detailed requirements in the acceptance document.
- [ ] Test the actual pinned tokenizer and all supported feedback formats, including no-action-history token independence.
- [ ] Run real Qwen/FLA zero-output equivalence, whole-sequence vs incremental fixed-history parity, and gradients with W_out deliberately opened on a tiny test checkpoint.
- [ ] Cover native prefill, multi-token continuation, single-token decode, Window8 eviction and FullContext reference. No backend-specific epsilon workaround without a new declared experiment.
- [ ] Run one-step/zero-use lane-gradient case on two ranks; run normal multi-episode optimizer and checkpoint/resume parity.
- [ ] Stress actual longest/high-token/long-instruction examples after optimizer slots exist, with original checkpoint/offload settings. Use R2R and the real RxR tail if the campaign is joint. No repeated shortest-episode benchmark extrapolation.
- [ ] Benchmark baseline and lane one process at a time on the same hardware, with diagnostics off, matching inputs/warmups/timing boundary. Separate model latency from simulator time.
- [ ] Record PASS/FAIL/SKIP; any GPU skip means the production gate is not passed. Commit reviewed implementation before long runs.
Task 10 — declare and execute the controlled study
Files: final resolved configs, immutable source snapshot, experiment manifest; no hot edits.
- [ ] Run N and L-step from exactly the same parent with matched additional schedule and episode batches. Both require fresh output directories.
- [ ] Run parameter-matched L-token-calibrated with frozen training-manifest n_ref and recorded gate initialization.
- [ ] Screen on the same scene-balanced panel; validate promising candidates on all1,839 R2R episodes and repeat matched seeds. Do not declare a target SR gain in advance as guaranteed power.
- [ ] Add FullContext × lane only as a separate matched study; constant lane bytes do not make FullContext bounded.
- [ ] If there is a reproducible benefit, consider DSR in a separate2×2 experiment. No pose/summaries as an undisclosed 'fix'.
- [ ] Produce a final report listing all runs, negative results, actual costs, provenance and limitations.
Completion definition
Implementation is complete only when contracts, real numerical/kernel tests, distributed/restore checks and runtime metrics are available. Research success is a separate question determined by controlled navigation/retention outcomes. This plan does not authorize job cancellation, data deletion or modification of a running campaign's source.
