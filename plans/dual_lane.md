# Parallel step-clocked lane: resolved implementation specification

Implementation source: `3c8f580`. The final 210-test H100 suite, FullContext/Window8 real-model replay and session isolation, R2R/RxR smoke, longest/tail resource gates, and exact optimizer/master/RNG restoration are verified. BF16 continuation is numerically bounded rather than bitwise reproducible; the initially failed tight resumed-logit threshold and follow-up controls are retained in `reports/DUAL_LANE_IMPLEMENTATION_2026-10-04.md`. The requested joint FullContext/no-history run is launched in Slurm step `4652.29` from a separate read-only source snapshot. Its actual live progress is recorded in that report; scientific benefit remains unevaluated.

## Resolution for the 2026-10-04 implementation

The user confirmed weights-only adaptation from the completed compatible **FullContext candidate-logits no-action-history policy**, followed by **one joint R2R + RxR pass**. This supersedes the preferred Window8 pilot parent and R2R-only schedule below for the requested launch. Both FullContext and Window8 are implemented and tested; FullContext weights are never relabeled as a trained Window8 policy.

The actual joint manifest contains 30,815 episodes (10,819 R2R and 19,996 RxR), 3,128,624 action states. Four ranks × one episode × GAS2 gives global batch8 and 3,852 updates with the existing repeated-tail policy. The final joint recipe uses116 warmup updates (ceil(3% ×3852)), backbone LR5e-6 and lane LR1e-4. R2R-only recipes retain1353/41 for separate use. No action history, auxiliary loss, new token, pose feature, or detached training state is introduced.

Source starts at inspected `origin/streaming_logits` commit `f194e23`. Existing server working tree and running campaign snapshots are protected. Implementation is in an isolated `streaming_logits_dual` worktree/checkout; the final training run uses an immutable source snapshot.

The exact equations and causal contract below remain binding. Production BF16 gradients are compared on four explicitly named native tensors against measured repeated-native-backward variation: relative norm ≤3% and ≤max(0.2%,2×native repeat), maximum absolute difference ≤max(1e-6,2×native repeat). This is a numerical gate, not bitwise production gradient equivalence. Exact zero-output logits/loss, exact CPU decoder gradients, and exact zero upstream gradients through the zero lane are independent checks.

Opt-in observed diagnostics report actual writer gates, per-head state/update norms and lane/native RMS ratios. The native RMS is reconstructed from the BF16 combined output minus lane output in FP32 and is explicitly labeled approximate. Ordinary training has no observation hooks. Parameter summaries label zero-input decay half-lives as references, not semantic retention.

Execution outcomes, precise commands, limitations and launch location are maintained in `reports/DUAL_LANE_IMPLEMENTATION_2026-10-04.md`; incomplete gates are never recorded as passed.

## Original scientific specification, retained with the resolutions above

1. Scientific question and falsifiable hypothesis
Can an already trained streaming navigation policy benefit from an additional, small associative state that is read at token rate but modified once per completed observation/feedback group?
This is not a claim that all native GDN heads have a defective clock. Token-level updates can be useful. The proposed lane complements rather than replaces them. It also does not guarantee that a slow state stores landmarks, geometry or instruction progress. Those are possible learned contents to measure, not hardcoded meanings.
The architecture and an eventual delayed-feature auxiliary must be evaluated independently. This specification implements only the architecture. A parallel lane is a transfer of known recurrent/residual-memory principles; novelty depends on diagnosis, controlled results and the particular integration, not on claiming the first two-timescale memory.
2. Locked pilot architecture
Field	Value
Host	Existing trained Qwen3.5-4B SimpleMemVLN policy
Preferred attention policy	An actually trained and validated Window8 parent; never reinterpret FullContext weights as Window8 silently
Supported output modes	qwen_text, candidate_logits
Feedback	Preserve the parent's exact format; candidate none remains none
Unsupported in v1	classification with no post-decision closing block
Zero-based decoder layers	[16, 20, 24, 28]; validate each is linear_attention
Native branch	Original callable, parameters, conv state and recurrent cache unchanged in structure
New lane per selected layer	2 query/key heads; 4 value heads; key/value dimensions 128
New state per selected layer	[B, 4, 128, 128], key-by-value, FP32
New state across four layers	16 matrices, 1,048,576 bytes per episode at B=1
New tokens	None
Write event	Last existing token of each complete observation/feedback group
Prefix behavior	Lane starts at zero and does not write during prefix
Read frequency	Every token at each selected layer
Pose / extra sensors / labels	None
Lane convolution	None; native convolution is retained unchanged
Output integration	Add a zero-initialized projected lane output to native GDN output
Initial decay-only half-lives	[8, 24, 64, 192] write events, one per value head
Initial beta	0.1 for all lane value heads
Objective	Existing navigation loss only
Training graph	Existing complete-episode, uncached full BPTT


All numerical choices are pilot design choices, not measured optima. In particular, beta=0.1 is a conservative proposed initialization; it is not a fact about the external proposal. Actual half-lives after BF16 casting must be logged rather than asserted exact.
2.1 Corrections to the informal explanation
Layer 15 is a full-attention layer in the pinned [L,L,L,F] pattern, not a GDN layer. Selected layer 16 follows it. Layer indices are locations for additional lanes, not times when a global memory is updated.
At each selected layer there are two banks, not two individual matrices: native bank [B,32,128,128] and new bank [B,4,128,128]. Different layers own independent states and weights. Their interaction is through the hidden stream, not copying matrices between layers.
A runtime state R is not an nn.Parameter: it is generated from the current episode, reset for a new episode and not included in the model weight checkpoint. Learned Q/K/V, gates, normalization and output projections are parameters and are saved.
3. Exact attachment point
For selected decoder layer l, let x be the native input_layernorm output supplied to linear_attn. The layer becomes:
residual = original layer input
x = existing input_layernorm(residual)
y_native = original native linear_attn(x, native_cache, ...)
y_lane   = step_lane(x, lane_state, token_roles)
z = residual + (y_native + y_lane)
output = z + existing_mlp(existing_post_attention_layernorm(z))
Do not add a second copy of the decoder residual, omit the existing MLP, or feed y_native into the lane instead of the agreed input x. Both paths read the same normalized x. Later layers naturally see the combined representation. Calling the native equations 'unchanged' does not mean later native activations remain identical after the lane learns.
At the same step-end token, layer16 writes its own state, affects downstream hidden representations, and layer20 later writes its distinct state. There is no global post-forward instruction that copies a final hidden vector into four matrices.
3.1 Parameter namespace and installation
Attach step_lane as a child of the existing selected linear_attn instance:
backbone.model.language_model.layers.16.linear_attn.step_lane.*
Preserve every pre-existing state-dict key. A project-owned bound-forward adapter calls the original native unbound forward, after removing lane-specific kwargs, and adds the lane branch. Do not wrap the native module as wrapper.native and silently rename all pretrained keys. Do not globally monkeypatch the Transformers class or FLA functions.
Installation is idempotent for the same immutable spec, and rejects incompatible repeated installation. Keep the original callable available for off-mode tests. Add no native-module alias as a registered child. Construct lanes before optimizer/DeepSpeed setup.
4. Lane equations and tensor contracts
4.1 Projections
For x of shape [B,L,2560]:
- in_proj_qkv: Linear(2560,1024,bias=False), followed by SiLU.
- Split widths 256/256/512 into q_raw/k_raw/v.
- q_raw/k_raw: [B,L,2,128]; repeat-interleave each head twice to obtain [B,L,4,128].
- v: [B,L,4,128].
- in_proj_z: Linear(2560,512,bias=False), reshaped [B,L,4,128].
- in_proj_a: Linear(2560,4,bias=False).
- in_proj_b: Linear(2560,4,bias=True).
- A_log, dt_bias: four scalar parameters each.
Let q and k be normalized using x * rsqrt(sum(x*x) + 1e-6). The additive-epsilon convention is deliberate; F.normalize(eps=...) uses a different formula. The production FLA path can normalize internally with use_qk_l2norm_in_kernel=True. Never normalize twice, and verify parity against the installed FLA version.
Use the native query scaling of 1/sqrt(D_k) exactly once. No lane RoPE is added in v1.
4.2 Roles and masks
Role tensor is uint8 [1,L]:
Role	ID	Meaning	Step-clock write/decay mask
PREFIX	0	Initial instruction/system prefix	0
OBSERVATION	1	Image, framing, decision cue	0
FEEDBACK	2	Action/closing tokens except final group token	0
STEP_END	3	Final existing token of completed group	1


Set roles from serialized spans, not token values. A newline may be a cue, prefix token or step-end in different locations.
For a complete group [s_t,e_t) and decision index d_t:
writer_t = e_t - 1 and writer_t > d_t.
The first condition identifies the event; the second proves the new write occurs after the decision. Full offline encoding has exactly T STEP_END positions, including final STOP. Prefix contains none. Candidate feedback none still appends a fixed terminator/newline block in the current code, so a post-decision STEP_END exists; its write is observation/closing-context memory, not action-conditioned memory.
4.3 Masked gates and update
Compute raw gates in the normal projection dtype, with decay algebra in FP32:
g_raw = -exp(A_log.float()) * softplus(a.float() + dt_bias.float())
beta_raw = sigmoid(b)
For the step-clock arm, let w=d=1 only at STEP_END:
g = where(d, g_raw, 0)
beta = where(w, beta_raw, 0).
Reject nonfinite gates/outputs; do not patch NaNs or introduce tiny writes to hide kernel failures. Use no in-place masking on tensors saved for backward.
For one value head with state R [D_k,D_v]:
\[
\bar R_i = \exp(g_i)R_{i-1},\qquad
\hat v_i = \bar R_i^T k_i,
\]
\[
\delta_i = \beta_i(v_i-\hat v_i),\qquad
R_i = \bar R_i+k_i\delta_i^T,
\]
\[
r_i=R_i^T(q_i/\sqrt{D_k}).
\]
Read is after update at writer tokens, matching the native delta-rule convention. At nonwriter tokens g=beta=0, so R is unchanged and read uses inherited state. Updating at the writer can affect higher layers at that same post-action token, but cannot affect the already computed decision at d_t.
The algebraic state identity does not mean every fused kernel must be bitwise identical. The sequential reference tests exact arithmetic identities; production numerical error is separately measured.
4.4 Output
For each 128-dimensional value-head read:
\[
u_i = \left[r_i\operatorname{rsqrt}(\operatorname{mean}(r_i^2)+10^{-6})\odot w_{norm}\right]\odot\operatorname{SiLU}(z_i).
\]
Use FP32 for RMS statistics and gating evaluation, then cast to projection dtype. norm.weight is one shared 128-vector across heads, initialized to ones. This is explicitly normalize-then-gate, not normalize after multiplying the gate.
Concatenate four heads (width512), then apply out_proj = Linear(512,2560,bias=False). All its weights start at zero. Native layer residual/MLP remain outside this function.
4.5 Initialization
- qkv/z: independent Normal(0,0.02) matrices.
- in_proj_a.weight=0, A_log=0.
- dt_bias[h]=log(expm1(log(2)/HL[h])), formed in FP32 then cast like other parameters.
- in_proj_b.weight=0; bias=log(0.1/0.9).
- norm.weight=1.
- out_proj.weight=0.
- runtime state=zero FP32; no learned initial state.
This yields intended initial decay half-lives but does not guarantee recall half-lives: beta/key-dependent overwriting can erase content sooner. The 8/24/64/192 values do not prove head specialization.
Create random lane weights inside a local RNG fork with a recorded lane_seed. Restore the pre-construction CPU/CUDA RNG states so native training shuffles/dropout are not changed by the extra initialization.
4.6 Counts
Per layer: 5,263,500 parameters under the exact bias/shared-norm conventions above. Four layers: 21,054,000 parameters. This count differs slightly from the other proposal because beta has an explicit bias here.
Four lane states: 4 layers * 4 heads * 128 * 128 * 4 bytes = 1,048,576 bytes = 1 MiB per episode. This excludes native recurrent/conv states, KV, activations and optimizer memory.
Small state does not imply tiny activation cost. qkv/z/read tensors are still produced over the token sequence. Measure time/VRAM; no claimed speedup or 'zero overhead'.
5. Causal lifecycle
5.1 Initialization/prefix
Create a separate per-session lane cache. Prefix is processed by the original model. New lanes read zero state and do not write. Their output is zero for this prefix, because normalization/gating contains no additive output bias.
5.2 Observation and decision
For step t, all current observation/cue roles are nonwriters. Lanes read state after step t-1. The action comes from the unchanged candidate/text head. With step-clock and zero initial state, lane contribution before the first episode decision is zero even after training, provided native weights are held fixed for that comparison.
5.3 Feedback and write
Candidate mode appends one deterministic feedback block; mark only its final token STEP_END. Text mode appends ordinary generated tokens with no writes, then appends EOS + separator; only the last token of that final append writes. Never mark EOS itself if another separator token follows. Invalid response handling still invalidates the complete session.
The existing session appends chosen feedback before the simulator executes the action. Preserve this contract; reset/abort if execution fails or differs in a nonterminal way. The state records the observation and chosen command, not measured action success. Forced terminal STOP can end an episode without a future policy call; do not silently carry that mismatched terminal history into another episode.
5.4 Native versus lane state
Native DynamicCache retains original FP32LinearAttentionLayer entries. A separate StepLaneCache owns only new R tensors, by zero-based layer index. Do not add fake layer entries or alter has_previous_state behavior of native caches.
Window8 eviction touches only native full-attention K/V. It must not delete lane R, native GDN/conv state, or reset logical position counters. FullContext can use the same lane but total serving memory remains unbounded due to native KV.
5.5 No deferred append optimization
Do not combine a previous step's writer with a next observation across Window8 eviction. Those queries can require different visibility. This version adds no writer tokens/calls, so there is no need for deferred summary append.
5.6 Sessions and retries
All episode state belongs to StreamSession, not the model/lane module. Reset all lane matrices/counters for every new episode. Same-step retry returns the existing action without a second write. A new step with the same image is still a new observation/write opportunity. Concurrent sessions on one model are unsupported unless explicitly isolated/tested; sequentially interleaved sessions must have disjoint caches.
6. Training execution and full BPTT
Training uses the existing whole-episode B=1 sequence with use_cache=False. At each selected layer the lane scans this entire causal sequence from zero. FLA chunk-parallel execution is an implementation of that scan; it is not TBPTT. No detach occurs between navigation steps.
The lane outputs are added to the native mixer before residual/MLP. Standard layer checkpointing recomputes the same pure computation during backward. Pass immutable roles as explicit kwargs captured by checkpointing. Pass no serving lane cache in a gradient-enabled forward. No global 'current step', callback-updated mask, hidden module state or writer counter may affect a recomputation.
A loss at step t may train the lane write at step t-1 and earlier. It cannot use the write after its own decision. The last STOP writer has no later navigation loss; it need not have a nonzero gradient for that specific write. Shared writer parameters are trained from earlier writes.
A one-step STOP-only episode can have zero lane gradients. All trainable lane parameters must still participate with finite zero gradients under the current ddp_find_unused_parameters=False; avoid training-time shortcuts that skip the branch entirely when R or W_out is zero.
7. Native kernel reuse and caching API
Production runner uses the installed, approved FLA chunk_gated_delta_rule and fused_recurrent_gated_delta_rule. The lane is small but uses the same argument conventions as native Qwen. B1 unpadded episodes only; no new packing/cu_seqlens scheme.
- Offline: initial_state=None, output_final_state=False, chunk kernel.
- Online: initial state read from the sidecar, FP32; request final state. Single-token append uses the recurrent kernel; multi-token append uses chunk continuation.
- Kernel calls must not corrupt the provided initial state. Verify with cloning checks; if a backend mutates, isolate it with an input clone inside the adapter and measure the cost.
- Session store receives only detached FP32 finite final state and advances its per-layer logical-token cursor once per append.
- No state mutation is permitted when grad is enabled or when training is active. Eval-mode teacher-forced full sequence remains uncached.
No custom Triton kernel is required for v1. Do not assume zero-g/beta or four-head layouts work numerically without testing installed kernels. Unsupported/fallback kernels fail closed in production.
8. Parent and training protocol
8.1 Do not confuse code branch with checkpoint identity
Use the latest inspected branch to implement both text and candidate modes. It does not require changing a trained text checkpoint into candidate mode. The report lists a validated R2R Window8 text checkpoint; joint candidate in the report is FullContext, not a validated bounded Window8 checkpoint [S7]. Use a completed matching Window8 parent, or explicitly run a FullContext study with its different claim.
The pilot has separate initialization and resume:
- weights-only adaptation: exact trained native parameters + freshly initialized lane + fresh optimizer/schedule;
- resume: strict same-model restoration of weights, optimizer, scheduler, RNG, dataset order and lane spec.
Never load a wrapper checkpoint as if it were a generic Hugging Face base model. Never recreate a trained copied-linear classifier from LM rows during warm-start.
8.2 Proposed additional-training recipe
One complete R2R data pass, global8 episodes/update (4 ranks * B1 * GAS2), existing state-mean navigation reduction, full BPTT and frozen vision. With the supplied 10,819-episode campaign this is expected to be 1,353 updates under its existing tail policy; recompute from the actual sampler/manifest and reject mismatches.
- Native trainable parameters: LR5e-6, same as matched continuation.
- Lane parameters: LR1e-4.
- Decay0.01 on lane matrix weights; no decay on bias, normalization, A_log, dt_bias.
- AdamW betas(0.9,0.95), global clip1.0.
- Warmup41 updates for this declared pilot; existing cosine_with_min_lr with min_lr_rate0.1.
- No action class reweighting in the primary experiment.
- No additional output gate initialized to zero; W_out is the sole zero bottleneck.
These are proposed defaults. Run the real memory gate before a full job. Short validation screens can execute a prefix of this fixed schedule, but must not redefine it and claim identical continuation.
9. Controls and success criteria
Primary arms: N (matched native continuation), L-step (this method), and L-token-calibrated (same lane/parameters/tokens with token-rate updates and calibrated initial decay/write rates). An uncalibrated token-clock control is optional; it alone does not isolate clock effects.
For L-token-calibrated, compute one train-manifest reference token count n_ref per step group before training and save it. Use HL_token=HL_step*n_ref and beta_token=1-(1-beta_step)^(1/n_ref). All nonprefix tokens write/decay. Never use future per-group length online. Calibration aligns initial rate approximately for repeated keys; it does not make the architectures equivalent.
Report paired SR/SPL and failure types, frozen lagged visual probes, later-action sensitivity to old evidence, lane-output norms/gates, parameter/state/activation costs and same-hardware latency. An opened projection or larger half-life is not sufficient evidence of useful memory. With repeated seeds, match parent and extra exposure and disclose selection protocol.
Do not promise an SR delta. If local control degrades or gains do not survive matched continuation/capacity controls, report the negative result and reconsider. Do not automatically add pose, summaries or DSR to conceal a failing architecture hypothesis.
10. Explicit non-goals and future versions
No pose integration, geometric equivariance claim, learned summary tokens, observation skipping, classifier-free guidance, action chunking, external teacher, DSR/QSR/WRR loss, RL, new data collection, pruning, or curriculum. A later DSR extension reads existing pre-action hidden state and must have its own native+DSR control. It is not part of the v1 acceptance definition.
