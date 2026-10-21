# Episodic Memory for SimpleMemVLN — Two-Phase Implementation and Training Plan

**Goal:** Test whether native GDN committed-write signals allocate a bounded historical KV cache better than FIFO recency and random admission, using the complete existing R2R + English RxR-CE 15-degree training corpus.

**Architecture:** Keep native Qwen3.5 GDN, convolution, language layers, vision encoder, positions, and canonical text-action interface. Each full-attention layer receives the instruction prefix, a recent contiguous window, and a bounded set of older visual K/V rows. Its admission score comes only from the immediately preceding GDN layer. No new memory lane, teacher, geometry labels, action chunking, or RL is included.

**Tech stack:** Existing pinned SimpleMemVLN model environment; PyTorch, Transformers 5.11.0, FLA/FLA-core 0.5.2, verified H100 backward backend, FlashAttention 2.8.3, DeepSpeed and Accelerate. Preserve the actual lockfile/environment after inspection; do not upgrade dependencies in an active training environment.

**Spec:** Sections 1–11 below are the normative design. Sections 12–21 define implementation, resource qualification, the two training phases, and operational acceptance.

**Intended repository path:** `plans/episodic_mem.md`.

**Prepared:** 2026-10-11. Proposed branch: `c2_episodic`. Audited starting reference: `streaming_text_dual@f6b8108f6ef8c7b4e3df035688d3210dbd3678a1`.

## Global constraints

- Train on all admitted R2R and English-guide RxR-CE 15-degree training episodes in the verified joint manifest. Do not replace RxR15 annotations with original 30-degree annotations.
- Every primary arm independently starts from the same pretrained Qwen3.5-4B revision, not from a trained M2 checkpoint or a diagnostic smoke model.
- Keep full-episode BPTT, complete episodes, action labels, image resolution, text prompt, canonical action history, EOS supervision, and global action normalization unchanged.
- No VGGT, V-JEPA, DINO teacher, GT depth/pose, extra trajectories, new action representation, learned router, or auxiliary loss in Phase 1.
- Episodic candidates are historical visual tokens only. Never select future observations or current/future gold action tokens. Native GDN/conv still processes all original tokens.
- Runtime episodic state resets at episode boundaries. Keep original logical and multimodal positions after eviction.
- Production retains only bounded recent/episodic KV, not an unreported full-history backing archive.
- Scores and hard selection indices are detached. Navigation gradients flow through selected K/V and the complete recurrent graph, not through top-k decisions.
- Every change to visibility, score, cap, backend, training initialization, or numerical qualification is versioned before a new research run.
- Do not cancel or mutate pre-existing jobs, allocations, checkpoints, environments, or source snapshots. Only this campaign's recorded redundant submissions may be cancelled under Section 17.

## Review focus

1. A source token is eligible exactly when its group leaves the recent window; E is fixed throughout a current observation/action group.
2. Correct committed residuals, head aggregation, normalized keys, prefix/history state, numerical segmentation, and no double multiplication by beta.
3. Checkpoint recomputation receives the same immutable selected indices and preserves late-loss gradients to old selected K/V and early GDN writes.
4. GPU admission, pinned activation offload, CPU optimizer state, prefetched complete episodes, cgroup limits, and checkpoint disk/transient host peaks are qualified together.
5. Queue racing selects one qualified runtime profile and one writer per logical run; rank-count changes are never treated as exact optimizer resume.

---

# 1. Scope, evidence, and the branch decision

## 1.1 What was actually checked

Live GitHub branch listing on the preparation date contained:

| Branch                    | Tip inspected/listed                           | Relevance                                                                                                   |
| ------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `main`                    | `53ac625611b5778df8a4eaeecbdeda028bfa0309`     | Independent explicit-history training; not the preferred base                                               |
| `streaming`               | `36a2c9e58ad1f5e75d672f4ed82c4010de4903b8`     | Original native streaming/text baseline                                                                     |
| `streaming_logits`        | `1058d090f58708a9f20d9ad40c7a561ff630a186`     | Candidate-policy controls; not the desired action interface                                                 |
| `streaming_logits_dual`   | `499f7d81348d0b7eb9a558dcbfae7991d7dba011`     | Lane infrastructure and runtime qualifications                                                              |
| **`streaming_text_dual`** | **`f6b8108f6ef8c7b4e3df035688d3210dbd3678a1`** | **Recommended engineering base: canonical text, bounded attention batching, text replay, resource reports** |
| `RL_baseline`             | `96eedf1a45fb3396bc94351d40347faafd9599b0`     | Not required for this IL experiment; no recommendation to merge RL code                                     |

This is a branch inventory plus focused reads of the relevant paths, not a claim that every file on every branch was executed. The text-dual and streaming branches have diverged; do not describe the former as containing every later streaming commit. Compare any missing fixes explicitly.

Create an isolated worktree from the pinned text-dual commit. Keep lane code available for legacy loading/tests, but disable lanes in every arm of this plan. Do not use the `vln_dual_window8_text_joint.yaml` file unchanged: it enables lanes and is a reference for inherited settings, not the experiment config.

The remote repository currently has no `c2_episodic` branch in the inspected branch list. Existing local work may have unpushed improvements. Inspect and preserve it; do not overwrite it. Record any deliberate replacement of this starting reference in a decision log and re-run the corresponding gates.

## 1.2 Results that motivate, but do not prove, the idea

The inspected consolidated report, updated 2026-10-08, records:

| Checkpoint                        | Full R2R SR |    SPL | Role here                               |
| --------------------------------- | ----------: | -----: | --------------------------------------- |
| Native joint Window8 text E1      |      49.32% | 45.56% | Historical quality reference            |
| Native joint Window8 text E2      |      52.20% | 48.19% | Frozen diagnostic/serving reference     |
| Window8 Text DualLane FromBase E1 |      46.55% | 41.73% | Negative result for that trained recipe |

The report explicitly treats these as checkpoint comparisons. Different execution and training trajectories prevent attributing all differences to a single module. There is no episodic-memory result yet.

The hypothesis is not “GDN is broken” or “additional lanes can never work.” It is: some old evidence may be better retained as uncompressed native K/V entries and read by pretrained attention than forced entirely through recurrent compression.

## 1.3 Changes relative to the supplied Group-2 proposal

Retained: V0/M1/M2/M2c, local GDN-to-next-attention scoring, write-time fixed priorities, full BPTT, native full-attention read path, B=300 first experiment.

Clarified/changed:

1. Phase 1 contains four primary configurations; geometry Group 1 and combined `F` are not included.
2. `Parent=M2` means configuration inheritance, never additional checkpoint training from M2.
3. Primary score is the mean head-wise norm of committed delta, specified below.
4. Main variants are uncapped. Early-frame concentration is diagnosed, not silently fixed with a per-source cap that would conflict with whole-frame M3. A separately named diversity variant is allowed only as a documented additional experiment.
5. M3 uses a separate group score per attention layer from its immediately preceding GDN, not an average across all eight future scoring layers.
6. Add M5c, the random control for R4+B1200.
7. M1 and M5 are near-budget comparisons until actual total KV and latency are measured. Complete groups contain more than 300 visual tokens.
8. Learned attention-popularity/utility routing and multi-budget training are deferred, not mandatory Phase-2 work.
9. First-token agreement or 98% set overlap is not sufficient implementation correctness.
10. Full-corpus training is fixed to two epochs from pretrained Qwen for the primary campaign. A warm-start pilot, if separately approved/documented, is a different campaign and cannot share these comparisons.

# 2. Scientific question and terminology

The model has two existing routes to history:

- **Recurrent path:** GDN and its convolution states, updated by the complete token stream.
- **Exact attention path:** retained native full-attention K/V entries.

“Exact” means that selected K/V vectors are kept rather than further compressed. It does not mean the cache contains pixel-perfect images or raw factual labels; these are contextualized, layer-specific representations.

Our questions are:

1. Does a small amount of old exact context improve closed-loop navigation?
2. Does committed-write admission outperform equally sized random old context and additional recent context?
3. Can a recent/episodic budget redistribution improve quality at a similar retained-KV budget?
4. Does patch-level selection outperform whole-view retention?

HOLA motivates committed-write selection and bounded exact memory, but its trainable within-GDN read branch and language-model results are not this implementation. DeltaS motivates intrinsic-state-guided KV retention, but is a training-free streaming-video method. This plan uses existing Qwen full-attention layers and trains the navigation policy under the new visibility rule.

Expected benefit is empirical. A novel frame is not necessarily important to the action; a large write is not proof that its value was poorly retained after the write. Source rank limits also do not imply a hard count of landmarks remembered.

# 3. Model, corpus, and primary training invariants

## 3.1 Backbone and actions

- Pretrained model: `Qwen/Qwen3.5-4B`.
- Base revision in current reports: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Output: `qwen_text`, canonical `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`.
- Preserve the actual action-order/Habitat-ID definitions in `contracts.py`; do not infer them from text sorting.
- Native vocabulary greedy decoding; existing maximum of 16 response tokens including EOS.
- Training teacher-forces prior expert response tokens. Deployment commits its own generated response exactly once.
- Frozen visual encoder **and merger**; train the language backbone and existing tied LM head.
- RGB remains 640×480 with the audited processor expansion `[1,30,40]` and 300 merged visual tokens/image. No repeated re-encoding of history is introduced.
- Native GDN matrix state remains FP32 in serving. BF16 model arithmetic and FP32 rotary preservation retain existing contracts.

## 3.2 Full training corpus

Use the complete existing joint manifest, expected from current reports:

- R2R: 10,819 training episodes.
- English-guide RxR-CE **15-degree**: 19,996 training episodes.
- Total: 30,815 episodes and 3,128,624 action targets per unique pass.
- Historical manifest SHA-256: `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.

These numbers are audit references. Recompute counts, action mapping, split/scene identifiers, file availability and encoded lengths before submission. Fail on unexplained mismatch; do not drop examples to recover the expected number.

“Full RxR” in this plan means all English-guide episodes of the existing 15-degree training corpus, not all languages/annotation roles of the original RxR release.

Preserve any unresolved collector-timing qualifications from earlier reports; filenames alone do not prove observation-before-action capture. Document the existing data provenance without pretending this campaign reconstructed the collector.

No action relabeling, crop/truncation, length cap below the actual corpus maximum, difficulty filtering, added oracle STOP, or TBPTT. Preserve all original labels and loss contributions. Repeated tails imposed by distributed loading must be logged.

## 3.3 Complete-episode objective

For each action, average CE over that action's canonical response targets, including the existing EOS target and excluding fixed unsupervised separators. Then sum action losses.

For a distributed accumulation window with total valid action count N:

    rank_loss = world_size * local_action_loss_sum / N

Preserve the existing DDP/DeepSpeed convention that avoids a second GAS division. Add no class weights or auxiliary loss in V0/M1/M2/M2c/M3/M5/M5c.

Selection is discrete and detached. The model is trained end-to-end conditional on the selected indices; top-k itself is not learned from a differentiable navigation gradient.

## 3.4 Primary initialization and schedule

All primary configurations independently load the same pretrained Qwen revision. They do **not** load the native E2 checkpoint and do **not** load a previously trained M2. The existing E2 is for Phase-1 preparation and exploratory zero-shot tests only.

Recommended inherited scientific recipe:

| Field                        | Value                                                  |
| ---------------------------- | ------------------------------------------------------ |
| Training seed                | 429 for primary arms                                   |
| Epochs                       | 1 over the full joint corpus                           |
| Nominal global episode batch | 8                                                      |
| Expected updates             | 7,704; recompute against actual sampler                |
| Warmup updates               | 232                                                    |
| Backbone learning rate       | 5.0e-06                                                |
| Optimizer                    | Existing AdamW, betas (0.9, 0.95)                      |
| Weight decay                 | 0.01 with inherited norm/bias exclusions               |
| Gradient clip                | 1.0                                                    |
| LR schedule                  | Existing cosine with minimum-LR ratio 0.1              |
| Checkpointing                | Non-reentrant; full BPTT                               |
| Recovery saves               | Every 250 updates, two complete rolling recovery saves |
| Research exports             | Epoch 1 and epoch 2                                    |

Do not shorten the actual LR schedule for screening and later stretch it. E1 is a mid-schedule snapshot of the same two-epoch trajectory. Primary reporting uses E2; E1 is diagnostic. A predeclared common checkpoint-selection rule may be used instead, but never pick different rules after seeing results.

# 4. Visibility and temporal indexing

## 4.1 Groups and budget

Let t be the zero-based environment/observation group index. A complete group contains the observation framing, expanded image tokens, generated/teacher-forced action text and closure.

- R: maximum recent groups, **including current group**.
- B: maximum old visual K/V rows per full-attention layer.
- P: instruction-prefix token count.
- G_t: logical token IDs of complete group t.
- V_t: its 300 visual token IDs.
- E_t^ell: old episodic token IDs visible in attention layer ell during group t.

At the beginning of step t, promote the group j=t-R if it exists, then keep at most B entries. Therefore every episodic candidate satisfies j<=t-R. No row can be both episodic and recent.

For any query position u in current group t:

    visible = prefix
              + E_t^ell
              + all tokens of groups max(0,t-R+1)..t-1
              + current-group tokens up to u

At t=20:

- V0: groups 13..20.
- M1: groups 12..20.
- M2: groups 13..20 plus <=300 selected visual rows from groups 0..12.
- M5: groups 17..20 plus <=1200 selected visual rows from groups 0..16.

Prefix remains visible. Native GDN/conv processes every original token and is never evicted by this policy.

## 4.2 Physical cache order versus logical positions

Store per full-attention layer as:

    [prefix | episodic rows sorted by original token ID | contiguous recent rows]

K has already received the appropriate native positional transform. Do not recompute RoPE at a new compressed index; do not reset the PositionLedger after eviction.

For causal FlashAttention, all selected historical rows precede the current query block in the **packed key layout**. The ordered current group is its final key suffix. Consequently bottom-right causal alignment exposes exactly the current causal prefix, despite omitted intermediate history.

Do not reorder the current suffix, duplicate prefix/current rows, or insert episodic rows after current queries. Permuting fully visible historical K/V together can be mathematically valid, but is not a substitute for correct masks, suffix order or deterministic packing.

## 4.3 Episodic set is frozen during a group

The same E_t is used for observation prefill, ordinary generated action tokens, EOS and the separator. A source's visual score is fixed when its image tokens are processed. Admission happens only after that source leaves the recent window, never midway through the same response.

At a repeated `observe(episode,t,rgb)` request, return the saved result without changing caches, scores, logical positions or selection. A partial append failure invalidates the session. Episode reset clears every cache and score record.

# 5. Committed-write score: normative definition and implementation

## 5.1 Which layers provide the score?

For full-attention layers ell in `{3,7,11,15,19,23,27,31}`, use native GDN layer ell-1.

Validate this against `text_config.layer_types`. No scoring from layers above the consumer, no all-layer aggregation for lower attention, and no new learned projection.

Full-episode layer-major execution is causal because scores at layer ell-1 depend only on earlier layers and earlier tokens. They can be computed for the full sequence before evaluating layer ell. The selector must nevertheless use only eligible source groups for each current group, not a global top-B over the full episode.

## 5.2 Mathematical score

Use native normalized keys and actual native gate transformations. For a head h and token i, with key-by-value S:

    S_decay = exp(g_i) * S_previous
    residual_i = v_i - transpose(S_decay) @ k_i
    delta_i = beta_i * residual_i
    S_next = S_decay + outer(k_i, delta_i)

The scalar priority for full-attention layer ell is:

    score_i^ell = mean_h( ||delta_i,h||_2 )

The norm and head reduction are in FP32. This is **mean head-wise norm**, not norm of a head-averaged vector, max-head norm, beta alone, raw residual norm, or state drift. There is no value-norm normalization in the main arm.

Prefix and text tokens still update S and influence later residuals, but only visual token scores become admission candidates. Never compute image scores from a state that skipped prefix or previous feedback.

If an exposed FLA intermediate already equals `delta = beta * residual`, do **not** multiply by beta again. Call the score committed-write magnitude; it is not ground-truth importance, a calibrated forgetting probability, or a learned utility score.

Scores are recomputed under the current weights on each training episode. Do not precompute them once from the frozen parent for production training and then claim a live native selector. Frozen-parent scores are allowed for diagnostics only.

## 5.3 FLA integration is a release blocker, not a trivial hook

The pinned FLA chunk implementation computes internal `v_new`, but the public operation does not return it as an extra public result. Existing code cannot be changed to unpack a third output without an actual implementation.

Preferred Phase-1 correctness route:

1. Leave the approved differentiable native mixer call and H100 backward backend intact.
2. Prepare Q/K/V/gates once in a project-owned, source-pinned adapter for selected GDN layers.
3. Compute detached scalar scores in a bounded, score-only block scan from those prepared tensors. The helper carries a numerical state through prefix, observations and feedback; it is not the policy state and is discarded after its scores are produced.
4. Reuse verified FLA forward primitives or implement a specialized scalar-score emitter. Never materialize the entire episode's per-token matrices, and never retain all dense `v_new` arrays in the autograd graph.
5. Profile the side-pass overhead. A later fused emitter may eliminate duplicate forward work, but must pass the same scalar-score and action-path tests and becomes a new backend version.

Independent block reference for the helper (not the production Python token loop): for a block of tokens with cumulative log decay c_i, block-entry state S0, normalized keys and post-activation beta:

    L_ij = beta_i * exp(c_i-c_j) * dot(k_i,k_j), for j<i; otherwise 0
    rhs_i = beta_i * v_i - beta_i * exp(c_i) * transpose(S0) @ k_i
    Delta = solve_triangular(I + L, rhs, lower=True)
    S_end = exp(c_last)*S0 + sum_i exp(c_last-c_i)*outer(k_i, Delta_i)

Evaluate decay ratios only on the permitted lower triangle: mask exponent arguments before exponentiation, never compute an enormous forbidden upper-triangle exponential and multiply it by zero afterward. Do not divide by underflowed cumulative decays. Each Delta_i is the committed delta in exact arithmetic. FP64 tests compare this block formulation against an independent sequential oracle with nonzero initial state, variable decay and multiple heads. Actual dtype/backend comparisons are a separate gate.

Pin score segmentation to the declared canonical serving segmentation: prefix, one observation block, each ordinary response token, then EOS+separator. For score-only processing, split these segments into bounded internal blocks without resetting the carried state. Do not assume HF/FA/FLA chunk indices mean separate episodes. If an optimized implementation uses a different segmentation, record it and qualify selection stability; do not silently change the science label.

If only beta or a cheap proxy is implementable initially, call that run `M2_beta_proxy`, not M2. It does not satisfy the committed-write arm. Never disable the existing Hopper backend guard to expose an intermediate.

## 5.4 Precision and top-k sensitivity

A reference oracle pass does not imply bitwise GPU/offline/streaming equality. Record scalar errors, rank margins at B/B+1 and exact selected IDs on real fixtures. Percent overlap alone is not an output-equivalence guarantee.

The deterministic selector must choose exactly the same IDs given identical scores. For backend scores that differ numerically, investigate disagreements near the threshold. Do not silently relax tolerance, replace the score, or turn 98% overlap into a correctness certificate.

A quantized/rank-bucket score is an allowed separately versioned stabilization experiment only after diagnostics. It is not part of the initial main method.

# 6. Selector policies and bounded storage

## 6.1 Token admission

Maintain per layer:

- Up to B selected old logical token IDs, their source-group IDs and fixed priorities.
- Scores for the currently recent visual rows, at most R groups.
- K/V tensors in the native full-attention cache, not a second full-history tensor.

Before a new group t, merge only the previous E with the newly expired group's visual rows. Keep top B with deterministic ordering. The running top-B equals batch top-B over all eligible source rows for fixed priorities and no additional changing constraints; reference-test that equivalence.

Tie-break: priority descending, original logical token ID ascending. Serve the selected set sorted by original token ID for deterministic key packing. Do not require the sets to be identical across layers; their counts and candidate populations must agree.

When fewer than B candidates exist, keep the available count. Do not pad attention with fake old tokens or zeros that change the softmax denominator.

## 6.2 Random control

Priority is derived from a deterministic SHA-256 key containing selection seed, episode UID, attention layer ID and original token ID. Use a defined 64-bit integer prefix, not Python's process-randomized `hash()` and not sequential RNG calls that depend on packing.

This specifies a per-layer random priority. It is a clarification relative to the original abbreviated hash example. Keep the same random-key policy across training and inference. Episode UID is used only by the randomized control, never as an input to the policy network.

## 6.3 Early-frame concentration

The main runs use `per_source_cap: null`. This keeps the first experiment interpretable and allows a valid token-vs-whole-frame comparison later.

During preparation, log on a fixed panel: source age histogram, turnover, fraction from steps 0..2, number of represented groups, concentration by layer and action type. A cap of 100 tokens/group cannot prevent steps 0..2 collectively filling a 300-row bank. It also conflicts with keeping a whole 300-token keyframe.

If the bank pins early frames, document it. A fixed-first-frame control or a capped/diversity variant may be added as one of the explicitly budgeted diagnostic runs. Do not silently activate it midway through V0/M1/M2/M2c. Applying a cap changes the effective method and requires matched token controls; M3 must not be mislabelled as a one-factor comparison if its cap differs.

## 6.4 Whole-frame selection for Phase 2

For M3, compute a group score separately at each scoring layer:

    group_score_j^ell = mean_{i in V_j}(score_i^ell)

Keep one complete historical visual group per attention layer for B=300. Do not average scores from layers 2..30 to create a shared keyframe for lower attention in one-pass layer-major training.

M3 retains 300 visual rows, not old action/framing tokens. It has the same old-visual budget as M2 and the same no-cap policy, but different granularity. Temporal scores are still causal and fixed when observed.

# 7. Offline training implementation and full BPTT

## 7.1 Pure score outputs and immutable plans

Do not place scores or selections in `module.latest_selection`, a global hook collector, or a mutable dictionary reused across episode forwards.

For each scoring GDN layer, provide a pure project wrapper:

    scored_decoder_forward(layer, hidden, positions, masks, score_spec, segmentation)
        -> (updated_hidden, detached_scalar_scores)

It preserves the native norm/mixer/residual/MLP equations and native parameter namespace. Return only scalar scores in addition to hidden state. Non-reentrant checkpointing captures this invocation; the returned scores from the first forward are used outside the checkpoint to build a plan.

Then call the consuming attention layer with an explicit immutable `LayerSelectionPlan`, including the original-token index list for every group. Backward checkpoint recomputation of that attention invocation receives the captured plan. Recomputed scorer outputs must not replace a plan used by the original forward.

This requires a project research language forward that retains native position/mask handling and all normal layers. Native-disabled equivalence is a release gate. Reusing native module parameters is mandatory; do not create a second set of language weights.

## 7.2 Attention packing

For group t with current query span [start,end), build the K/V source index vector:

    [0..prefix_end)
    + sorted(E_t)
    + [recent_first_token..end)

Gather Q from the current span and K/V from that list. Pack a bounded number of independent windows into FA2 varlen; never pack all duplicate windows for a long episode at once.

- Prefix queries form their own causal sequence.
- Current query block is the key suffix.
- GQA head mapping is native, not manually replicated inconsistently.
- Original token IDs select already position-transformed K.
- The plan and actual per-window lengths are immutable, not reconstructed from mutable serving state.

Start with `window_attention_batch_steps=8` as a conservative proposed scratch-space setting. Qualify 4/8/16 as engineering choices on real long sequences and select one before primary runs. All primary arms on one runtime profile use the same value. The historical batch16 result is not an episodic-memory qualification.

## 7.3 Backward

Save original Q/K/V and compact integer plans. Recompute only the current bounded packed batch during backward. Obtain local dQ/dK/dV and scatter-add into full-sequence gradient buffers by original token IDs.

An old selected K/V row may appear in many future windows. Contributions must sum, not overwrite. Prefix gradients also sum. Every query row appears exactly once in its own prefix/group sequence.

No detach of original historical K/V. Local detached packed tensors inside a custom VJP are allowed only if the computed gradients are explicitly returned to the original inputs. Keep first-order derivative limitation declared if inheriting `once_differentiable`.

Mathematical tests must show a loss after the recent window can backpropagate through an old episodic row, and through native GDN temporal paths. This is not TBPTT.

## 7.4 Loss, scoring, and optimizer

No auxiliary score loss. Detached rankings mean the selector rule is fixed; weights change the scored representations indirectly through the policy training path. Do not call it a differentiably learned eviction policy.

Reuse the existing global action normalization and optimizer. Assert every trainable parameter appears exactly once. M2 introduces no new learned parameter by default. M1/M2c need not compute unused write scores in headline performance runs; their actual cheaper selector overhead must be reported, not hidden by charging artificial work.

# 8. Serving state machine

## 8.1 Session-owned metadata

Introduce `EpisodicSessionState` owned by one StreamSession. It contains per-layer selected IDs/scores, recent source scores, group spans, original positions and counts. It must not contain full-history K/V backups.

Native GDN/conv and full-attention cache remain session-owned. `kv_length()` may continue to check equal resident row counts across full-attention layers, but equality of counts is not a correctness proof. Assert prefix integrity, no duplicates, disjoint recent/episodic IDs and exact current suffix.

## 8.2 Beginning a step

Before processing observation t:

1. Validate episode ID, expected next step and retry status.
2. Freeze the previously committed cache as the source of promotion.
3. Identify old group t-R, if present.
4. For each attention layer, merge its episodic candidates with that group's visual rows and fixed scores.
5. Rebuild cache `[prefix|E_t|recent survivors]`, preserving original K/V values and positions.
6. Discard unselected expired rows and old text rows.
7. Record selection digest and make E_t immutable for this group.

There is no need to recompute old frames or reconstruct an already discarded row. Running top-B only admits candidates at the time their original resident rows are still available.

## 8.3 Observation and response

Observation append computes source scores for its image tokens at the eight designated GDN layers. These scores are retained with recent metadata for future promotion, not used to rewrite E_t.

Response generation keeps original greedy vocabulary/parse behavior. All response tokens update native state exactly as before; they never enter the old episodic candidate set. Final EOS/separator closes the group; store the actual complete group length and step identity.

Exception at any intermediate point invalidates the session. Do not partly roll back only KV while leaving GDN/conv ahead. Reset recreates all state. Single-session retries are idempotent. Concurrent multi-thread model serving remains outside scope unless explicitly qualified.

## 8.4 Cache bounds

Structural bound:

    retained_KV <= actual_prefix + B + sum(lengths of <=R resident groups)

A conservative static guard can be derived from declared caps:

    max_prefix_tokens + R * max_step_group_tokens + B

Use an integer guard derived from the mode; do not keep a hardcoded 5120 if R/B change. Separately retain the logical-token cap and full-episode training admission.

For the native attention head layout reported (8 full-attention layers, 4 KV heads, width256, BF16), B300 adds approximately 9.375 MiB of K/V tensors; B1200 adds 37.5 MiB. Scores, positions, temporary compaction buffers, weights and training activations are extra. These are shape calculations, not measured allocator peaks.

# 9. Phase 1 — four independent primary runs

Preparation and smoke gates are part of Phase 1, not a separate research phase or counted training ablation.

| Run     |   R |   B | Policy                            | Initialization            | Scientific question                                    |
| ------- | --: | --: | --------------------------------- | ------------------------- | ------------------------------------------------------ |
| **V0**  |   8 |   0 | Recent FIFO                       | Same pretrained Qwen base | Fresh matched native baseline                          |
| **M1**  |   9 |   0 | Recent FIFO                       | Same base                 | Does comparable extra recent context explain the gain? |
| **M2**  |   8 | 300 | Layer-local committed-write top-B | Same base                 | Does selected old exact context help?                  |
| **M2c** |   8 | 300 | Per-layer fixed random priority   | Same base                 | Is the score better than generic old-context access?   |

All use the full joint corpus, one epoch, seed429 and the same chosen execution profile. V0 should use the generalized code path with B0 plus a tested native fallback; it is not simply the previously published 52.20 result reused as a matched baseline.

## 9.1 V0 — establish reproducible control

- New code, no episodic bank and no lane.
- Validate old native text E2 loading and serving independently as a regression fixture.
- Train V0 from pretrained Qwen, not from that E2 reference.
- Preserve all original data and objective choices.
- Record deviation from the historical 52.20 result without assuming every difference is a bug; training execution differs and repeated seeds have uncertainty.

## 9.2 M1 — recent context control

Increase R to9, keep B0. No score helper or old episodic list is needed.

It retains one extra **complete** observation/feedback group, whereas M2 adds visual-only entries. Treat it as near-budget recency control. Report total K/V rows and measured runtime, not only 9×300.

## 9.3 M2 — proposed main candidate

Keep R8, add B300. A token may remain far beyond eight groups if its fixed priority remains high. Use local scoring GDN below each attention layer, current model weights and all native history updates.

No recurrent snapshots are stored in the episodic cache; the selected objects are the attention layer's native K/V, not the preceding GDN's k/v. These two spaces are different. Score a token at GDN ell-1, then keep that same original token's **full-attention K/V at ell**.

The score is an admission heuristic. The pretrained full-attention layer supplies the read mechanism. Do not add HOLA's RMSNorm/cache gate by default; this is not an internal GDN softmax path.

## 9.4 M2c — required random control

Everything that determines K/V budget, candidate eligibility, text/history semantics and training exposure matches M2. Only the fixed priority changes to deterministic random ranking.

Keep this control even if M2 looks promising. Without it, extra old context and the write-score mechanism cannot be distinguished.

## 9.5 Phase-1 evaluation and transition criteria

Qualification checks happen before any full run. Evaluate E1 and E2 at matched progress; do not kill a finite, correctly running arm solely on early training CE.

Use a predeclared scene-stratified val_seen panel for development. Primary Phase-1 comparisons use the full 1,839 R2R val_unseen IDs at E2; also run a fixed 1,000-episode English RxR15 val_unseen development panel (if the verified split contains >=1000), same IDs for all arms. This is a subset result, never label it full RxR. Finalists receive the full split.

Proposed operational gate, frozen before primary results (thresholds are planning choices, not significance claims):

**A — selector-positive:** M2 gains >=1.0 SR percentage point over V0 on full R2R, is >=0.5 point above both M1 and M2c, and SPL is not >0.5 point below V0; OR it gains >=2.0 SR points on the predeclared RxR panel, is >=1.0 point above both controls there, and R2R SR/SPL are each no worse than V0 by >0.5 point. Require no new protocol, failure-accounting or numerical correctness violation and actual bounded-cache/resource evidence.

**B — generic episodic signal:** M2 and/or M2c improve over V0 under the same quality guard, but no robust selector advantage is visible. Phase 2 may run M5/M5c to investigate allocation; do not claim committed write is the winning mechanism. Defer score-specific refinements until the control gap is explained.

**C — inconclusive:** point differences are small, confidence intervals broad, or gains depend only on a tiny selected subgroup. Prefer one diagnostic/seed follow-up over opening all variants. Do not redefine the gate after seeing results.

**D — negative/invalid:** no useful quality/cost signal, or correctness/resource qualification fails. Stop expansion. Diagnose first; do not add unrelated modules to rescue the story.

Use paired episode outcomes and scene-cluster bootstrap intervals. An exploratory gate passing with an interval containing zero authorizes confirmation, not a final scientific significance claim. Predeclare subgroup definitions such as route length from reference metadata rather than selecting successful examples afterward.

# 10. Phase 2 — conditional allocation, granularity, and confirmation

Default Phase-2 additions: M5, M5c, V0 at seed430, and one selected winner at seed430. Add corrected M3 when the score/granularity question remains relevant. Thus default total is 8 training configurations, or 9 with M3.

| Run                  |      R |      B | Selector/granularity                  | Compare with | Main question                                             |
| -------------------- | -----: | -----: | ------------------------------------- | ------------ | --------------------------------------------------------- |
| **M5**               |      4 |   1200 | Committed-write tokens                | V0, M2, M5c  | Better recent/old allocation at similar visual-KV budget? |
| **M5c**              |      4 |   1200 | Random tokens                         | M5           | Does score matter at the redistributed budget?            |
| **M3** (conditional) |      8 |    300 | One whole old visual frame, per layer | M2           | Complete spatial context versus scattered patches?        |
| **V0-s430**          |      8 |      0 | FIFO                                  | V0-s429      | Baseline training-seed repeat                             |
| **winner-s430**      | winner | winner | Exact frozen winner config            | Same seed V0 | Confirm observed effect                                   |

Do not initialize M5 or M3 from trained M2. Configuration parentage is not weight parentage. Each independently trains from the same pretrained revision for the same one-epoch schedule.

## 10.1 M5 — budget redistribution

At a full window, visual entries are:

    V0: 8*300 = 2400
    M5: 4*300 + 1200 = 2400

Total K/V differs because complete recent groups also contain action/framing tokens. Early episode bank fill also differs. Label it equal visual-KV / near-total-KV budget; log actual total occupancy and latency over time.

It trades local detailed history for broader old context. Negative results are informative: recent temporal structure may be more valuable than selected old evidence. Do not automatically increase B to mask this trade-off.

## 10.2 M5c — do not omit this control

A gain by M5 alone could come from the redistribution rather than the score. M5c has identical R/B and randomized old entries. The most valuable result is a consistent M5 improvement over both V0 and M5c at acceptable measured inference cost.

## 10.3 Corrected M3

Use local per-layer group scores, one retained historical view, all 300 of its visual K/V rows. No all-layer aggregation and no cap100. A single source group may differ across full-attention layers.

If a separate capped token variant was adopted, M3 is not automatically a one-factor comparison with that variant. Restore an uncapped token control or report the joint change explicitly.

## 10.4 Optional, bounded autonomy

Within an explicit budget declared in the campaign manifest, the coding agent may add at most **two** additional full-training configurations beyond the nine-run default. Each requires a written hypothesis, one changed factor, its matched control, estimated GPU/node-hours, release tests and a unique experiment ID before submission. Examples: a pinned-first control after clear cache concentration, or an exact-token recency control when M1's framing difference matters.

M4 (R8+B1200), M3b, M6a, M6b, M7, teacher losses, RL and combined geometry model F are not automatic work. A proposed extra that changes the question or exceeds the budget requires user approval. No guessed conference deadline overrides safety or the locked evaluation protocol.

# 11. Diagnostics that explain outcomes

Collect bounded, opt-in records; no dense attention matrices or all-step state dumps in production.

Per attention layer and selected sampled steps:

- Candidate count and admitted/removed original IDs.
- Episodic occupancy and age histogram; number of distinct source groups.
- Fraction from steps 0..2 and first-frame fraction.
- Entry turnover and how long slots remain occupied.
- Score quantiles; beta, residual and committed-score distributions in diagnostic mode.
- Cutoff margin and offline/online ranking disagreements.
- Attention mass on episodic keys at decision rows; selected visual rows may also be examined in a bounded probe.
- Model-generated action and forced-cap/invalid/failure classification.

Low decision-row mass alone does not prove nonuse; past evidence can act through image tokens, lower layers and native recurrent states. Use episodic-output/visibility disabled **from episode reset** as a labelled intervention on a trained checkpoint, not as a separately trained baseline. An intervention after many steps retains earlier episodic influence in native state.

Large committed scores can reflect viewpoint changes or writes already absorbed correctly. Report score correlation with action type/view change and compare random/recency, not just appealing heatmaps.

No automatic STOP intervention, oracle goal input, or loop-breaking heuristic is added to the policy by this plan.

# 12. File architecture and implementation interfaces

All entries below marked NEW are proposed code. Do not run commands expecting these files to exist before implementation.

| File                                                | Responsibility                                                               |
| --------------------------------------------------- | ---------------------------------------------------------------------------- |
| `src/qwen_vl/research/episodic_contract.py` **NEW** | Immutable spec, validation, score/backend IDs, checkpoint metadata           |
| `src/qwen_vl/data/episodic_metadata.py` **NEW**     | Original-token image indices, group spans, canonical append segmentation     |
| `src/qwen_vl/models/committed_write.py` **NEW**     | Pinned prepared-tensor score helper, independent reference is test-only      |
| `src/qwen_vl/models/episodic_forward.py` **NEW**    | Pure scored GDN decoder and project language forward, same native parameters |
| `src/qwen_vl/stream/episodic.py` **NEW**            | Shared deterministic selector and immutable per-layer plans                  |
| `src/qwen_vl/stream/episodic_cache.py` **NEW**      | Session metadata, KV promotion/repacking, invariants and reset               |
| `src/qwen_vl/stream/window_attention.py`            | Parameterized R/B, gathered windows, VJP scatter, native B0 path             |
| `src/qwen_vl/stream/session.py`                     | Step-boundary promotion, source-score commit and retry/failure semantics     |
| `src/qwen_vl/stream/cache.py`                       | Episodic-aware guards/count checking without changing GDN precision          |
| `src/qwen_vl/models/nav_model.py`                   | Metadata routing; same native vision and action loss                         |
| `src/qwen_vl/data/episode_serializer.py`            | Additional metadata only; no changed tokens/targets                          |
| `src/qwen_vl/contracts.py`, `train/vln_runtime.py`  | New memory-mode validation and strict loader compatibility                   |
| `src/qwen_vl/train/train_episode.py`                | Resource-safe staging/prefetch configuration, same objective                 |
| `scripts/vln/check_episodic.py` **NEW**             | CPU/GPU/real-model qualification orchestrator                                |
| `scripts/vln/profile_episodic.py` **NEW**           | Real complete-episode and checkpoint resource gates                          |
| `scripts/vln/diagnose_episodic.py` **NEW**          | Bounded score/cache/intervention diagnostics                                 |
| `scripts/vln/submit_episodic.py` **NEW**            | Manifest, job ownership, queue candidates and exclusive-run lease            |
| `scripts/vln/run_episodic_gate.py` **NEW**          | Neutral owned-command supervisor, cgroup/disk/all-rank checks                |
| `configs/episodic/*.yaml` **NEW**                   | Scientific configs separated from 2/4-GPU runtime profiles                   |
| `reports/episodic_idea/` **NEW**                    | All decisions, implementation/testing evidence, run/results records          |

Suggested interfaces, whose exact names must stay consistent across tasks:

```python
@dataclass(frozen=True)
class EpisodicSpec:
    version: str
    recent_groups: int
    budget_tokens: int
    granularity: str       # "token" or "visual_group"
    score: str             # "none" (B0), "committed_mean_l2_v1", "random_hash_v1"
    scoring_pairs: tuple[tuple[int, int], ...]  # (GDN, attention)
    selector_seed: int
    per_source_cap: int | None
    score_backend: str
    score_segmentation: str

@dataclass(frozen=True)
class LayerSelectionPlan:
    layer_idx: int
    group_offsets: torch.Tensor   # packed episodic-index offsets, CPU int64
    token_ids: torch.Tensor       # packed original episodic token IDs, CPU int64
    source_digest: str

def compute_committed_scores(prepared_k, prepared_v, log_decay, beta,
                             initial_state, append_segments, backend_spec):
    # returns FP32 detached scalar scores; state layout/version is explicit
    ...

def build_layer_plan(spec, episode_uid, full_attention_layer,
                     prefix_length, group_spans, visual_ids_by_group, scores):
    # uses ONLY prefix-eligible scores for each group's selection
    ...

def promote_expired_group(session_state, native_cache, step_id):
    # atomic at logical group boundary; no current-append KV involved
    ...
```

Validate types, nonfinite values, duplicate IDs, negative lengths, mismatched devices and unsupported modes. B0 must not require scores or a score helper. `visual_group` requires budget divisible by300 for this corpus. Do not silently round a token budget into a frame budget.

# 13. Implementation tasks and review gates

Each task has a test cycle: write the test, verify it fails for the intended missing behavior, implement minimal code, re-run, then commit. Use new files/names only after comparing local work.

## Task 0 — provenance and environment audit

**Read:** `reports/R2R_ALL_VERSIONS_COMPARISON_2026-10-03.md`, `reports/STREAMING_TEXT_DUAL_IMPLEMENTATION_2026-10-07.md`, `EVALUATION_SETUP.md`, original Window8/RAM incident records, and current local plans.

- [ ] Record remote branches, local unpushed commits and dirty work without overwriting anything.
- [ ] Create an isolated `c2_episodic` worktree; record source SHA and file hashes.
- [ ] Discover current Slurm allocations/jobs/resources; create a protected-job manifest.
- [ ] Verify pinned model files, manifest hashes, actual training lengths and evaluation assets.
- [ ] Run existing native text unit/reload/replay checks on the actual environment.
- [ ] Commit `reports/episodic_idea/00_audit.md`; record missing access/evidence rather than assuming it exists.

**Gate:** a complete baseline/environment identity. No current training/evaluation result is fabricated from historical logs.

## Task 1 — spec, serialization, and sampler contracts

**Create:** `research/episodic_contract.py`, `data/episodic_metadata.py`; modify serializer/loader/config validation.

- [ ] Test B0/R8 serialized token IDs, image tensors, response targets and action loss positions match original serialization.
- [ ] Test one-step STOP, long instruction, variable action-token lengths, final partial group rejection and exact image indices.
- [ ] Define a new versioned memory mode (proposed `episodic_window_v1`); retain legacy `window8` behavior unchanged.
- [ ] Add original-token IDs and canonical append segments as metadata only.
- [ ] Test strict save/reload/resume rejects memory-policy mismatches and scorer/cap/backend changes.

**Gate:** no accidental change to labels, vocabulary, prefixes or geometry inputs.

## Task 2 — selector and plan builder

**Create:** `stream/episodic.py`; tests `tests/vln/test_episodic_selector.py`.

- [ ] Test incremental running top-B against prefix-only reference for every step.
- [ ] Test t=R-1/R/R+1, B0, fewer-than-B entries, duplicates, ties and random reproducibility across chunking/ranks.
- [ ] Test future scores cannot change earlier plans.
- [ ] Test per-layer selected identities may differ while counts agree.
- [ ] Test corrected per-layer keyframe granularity and no-cap policy.

**Gate:** exact selected IDs with identical score inputs.

## Task 3 — committed-score helper

**Create:** `models/committed_write.py`; tests `test_committed_write_reference.py` and `test_committed_write_cuda.py`.

- [ ] Derive/test sequential and block-WY delta identities with nonzero initial state and unequal key/value widths.
- [ ] Test beta0, beta1, near-zero key, prefix effects, varying alpha, nonvisual writes and no double-beta application.
- [ ] Identify actual FLA dispatched forward/backward backend and source hashes; preserve guards.
- [ ] Implement a bounded no-grad helper or scalar emitter; delete dense intermediates promptly.
- [ ] Verify scalar norms/head aggregation, sequence versus serving segmentation, timing and memory.

**Gate:** no silent proxy substitution; actual H100 helper correctness/resource evidence. If blocked, M2 remains blocked, not automatically downgraded to beta-only.

## Task 4 — pure layer-major scored forward

**Create:** `models/episodic_forward.py`; extend model metadata plumbing.

- [ ] Test feature-disabled full sequence equals original native model within fixed numerical budgets.
- [ ] Return score tensors from scoring-layer checkpoint calls, then build immutable plans outside them.
- [ ] Test two episodes interleaved at the Python call level and gradient accumulation cannot overwrite another invocation's plan.
- [ ] Test checkpoint on/off outputs/gradients and unchanged early outputs when future action labels change.
- [ ] Keep native GDN/conv preparation and ordinary non-scoring layers intact.

**Gate:** no global state collector in training; no future-layer dependency.

## Task 5 — episodic attention forward/backward

**Modify:** `stream/window_attention.py`; tests `test_episodic_attention.py`.

- [ ] Build FP64 dense-mask oracle with arbitrary selected past IDs, uneven groups, GQA and partial final batches.
- [ ] Implement index-gather packing and bounded-batch FA2 path.
- [ ] Implement additive VJP scatter to original Q/K/V; verify repeated old rows receive summed gradients.
- [ ] Test causal key suffix, disjoint stores, B0/R8 parity, R9 and R4+B1200.
- [ ] Profile packed-batch scratch versus 4/8/16 batch steps, including backward.

**Gate:** correctness independent of native FLA memory and actual CUDA FA2 qualification.

## Task 6 — online cache and session

**Create:** `stream/episodic_cache.py`; modify session/cache.

- [ ] Test promotion before new observation, per-layer K/V index alignment and no re-encoding old RGB.
- [ ] Preserve logical/MRoPE positions and native recurrent states through promotion.
- [ ] Keep E fixed throughout all response tokens; store source scores only once.
- [ ] Test reset, same-step retry, partial failure, prefix, first admission and repeated turnover.
- [ ] Run >=500 steps of a declared varied/cyclic real-RGB replay to verify bounded cache and no hidden full-history archive. This is not a navigation score.

**Gate:** actual bounded storage and complete action-string replay parity, not only first-token logits.

## Task 7 — training and resource-safe configuration

**Modify:** trainer/config/staging only where needed; do not change navigation objective.

- [ ] Keep all GAS batches on CPU until their own forward; test only one current episode's pixels are on GPU per rank.
- [ ] Expose and record worker count, prefetch factor, persistent workers and pinning; start conservatively with no loader workers.
- [ ] Verify global action normalization, zero extra scorer parameters and exact optimizer-group coverage.
- [ ] Test distributed tail/exposure accounting and strict same-runtime checkpoint resume.
- [ ] Separate scientific config from validated 2H100/4H100 runtime profiles.

**Gate:** complete optimizer updates and serialization/reload on both qualified profiles, not import/forward success alone.

## Task 8 — diagnostics, full evaluation and report generation

**Create:** diagnostic/check/profile/report scripts; re-use validated Habitat entrypoints.

- [ ] Score and selection audit panels do not update weights or alter primary inference policy.
- [ ] Exported metadata drives evaluator memory behavior; mismatches fail closed.
- [ ] Log exact episode IDs, learned checkpoint hashes, failures, model STOP and cap STOP separately.
- [ ] Replay the exact committed response strings for offline/online comparison, including EOS.
- [ ] Generate paired outcomes and efficiency tables from journals; no smoke/partial result labelled complete.

**Gate:** full data/contract integrity and trained-episodic-checkpoint replay, not fresh-base-only tests.

## Task 9 — safe launch coordinator

**Create:** `scripts/vln/submit_episodic.py`, owned-command supervisor, profile templates.

- [ ] Dry-run creates manifests and plans only; it does not submit/cancel jobs.
- [ ] Test atomic one-writer ownership and simultaneous 2/4-candidate startup using local subprocess fixtures.
- [ ] Test duplicate candidates cannot write the same checkpoint directory.
- [ ] Test cancellation allowlist requires exact campaign/run/job identity; unrelated jobs are rejected.
- [ ] Test stale lease handling requires scheduler/process evidence, not deleting a lock blindly.
- [ ] Test winner-health promotion and failed-initialization recovery without cross-world optimizer resume.

**Gate:** no broad user-wide cancellation commands; all scheduling actions journaled.

# 14. GPU and host-RAM qualification: mandatory before full training

## 14.1 Why two GPUs is not automatically safer

With the same global batch8 and B1/rank:

| Profile | Ranks | GAS | Local model parameters          | Important difference                                          |
| ------- | ----: | --: | ------------------------------- | ------------------------------------------------------------- |
| **H4**  |     4 |   2 | Full replicated forward weights | More ZeRO-2 optimizer/gradient partitioning                   |
| **H2**  |     2 |   4 | Full replicated forward weights | Larger local optimizer partition and more staged CPU episodes |

Increasing GAS preserves global batch, not per-rank memory. CPU optimizer offload shifts pressure to host RAM and transfer bandwidth. Activation offload also consumes pinned host memory. Lowering its threshold may reduce GPU pressure while **increasing** RAM pressure.

The repo records a two-GPU GPU-optimizer run that passed update1 and OOMed at update2 after optimizer states existed. A 240-GiB host allocation then OOM-killed a CPU-offload attempt. The later text-dual four-H100 qualification used a 768-GiB allocation and observed about 417.9 GiB guarded host working set on the longest repeated fixture. These are historical measurements, not proof the new episodic scorer/attention fits.

## 14.2 Discover current resources; never reuse historical job IDs

Read current scheduler/environment state, including:

```bash
squeue -u "$USER" -o "%.18i %.9P %.24j %.8T %.12M %.12l %.6D %R"
scontrol show job <VERIFIED_CURRENT_ALLOCATION_ID>
squeue --steps -j <VERIFIED_CURRENT_ALLOCATION_ID>
sacct -j <VERIFIED_CURRENT_ALLOCATION_ID> --format=JobID,State,ExitCode,ReqMem,AllocTRES,MaxRSS,Elapsed
nvidia-smi
```

`<...>` fields are placeholders, not runnable IDs. Confirm ownership, remaining allocation time, GPU indices and process usage. `--overlap` does not grant permission to use GPUs occupied by another job. Do not cancel the user's interactive shell/allocation.

Read cgroup-v2 `memory.max`, `memory.current`, `memory.peak` when available, `memory.stat`, `memory.events`; inspect the effective enclosing job limit, not only host `free`. MaxRSS is per reported task/step and is not a complete synchronized job peak.

If only two GPUs are available, H2 can be tested there; H4 remains unqualified until a real four-GPU allocation exists. If the current allocation has four idle GPUs, run H4 and H2 **sequentially**, never concurrently on the same devices or by overlapping their long offload peaks.

## 14.3 Initial runtime profiles

Proposed starting points, to be adapted only after measured resource evidence:

| Setting                           | H4                                               | H2                                                                                      |
| --------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| GPU request                       | 4×H100 80GB on one node                          | 2×H100 80GB on one node                                                                 |
| B1 per rank / GAS                 | 1 / 2                                            | 1 / 4                                                                                   |
| ZeRO                              | Existing ZeRO-2                                  | Existing ZeRO-2                                                                         |
| Optimizer offload                 | GPU initially                                    | CPU optimizer offload initially                                                         |
| CPU optimizer implementation      | Native approved optimizer                        | Existing `zero_force_ds_cpu_optimizer:false` profile; do not silently replace optimizer |
| Activation offload threshold      | 65,536 tokens initial                            | 65,536 initial; reprofile all changes                                                   |
| Loader workers                    | 0 initial                                        | 0 initial                                                                               |
| Prefetch if workers later enabled | 1; no persistent workers initially               | 1; no persistent workers initially                                                      |
| DataLoader pinned prefetch        | False initially                                  | False initially                                                                         |
| Vision microbatch                 | 4 initial, lower only if qualified               | 4 initial, lower only if qualified                                                      |
| Attention batch groups            | 8 initial; qualify 4/8/16                        | same science/runtime choice within profile                                              |
| Host allocation request           | 768 GiB preferred starting qualification request | 768 GiB preferred starting qualification request                                        |
| Model activation pinning          | Retain required offload behavior                 | Retain required offload behavior                                                        |

A 768-GiB request is not a promise it fits every cluster, nor an instruction to exceed local policy. Determine available nodes. If actual qualified host working set plus safety margin requires more RAM, request more on a permitted node or stop. A smaller host request is allowed only after a fresh measured gate establishes margin. Do not infer a safe host amount from GPU count.

Reducing loader workers protects against multiple serialized long episodes per worker queue. Disabling DataLoader pinning does not disable the pinned activation-offload mechanism; report them separately.

## 14.4 Resource stress suite for **each** runtime profile

Use native V0 and new M2. Phase2 large-cache arms require additional M5 profile. Fresh profiles allocate optimizer states and exercise save/reload.

1. **Integration smoke:** mixed real short R2R/RxR examples, 3 complete optimizer updates; tokenizer, loss, scorer, backward, reload. Diagnostic truncation is allowed only for this labelled fixture and cannot become training data.
2. **Longest R2R:** expected reference 183 observations / about58,290 text tokens; all global slots use the audited representative, 3 updates.
3. **Just below offload:** actual maximum episode below65,536 tokens, expected reference 205 observations /65,526; 3 updates.
4. **Just above offload:** smallest representative above the threshold plus a large-prefix case; 3 updates.
5. **Longest RxR:** actual maximum, historical 627 observations /199,735 text tokens; repeat it to fill global8; 3 updates.
6. **Mixed-tail:** eight distinct long/medium episodes including longest and alternating below/above-threshold episodes; >=3 updates to catch asynchronous offload and changing allocations.
7. **Checkpoint peak:** save a complete distributed recovery checkpoint during a stressed allocation, verify marker/files, exit, reload in a fresh process and continue the next update with the same topology.
8. **Representative speed:** >=20–50 ordinary-manifest updates with the declared sampler, separate output and no use of smoke weights as the production initialization. Record lengths and queue/cache conditions; no guaranteed ETA from short examples.

Tests on a different number of ranks, offload policy, attention pack size, score backend or dataset manifest do not qualify this profile.

## 14.5 Pass/fail rules

All ranks must finish each update and report finite losses/gradients. Confirm actual optimizer step counts and expected exposures; a process returning exit0 without training is FAIL.

GPU: hard gate <=78 GiB maximum reserved on the reported 80GB H100 setup, and preferably <=76 GiB target for additional margin. Use actual device capacity; no assertion that all hardware labels imply identical bytes. Both allocated and reserved peaks are recorded on every rank/update.

Host: derive an owned-command working-set guard from actual finite cgroup limit, proposed `min(0.9*limit, limit-48 GiB)` for large allocations. Record total and working-set values. A conservative estimate is:

    clean_inactive = max(0, inactive_file - file_dirty - file_writeback)
    working_set = max(0, memory.current - clean_inactive)

This is a heuristic, not reclaimability proof. Record anonymous/pinned/unevictable memory, swap, dirty/writeback and event counters. Any new `oom`/`oom_kill` event, OS kill, guard termination or missing-rank step fails the profile. Nonzero `high/max` reclaim events are recorded and investigated, not automatically called OOM or ignored.

Sample host usage about every second in the owned supervisor, including load, prefetch, optimizer allocation, save and teardown. Respect job-aggregate usage if the interactive allocation contains other processes. Such a shared allocation cannot establish an isolated benchmark unless those contributions are separated or conservatively included.

Disk: preflight reserves the larger of400GiB and a measured projected footprint for two rolling recovery checkpoints, both epoch exports, one transient checkpoint save and bounded logs. Parallel experiments require the sum of their reservations. Retain trained weights/reports unless the user explicitly authorizes cleanup; never delete unrelated checkpoints to force launch.

A guard sends SIGTERM only to the process group created by this supervisor, then owned-process escalation if necessary. It never changes cgroup limits, drops shared filesystem caches, or cancels the entire Slurm allocation.

## 14.6 Allowed responses to memory failure

Diagnose the actual failing allocation and stage. In order, consider bounded packed attention batches; score-buffer reduction; loader/prefetch reduction; avoiding all-GAS GPU staging; bounded CPU offload queues; optimizer offload profile; an allocation with more host RAM; adjusted activation threshold with GPU/RAM requalification.

Not allowed as silent fixes: dropping long episodes, shorter training data, pixel downsampling, truncating instructions/actions, TBPTT, changing global objective normalization, moving offload to unaccounted disk, bypassing numerical/backend guards, or repeated blind resubmission.

# 15. Numerical and learning release gates

1. **CPU mathematical oracle:** FP64 sequence/block delta, top-B causality, dense attention with arbitrary original indices and first-order gradients. Unequal key/value widths prevent hidden transpose errors.
2. **Fixed-score selector:** exact batch/incremental IDs for every prefix, including ties, empty bank and turnover.
3. **Production kernels:** actual H100 FLA/FA2 forward/backward; verify dispatched approved Hopper backend and its actual source hash.
4. **Feature disabled:** B0/R8 parity against unchanged native model, including frozen visual/merger, parameter identity and action-token loss.
5. **Real scored replay:** R2R and RxR varied 60–128-step prefixes crossing multiple admissions. Report raw score error, cutoff margins, ID mismatches, all response-token logits and complete decoded actions. Do not modify an episode's terminal labels and then call it a navigation result; replay fixtures may terminate without inventing a training target.
6. **Checkpointing:** first-forward selected plan equals the one used in VJP/recompute; compare direct and checkpointed gradients with a measured native-repeat budget. Never change thresholds after seeing one failure without retaining that failure and declaring a new protocol.
7. **Trained checkpoint:** repeat the serving parity gate on actual trained M2, not just base weights or artificially opened projections.
8. **Distributed recovery:** restore same-world optimizer/master weights, RNG, sampler and next exposures; acknowledge BF16 nondeterminism instead of claiming bitwise training identity from matching losses alone.
9. **Full rollout:** real Habitat closed-loop test with strict episode identity and failure accounting; numerical replay is not SR/SPL.

Use tight FP64 tolerances for algebra (for example1e-10 output/1e-9 gradient on small fixtures). Existing BF16 text replay uses absolute0.75, relative0.03 and RMS0.15 budgets; these are inherited qualification references, not universal guarantees. Preserve all full-response argmax disagreements and investigate. A correct selector with input score differences may still alter policy; report that sensitivity rather than declaring98% overlap sufficient.

# 16. Full evaluation and Phase-2 promotion

## 16.1 Quality

Use the repository's new-server evaluation guide and exact simulator/model dependency separation. Preserve the 1,839 R2R episode-ID set, data/config hashes, seed42, 640×480/79-degree camera, 0.25m forward/15-degree turns,3m radius and500-action cap. Invalid responses and infrastructure failures remain accounted for according to the existing evaluator contract; do not silently use an oracle fallback.

For RxR, load the verified English15-degree split and its required registrations/config. Historical reference count is3,669 val_unseen episodes; verify actual identity. Distinguish complete full-split and fixed-panel results. No guessed nDTW/SDTW values if the active evaluator does not compute them.

At each compared epoch record SR/SPL/NE/OS, model STOP, forced STOP, invalid/failure count, route length and per-episode outcomes. Count every scheduled episode exactly once. Preserve incomplete journal fragments when resuming instead of deleting evidence.

## 16.2 Efficiency

Measure isolated native/Epi checkpoints on the same GPU/runtime. Separate:

- Full decision latency including score extraction, sorting, cache packing, vision, text prefill/decode and feedback append.
- Simulator/IPC/disk timing.
- Training update compute and end-to-end node-hours, including checkpoint overhead separately.
- Retained KV tokens/bytes and state/metadata memory.

Use real varied recorded trajectories and actual closed-loop lengths in addition to the historical repeated-frame timing control. Repeated identical RGB must not be the sole evidence for a cache method. Repeat and randomize benchmark order; no simultaneous timing competitors on one GPU.

## 16.3 Interpretation

M2>M1 and M2>M2c supports the selector's value under this budget if the result survives confirmation. M2≈M1>V0 means two ways of adding context help similarly, not “M2 is recent memory.” M2≈M2c>V0 supports generic episodic access but not score superiority. M5 versus V0 is not exact equal-cost until total K/V and measured runtime are compared.

A score is fixed at original write time; it is not a query-aware utility estimate. Report pinning, poor turnover and negative outcomes rather than hiding them behind a learned scorer introduced after seeing results.

# 17. Safe 4H100 / 2H100 submission race

## 17.1 What the user is authorizing

The requested race is between **two resource profiles for the same logical training run**, not permission to cancel arbitrary other jobs. Smoke/qualify H4 and H2 first. Only profiles that passed their own full gates may be production candidates.

Recommended policy: race the first logical run V0 to select an available profile, then keep that profile for all primary Phase-1 comparisons. This avoids confounding method differences with2-rank CPU-offload versus4-rank GPU-optimizer trajectories. If queue reality forces a later runtime change, version it, report it and re-run the necessary matched baseline on that profile. Same global batch is not numerical identity.

No exact optimizer-state resume from4 ranks into2 ranks or vice versa is permitted by this plan. A failed run resumes on the same topology; topology-changing recovery would need a separately validated conversion method.

## 17.2 Launch identities and owned jobs

Create a campaign UUID, scientific run ID, resolved config hash, source SHA, dataset hash and profile hash before `sbatch --parsable`.

Maintain an allowlist:

    campaign/run/job_id/profile/output_dir/source_sha/config_sha

Create distinct candidate directories. No two jobs write the same checkpoint/output directory. A shared campaign coordinator owns the logical-run lease; physical candidate outputs remain isolated.

Only exact job IDs from this allowlist, whose live scheduler records still match owner/campaign, can be cancelled. Never use `scancel -u`, wildcard job names, `pkill python`, or cancellation of a parent interactive allocation.

## 17.3 Winner lifecycle

States:

    SUBMITTED -> ALLOCATED -> PREFLIGHT_PASSED -> CLAIMED
              -> TRAINING_VERIFIED -> ACTIVE -> COMPLETE

A job being `RUNNING` means resources were allocated; it is not yet a healthy training run.

The first qualified candidate to pass live startup preflight atomically claims a provisional lease before allocating the full model/trainer. Use a tested shared-filesystem atomic operation/locking mechanism. Do not rely on two processes checking “file exists” and then both writing. Record hostname/PID, job ID, timestamp and heartbeat.

A competing candidate that sees an active lease must not launch its Trainer. It may exit cleanly with `DUPLICATE_STANDBY` evidence after a bounded wait. A pending sibling is left pending until winner verification, then cancelled. If an already-started sibling has its own isolated smoke process, stop only that owned candidate after recording its state; it must not run a duplicate full training trajectory.

Promote the winner to TRAINING_VERIFIED only after:

- Source/model/data/runtime identity checks pass on the actual allocation.
- All ranks perform at least50 production updates with finite values and correct exposures.
- One complete recovery checkpoint is written and its completeness/reference reload check passes (request an initial save at update50; subsequent250 cadence is unchanged).
- Host/GPU/disk guards stay below qualified limits and no OOM events occur.
- The remaining wall-time/checkpoint plan is viable.

Do not load a second4B model on the active Trainer GPUs to validate the winner. Use the separately qualified recovery procedure with complete-shard/hash/reference checks, or pause at a safe checkpoint boundary and restart the same owned attempt under its lease. Account for any CPU checkpoint inspection in the host guard.

Then cancel only the recorded redundant sibling job. If the sibling exited voluntarily earlier and the winner fails before verification, diagnose and resubmit one qualified candidate under a new recorded attempt; do not resurrect stale checkpoints or assume the old pending job still exists.

If a job fails after producing a complete optimizer checkpoint, recover with the same profile and exact source. If it fails before any valid checkpoint, restart from the declared base with a new attempt record. Do not delete lease/checkpoint evidence merely to make a launch pass.

## 17.4 Interruption, recovery, and time limits

Resolve current Slurm time limits; do not assume an interactive allocation lives long enough for two epochs. Production should normally use durable batch jobs after the interactive qualification. Request advance termination signaling with sufficient time for a checkpoint, and implement checkpoint requests at optimizer boundaries. Never try to serialize model state directly inside an asynchronous signal handler.

External Slurm signaling/cancellation semantics must be checked against the installed scheduler. Scripts remain placeholders until account/partition/QOS, GPU resources and RAM limits are discovered. An accepted `sbatch` submission is not training completion.

## 17.5 Command skeletons — only after implementation and qualification

The existing public training entry is `qwen_vl.train.train_qwen`; `train_episode` is an implementation helper, not a supported `python -m` training command. The following uses **new proposed configs**, which must be generated and validated first. Paths are discovered locally, not copied from obsolete server reports.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
# Set MODEL_PATH, JOINT_MANIFEST, OUTPUT_ROOT and the validated CUDA/model environment.
# Do not invoke before configs/episodic and the memory implementation exist.
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/episodic/m2_joint_4h100.yaml \
  --output_config configs/vln_empty.yaml \
  --manifest "$JOINT_MANIFEST" --model_name_or_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_ROOT/train" --run-name episodic-M2-s429 \
  --recovery-save-steps 250 --deepspeed deepspeed.json
```

For the H2 candidate, use two ranks and its resolved GAS4/CPU-optimizer profile, including `configs/deepspeed_zero2_cpu_optimizer.json`. Do not merely edit nproc while retaining GAS2. The initial save-at50 behavior is a proposed campaign callback added in Task9, not an existing CLI flag assumed to work. The real coordinator wraps this command with the owned supervisor, lease and identity checks; the skeleton is not a safe replacement for those gates.

Create Slurm batch templates from actual permitted partition/account/QOS and resource syntax. Submit both qualified profiles with `sbatch --parsable`, record returned IDs, and then apply the lease rules. Do not use historical allocation IDs, hardcoded IPs or credentials in the plan or committed launcher.

# 18. Reporting layout and required evidence

Use `reports/episodic_idea/` for small, committed Markdown/JSON summaries. Large logs, checkpoint binaries and journals live under a separate campaign output root with recorded paths and hashes.

```text
reports/episodic_idea/
  00_audit.md
  01_design_decisions.md
  02_implementation.md
  03_score_validation.md
  04_attention_and_replay.md
  05_resource_4h100.md
  06_resource_2h100.md
  07_phase1_results.md
  08_phase2_decision.md
  09_phase2_results.md
  run_index.json

<campaign_root>/
  source/                     # detached, immutable source snapshot
  campaign_manifest.json
  protected_jobs.json
  submission_ledger.jsonl
  leases/<logical_run>/
  profiles/H4/ and H2/
  runs/<logical_run>/<attempt>/
    launch_manifest.json
    resolved_config.yaml
    source_hashes.json
    supervisor/
    train/
    evaluation/
    status.json
```

Every report differentiates proposed, implemented, CPU-tested, GPU-qualified, submitted, started, completed, evaluated and failed. Log actual profiler fixture IDs and whether examples were repeated for stress. Never represent smokes as full-dataset results.

Keep a decision log for any flexible change: observation prompting the change, alternatives, selected fix, whether it changes science or only engineering, tests, cost and new run IDs. Do not overwrite the original plan to hide failed assumptions.

# 19. Run count and planning budget

| Stage                     | Configurations                |   Number |
| ------------------------- | ----------------------------- | -------: |
| Phase1                    | V0, M1, M2, M2c at seed429    |        4 |
| Phase2 default            | M5, M5c, V0-s430, winner-s430 |       +4 |
| Phase2 granularity        | corrected M3, when useful     |       +1 |
| Planned maximum           | with corrected M3             |    **9** |
| Bounded extra experiments | at most2 documented full runs | **<=11** |

Smokes, CPU tests, frozen probes, repeated evaluation and redundant queue requests are not additional trained scientific configurations. They still consume resources and belong in cost reporting.

Use measured representative full-corpus update timings to estimate hours. The supplied proposal's26h per four-H100 run is an estimate, not qualification for this scorer/cache. If it were achieved, four Phase1 runs would be104 node-hours or416 GPU-hours of training, excluding queue/evaluation. Actual H2 and H4 hours must be measured separately.

Compute a forecast with episode/token-length strata, actual data-loading/offload cost and checkpoint I/O. No promised completion date. The user authorizes scientific flexibility within the declared budget, not unlimited retries or arbitrary new projects.

# 20. Stop rules and success criteria

Stop implementation promotion on causality errors, incorrect score semantics, hidden full-history storage, missing gradient contributions, unsupported kernel fallback, invalid checkpoint export or uncontrolled duplicates.

Stop a resource attempt on OOM/guard/failed rank rather than shrinking the science data. Keep diagnostics and fix the cause before a new attempt.

Stop research expansion if Phase1 yields no reliable useful signal. A meaningful negative result is better than an uncontrolled collection of extra modules.

A successful experiment is not merely lower training loss. It requires correct training/serving, a completed full-data trajectory, qualified resource use, valid closed-loop results, meaningful controls and honest quality/cost interpretation. A CVPR/ICML-level claim depends on that evidence and prior-work positioning; no score or acceptance outcome is guaranteed.

# 21. Sources and what they support

Repository sources were read through the live GitHub connector. Direct raw download from the container failed DNS; no full repository execution or remote Slurm access was performed while writing this plan.

- [Branch inventory](https://api.github.com/repos/anhdao69/SimpleMemVLN/branches?per_page=100): branch identities at preparation time.
- [Pinned branch](https://github.com/anhdao69/SimpleMemVLN/tree/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1): source base.
- [Text implementation/resource report](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/reports/STREAMING_TEXT_DUAL_IMPLEMENTATION_2026-10-07.md): existing batched attention, text gates,627-frame and host-memory profiles.
- [Consolidated results](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/reports/R2R_ALL_VERSIONS_COMPARISON_2026-10-03.md): historical quality, runtime and OOM records; not results of this proposed method.
- [Attention code](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/src/qwen_vl/stream/window_attention.py): existing bounded varlen packing and VJP structure.
- [Session code](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/src/qwen_vl/stream/session.py): current text decoding, cache, positions and retries.
- [Episode trainer](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/src/qwen_vl/train/train_episode.py): CPU staging, recovery and full-episode trainer.
- [Activation offload](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/src/qwen_vl/models/activation_offload.py): non-reentrant checkpoint-input offload.
- [CPU optimizer profile](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/configs/deepspeed_zero2_cpu_optimizer.json): existing two-GPU candidate profile, not a new qualification.
- [Owned resource supervisor](https://github.com/anhdao69/SimpleMemVLN/blob/f6b8108f6ef8c7b4e3df035688d3210dbd3678a1/scripts/vln/run_step_lane_gate.py): working-set accounting and owned-process termination to adapt.
- [HOLA](https://arxiv.org/html/2607.02303v1): committed-write-selected exact memory and its language-model experiments; not a VLN guarantee.
- [DeltaS](https://arxiv.org/html/2609.27470v1): recurrent-state-guided KV retention and early-history bias.
- [FLA chunk source](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py): public versus internal outputs and native recurrence integration; inspect actual dispatched backend.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): unequal-query/key bottom-right causal convention.
- [Slurm sbatch](https://slurm.schedmd.com/sbatch.html), [srun](https://slurm.schedmd.com/srun.html), [scancel](https://slurm.schedmd.com/scancel.html): scheduler semantics, not authorization to touch other jobs.

