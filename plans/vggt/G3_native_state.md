# G3 — Delayed native GDN-state supervision: implementation plan

> **For agentic workers:** use the common pure forward/probe interfaces, not inference-cache snapshots or side-effect hooks. G3 is one new run; no recent-control policy run is added.

**Goal:** test whether training the existing recurrent state to retrieve old VGGT evidence helps navigation without adding deployed state or parameters.
**Architecture:** context-conditioned auxiliary queries read native `S24`; five reads per delayed source/read pair; all auxiliary machinery removed for deployment.
**Tech stack:** pinned native GDN/FLA differentiable chunk path, shared offline teacher.
**Spec:** master, common infrastructure and teacher documents.

## Global constraints

Only GDN layer24 supervised. No extra lane; no Q/K adapters; no current-alignment loss; single-view teacher only. Full BPTT. Probe rows exist only inside the selected native GDN kernel, after convolution, and never in Qwen input tokens, KV, positions or labels.

## Review focus

- Read exactly after the final observation/cue token, not after the current gold action.
- A detached native query still contains current-context information; recall is not a pure matrix-information proof.
- Constant zero beta/log-decay must be effective kernel inputs, not zero raw gate preactivations.
- No-op probe insertion must not alter real-token residual/MLP length or native z indexing.
- State loss should train earlier writes; querying a detached serving cache would not do that.

## 1. Observation-time semantics

Let action t's first response target position be j_t. Anchor a_t=j_t-1. Native state at that location includes prefix, all past tokens and current observation/cue, but not current action.

Old source tau must satisfy 8<=t-tau<=32. Under plain Window8, source visual K/V is no longer retained directly. This does NOT mean all old information exists solely in S24: lower layers/nearby contextualized tokens can carry old evidence indirectly. The read head is intentionally constrained to a matrix read, plus the stated query context.

## 2. Query network, exact dimensions

Native post-convolution q at anchor is normalized using the native additive-epsilon rule, repeated to32 value heads, flattened to4096, then detached. Concatenate:

- detached q-native context4096;
- learned region embedding32, five region IDs;
- fixed lag sin/cos32: frequencies `10000**(-arange(16)/16)` and argument `delta*frequency`.

Query network:

`LayerNorm(4160,elementwise_affine=False,eps=1e-6)`
`-> Linear(4160,512) -> SiLU -> Linear(512,4096)`
`-> reshape [32,128]`.

Kernel L2-normalizes learned probe queries once. Do not feed target features, episode IDs, source images, absolute file paths or teacher similarity scores into this head. Context detach prevents a direct loss gradient through the supplied native q, not information flow.

## 3. Read-only probe construction

For each of up to8 pairs, insert five query rows at its anchor, in deterministic `(anchor,tau,region)` order. Total at most40 inserted rows in this one layer.

Per row/head:

`q=Q(context,lag,region), k=[1,0,...,0], v=0, beta=0, g=0`.

With S key-by-value, output is `S_anchor.T @ normalized(q) / sqrt(128)`. No writes/decay. Prepared Q/K/V for every real token are produced before insertion. Existing native conv state is not advanced by probes. Real-token output rows are gathered back before gated RMSNorm/out_proj; probe rows go only to the auxiliary decoder.

The functional `insert_probes(prepared, anchors, q_probe)` must return both expanded prepared tensors and immutable real/probe output indices. Repeated anchors are legal; repeated identical source/read pairs are not. No dense [L,L] matrix or all-step state snapshots.

## 4. Decoder and objective

Read tensor per pair `[5,32,128]`. Apply independent per-head RMS normalization (FP32, eps1e-6, no affine), concatenate heads to4096, then:

`Linear(4096,1024) -> GELU -> Linear(1024,256)`.

Target `[5,256]` is four quadrant means and global mean of source tau's unpositioned shared PCA tokens. No added 2D sinusoid.

Average cosine loss across five regions; sum over valid pairs. Global count is valid pair count across ranks/GAS. Lambda max0.1, common linear ramp. Decoder/query linear matrices Normal(0,.02), biases zero, region embedding Normal(0,.02), isolated RNG.

## 5. Expected gradients and non-claims

L_mem flows through the probe reads into S_anchor and earlier K/V/beta/g variables contributing to it. With stop-grad query context, local native q is not directly trained by this loss through the query network. Local native query and later decoder layers still train through navigation. Model dependencies may send state-loss gradients into earlier layers, including their read pathways.

Do not require every earlier gate to have a nonzero gradient on every fixture; test an explicitly nondegenerate fixture with evidence of earlier-write gradients. Loss at an isolated anchor must not have a path through real tokens after the anchor. Future anchors' losses naturally can train those later tokens.

Probe row normalization/auxiliary decoder can learn shortcuts from current query and scene priors. Success of cosine reconstruction alone is not evidence of causal navigation memory.

## 6. Frozen-policy diagnostics, not extra full runs

Before or after training, on fixed held-out scene/episode panels:

1. Normal context-conditioned read.
2. Same query with zero state (sanity only; output collapse is not proof of useful memory).
3. Same query with a state captured from a different already-processed episode, reporting that intervention clearly.
4. A small query-only diagnostic decoder fit on TRAIN features only, with comparable evaluation queries.
5. Lag bins1–7,8–15,16–23,24–32; train-mean/current-view/recent-view teacher target baselines.

Collect extra teacher targets needed for diagnostic current/recent baselines; do not substitute missing cached frames. Probe fitting is logged separately and does not update the navigation policy. Do not call a fitted diagnostic probe “training free” without qualification.

## 7. Tasks

### G3.1: head and target tests
**Files:** `models/vggt_aux_heads.py:ContextStateQuery, StateReadDecoder`; `tests/vln/test_vggt_state_heads.py`.
**Interfaces:** `query(context_q: Tensor[P,4096], delta: Tensor[P], region: Tensor[P,5]) -> Tensor[P,5,32,128]`; `decode(read: Tensor[P,5,32,128]) -> Tensor[P,5,256]`.
- [ ] Shape/count tests and detached-query context test.
- [ ] A fixed-identity-state counterexample demonstrates why context provides information; document not a failure of the code.
- [ ] Implement shared heads; commit.

### G3.2: probe injection
**Files:** `models/gdn_research.py`; `tests/vln/test_gdn_readonly_probes.py`.
- [ ] FP64 independent recurrence Dk3/Dv5, compare inserted read to explicit snapshot read.
- [ ] Check real outputs/final state/navigation gradients with zero-weight probes.
- [ ] Check zero beta/g, current-action causality, multiple pairs at one anchor and empty plan.
- [ ] Implement production adapter and test actual pinned FLA at heads32/dims128; no fallback to slow reference in production.

### G3.3: checkpointed layer24
**Files:** `models/research_forward.py` and common loss integration.
- [ ] One native GDN call, no double decoder checkpointing, aux sums returned explicitly.
- [ ] Checkpoint on/off gradients, full episode vs streamed no-probe native outputs, loss counted once.
- [ ] Deleting probes restores original sequence length/positions before every upper layer.

### G3.4: production qualification
- [ ] Profile40 rows added to the longest real GDN layer, all8 pairs valid, plus dense language graph and teacher-region tensors.
- [ ] Save/reload recovery includes query/decoder; served export excludes them.
- [ ] Train fresh G3; compare equal-progress V0 and G1 as method-level comparisons.

## 8. Criteria

A helpful result improves action metrics/closed-loop navigation, not only recall. G3 vs G1 is not a clean delay ablation: layer interface/target density/decoder differ. With no extra budget for a trained recent arm, limit the claim to a promising fusion screen. Negative performance is retained; do not add extra query modes as undeclared full runs.
