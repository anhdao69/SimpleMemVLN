# M5 GDN-Only Selection and Learned Ranking — Final Implementation Plan (Revision 2)

> **Status:** finalized handoff specification for implementation; not an implemented SimpleMemVLN feature. This revision incorporates the approved F1_v2 feature change, dual-horizon labeling, and S-first execution order. No navigation training or Habitat evaluation was run for this update. Accompanying Python files are standalone mathematical/data-contract references, not production FLA or simulator integrations.
>
> **For implementing agents:** start with `START_IMPLEMENTATION.md`, then read this full specification and the pinned repository code. Implement and review one task at a time using the executing-plans or subagent-driven-development workflow available in your environment. Begin with Task 0 and Phase A. Do not launch a full experiment until its mathematical, causality, memory, and serving gates pass.

**Goal:** determine which native GDN signals predict the need for retaining historical visual KV, then test whether a small GDN-only learned ranker improves M5 under the same causal memory budget.

**Architecture:** retain the pretrained navigation backbone and native full-attention read path. Replace only the scoring/selection component through an explicitly versioned interface. Start with static priorities and fixed per-layer budgets; defer dynamic reranking and navigation-outcome labels to separately named extensions.

**Tech stack:** the repository's qualified Python/PyTorch/Transformers/FlashAttention/FLA environment, not an independently upgraded stack. Mathematical references use only Python, PyTorch, and pytest. Specification revision date: 2026-10-20. This is a source-grounded update of the supplied plan/package, not a new live repository or literature audit.

**Source baseline:** `anhdao69/SimpleMemVLN` at `fde1cbef213e27320920400a826addea8d5e114a`, branch snapshot `c2_vggt_omega_g3_m5_layers`. Pin exact checkpoint bytes separately. The branch name alone is not sufficient provenance.

**Scope:** S0–S9, with S8-N/S8-W; random and recency controls; L0–L3 and offline diagnostic controls; conditional L4–L6 extensions. No VGGT/G3 supervision, new vision encoder, map, pose/depth input, RL algorithm change, or generic KV mechanism in the core sweep.

## Start here: what changes, and what does not

**Implementation begins with the S arms. They require no learned-ranker utility dataset.** Existing R2R/RxR images, instructions, expert actions, dataset versions and checkpoint bytes remain unchanged. Training-only calibration for S6/S8 and optional diagnostic traces are not a new navigation dataset.

The derived feature/utility dataset is needed only for Phase B learned rankers. It is created by replaying existing training trajectories through a frozen model and computing offline labels. Do not re-render images, rewrite manifests, or collect new simulator trajectories unless a separately documented asset problem or later L4-return experiment requires it.

| Revision-2 decision | Exact effect |
|---|---|
| Replace F1 feature 5 | `log(max(||beta*e||,1e-6))` becomes `log(max(||vhat||,1e-6))`; `vhat` is the decayed-prior prediction |
| Preserve diagnostic information | Store both raw `committed_norm` and `prediction_norm` in `gdn_raw_v2` |
| Keep architecture widths | L0/L1/L2 unchanged; L3 stays 576-D, now using `F1_v2` |
| Collect two horizon strata together | Up to four near offsets in the first 16 post-admission decisions, plus up to four tail offsets after them |
| Keep targets separate | Store near mean, tail mean, and support-length-weighted remaining mean; missing is not zero |
| Preserve primary comparisons | All first-round L0–L3 fits use the near-16 target; only one or two selected architectures enter the remaining-horizon comparison |
| No serving leakage | Remaining length, sampled future offsets and future utilities are label-only metadata, never runtime ranker inputs |
| No promise of free extra labels | The shared model forward is reusable; candidate/query LOO work and archival I/O can increase substantially |

**Version names are not new model releases.** `plan_revision=2`, `gdn_raw_v2`, `F1_v2`, and utility-schema v2 describe documents or derived-data contracts. They do not mean a new M5 architecture, M5 checkpoint, R2R version or RxR version. The legacy M5 and S0 contracts remain unchanged. The proposed `episodic_window_gdn_sweep_v1` mode remains the first implementation version of the new harness.

**Source scope:** repository/paper access statements in Sections 2 and 17 are preserved from the supplied plan's earlier audit. This update has not re-verified their live availability. Recheck the pinned source and native fingerprint on the execution host; report discrepancies rather than silently changing the baseline.

**Global constraints:** original data and checkpoint bytes are immutable; R4/B1200 and native KV read/positions stay fixed; no new G3/VGGT/RL objective enters the core sweep; feature and target schemas are explicit and hashed; no future serving features; no token resurrection; no full run before qualification; no reinterpretation of old arrays under new feature names.

**Review focus:** fifth-channel schema mismatch; truncated/zero-length label support; censored rows mistaken for zero utility; unweighted or log-space averaging of near/tail labels; trace outputs that leak future metadata into serving. Tasks 1/3/5/6 below include corresponding tests.

## Contents

1. Decision record and review of the supplied comments
2. Reference code and literature map
3. Baseline contract and experimental axes
4. Exact GDN mathematics and numerical conventions
5. Phase A: S arms and controls
6. Extraction backend and stable residual scan
7. Offline trace replay versus actual policy recomputation
8. Checkpoints, splits, uncertainty, and screening
9. Phase B: utility-label definition, near/tail horizons and reference contexts
10. Censoring, random banks, and label coverage
11. Learned features (F1_v2), models, ages, objectives, and calibration
12. Oracle headroom, hidden-state controls, and promotion gates
13. Conditional L4–L6 extensions
14. Integration interfaces and file-by-file plan
15. Verification tasks and acceptance tests
16. Execution order, resource accounting, and deliverables
17. References and source-access status

---

# 1. Decision record: which comments are correct?

The original review decisions below remain in force. Revision 2 adds the two subsequently approved refinements: prediction magnitude replaces the redundant F1 committed-magnitude input, and the first main label-collection pass measures near and tail query strata. The data-collection clarification in the start section is normative.

| Comment | Decision | Technical resolution |
|---|---|---|
| S0–S3 form a gate × state-prediction factorial | Accept | Use all four pairwise contrasts and the interaction, not only S0 versus each arm. |
| HAM cosine needs an epsilon and tie diagnostics | Accept | Fix the exact zero-value/zero-prediction conventions. An instruction-prefilled state is not necessarily near zero at frame 0; measure rather than assume. |
| Slow-head gated-value is meaningless because it does not read state | Reject that absolute claim | Slow decay plus effective incoming value can still predict future utility. Nevertheless make S6 apply the same slow-head mask to S0, with an optional S6-V paired with S2. This isolates head selection cleanly. |
| Whole-frame bank keeps four of five frames at each saturated admission | Accept | Before saturation keep all eligible frames. Each attention layer maintains its own four frame identities. No pooling or merging of native K/V. |
| Net state drift includes decay | Accept | Add an exact decay-transported-write score; also log untransported write activity, but do not conflate these quantities. |
| Write-only drift is `norm(sum k delta^T)` | Correct | The exact decomposition discounts each write by all subsequent decays in the segment. An undiscounted sum is a different statistic. |
| Add `(1-beta)*norm(e)` | Accept with qualification | It complements S0 algebraically, and approximates immediate self-read residual for unit keys. Epsilon-normalized keys require the exact factor `abs(1-beta*||k||^2)` for equality. It is not itself a faithful HAM or LoLA reproduction. |
| Add random-at-serving on the same checkpoint | Accept | Use deterministic token priorities for each selector seed; separate selector seeds from training seeds. |
| Add recency | Accept | Retain the latest 1,200 eligible old visual rows. This is not full Window8: older action/text rows remain absent. |
| Exclude VaN/KeyDiff/H2O | Accept for the core development sweep | A conditional full-attention-value-norm control is reasonable if S3 looks sufficient. It answers a distinct question and is not mandatory before the core sweep. |
| Skip closed-loop when selected-set overlap exceeds 90% | Reject | A 10% difference is 120 bank rows per layer; one critical row can change an action. Use overlap to identify redundancy, never as a behavioral guarantee. |
| M5c is a neutral primary testbed | Reject | Its weights were optimized for random retention, and historical schedules may differ from M5_E1. It is an alternate adaptation control, not neutral weights. |
| Use val_seen for development and val_unseen once | Accept prospectively | Freeze finalists before the next unseen evaluation. Previously inspected unseen results do not become an untouched test again. |
| 778 episodes have weak power for 2-point effects | Qualify | Maximum independent-Bernoulli SE is 1.79 pp. Paired precision depends on discordant pairs; scene correlation and training seeds matter. Verify actual split counts. |
| LOO output after o_proj at the pre-action query | Accept, with native output gate included | Remove one logical row across all relevant attention heads; aggregate after sigmoid gate and output projection. |
| Mean over the token's observed bank lifetime is unbiased | Reject | Retention causes informative censoring and age-dependent exposure. Use a fixed target horizon and distinguish missing labels from zero utility. |
| Random-bank traces remove label bias | Reject | They randomize selection priorities, but survival, downstream hidden states, and exposure are still policy-dependent. |
| Never use M5 traces | Reject | Do not use them exclusively; they remain relevant deployment-distribution data. Combine explicit randomized logging with controlled label coverage. |
| GateSight max is inherently lifetime-biased | Qualify | Original GateSight uses a common continuation query set. In this streaming adaptation unequal exposure is the problem; mean aggregation alone does not fix it. |
| Replace age with absolute birth time | Optional, not mandatory | Static priorities are comparable when trained to a common globally calibrated target. Birth time adds a recency prior and must be ablated. |
| Linear birth-time bonus equals exponential aging | Correct only in log-score space | `log u_i + lambda*t_birth` ranks like `u_i*exp[-lambda(t-t_birth)]` for positive u and common lambda. Adding birth time to raw utility is not equivalent. |
| Log beta/value inputs let a linear model express products | Partly accept | They express multiplicative per-head forms, not arbitrary sums of products across heads. Keep explicit products or an S0-residual control. |
| Native feature hooks remove L3 overhead | Qualify | They can avoid duplicate projections, but 24-layer state scans, capture, storage, synchronization and parity remain. Implement a correct side-pass reference first. |
| Hidden-state control is cheap and invalidates GDN if it ties | Accept the control; reject the inference | Measure information versus cost. A stronger hidden-state probe challenges exclusivity, not necessarily efficiency or utility of GDN features. Runtime is not known in advance. |
| Oracle-label plan must beat S0 before training | Useful diagnostic, not a theorem | Future-label plans test the proposed surrogate's headroom. Marginal LOO labels are not additive or globally optimal; zero gain does not prove all learned memory methods are useless. |
| Spearman(S0,label)>0 is mandatory because M5 beats random | Reject | The historical comparison used separately trained policies and closed-loop returns, not this particular local-LOO surrogate. Report correlation; do not make it a correctness gate. |
| Fit L0–L3 in parallel offline | Accept | Use identical labels, folds, preprocessing and comparable tuning. Do not promise a few hours without profiling. |
| Only best L and L2 enter closed loop | Revise | Include the best fixed scorer, best learned ranker, L2, and best simple learned model if distinct. Selection rules are frozen before unseen results. |
| Two SFT seeds only for one winner | Qualify | Baseline needs corresponding matched seeds as well. Two seeds are initial replication, not a robust variance estimate. |
| L4 immediately after L3 | Accept conditionally | First validate that the local-output label predicts any useful retention behavior. Keep the model/features fixed when changing the label. |
| L5 dynamic last | Accept | It changes selection semantics and invalidates fixed-priority running-top-B guarantees. |
| L6 admission-only costs about 118 MB and 0.5% | Split the claim | 117,964,800 bytes is correct for eight layers, four frames, 16 key heads + 32 value heads, width 128, two-byte storage, excluding overhead. 0.5% wall-time is unverified. |
| Replace redundant F1 committed-magnitude input with prediction magnitude | Accept | Redundancy is exact before clipping/floors; expose decayed-prior prediction norm instead and preserve both raw scalars |
| Changing feature 5 always requires collecting all traces again | Reject | Reuse raw prediction norms or original value/error vectors when available; otherwise regenerate only missing feature shards, not original data or unaffected labels |
| Collect a second horizon in the first labeling campaign | Accept with stratification | Four near offsets plus up to four disjoint tail offsets; derive the remaining mean with support-length weights |
| A second horizon is almost free | Not guaranteed | Reuse model forward/extraction, but profile additional local LOO/projection work, archival I/O and retention |
| A long-horizon mean always detects rare critical tokens | Reject | Sparse sampling may miss rare events; keep per-query values and a dense quality subset |

**Research decision:** accept the tightly GDN-focused sweep, add S9 and same-checkpoint controls, and correct the label, timing, statistical and cost assumptions before implementation.

---

# 2. Source and reference-code map

## 2.1 Existing project code to reuse or extend

All project links below are pinned to the source baseline.

| File | Current responsibility | Planned use |
|---|---|---|
| `src/qwen_vl/models/committed_write.py` | Native GDN preparation, FP32 block-64 scorer, carried state | S0 oracle; feature extractor; stable unscaled-error scan reference |
| `src/qwen_vl/stream/episodic.py` | Hash priorities, RunningTopB, immutable LayerSelectionPlan, diagnostics | Preserve S0; implement separately named token/frame priority builders |
| `src/qwen_vl/research/episodic_contract.py` | Versioned visibility/scoring contract | New versioned score IDs, artifact hashes and numeric conventions |
| `src/qwen_vl/models/episodic_forward.py` | Layer-major offline execution and serving routing | Side-pass/tap interfaces; freeze immutable plans for backward |
| `src/qwen_vl/stream/episodic_cache.py` | Session metadata and atomic KV promotion | Frame selection, static learned scores, later admission-deficit rings |
| `src/qwen_vl/stream/window_attention.py` | Causal gathered FA2 and backward scatter | Exact visible-index recovery; local utility-label reconstruction |
| `src/qwen_vl/stream/session.py` | Native streaming state lifecycle | Reset/retry/snapshot integration and no-resurrection checks |
| `src/qwen_vl/models/nav_model.py` | Model dispatch | Explicit versioned backend dispatch; no silent fallback |
| `reports/vggt_five/M5_E1/implementation_report.md` | One-epoch reference recipe | Matched full-training contract |
| `evaluation_reports/EPISODIC_MEMORY_EVALUATION.md` | Results and evaluation identities | Do not mix historical checkpoints/schedules with new runtime controls |

The literal project source base URL is:

`https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/`

The installed native implementation must match the pinned project fingerprint. I also inspected the reference at:

`https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py`

Its full-attention output is **concatenated attention heads × sigmoid gate, then o_proj**. Label code must retain that gate if claiming actual layer-output removal sensitivity.

## 2.2 External mechanisms: what is adopted, not reproduced

- **HOLA:** committed-write selection is the S0-family reference. Official README and code layout were accessible at `https://github.com/scale-lab-ai/HOLA`; relevant files listed by the project include `fla/layers/gated_deltanet.py`, `gdn_swa_ops.py`, `gdn_swa_decode.py`, and `gdn_swa_fused.py`. Do not transplant its separate cache-read normalization into pretrained M5 attention.
- **HAM:** use its prediction-error-routing motivation for S4/S5. This plan uses fixed-budget top-B, not HAM's full parallel architecture or threshold-controlled cache. The paper and Zyphra research page were accessible; a dedicated official implementation was not verified.
- **DeltaS:** use local normalized state drift for S8-N. Original DeltaS aggregates layers and includes temporal bucketing; the proposed whole-frame, pair-local comparison is explicitly an adaptation.
- **GateSight:** use the feature-to-utility-prediction design, not released pretrained weights. The paper advertises `https://github.com/hideinsewer/GateSight`; the connector returned 404 in this review. Do not claim that its source code was inspected.
- **LoLA:** related self-recall/sparse-cache work. S9 is a GDN-derived complement control, not a LoLA reproduction. No dedicated official code release was verified in this review.
- **KVzap:** hidden-state control only. `https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/kvzap_press.py` was inspected. It demonstrates linear/MLP hidden-state score predictors; no published predictor is presumed compatible with this backbone.
- **DeltaS code access:** the paper advertises `https://github.com/MaumAI-Company/DeltaS`; the connector returned an empty-repository response. Use the paper to specify the adaptation, not imaginary implementation details.

Recheck accessibility at execution time, pin retrieved commits, retain licenses, and record which mechanism was actually reproduced. The mathematical reference supplied with this plan is original educational code and does not include third-party model code.

---

# 3. Baseline contract and invariants

## 3.1 Locked quantities

- Architecture: pretrained Qwen3.5-4B hybrid, verified layer types; 24 GDN and 8 full-attention layers.
- Attention indices: 3, 7, 11, 15, 19, 23, 27, 31. Scoring GDN indices: 2, 6, 10, 14, 18, 22, 26, 30.
- GDN geometry: 16 key heads, 32 value heads, Dk=Dv=128. Expand keys to value-head layout with the native contiguous repeat mapping, not an arbitrary reshape.
- R=4 recent groups including current; B=1200 old visual rows per attention layer; prefix retained unchanged.
- Observation grid: 15×20=300 visual rows.
- Recent groups retain their original text/action/framing content; old bank admits visual rows only.
- Exact native attention K/V retained with original logical positions/MRoPE. No merging, re-rotation, or replacement by GDN K/V.
- Canonical text-action protocol remains unchanged. Resolve action IDs from the actual serializer; do not invent a four-logit head.
- No teacher/auxiliary loss or RL change in the main S/L sweep.
- No external images, geometry teacher, GT pose or GT depth as serving inputs.
- Every experiment records checkpoint weight hash, source hash, feature schema, selector seed, numerical constants, data manifest, evaluator contract and learned-artifact hash.

## 3.2 Time conventions

- `i`: serialized token index.
- `f` or `s`: source observation-group index.
- `t`: group about to be processed.
- Admission is **before observation t**; newly eligible source is `s=t-R`.
- State `S_(t-)` includes the completed prefix and all prior groups including their responses, but no part of current observation t.
- A frame-level score is captured after that source's complete observation block and before its action. The exact block boundaries must match the serializer.
- Predictor training labels may use future training traces. Predictor runtime features and all selector decisions may not.

## 3.3 Experimental axes that must not be silently coupled

1. Signal: value, residual, gate, drift, learned utility.
2. Head aggregation: all-head mean, min, frozen slow-head subset.
3. Selection unit: token versus entire frame.
4. Score timing: birth, admission, repeated online updates.
5. Parameters: fixed heuristic versus independently trained ranker.
6. Weight adaptation: frozen-checkpoint swap, matched continuation, or full matched training.

A table mixing these axes is not automatically a one-factor ablation. The arm registry below includes the necessary matched comparisons.

---

# 4. GDN mathematics and conventions

Use key-by-value state orientation:

\[
\bar S_i=\alpha_i S_{i-1},\quad \alpha_i=e^{g_i},\quad
\hat v_i=\bar S_i^\top k_i,\quad e_i=v_i-\hat v_i,
\]
\[
\delta_i=\beta_i e_i,\qquad S_i=\bar S_i+k_i\delta_i^\top.
\]

Scores are computed in FP32 from the same native preparation as S0. FP64 sequential math is the test oracle. Do not confuse the GDN query-output `1/sqrt(Dk)` scale with the key-to-value prediction used in the delta update: **do not insert that query scale into e**.

## 4.1 Key normalization

Match the pinned native rule:

\[
k\leftarrow k/\sqrt{\|k\|^2+10^{-6}}.
\]

These keys have norm slightly below one. Keep the exact squared norm when checking self-read identities.

## 4.2 Immediate post-write self-error

\[
v_i-S_i^\top k_i=(1-\beta_i\|k_i\|^2)e_i.
\]

For mathematically unit keys this becomes `(1-beta)*e`. The factor is nonnegative for the verified sigmoid beta and this normalization. The exact expression must still use `abs` when converted to a norm in the reference.

For the requested S9 gate-complement score:

\[
\|e_i\|=\beta_i\|e_i\|+(1-\beta_i)\|e_i\|.
\]

This equality does not imply a partition of semantic information, storage capacity, or navigation utility.

## 4.3 Exact frame-state decomposition

For a segment i=a,...,b and each head, define:

\[
A_{a:b}=\prod_{i=a}^b\alpha_i,
\]
\[
W_{a:b}=\sum_{i=a}^b\left(\prod_{j=i+1}^b\alpha_j\right)k_i\delta_i^\top.
\]

Then:

\[
S_b=A_{a:b}S_{a-1}+W_{a:b},
\]
\[
S_b-S_{a-1}=(A_{a:b}-1)S_{a-1}+W_{a:b}.
\]

`W` is the decay-transported contribution of the **actual committed deltas**. It is not obtained by recomputing all deltas from a zero start state. The undiscounted `sum k delta^T` is a different quantity.

Useful diagnostic identity:

\[
\|\Delta S\|_F^2=\|D\|_F^2+\|W\|_F^2+2\langle D,W\rangle,
\quad D=(A-1)S_{before}.
\]

Log the cross term; net state drift can be small because components cancel, not because little computation happened.

## 4.4 HAM-inspired cosine conventions

For shared implementation, primary S4/S5 use the **decayed prior prediction** `vhat=v-e`. The HAM equation uses its own pre-update predictor convention; finite-epsilon and decay differences must be declared. Name these arms `HAM-cos-local@M5`, not full HAM.

\[
c_{i,h}=1-\operatorname{clip}\left(
\frac{\hat v_{i,h}\cdot v_{i,h}}
{\|\hat v_{i,h}\|\|v_{i,h}\|+10^{-6}},-1,1\right).
\]

Explicit convention: when `norm(v)<=1e-6`, set its cosine-error score to zero and log that event; a zero prediction with a nonzero value gives error one. Log prediction/value near-zero fractions separately. Fail on nonfinite inputs; do not replace NaNs by priorities.

The exact threshold and formula are part of the scorer artifact. Do not tune epsilon on navigation validation results.

---

# 5. Phase A registry

## 5.1 Main S arms

| ID | Exact scalar per candidate | Unit/timing | Required contrast |
|---|---|---|---|
| S0 | mean_h beta_h ||e_h||2 | token/birth | frozen reference |
| S1 | mean_h ||e_h||2 | token/birth | S0 and S3 |
| S2 | mean_h beta_h ||v_h||2 | token/birth | S0 and S3 |
| S3 | mean_h ||v_h||2 | token/birth | S1 and S2 |
| S4 | min_h cosine_error_h | token/birth | S5 and S1 |
| S5 | mean_h cosine_error_h | token/birth | S4 and S1 |
| S6 | mean over frozen slow-head subset of beta_h ||e_h||2 | token/birth | S0 |
| S7 | arithmetic mean of 300 visual S0 scores | frame/source-observation end | S0: granularity effect |
| S8-N | ||S_after-S_before||F / denominator | frame/source-observation end | S7 |
| S8-W | ||W||F / same denominator | frame/source-observation end | S8-N and S7 |
| S9 | mean_h (1-beta_h)||e_h||2 | token/birth | S0, S1 |

S8-N and S8-W use the full layer state, including all heads, in their Frobenius norm. They do not average headwise ratios; that would be a separate arm.

Optional offline metrics: S9 exact self-error with `abs(1-beta*||k||^2)`; untransported write norm; sum of individual write norms; cancellation ratio. Only promote an optional score to a full run after preregistering it.

### Factorial contrasts for S0–S3

For any reported endpoint Y:

- Gate effect with prediction: Y(S0)-Y(S1).
- Gate effect without prediction: Y(S2)-Y(S3).
- Prediction effect with gate: Y(S0)-Y(S2).
- Prediction effect without gate: Y(S1)-Y(S3).
- Interaction: [Y(S0)-Y(S1)]-[Y(S2)-Y(S3)].

Call this an implementation-level gate × residualization factorial; it does not prove the state is semantically unnecessary when one contrast is small. S2/S3 do not explicitly re-read the scorer state, but their native values/gates are still produced from history-dependent backbone hidden states; they are not memory-free policies.

## 5.2 Slow-head mask for S6

Calibration data: 128 training episodes, 64 R2R and 64 English-guide RxR where available, scene/length stratified, selected before inspecting navigation results. No validation labels enter calibration.

For each head and episode compute mean g over all actual processed tokens, excluding padding. Average those per-episode means with equal R2R/RxR domain weight. Define:

\[
\tau_h=\log 2/\max(-\operatorname{mean}(g_h),10^{-8}).
\]

Select the eight largest tau heads per scoring layer (25% of 32), tie-breaking by head index. Save IDs, calibration rows, bootstrap mask stability and hash. This is a decay timescale estimate, not a measured information lifetime.

Keep the mask frozen throughout an arm's policy continuation/full training. Recalibrating after policy updates creates a distinct adaptive-mask method.

Optional S6-V applies that exact mask to beta*||v|| and compares with S2. The supplied critique that this has no meaning is too strong; it remains a valid test of head-timescale specialization.

## 5.3 Drift denominator and early frames

Use:

`denom_l = max(norm(S_before_l), eps_l)`

with `eps_l = max(1e-6, 1e-3 * median_training(norm(S_before_l)))`, calibrated on the same training-only calibration split and then frozen. Record raw, unfloored drift and the floor-hit fraction.

This floor is an adaptation to handle near-zero states, not an unreported reproduction of the original DeltaS formula. An instruction prefix usually makes the first visual boundary nonzero; do not assume the first frame starts from a zero state.

Compute alpha products per head. Do not use a single mean alpha to remove decay.

## 5.4 Frame selection semantics

S7/S8 retain complete sets of 300 original rows per selected frame. With B=1200, retain all eligible frames until four exist; afterwards compare the four retained frames with the newly eligible frame and keep the top four. Fixed scalar frame priority makes this exact running top-4 over eligible frames.

Tie order is original source group index. Never resolve a tied frame score by selecting only the first raster rows. No K/V averaging and no shared ranking across layers.

Scores for S7 average **visual** token contributions only. S8's before/after snapshots cover the entire encoded observation block (including its formatting tokens), excluding the action. State this small support difference explicitly; optionally log visual-only drift when native boundaries allow it, but do not silently change primary S8.

## 5.5 Same-checkpoint controls

Use names `C_RANDOM` and `C_RECENCY`, not R0/R1, because those names already identify other model architectures in this project.

**C_RANDOM:** stable random hash priority per token and layer; no per-step random redraw. Use selector seeds 429, 430, 431 in finalist sensitivity checks. Freeze canonical dataset/episode identity. Existing project random priorities include episode UID: a changed absolute scene path can otherwise alter the experiment.

**C_RECENCY:** top latest eligible visual rows by source group/token ID. Under contiguous 300-row admissions and budget 1200 this retains the four latest old visual frames. It is not Window8 because old text/action/framing rows are absent.

For each control, start a fresh episode/session. Do not switch selectors halfway through a run and restore previously deleted rows.

Optional conditional control: native full-attention value norm (VaN-like). Run only if S3 nearly matches S0 and the paper will make a claim that the signal is specifically recurrent. It stays outside the core GDN method set.

---

# 6. Scoring implementation: stable residuals and extraction backends

## 6.1 Do not obtain residual-only scores by unsafe division

The current block scanner returns committed deltas, which already contain beta. `delta/beta` is not a robust residual implementation: beta can be zero after finite-precision sigmoid or extremely small.

For a block with cumulative log-decay c_t, solve the unscaled residual system:

\[
E_t + \sum_{j<t}e^{c_t-c_j}(k_t^\top k_j)\beta_j E_j
= v_t-e^{c_t}S_0^\top k_t.
\]

For each head:

\[
A_{tj}=1[t=j]+1[j<t]e^{c_t-c_j}(k_t^\top k_j)\beta_j,
\]
\[
AE=V-\operatorname{diag}(e^c)KS_0.
\]

**The beta belongs to column j.** Then delta_t=beta_t E_t. This system stays defined for beta=0. `reference/reference_math.py::unscaled_block_errors` implements the FP64/FP32 mathematical reference.

Respect the original prefix/observation/response segment boundaries and block length 64. An arbitrary re-chunking can alter FP32/BF16 results and top-B near ties.

## 6.2 Preserve S0 exactly

Keep the existing `CommittedWriteScorer` path available and untouched for S0. Do not silently substitute a mathematically equivalent but numerically different residual scan and reuse the S0 label.

The extended backend must compare its committed-score result against S0. A mismatch is reported with score error, selected IDs, and action consequences. If a deliberately different backend is used, give it a new backend version and rerun the relevant controls.

## 6.3 Correctness-first side pass

First implement all scores using the project's current native-input reconstruction route. This is slower but establishes a readable oracle. Store per-token **raw scalar features**, including both `committed_norm` and `prediction_norm`, not a state matrix for every token. In Phase A collect only the layers needed by its selected S arms; full 24-layer feature collection is a separate Phase B workload and must not block S execution.

For S8, retain only before/after state at observation boundaries and headwise cumulative decay. Transported W can be computed as `S_after - A*S_before`; verify cancellation-sensitive cases against a carried write accumulator in FP64.

## 6.4 Optional native-preparation tap

L0–L2 use gates/value magnitudes and do not require a recurrent scan if native prepared tensors are captured correctly. L3 needs residuals from three preceding GDN layers for each of eight attention layers: this covers all 24 GDN layers.

An optional explicit project-owned wrapper may capture normalized keys, post-convolution values, beta and g from the original forward, avoiding duplicate projections/convolution. A generic input hook alone does not expose post-convolution tensors.

Rules:

- No global mutable monkeypatch and no change to the checkpoint parameter namespace.
- Capture must be owned by an episode/invocation and receive original token IDs.
- Detach diagnostic features; preserve the navigation autograd path.
- Suppress repeated capture during activation-checkpoint recomputation.
- Native gates must use the same dtype/order as the qualified model; compare each captured tensor against `prepare_gdn_inputs`.
- The output/logits and native states must be unchanged with capture enabled.
- In serving, streaming append segmentation and convolution-tail lifecycle must match the offline reference.

Native taps are an optimization milestone after the reference passes, not an excuse to skip parity.

## 6.5 Storage and transfer

Compute scores on GPU. Transfer only FP32 priorities/compact metadata needed by the selector; avoid storing all 24-layer K/V tensors for an episode in RAM. For offline datasets, write bounded per-episode shards and release tensors promptly.

Report projection, scan, scoring, CPU transfer, sorting, KV gather and end-to-end timing separately. Count side-helper memory as well as retained attention KV.

---

# 7. Offline screening is not a substitute for executing a new policy

There are three distinct computations; never combine their scores in one unlabeled table.

| Level | What is recomputed? | What it establishes |
|---|---|---|
| Frozen-trace plan replay | Only candidate scores/plans from already saved features | Mechanism similarity on one reference distribution |
| Fixed-history policy replay | Full backbone on the same RGB/instruction/forced-action sequence, using the new selector | Action loss and hidden changes without trajectory divergence |
| Closed-loop rollout | New selector and policy choose their own actions | Actual navigation behavior |

Changing an upstream attention bank changes downstream hidden states and GDN scores. Therefore scores logged under S0 are not automatically the scores that S1 or a learned arm would produce under its own evolving memory. Reusing traces is legitimate for inexpensive analysis, but actual forward/rollout is required for performance conclusions.

## 7.1 Overlap metrics

For equal-sized banks report both:

- overlap fraction: `|A intersect B| / |A|`;
- Jaccard: `|A intersect B| / |A union B|`.

Aggregate per layer, admission and episode, including lower quantiles and maximal-disagreement windows, not only a global mean. Record source coverage and boundary-score margins.

**No 90% pruning rule.** At B=1200, 90% overlap still changes 120 rows. Their utility can be arbitrarily large. A reference test in the package constructs a 90%-overlap pair that loses its only useful row.

An exactly identical score ranking/plan can be deduplicated for the particular fixed trace panel. It does not prove two algorithms will agree on all future observations. Only structural equivalence plus matching numerical conventions justifies a general redundancy claim.

## 7.2 Suggested first screening endpoints

- Action-sequence negative log likelihood, normalized exactly as the reference model.
- Pre-action canonical-action disagreement where the existing serializer supports its original text action protocol.
- Per-layer selected-row identities and score statistics.
- Early-source fraction (frames 0–2), age, turnover, top-source concentration, per-spatial-cell occupancy.
- Scorer numerical stability and true runtime/memory.

Do not choose a method solely because CE improved. The final promotion decision includes closed-loop development results and uncertainty.

---

# 8. Checkpoints, splits and statistical protocol

## 8.1 Checkpoint roles

**Primary runtime-swap checkpoint:** M5_E1, hash verified against the repository's evaluation evidence. It is the immediate deployment target. A selector-seed change does not change model weights.

**Secondary:** the available M5c checkpoint, explicitly labeled random-retention-trained and with its original LR schedule. It is not neutral. For a schedule-matched factorial comparison, train M5c_E1 from the same base and recipe; do not silently substitute the historical M5c.

Recommended economical matrix:

1. All cheap heuristic arms: fixed-history replay on M5_E1 and, where accessible, M5c.
2. Closed-loop pilot: S0, C_RANDOM, C_RECENCY and nonredundant candidate finalists on M5_E1.
3. Cross-check top candidates on M5c as a sensitivity analysis.
4. Final matched training: baseline S0 and winning fixed/learned mechanisms using matched seeds/initialization/data/schedule.

A winner only on M5c may reflect adaptation to that policy, not an architecture-independent improvement.

## 8.2 Dataset partitions

Generate and hash partition manifests before label creation or hyperparameter selection.

- `calibration`: the 128 training episodes for scale/head diagnostics. These can be a declared subset of ranker fit data, never of selector holdout/test.
- `ranker_fit`, `ranker_dev`, `ranker_holdout`: disjoint **training-scene** partitions, approximately 70/15/15 by scene with episode-count balancing. Keep all instructions of a physical trajectory in the same partition. The selector's holdout is not a new unseen-backbone test if the SFT backbone already saw those scenes.
- `navigation_dev`: official R2R val_seen and a preselected English-guide RxR val_seen panel. Enumerate actual availability and IDs; do not hardcode 778 without checking files.
- `navigation_confirm`: R2R val_unseen and the pinned RxR evaluation protocol, used only after methods/hyperparameters are frozen for this campaign. Disclose prior development exposure to these benchmarks.

Initial label pilot: 256 training episodes, balanced across R2R/RxR and length strata, drawn only from ranker fit/dev partitions. Expand after label quality and throughput are measured; do not assume a few minutes of trace generation.

Proposed full label budget: up to 1,024 trajectories total, split by the predeclared scene partitions. This is a starting budget, not a requirement to exceed available data or balanced support. Record trajectory counts per domain and scene.

For a first closed-loop pilot, 256 R2R plus 256 RxR development episodes is reasonable if available. Promote finalists to the full development split or a larger preregistered panel before interpreting a two-point difference. Never take the first N scene-sorted rows as a random sample.

## 8.3 Statistical interpretation

For n=778, the maximum Bernoulli standard error is `sqrt(0.25/n)=1.79 pp`. This is neither a universal significance threshold nor the SE of a paired difference.

For paired binary outcomes, with discordance probability q and mean difference d, an independence approximation is:

`Var(mean difference) = (q - d^2)/n`.

Use raw paired outcomes to estimate it; cluster structure can increase uncertainty. Report:

- paired SR/SPL/NE differences;
- scene-cluster bootstrap with trajectory grouping and 5,000 resamples;
- per-scene and per-domain effects;
- exact McNemar as an additional episode-level analysis, labeled with its assumptions;
- training-seed results separately, not absorbed into an episode-only CI;
- selector-seed sensitivity for C_RANDOM and randomized logging, distinct from training seeds.

Do not average R2R and RxR SR without separately reporting each domain. A predeclared equal-domain development composite may be used for selecting a finalist, but it is not an official benchmark metric.

## 8.4 Finalist rule

Freeze the rule before confirmation:

- retain S0;
- retain best fixed GDN mechanism by the declared navigation-development endpoint;
- retain best simple learned mechanism (linear or residual-linear);
- retain best learned overall mechanism if distinct;
- include L2 when its causal role in the feature ablation remains unresolved.

If more than four learned/fixed finalists remain, use development utility/cost Pareto dominance to reduce the set. Do not promote all offline model variants to full SFT.

Two matched policy-training seeds (429/430) are the minimum initial confirmation for the baseline and chosen new method. Add a third if the intended claim is robustness or gains are small. Old S0-s429 can be reused only if its full recipe/source/runtime contract matches exactly.

---

# 9. Phase B: define the utility label before fitting rankers

## 9.1 What a local LOO label means

The primary computation-level target measures the effect of removing **one native visual KV row** from one attention layer while holding its query input and all other rows fixed. It is not a complete trajectory intervention, a memory capacity measure, or a guarantee of better action choices.

For query q and query head h, let:

`a_hi = softmax(logits_h)[i]`,

`o_h = sum_j a_hj V_hj`.

Removing logical row i and renormalizing gives:

\[
\Delta o_h=o_h-o_h^{(-i)}=\frac{a_{hi}}{1-a_{hi}}(V_{hi}-o_h).
\]

The numerically stable equivalent is:

\[
\Delta o_h=a_{hi}(V_{hi}-o_h^{(-i)}),
\]

where `o_h^(-i)` is evaluated with a masked softmax. Use the stable form near a_hi≈1.

With native elementwise output gate z_q and output projection W_O:

\[
\Delta y_{i,q}=W_O\,\operatorname{concat}_h(z_{q,h}\odot\Delta o_h).
\]

Primary local utility:

\[
\ell_{i,q}=\|\Delta y_{i,q}\|_2/\sqrt{d_{model}}.
\]

Bias in o_proj cancels. Cross-head contributions can cancel or reinforce: **do not sum per-head norm magnitudes as a substitute**. Expand GQA values through the actual head mapping. Remove the same logical token across all affected query heads because M5 evicts shared rows.

GateSight's published label omits its output gate for reasons specific to that analysis. Including the native gate here is intentional: this experiment targets actual Qwen3.5 attention-module output sensitivity, not a literal replication of its target.

## 9.2 Which queries?

Primary query is the serializer-defined pre-action anchor: the position immediately preceding the first action-response token. Do not assume it is the last RGB visual token; framing can intervene.

Use one query per observation/action decision for primary labels. A secondary sensitivity metric may include 12 stratified current visual queries, but it must not be mixed into the primary target without a new target ID.

## 9.3 Near/tail collection and separately versioned targets

This is an **offline training-label** design. The remaining trajectory length and query schedule are never supplied to the serving ranker.

For source group s, let first eligibility be a=s+R. Let T be the number of recorded observation/action decisions (including the final recorded decision), and set:

\[
M=\max(T-a,0),\qquad N=\min(16,M),\qquad F=M-N.
\]

A decision step u has a real pre-action query only if 0<=u<T. No artificial post-terminal queries are generated. If M=0, the source has no eligible future query support: all aggregate labels are null and the source is excluded from supervised fitting.

Use these three raw targets:

\[
Y_i^{near16}=\frac1N\sum_{u=0}^{N-1}\ell_{i,q_{a+u}},\quad (N>0),
\]
\[
Y_i^{tail16+}=\frac1F\sum_{u=N}^{M-1}\ell_{i,q_{a+u}},\quad (F>0),
\]
\[
Y_i^{remaining}=\frac1M\sum_{u=0}^{M-1}\ell_{i,q_{a+u}}
=\frac{N}{M}Y_i^{near16}+\frac{F}{M}Y_i^{tail16+}.
\]

For F=0, `tail_mean=null` with `tail_status=not_applicable`; the remaining mean equals the near mean. A nonexistent tail is not a zero-utility observation and must not contribute a training example for a tail model.

The existing H=16 learning question is preserved: first-round L0–L3 use **near16 only**. Collecting the tail now creates a separately testable label, not a compulsory two-output model or an implicit change to the primary objective.

### Sampling contract

- Near population offsets: `[0, N)` relative to admission.
- Tail population offsets: `[N, M)` relative to admission.
- Draw `min(4,N)` near offsets and `min(4,F)` tail offsets uniformly without replacement **within each stratum**.
- Use independent local deterministic RNG streams keyed by label seed, canonical episode/trajectory key, source group, and stratum. Never use utilities, S0 scores, candidate residency, patch ID, attention layer, or global mutable RNG state in offset selection.
- Reuse exactly the same source-level query offsets for every sampled patch and compared attention layer/checkpoint where the recorded history is the same. Store actual offsets, not just a seed.
- Existing valid near-offset lists may be imported to preserve already computed H=16 labels. Record `near_origin=imported`; do not redraw them just to add a tail. Otherwise use the versioned v2 sampler.
- Small supports (size <=4) are evaluated exhaustively.

Let `near_hat` and `tail_hat` be the sample means of the respective selected queries. Then:

\[
\boxed{\widehat Y_i^{remaining}=\frac{N}{M}\,near\_hat+\frac{F}{M}\,tail\_hat.}
\]

**Do not average eight queried values with equal weights when N and F differ.** That estimates an artificially balanced near/tail mixture, not the remaining-episode mean. Store per-query estimator weights: `1/n_near` for the near mean, `1/n_tail` for the tail mean, and `N/(M*n_near)` or `F/(M*n_tail)` for the remaining mean.

The remaining-horizon estimate is unbiased for the specified finite reference-context mean under the uniform stratum sampling and complete label access. This does **not** imply an unbiased full-policy return or remove reference-context dependence.

### Missingness and validity

Store each requested query separately. `raw_utility=null` means the label could not be evaluated; it is never treated as zero. If any requested query in a stratum is missing, that stratum's aggregate is null rather than a mean over surviving rows. The remaining aggregate is null unless every nonempty stratum needed by its weights is complete. A valid near aggregate can still be used when the tail is incomplete. Never silently redraw missing points until a candidate happens to be resident.

Separate source/trajectory truncation from label-computation failure. For a capped or incomplete reference trajectory, `remaining` means the remaining **recorded** decisions, not the unknown continuation after the cap. Save `trajectory_end_reason`, `trajectory_truncated`, T and support sizes. Report sensitivity to excluding capped traces.

### Rare evidence and dense quality checks

Four tail samples guarantee tail coverage, not detection of every important event. A single decisive query among 100 possible queries is sampled with probability only 4/100 under a uniform four-query sample. A long-horizon mean also dilutes brief high-utility events.

Keep raw per-query labels and log sampled maxima/quantiles only as **sample statistics**. They are not unbiased estimates of full-horizon maxima/quantiles. On the 64-episode dense-quality subset, choose source frames before looking at utilities and evaluate all near and tail queries for those selected sources within a measured resource budget. Compare sparse versus dense means, ranking agreement, Monte Carlo variation and the rate of missed rare events. Any cap on this dense subset must be fixed and reported, not applied after seeing utilities.

### Reference implementation

`reference/reference_features_labels.py` supplies `make_horizon_plan`, `remaining_query_weights`, and `aggregate_horizon_labels`. Tests cover no tail, short episodes, support exhaustion, imported near offsets, deterministic sampling without global RNG mutation, missing labels, and exact enumeration of the stratified estimator. It does not run attention or the simulator.

## 9.4 Calibration and artifact IDs

Version the label protocol as `gdn_utility_near_tail_v2`. Separate target IDs are:

- `shadow_fixed_query_loo_near16_mean_v2` — primary for all initial L0–L3 fits.
- `shadow_fixed_query_loo_tail_after16_mean_v2` — auxiliary diagnostic target, null when F=0.
- `shadow_fixed_query_loo_remaining_mean_v2` — secondary horizon comparison for at most two development-selected ranker architectures.

The `_v2` IDs describe the derived-data schema/protocol, not model or benchmark versions. Do not relabel censored `OBSERVED_LOO` samples as `SHADOW_FIXED_QUERY_LOO` targets.

Fit per-layer, **per-target** global training constants:

`kappa[layer,target] = max(median_positive_fit_raw_target, 1e-12)`; if there is no positive fit label, use the floor and explicitly flag the degenerate target.

`target_z = log1p(raw_target / kappa[layer,target])`.

**Aggregate raw utility first, transform afterwards.** A support-weighted average of separately log-transformed near/tail values is not the log-transformed remaining mean. Never reuse the near target's output calibration for a remaining-target checkpoint without an explicit documented control.

Store raw means, transformed target, support lengths, sampled counts, query steps, sampling metadata, context-generation IDs, valid/missing status and calibration hash. No per-frame/per-episode normalization that changes the meaning of fixed priorities across admission times.

## 9.5 Cost and future-data boundary

Collect near and tail labels in the same trace/label campaign so frozen backbone forwards and source-feature extraction can be reused. Cache or archive source candidates until their last selected tail query; free them afterwards. This is training-only I/O, not a larger deployed memory.

The number of source/query offsets grows from at most four to at most eight. Candidate LOO, gate/o_proj application and archive access may approach twice their previous cost, even though the shared forward is unchanged. Profile near and tail work separately. Do not promise negligible overhead or eliminate the tail silently on long RxR sequences.

Future query steps, T, N, F, end reasons and labels are forbidden in the serving `FeatureSchema`. Keep them in a separate label record and test that changing them cannot affect a runtime priority.

---

# 10. Random-bank logging, selection bias, and complete labels

## 10.1 Why random retention is not automatically unbiased supervision

For ideal independent continuous hash priorities and N eligible old rows, fixed top-B gives a particular row marginal inclusion probability `min(1,B/N)` at a specified boundary. This is a useful exploration property.

It does not imply that the empirical mean utility over the row's observed survival is an unbiased estimate of its specified near-16 or remaining-horizon utility:

- rows have different exposure patterns and may be censored before a decisive query;
- utility changes with age/query;
- retaining or deleting rows changes later hidden states and queries;
- averaging only observed queries uses a random denominator;
- a random-retention-trained checkpoint has a different representation distribution from M5_E1.

Inverse-propensity weighting can address a narrowly defined missing-observation problem under an explicitly fixed reference distribution. It does not recover arbitrary counterfactual hidden states, an uncompressed policy, or navigation rewards merely from B/N.

## 10.2 Logging policy versus checkpoint

Primary logger uses frozen M5_E1 weights with an explicitly versioned randomized old-bank selector on **teacher-forced training histories**, plus a smaller S0 logging mixture. This separates model weights from exploration mechanism. M5c is an optional alternate logger/checkpoint; record it separately.

Suggested mixture for the fit pilot: two deterministic random selector seeds and one S0 trace for selected training trajectories. A full threefold expansion of every trajectory is not mandatory: sample the mixture before measuring labels and record coverage.

Never treat an absent row as utility zero. Never forbid all S0 data: it is relevant deployment-distribution evidence, just insufficient by itself for broad alternative-selection labels.

## 10.3 Two supported label backends

### Backend OBSERVED_LOO — inexpensive pilot only

Compute exact LOO for rows actually resident at each reference query. Record a missing mask for every requested row/query. Report observed-lifetime means only as descriptive statistics. Fit pilot models with declared age/exposure weighting and missingness diagnostics.

Do not call this a complete oracle dataset or use it to assign utilities to every historical token. Failure of a pilot trained on these labels may reflect censoring.

### Backend SHADOW_FIXED_QUERY_LOO — primary confirmation dataset

To provide labels even when reference M5/random selection evicted the row:

1. On a training-only trace, archive the native attention K/V for selected source rows at their original birth computation. Archive original positions and relevant GDN features. This archive is an **offline label-generation artifact**, never available to a serving selector.
2. At each selected future pre-action query, freeze the reference layer input, post-RoPE Q, sigmoid gate, prefix/recent K/V, and its live old bank.
3. Construct a reference old-bank context containing candidate i at exactly the same pre-removal occupancy:
   - if i is already resident, use that context;
   - otherwise replace one old-bank row by i, with the displaced row chosen by a predeclared deterministic random rule independent of utility. Keep the total old rows at the reference occupancy, no more than B.
4. Compute output with i present, then the same fixed-query output with i removed. This is an actual row-LOO contrast, not an equal-budget utility swap; the equal-budget construction only avoids inserting i into an oversized reference context.
5. Use exact GQA mapping, causal eligible rows and native gates/o_proj. Do not alter later policy states or propagate this shadow query into the reference trajectory.
6. Use a background-context seed keyed by source/query/layer/context protocol, not by near-versus-tail target ID. If a query is reused, reuse its context and raw LOO result. Repeat with two background-context seeds on the label-quality subset to estimate context dependence.

Interpretation: these are local computation-sensitivity labels under reference queries and controlled candidate access. They do not describe full future trajectories generated with each candidate retained. That distinction is deliberate and is evaluated by the headroom/closed-loop gates.

The labels can still depend on the reference policy and background bank. Use held-out scenes and a second logger distribution to test transfer; do not claim an unbiased full-policy target.

## 10.4 Sampling rows without selection-score bias

For the first label dataset, sample 24 spatial positions per eligible source: two from each 5×5 tile of the 15×20 grid, using a stable seed. Select positions before computing utility and regardless of S0 score or residency.

On a 64-episode dense-quality subset, label all 300 positions for a preregistered, resource-profiled set of source frames spanning the episode, and evaluate their complete near/tail query supports. This supports sparse-versus-dense label-quality checks. It is NOT a full deployment top-B oracle unless every competing source/candidate required at the evaluated boundary is covered; sparse positions or missing source frames cannot produce a complete dense oracle plan.

For a full frozen-oracle policy diagnostic, either label every candidate required by that diagnostic horizon or restrict its claim to the completely covered subset. Missing labels must not be filled with zero and described as ground truth.

Candidate archive I/O and output-projection label compute are likely material costs. Measure them. Optimizations using output Gram matrices or sketches are allowed only after comparison to exact labels and ranking agreement.

---

# 11. Learned rankers L0–L3

## 11.1 Feature schema: F0_v1 unchanged, F1_v2 adopted

Features are available at token birth and pooled only by the predictor, not averaged over heads in advance. The fifth F1 channel is now **decayed-prior prediction magnitude**. All normalizers are fitted only on ranker_fit and frozen.

Primary F0 per GDN head (`F0_v1`) remains:

1. `log(max(beta,1e-6))`;
2. `log(max(-g,1e-8))` (decay magnitude, not the logarithm of a negative number);
3. `log(max(value_norm,1e-6))`.

F1_v2 appends:

4. `log(max(error_norm,1e-6))`;
5. **`log(max(prediction_norm,1e-6))`**;
6. `cosine_error` under the Section 4.4 convention.

Here `prediction = S_decay.T @ k`, `error = value - prediction`, and `prediction_norm=||prediction||2`. It is not a post-write state read. The mathematical identity `prediction=value-error` may be used when vectors are available, but validate cancellation-sensitive small-prediction cases against the direct read reference.

Why replace the old feature: for positive beta before floors, `log(||beta*e||)=log(beta)+log(||e||)`. Independent flooring means old stored channels are not exactly identical near zero, but the product supplies no new unfloored information. Exposing prediction magnitude is an explicit feature-design hypothesis, not a guaranteed gain. Keep the committed norm in raw storage for S0 and diagnostics.

### Raw feature record: gdn_raw_v2

For each original visual token and collected GDN layer, save FP32 per-value-head scalar arrays:

| Raw field | Definition |
|---|---|
| `beta` | native write gate |
| `log_decay_g` | native log decay g |
| `value_norm` | ||v||2 |
| `error_norm` | ||e||2 from the stable residual scan |
| `committed_norm` | ||beta*e||2 |
| `prediction_norm` | ||S_decay.T k||2 |
| `prediction_dot_value` | prediction dot value, for cosine convention auditing |
| `cosine_error` | Section 4.4 score, including its zero-value rule |
| `key_norm_sq` | native epsilon-normalized key norm squared |
| `prediction_near_zero` / `value_near_zero` | boolean diagnostic flags |

Do not average heads before storage. Keep canonical trajectory/token IDs, layer IDs, extraction backend, numeric constants and schema hash. Values above are small scalar arrays, not the full state or per-token K/V vectors. FP64 is the mathematical oracle, FP32 is the primary persisted feature format; compression to lower precision is a separate qualified approximation.

Save both `committed_norm` and `prediction_norm` during each extraction pass at the layers being collected. Phase A does not need all 24 layers; Phase B L2/L3 explicitly requires its wider layer scope. Do not claim an eight-layer Phase A shard is sufficient for a 24-layer L3 fit.

Flatten input in increasing GDN-layer order, then native value-head index, then the listed feature order. Center/scale each dimension with fit-partition mean/std and std floor 1e-4. No whole-episode or per-admission normalization of priorities.

Optional quality metadata, not primary model inputs: source group, patch row/column, near-zero flags, key norm and extraction error. No age, birth step, final episode length, future offsets, label values, instruction embeddings or spatial coordinates are silently added to L0–L3.

### Migration: do not recollect original navigation data

- If `prediction_norm` already exists, regenerate F1_v2 tensors from raw shards; no backbone replay is needed.
- If original value and error vectors are available, compute `prediction=value-error`, validate precision, and derive the new scalar offline.
- If suitable states and original keys are available, recompute the read from those artifacts.
- If only old scalar features remain, exact prediction-norm recovery is not generally guaranteed. Regenerate only the required feature shards from existing episodes; utility labels remain reusable if token identities, checkpoint and reference trace/backend are unchanged.
- Never overwrite old F1 arrays under the same schema name. The previous ordering is `F1_v1_legacy` and is not a valid input to an F1_v2 ranker.
- An existing L3 normalizer/predictor must not be reused with the replaced channel. Refit the feature normalizer and ranker under F1_v2. L0–L2 are unchanged.
- An existing near-label shard can be reused with its recorded query offsets and contexts; append independently sampled tail measurements without rewriting its raw near measurements.

`reference/reference_features_labels.py::raw_gdn_features` and `encode_features` provide standalone reference transformations. Their presence does not implement production trace collection or ranker fitting.

## 11.2 Exact dimensions and layer windows

Primary static models use **no age/birth-time feature**. This deliberately changes the earlier 99/291/579 dimension examples and avoids conflating learned GDN ranking with a recency prior.

| Arm | GDN layers per attention | Features | Width |
|---|---:|---|---:|
| L0 | immediate previous layer | F0 | 96 |
| L1 | immediate previous layer | F0 | 96 |
| L2 | previous three consecutive GDN layers | F0 | 288 |
| L3 | previous three consecutive GDN layers | F1_v2 | 576 |

For attention indices 3,7,...,31, the three-layer windows are [0,1,2], [4,5,6], ..., [28,29,30]. L3 therefore needs residual and decayed-prior prediction-magnitude features from all 24 GDN layers. L2 needs their gate/value outputs but not state reconstruction for the primary features.

All token identities must align across those layers. Do not concatenate features for different raster positions because of a reshape or padding offset.

## 11.3 Models

One predictor per full-attention layer; one scalar score per logical KV row shared across KV heads, matching M5's eviction operation.

- **L0:** `Linear(96,1)`; log-feature linear baseline.
- **L1:** `Linear(96,64) -> SiLU -> Linear(64,1)`.
- **L2:** `Linear(288,64) -> SiLU -> Linear(64,1)`.
- **L3:** `Linear(576,64) -> SiLU -> Linear(64,1)`.

No output sigmoid is required for the primary log-utility regression; rankings may use real-valued scores. Estimated counts across eight independent layers: L1 50,184; L2 148,488; L3 295,944 trainable parameters, excluding stored normalizers. Verify these programmatically.

Do not average heads before L0: a linear predictor can learn head-specific weights.

## 11.4 Linear-model controls

Fit these offline, with the same labels/folds:

- `L0_raw`: beta, g, value norm with global standardization.
- `L0_log`: the primary log-feature L0.
- `L0_S0res`: `score = log1p(S0/kappa_S0_l) + f_phi(x)`, where f is a zero-initialized linear correction and kappa_S0 is a positive training-only layer constant.

`L0_S0res` has the same ordering as S0 at initialization, subject to the floating-point rank-parity test. It is NOT a gate-only cheap predictor because computing its baseline requires S0 residual information.

Log inputs permit multiplicative forms after exponentiating a linear score. They do not exactly implement `mean_h beta_h ||e_h||` across heterogeneous heads without explicit nonlinear structure. The residual control supplies that baseline directly.

## 11.5 Age handling

Do not assert static features from different birth times are incomparable. They are comparable if the prediction target is defined and globally calibrated; what can fail is stale utility or inconsistent target normalization.

Primary static L0–L3 score once at first eligibility, using birth features and fixed normalizers; then retain that score for the token's remaining lifetime.

Optional log-recency arm:

\[
s_i=\log(u_i+\epsilon_u)+\lambda b_i,
\]

which ranks like `(u_i+epsilon_u)*exp[-lambda(t-b_i)]` for common lambda at current t. Birth index b_i is an observation index. A linear bonus on **raw** u is not equivalent.

This is an explicit recency prior, not a feature that must automatically be present. Compare lambda=0 against one train/dev-selected, preregistered lambda in a separate arm. Long-episode extrapolation and timing scale must be measured. Do not normalize b_i by final episode length.

Dynamic ages belong to L5. Observation age at initial R4 admission is constant four, so treating it as a major informative feature in a static-admission model is misleading.

## 11.6 Training objective and tuning budget

Primary target for the first architecture/feature sweep: `shadow_fixed_query_loo_near16_mean_v2` for **every L0–L3 model**. Primary loss: Huber with delta=1 on globally transformed log utility, equal weight per episode and equal total weight per R2R/RxR domain. Within an episode, average over sampled candidate rows and layers rather than letting the longest RxR trajectories dominate.

Optimizer proposal: AdamW, LR candidates {3e-4,1e-3}, weight decay 1e-2, batch 4096 labeled rows if memory permits, maximum 20 epochs, gradient clip 1.0, seed 429. Use early stopping on ranker_dev with patience 3, selecting by a predeclared top-budget utility-capture metric; use dev loss as tie-break. Report actual wall time and dataset rows.

Apply the same tuning grid and seed protocol to L0–L3. Run fitting jobs concurrently only if it does not change data/tuning fairness. A ranker-training seed is not a policy-training seed. Once the near16 comparison is complete, fit at most two development-selected architectures against `shadow_fixed_query_loo_remaining_mean_v2`, with the same features, data partition, optimizer grid and ranker seed. Refit output calibration for that target. Name runs with a target suffix (e.g. `L3__F1_v2__remaining`) rather than redefining L3 or confusing the horizon experiment with L4 navigation labels. Tail collection alone does not authorize doubling all closed-loop or policy-training runs.

Secondary objective, only after primary comparison: pairwise logistic ranking within the same reference decision/layer candidate set, weighted by a capped absolute utility gap. Do not change regression/ranking objective while also changing feature width and call the result a one-factor ablation.

## 11.7 Frozen ranker during matched policy training

During initial matched continuation/full SFT, freeze ranker weights and normalizers, compute feature inputs under no_grad, and let navigation gradients update the original backbone through retained K/V as in M5. The ranker's input distribution will drift as the backbone changes; measure prediction calibration/top-B agreement on a fixed training diagnostic panel at predetermined checkpoints.

If the drift is material, a second named experiment may recollect labels/refit between training stages. Do not silently refit a controller during one arm while leaving the control fixed.

---

# 12. Headroom and specificity controls

## 12.1 Oracle-label headroom is a three-stage diagnostic

**Stage O1 — local set-quality headroom.** On completely labeled candidate pools, compare S0, random, recency and label-top-B by captured utility. LOO labels measure a marginal deletion in a reference set; they are not additive utilities. Therefore also evaluate the *combined removal* of the discarded set in the frozen-query attention computation. Report both and their disagreement.

**Stage O2 — fixed-history causal-policy replay.** Construct static birth/admission priorities from future labels, maintain causal candidate eligibility/no resurrection, and replay the entire policy on the same observations/actions. Use actual original K/V recomputation, not merely the original Q and a new mask at the final layer. The policy uses a future-informed selector and is labeled `oracle_future=true, qualified=false`.

**Stage O3 — decision relevance.** Measure action CE and greedy action changes for those plans. Include an intervention choosing high-versus-low-label candidates within a fixed cardinality swap, because wholesale top-B plans can create large distribution shifts.

If labels give no headroom in O1, suspect an uninformative label or a scoring/coverage defect. If O1 improves but O2/O3 do not, the local-output surrogate may not align with navigation or may fail under joint deletion. Improve/replace the label before a large learned-ranking campaign. This is not a proof that all possible memory methods are useless.

Do not require Spearman(S0,Y)>0. It is a reported diagnostic, not a correctness condition implied by SFT M5-versus-M5c navigation scores.

## 12.2 Hidden-state predictor control

Fit on exactly the same rows/labels/folds:

- `H_linear`: ridge/linear predictor from the causal attention-layer input hidden state.
- `H_mlp_matched`: project that hidden state with a fixed seed-0 training-independent random projection to 576 dimensions, then use the same 64-wide MLP as L3.

The raw hidden dimension must come from the checkpoint config, not an unvalidated constant. Save projection seed/hash. Also record a raw-hidden linear result if memory permits; its capacity is different and must be labeled.

Interpretation:

- Hidden predictors much stronger at comparable cost weaken a claim that only GDN features make utility accessible.
- GDN predictors at equal quality with lower metadata or compute can still be useful.
- This control does not prove which representation carries causal navigation memory, because both are derived from the backbone.

Use the published KVzap code as a predictor-design reference, not pretrained weights for this model. "A few minutes" is not a guaranteed runtime: feature collection and labels can dominate fitting.

## 12.3 Offline metrics and uncertainty

Report Spearman, utility-weighted top-k capture, normalized regret, cutoff precision, and combined-output removal error. Compute at the deployment budget where complete candidate pools exist; for sparse sampled candidates, report explicitly sample-relative metrics rather than pretending they are top-1200 results.

Macro-average by episode and domain. Include per-layer results and scene-cluster intervals. One globally pooled correlation over millions of correlated token rows is not a convincing generalization result.

## 12.4 Promotion to closed loop

Predeclared learned finalists:

1. best L0/L0-residual simple model;
2. best L0–L3 overall model;
3. L2 if distinct, to resolve whether residual scan overhead is necessary;
4. best fixed S arm and S0 as method controls.

Not every offline predictor needs a closed-loop run. Use a fixed model-selection rule on ranker_dev and navigation_dev; do not choose a ranker architecture on val_unseen.

---

# 13. Conditional L4–L6 extensions

## 13.1 L4 — navigation-consequence labels

Keep the winning architecture/features fixed. Change only the target initially.

### L4-CE: local action-loss intervention

At a causal retention boundary, replace a candidate group C with equal-cost group D. Hold instruction, observation/action history and other cache/state elements fixed, replay the policy, and calculate the entire canonical expert action sequence loss:

\[
U_{C,D}=L_{action}(B\setminus C\cup D)-L_{action}(B).
\]

Positive U favors C. Grouping can use fixed spatial tiles so computation remains manageable. Record group size; no silent increase in memory when adding a candidate.

The CE target is supervised task relevance, not a closed-loop return. Fit a pairwise controller or aggregate token scores consistently with the group intervention. Do not divide a group utility equally across its members and call those independent ground-truth token utilities without a validating assumption.

### L4-return: optional later experiment

Use actual simulator continuations from the same state/history and equal-budget memory variants. Generate labels only on training scenes. Frozen navigation policy first; then optionally alternate recollection and policy adaptation in a separately named experiment.

Branches with different memories are not IID continuations from the same policy state. They do not inherit GRPO+F-B's fixed-state sibling-baseline argument. Keep selector supervision separate unless a correct joint retention/action policy objective is explicitly derived.

## 13.2 L5 — dynamic reranking

At each boundary reevaluate only:

`active_old_bank union newly_expired_visual_rows`.

Add current observation-age/token-age and causal memory-summary features if specified. Preserve original logical positions. A dropped row never returns unless a separately budgeted external archive is introduced (outside this plan).

Replace `RunningTopB` with a dynamic-candidate selector. Do not describe it as global top-B over all ever-seen tokens. Log re-ranking churn and metadata overhead. Use the same ranker/target when isolating static versus dynamic timing where feasible; retrain when feature semantics change.

## 13.3 L6 — state deficit measured once at admission

This is admitted earlier than full L5 when Phase A suggests prediction/state features add value. S1 beating S3 is a useful trigger for a pilot, not proof that delayed deficits will be helpful; check admission-deficit association with the target utility before a full run.

For each source visual token i, store its original normalized GDN key and value while its source is recent. At source expiry before observation t, compute:

\[
D_i^{adm}=\operatorname{mean}_h
\frac{\|v_{i,h}-S_{t^-,h}^\top k_{i,h}\|^2}
{\|v_{i,h}\|^2+10^{-6}}.
\]

Calculate once, attach it to the candidate's static feature record, and then free that source's key/value ring entries. The priority remains fixed after admission. L6 is therefore not automatically a dynamic-bank method.

Compare three same-metric timings when affordable: pre-write, immediate post-write, and admission. A result from changing both metric and timing cannot be attributed to timing alone.

### Memory calculation

With R=4, 300 visual rows/group, eight local scoring layers, Hk=16, Hv=32, D=128 and two-byte storage:

\[
4\cdot300\cdot8\cdot(16+32)\cdot128\cdot2
=117{,}964{,}800\ \text{bytes}.
\]

That is **117.965 decimal MB = 112.5 MiB**, for key/value ring payload only.

- FP32 payload: 225 MiB.
- All 24 layers at two bytes: 337.5 MiB.
- Expanding keys from 16 to 32 heads before storing increases the eight-layer two-byte payload to 150 MiB; avoid this.
- Add existing state helpers, convolution tails, IDs, candidate features and transient readouts separately.

A full read for one newly eligible 300-token source at eight scoring layers costs approximately:

\[
300\cdot8\cdot32\cdot128\cdot128
\approx1.258\times10^9\ \text{multiply-adds},
\]

or roughly 2.52 GFLOPs if a multiply-add counts as two FLOPs, excluding feature preparation and memory traffic. This does not determine latency percentage. Ring transfers, layout and synchronization can dominate. Profile isolated kernels and end-to-end policy latency before accepting a 0.5% claim.

FP32 is the diagnostic reference. Two-byte ring storage is an explicit approximation that must pass error/rank/behavior checks; do not compare a quantized-key admission metric to an FP32 birth metric and attribute all differences to time.

---

# 14. Proposed integration structure and interfaces

Paths below are **to be created**. Commands referencing them are specifications for the implementing agent, not existing executable tools.

```text
src/qwen_vl/research/gdn_sweep_contract.py
src/qwen_vl/models/gdn_features.py
src/qwen_vl/models/gdn_feature_tap.py
src/qwen_vl/stream/gdn_selectors.py
src/qwen_vl/stream/gdn_sweep_state.py
src/qwen_vl/research/kv_utility_labels.py
src/qwen_vl/research/gdn_label_horizons.py
src/qwen_vl/research/gdn_ranker.py
src/qwen_vl/research/gdn_ranker_dataset.py
scripts/memory_sweep/audit_manifest.py
scripts/memory_sweep/collect_traces.py
scripts/memory_sweep/replay_scores.py
scripts/memory_sweep/build_labels.py
scripts/memory_sweep/train_ranker.py
scripts/memory_sweep/evaluate_ranker.py
scripts/memory_sweep/oracle_headroom.py
scripts/memory_sweep/evaluate_policy.py
scripts/memory_sweep/aggregate.py
configs/memory_sweep/*.yaml
tests/memory_sweep/test_*.py
reports/memory_sweep/<campaign_id>/
```

## 14.1 Versioned contract

A new memory version may extend the existing spec, but old configs must load identically. Proposed fields:

```yaml
memory:
  mode: episodic_window_gdn_sweep_v1
  recent_groups_including_current: 4
  episodic_budget_tokens: 1200
  selection_unit: token          # token | frame300
  priority_timing: birth_fixed  # birth_fixed | admission_fixed | dynamic_boundary
  score_id: S0
  head_reduction: mean
  score_dtype: float32
  key_normalization: native_additive_eps_1e-6
  cosine_epsilon: 1.0e-6
  zero_value_threshold: 1.0e-6
  state_layout: value_heads_key_by_value
  scorer_backend: qualified_m5_legacy_v1
  logical_position_policy: preserve
  candidate_policy: resident_plus_newly_eligible
  tie_break: priority_desc_original_id_asc
  selector_seed: 429
  artifact:
    raw_feature_schema: null         # gdn_raw_v2 for derived traces, not a policy dataset
    feature_schema: null             # F0_v1 or F1_v2 when a learned arm requires it
    target_id: null                  # explicit near16 or remaining target
    feature_schema_sha256: null
    slow_head_mask_sha256: null
    normalizer_sha256: null
    ranker_weights_sha256: null
    target_definition_sha256: null
```

Do not stuff these settings into the G3 `research` section, which dispatches unrelated auxiliary training. The policy checkpoint stores the exact selector contract. Missing required learned artifacts must raise an error, never silently fall back to S0 or random.

For experiments on legacy checkpoints, use an explicit diagnostic override with `qualified=false`; do not rewrite the original checkpoint's navigation.json and claim existing qualification carries over.

## 14.2 Core typed records

```python
@dataclass(frozen=True)
class TokenIdentity:
    episode_uid: str
    source_group: int
    logical_token_id: int
    patch_row: int
    patch_col: int

@dataclass(frozen=True)
class FeatureSchema:
    version: str
    gdn_layers: tuple[int, ...]
    value_heads: int
    feature_names: tuple[str, ...]
    timing: str
    transforms: tuple[str, ...]
    normalizer_hash: str | None

@dataclass(frozen=True)
class BoundaryRecord:
    attention_layer: int
    current_group: int
    candidate_ids: tuple[int, ...]
    candidate_sources: tuple[int, ...]
    prior_bank_ids: tuple[int, ...]
    priority_version: str

@dataclass(frozen=True)
class SourceHorizonPlan:
    episode_key: str
    source_group: int
    admission_group: int
    episode_steps: int               # OFFLINE LABEL METADATA ONLY
    near_support: int
    tail_support: int
    near_offsets: tuple[int, ...]
    tail_offsets: tuple[int, ...]
    seed: int
    near_origin: str                 # sampled_v2 | imported
    recent_groups: int = 4
    near_horizon: int = 16
    samples_per_stratum: int = 4

@dataclass(frozen=True)
class UtilityQueryRecord:
    episode_uid: str
    reference_trace_id: str
    checkpoint_sha256: str
    attention_layer: int
    source_group: int
    token_id: int
    admission_group: int
    query_step: int
    offset_from_admission: int
    stratum: str                     # near | tail
    stratum_population: int
    stratum_sample_count: int
    inclusion_probability: float
    remaining_estimator_weight: float
    reference_context_id: str
    label_backend: str
    raw_utility: float | None
    missing_reason: str | None
    horizon_plan_sha256: str

@dataclass(frozen=True)
class UtilityAggregateRecord:
    episode_uid: str
    reference_trace_id: str
    attention_layer: int
    token_id: int
    horizon_plan_sha256: str
    near_mean: float | None
    tail_mean: float | None
    remaining_mean: float | None
    near_status: str                 # complete | missing | not_eligible
    tail_status: str                 # complete | missing | not_applicable
    remaining_status: str
    target_ids: tuple[str, ...]
    target_calibration_sha256: str | None

@dataclass(frozen=True)
class HorizonAggregate:
    near_mean: float | None
    tail_mean: float | None
    remaining_mean: float | None
    near_status: str
    tail_status: str
    remaining_status: str

```

Actual storage should use columnar arrays/Parquet rather than Python objects per token. Store source-level horizon plans once and reference their hashes from query/aggregate rows. Keep raw-feature, serving-input and offline-label schemas separate. The records define semantics and hashes; trajectory end/truncation metadata lives in the reference-trace manifest.

## 14.3 Function contracts

```python
def extract_features(prepared, initial_state, segments, schema) -> FeatureShard:
    """Causal reference features and boundary summaries; detached; bounded workspace."""

def score_static(shard: FeatureShard, spec: SweepSpec,
                 artifacts: ScoringArtifacts) -> torch.Tensor:
    """FP32 priority per original visual token, once at declared timing."""

def build_token_plan(layout, priorities, spec) -> LayerSelectionPlan:
    """Only original eligible IDs; deterministic; no resurrection."""

def build_frame_plan(layout, frame_priorities, spec) -> LayerSelectionPlan:
    """Atomically retain complete 300-row sources; same external plan interface."""

def local_loo_utility(query_bundle, original_rows, candidate_id,
                      include_output_gate: bool = True) -> UtilityResult:
    """Fixed-query, whole-row deletion across GQA groups after gate/o_proj."""

def make_shadow_context(reference_context, archived_candidate,
                         context_seed: int) -> ShadowContext:
    """At most B old rows before LOO; original positions; no live-policy mutation."""

def make_horizon_plan(episode_key: str, source_group: int, episode_steps: int,
                      seed: int, recent_groups: int = 4, near_horizon: int = 16,
                      samples_per_stratum: int = 4,
                      existing_near_offsets: tuple[int, ...] | None = None) -> SourceHorizonPlan:
    """Offline only; deterministic local uniform near/tail sampling; no utility inputs."""

def aggregate_horizon_labels(plan: SourceHorizonPlan,
                             utilities_by_query: dict[int, float | None]) -> HorizonAggregate:
    """Strict requested-query coverage; support-weight raw means; missing is not zero."""

def predict_priority(features, feature_schema, artifact) -> torch.Tensor:
    """Validate exact feature schema/hash; no gradients into backbone."""
```

All stateful scoring belongs to a session, not to a global module variable. S0 must remain able to use its legacy path and original state_dict names.

## 14.4 Files that require minimal controlled edits

- `episodic_contract.py`: dispatch new version/score IDs without weakening legacy validation.
- `episodic_forward.py`: feature/backend interface and per-layer cached features; no change to native parameter computations.
- `episodic_cache.py`: selector factory and immutable group priorities; admission rings only for L6.
- `nav_model.py` / session loading: exact learned-artifact loading and contract verification.

Do not refactor unrelated attention, serializer, data preparation or RL code in the same changeset.

---

# 15. Implementation tasks and acceptance tests

Implement Tasks 0–4 first for the S sweep; no utility-label shard or learned ranker is a prerequisite. Perform the applicable Task 8 qualification before any S closed-loop pilot. Only then enter Tasks 5–7 for Phase B and repeat Task 8 for learned finalists. Every task ends with a reviewable diff, tests and a saved result. The accompanying reference tests validate algebra only; production tests below still need implementation and execution on the user's environment.

## Task 0 — Freeze sources, identities and manifests

**Create:** audit script, campaign directory, source/checkpoint/data manifest JSON.

- [ ] Resolve/pin repo commit and expected native forward fingerprint.
- [ ] Verify checkpoint bytes; distinguish M5_E1, historical M5 and M5c.
- [ ] Enumerate R2R/RxR split counts and duplicate physical trajectories.
- [ ] Freeze scene partitions, development panels and selectors' canonical episode identity.
- [ ] Unit-test that changed scene paths do not silently change the new canonical hash control, or preserve the legacy identity and explicitly record it.
- [ ] Run S0 smoke through the frozen evaluator without overrides.

**Gate:** provenance complete; no assets assumed available merely because an absolute path appears in a report.

## Task 1 — Mathematical feature extractor

**Create:** `gdn_features.py`; tests for sequential/block scans, GQA head alignment and score formulas.

- [ ] Test beta=0, beta≈0, beta=1; g=0 and strong decay; zero/nonzero initial state.
- [ ] Test source lengths 1/63/64/65 and multiple observation/response boundaries.
- [ ] Implement residual solve without dividing by beta.
- [ ] Verify exact post-write identity and S1=S0+S9 at floating tolerance.
- [ ] Verify decay-transported decomposition, pure-decay case and cancellation cross term.
- [ ] Verify no input mutation and causal prefix invariance under future perturbations.
- [ ] Save both committed and predicted-value norms in gdn_raw_v2; verify the latter uses the decayed prior, not post-write state.
- [ ] Test F0_v1 is unchanged and F1_v2 channel 5 matches prediction norm, including zero/floored values.
- [ ] Test raw v2 can reconstruct F1_v1_legacy for analysis but an old-schema artifact is rejected by an F1_v2 ranker.

**Gate:** FP64 versus oracle `rtol=atol=1e-10` on small tests; FP32 errors reported with predeclared scale-aware thresholds. Production S0 remains the legacy scorer.

## Task 2 — Token/frame selectors and controls

**Create:** `gdn_selectors.py`; extend contracts and plan builder.

- [ ] Test fixed running top-B against independent batch top-B for every prefix.
- [ ] Test ties, B=0, undersaturated/saturated banks, source counts and sorted logical IDs.
- [ ] Test frame bank contains all 300 rows or none from each retained source.
- [ ] Test C_RECENCY semantics and confirm it differs from full Window8 in old text visibility.
- [ ] Test stable random scores are reproducible and distinct across selector seeds.
- [ ] Test no future IDs and no previously evicted row reappearance.

**Gate:** all selections match an independently implemented CPU oracle.

## Task 3 — Passive trace collection and optional native tap

**Create:** `collect_traces.py`, `gdn_feature_tap.py`, feature-shard schema.

- [ ] With passive diagnostics enabled, native logits and outputs are identical to diagnostics-disabled execution at the same runtime.
- [ ] Validate taps versus `prepare_gdn_inputs`, including conv tails and beta/g dtype.
- [ ] Verify no repeated writes to the trace during checkpoint recomputation.
- [ ] Verify bounded device and host working sets on long RxR fixtures.
- [ ] Confirm Phase A can run without any utility labels; record actual GDN layer coverage in every shard.
- [ ] Preserve original image/instruction/action files and manifests; all derived data go to a new campaign root.
- [ ] Audit that future episode length/query offsets cannot enter runtime feature tensors.
- [ ] Profile side-pass versus native-tap versions; report whether claimed savings actually occur.

**Gate:** no trace feature contains future data; all sampled row IDs align across layers.

## Task 4 — Source-aware offline score replay

**Create:** `replay_scores.py`, scorer-metric report.

- [ ] Replay S0–S9 and controls from common reference traces.
- [ ] Report overlap fraction/Jaccard separately, including local minima.
- [ ] Verify score-only replay is labeled as frozen-reference distribution, not a new policy rollout.
- [ ] Do not discard arms solely for >90% overlap.

**Gate:** offline diagnostics complete; shortlist rule preregistered.

## Task 5 — Exact local LOO labels

**Create:** `kv_utility_labels.py`, `build_labels.py`.

- [ ] Dense FP64 test: analytic row removal equals brute-force masked attention after native-style gate/o_proj.
- [ ] Test GQA shared-row removal, nontrivial cross-head projection and cross-head cancellation.
- [ ] Test attention probability near one using stable masked softmax rather than denominator clamping that changes the target.
- [ ] Native BF16 check: reconstructed unmodified module output matches the actual module under its qualified tolerance; record errors.
- [ ] Verify absence is marked missing, never zero; every shadow candidate is already temporally eligible.
- [ ] Verify fixed query support and equal occupancy before local deletion.
- [ ] Implement near/tail sampling with independent local RNGs, source-shared offsets and optional imported near offsets.
- [ ] Test M=0, M=1..4, M=16, M=17, M=20 and long episodes; no out-of-range/post-terminal query.
- [ ] Verify near/tail raw means and length-weighted remaining mean by exact enumeration on small finite populations.
- [ ] Reject missing planned queries or represent them explicitly as null; do not average only observed survivors.
- [ ] Test absent tail is null/not_applicable, remaining equals near for M<=16, and raw weighting occurs before log1p.
- [ ] Verify a repeated source/query/context is evaluated once and reused across target IDs; archive I/O never changes live KV/state.
- [ ] Profile forward, near-LOO, tail-LOO, archive I/O and storage separately; do not claim the added horizon is free.
- [ ] Test archiving/shadow probes do not mutate real policy cache or recurrent state.

**Gate:** no learned model fitting until label correctness and coverage report pass.

## Task 6 — Ranker fitting and hidden controls

**Create:** dataset/model/train/evaluation modules.

- [ ] Reject train/dev/holdout scene or physical-trajectory leakage.
- [ ] Fit training-only normalizers; verify transformed dev data does not alter fitted constants.
- [ ] Check L0/L1/L2/L3 feature dimensions and parameter counts; L3 uses F1_v2 at width 576.
- [ ] Make the first L0–L3 comparison use exactly the same near16 target, sample identities, folds and tuning budget.
- [ ] Refuse schema/hash or target-normalizer mismatches; do not reuse a legacy L3 artifact with the new channel.
- [ ] Store all three raw horizon aggregates; a remaining-target fit is a separately named experiment, not L4.
- [ ] Limit the secondary remaining-horizon comparison to at most two development-selected architectures.
- [ ] Verify zero-initialized residual predictor gives S0's selected IDs.
- [ ] Compare raw/log L0 inputs; explicit products are documented.
- [ ] Fit hidden controls and report parameter/compute differences.
- [ ] Save artifact schema, target version, hashes, fitting seed and tuning ledger.

**Gate:** selection-quality and uncertainty report available; no navigation-model parameters changed.

## Task 7 — Oracle headroom and fixed-history action replay

**Create:** `oracle_headroom.py`.

- [ ] Only completely labeled pools enter complete-oracle claims.
- [ ] Report marginal captured utility and combined-deletion output error separately.
- [ ] Execute true full-backbone replay for oracle plans; mark future-informed method unqualified for serving.
- [ ] Record action loss/argmax changes and shadow/reference distribution shifts.
- [ ] Apply the diagnostic decision tree rather than an unjustified universal positive-correlation gate.

## Task 8 — Serving qualification and closed-loop development

**Create:** explicit diagnostic override launcher; no overwriting old evaluation journals.

- [ ] Run 2-episode smoke, then replay fixtures long enough to saturate the bank and cover long histories.
- [ ] Verify reset/retry/session isolation, KV row bounds, logical positions and checkpoint reload.
- [ ] Compare offline/streaming per-layer bank IDs, score differences and action outputs.
- [ ] Re-run S0 on a fixed small panel to verify evaluator consistency.
- [ ] Execute predeclared development arms and paired analyses.

**Gate:** existing qualification tolerances are not relaxed to pass a method; failures are investigated and reported. Deliberately different policies are diagnostic runs, not existing-checkpoint qualification results.

## Task 9 — Matched policy-training confirmation

- [ ] Freeze the selected score/ranker contract before training.
- [ ] Baseline and finalists use identical initialization exposure, data order per seed, optimizer schedule and training action protocol.
- [ ] Resume validation includes ranker/head-mask/normalizer hashes.
- [ ] Record ranker feature-distribution drift during SFT at predetermined checkpoints.
- [ ] Check exported policy requires exactly its selected learned artifacts; no silent fallback.
- [ ] Evaluate final locked models on confirmation splits once for this campaign.

**Gate:** a final method claim requires closed-loop results, resource costs and matched controls, not just offline labels.

## Task 10 — Optional L4/L6 before L5

- [ ] Implement L6 ring with payload-size accounting, causality and dtype parity tests.
- [ ] Add L4 action-loss labels on training histories, exact text-action normalization and equal-budget swaps.
- [ ] Keep estimator/features fixed when testing a new target.
- [ ] Only then consider L5 dynamic selection and a new dynamic contract.

---

# 16. Execution budget and outputs

## 16.1 Recommended order

1. Read START_IMPLEMENTATION.md; freeze sources/data identities; run mathematical/scoring tests.
2. **Phase A first:** collect only the causal features needed for S0–S9 and same-checkpoint controls. No utility-label dataset is required.
3. Execute score replay, true fixed-history policy replay and qualified development rollouts for S finalists.
4. **Phase B only after its entry gate:** collect gdn_raw_v2 for the required 24-layer feature coverage and build near/tail labels together, with shadow-label coverage and headroom checks.
5. Fit L0–L3 and hidden controls against the **near16** target on identical partitions; L3 uses F1_v2.
6. Refit at most two chosen architectures against the separately stored remaining target; do not run a full architecture × horizon grid by default.
7. Closed-loop learned finalists plus S0 and best fixed scorer.
8. Matched continuation/full SFT of a small final set with matched baseline seeds; freeze contracts before confirmation.
9. L4 and/or L6 only if mechanism evidence justifies them; L5 last.
10. Locked full R2R/RxR confirmation and reporting.

This is not a promise to train every S and L arm end-to-end. Most arm discovery happens in cheap math/offline checks; performance claims require proper policy execution and training controls.

## 16.2 Profiling instead of speculative hours

Pilot the shortest, median and long-tail R2R/RxR sequences. Record:

- seconds per 1,000 original tokens and per observation;
- side-pass projection/scan time versus native tap;
- feature collection, transfer and label-generation throughput, with near/tail LOO and shared-forward costs separated;
- controller CPU/GPU time and number of synchronizations;
- training forward/backward tokens, not just number of optimizer steps;
- peak allocated/reserved GPU and sampled host memory;
- artifact storage size per episode.

Extrapolate with the actual sequence-length histogram and report a range. Historical M5_E1 wall time is a reference only; it is not a budget guarantee for 24-layer scans or full-output LOO labels.

For L3, explicitly compare extra utility/ranking quality to L2 against the cost of residual scans. For L6, report both equal-native-KV-row and approximately equal-total-memory comparisons if metadata is material.

## 16.3 Output layout

```text
reports/memory_sweep/<campaign_id>/
  README.md
  comment_decisions.md
  source_manifest.json
  checkpoint_manifest.json
  split_manifest.json
  environment.json
  calibration/
    slow_heads.json
    numeric_constants.json
  traces_manifest.json
  labels_manifest.json
  qualification/
  score_replay/
  ranker_fit/
  headroom/
  development/
  confirmation/
  metrics.parquet
  results.json
  REPORT.md
```

Every row of reported results identifies plan revision, checkpoint, selector/ranker ID, ranker training seed, policy training seed, selector seed, data split, raw/input feature schema hashes, target ID/calibration hash, reference-label backend, horizon support/sample plan, runtime source and qualifier status. Report available near/tail/remaining labels separately and preserve every missingness status.

## 16.4 Final paper tables implied by this plan

1. Gate × residualization factorial S0–S3 plus S9 complement.
2. HAM head reduction and slow-head controls.
3. Frame granularity and net-versus-write drift.
4. Learned ranking: linear/nonlinear, one/three layers, with/without residual state signals.
5. Label and state-conditioning controls for L4/L6 if used.
6. Matched closed-loop navigation, seeds and budget/latency.

Do not claim that a score improved memory retention solely because SR increased. Do not claim uniqueness over HAM/LoLA/GateSight from a renamed scalar. The defensible contribution, if experiments support it, is the specific relation between native GDN signals, bounded retention decisions and navigation consequences.

---

# 17. References and verified access status

## Project / implementation

**[R1] M5 scorer:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/models/committed_write.py

**[R2] Selection and plan invariants:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/stream/episodic.py

**[R3] Serving promotion:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/stream/episodic_cache.py

**[R4] Episodic training forward:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/models/episodic_forward.py

**[R5] Visibility contract:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/research/episodic_contract.py

**[R6] Native attention integration/backward:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/src/qwen_vl/stream/window_attention.py

**[R7] Pinned native Qwen implementation:**
https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py

**[R8] Experiment provenance:**
https://github.com/anhdao69/SimpleMemVLN/blob/fde1cbef213e27320920400a826addea8d5e114a/evaluation_reports/EPISODIC_MEMORY_EVALUATION.md

## Papers and reference implementations

**[P1] HOLA:** A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets.
Paper: https://arxiv.org/abs/2607.02303
Official code: https://github.com/scale-lab-ai/HOLA
Access: official README retrieved; relevant source paths advertised by that README. Re-pin the external repository before reuse.

**[P2] HAM:** Hybrid Associative Memories.
Paper reviewed: https://arxiv.org/html/2603.22325v2
Author research page: https://www.zyphra.com/our-work/ham
Access: paper and author page available; dedicated official implementation not verified.

**[P3] DeltaS:** Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video.
Paper reviewed: https://arxiv.org/html/2609.27470v1
Advertised code: https://github.com/MaumAI-Company/DeltaS
Access in this review: repository contents request returned an empty-repository error. Paper-defined local adaptation only; do not claim a code reproduction.

**[P4] GateSight:** Recurrent Memory Anticipates Attention Memory for Query-Agnostic KV Eviction in Hybrid LLMs.
Paper reviewed: https://arxiv.org/html/2610.14754v1
Advertised code: https://github.com/hideinsewer/GateSight
Access in this review: repository contents request returned 404. The estimator/utility-label specification comes from the paper, not inspected source.

**[P5] LoLA:** Low-Rank Linear Attention with Sparse Caching.
Paper edition reviewed: https://arxiv.org/html/2505.23666v2
Access: paper available. Used to locate related self-recall work, not as evidence that S9 duplicates LoLA's algorithm.

**[P6] KVzap:** Fast, Adaptive, and Faithful KV Cache Pruning.
Paper: https://arxiv.org/abs/2601.07891
Inspected official predictor code: https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/kvzap_press.py
File blob SHA observed: `f187f95f27dd8238b2efc135ff3bf3d8e7ba7a1e`.

## Supplied reference package

- `START_IMPLEMENTATION.md`: implementation-scope handoff and S-first task sequence.
- `HANDOFF_PROMPT.txt`: compact coding-agent instruction.
- `CHANGELOG.md`: approved revision-2 changes and migration rules.
- `reference/reference_math.py`: independent mathematical implementations, not a production integration.
- `reference/test_reference_math.py`: unchanged reference tests for residual block solve, write transport, immediate residual identity, local LOO after gate/projection, static aging equivalence, top-B and cost arithmetic.
- `reference/reference_features_labels.py` and its tests: F1_v2 transformations, raw-schema checks, near/tail scheduling, weighted aggregates and missingness.
- `reference/test_package_contract.py`: plan/config/schema consistency tests.
- `configs/data_schemas.yaml`: explicit raw-feature, input-feature and offline-label contracts.
- `configs/arm_registry.yaml`: machine-readable proposed arm metadata.
- `configs/campaign.example.yaml`: proposed campaign contract; scripts are not yet implemented in the repository.
- `evidence/reference_tests.txt`: freshly run reference-test output.

**Final handoff:** start implementing and qualifying the S arms now using the existing datasets/checkpoint. Phase B then uses gdn_raw_v2 and F1_v2 with near/tail collection, preserving near16 as the initial common target. Reserve horizon comparisons and expensive matched policy training for a small finalist set. No original navigation dataset correction or new image collection is required by these changes.
