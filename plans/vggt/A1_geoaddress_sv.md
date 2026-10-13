# A1 — GeoAddress-SV: single-view descriptor relations on actual native Q/K

> **For agentic workers:** this is an extension of A0's CONFIGURATION, not training from A0's finished checkpoint. Keep A0/A1 runtime modules identical.

**Goal:** test whether geometry-informed teacher descriptor relations improve learned native memory addressing under the user's single-view restriction.
**Architecture:** exactly A0 at layer24; a training-only relation objective on selected native visual Q/K.
**Tech stack:** shared offline single-view VGGT cache plus native differentiable policy.
**Spec:** master, teacher and A0 plans.

## Global constraints

Same one-layer/rank/head adapter as A0. No simultaneous-view VGGT call, new pose/depth head, hard physical-correspondence label, protection loss, state decoder, lane or extra runtime router. Teacher descriptors for two times are obtained independently. No target features enter student inputs.

## Review focus

- A1 is descriptor-relational GeoAddress, NOT certified place correspondence.
- Student old key is the causal key computed at its own source position, not re-encoded with current/future observations.
- Teacher similarity uses unpositioned PCA descriptors; do not use G1's added sin/cos targets.
- No source key or selected adapter gradient is detached inadvertently.
- Pair/read positions must map to visual tokens, not the action cue or same grid offsets in packed storage.

## 1. Why this version is explicit

Full physical GeoAddress would need reliable geometric cross-view labels. The user excluded multiview extraction now. Silently using same-pixel coordinates or nearest descriptors as ground-truth same-place labels would be wrong.

A1 therefore transfers a SOFT distribution of teacher descriptor similarity. It is implementable with the exact shared single-view cache and does not invent correspondences. It tests fusion at the address operator, with a narrower interpretation. A failure does not settle whether a true geometry-correspondence version would work; success does not prove metric spatial keys.

## 2. Teacher distribution

For each common delayed pair `(tau,t)`, independently cached descriptors Z_tau,Z_t have shape[300,256]. Choose32 current image query indices by the shared 4x8-rectangle spatial selector. All300 source cells remain candidates.

`P_T[i,j]=softmax_j(cos(Z_t[i],Z_tau[j])/T_T)` with `T_T=0.1`.

Targets FP32/detached. No hard negatives/thresholded matches. No teacher view ordering because each descriptor extraction had S1. A shuffled-source diagnostic must permute teacher/student indexing deliberately and record it; do not silently train another version under A1.

## 3. Student distribution

At GDN24 after A0's residual and before repetition into value heads, take raw heads0/1. Apply the SAME additive-epsilon L2 normalization used by native FLA for the diagnostic distribution, without modifying the runtime arrays twice.

For each head h:

`L_h[i,j]=(q_hat[t,i,h] dot k_hat[tau,j,h])/T_S`, `T_S=0.1`.

`P_S,h=softmax_j(L_h)`.

Loss per pair:

`l_pair = mean_{h,i} KL(P_T[i,:] || P_S,h[i,:])`.

Use `log_softmax` and a numerically stable teacher entropy term. Sum over pairs, normalize by global valid-pair count, lambda max0.1 and common ramp. No native sqrt(128) factor added to this auxiliary cosine-softmax beyond the explicitly chosen temperature; native kernel still uses its original scale in forward. This distribution is a training probe of address relations, not an assertion that GDN uses softmax attention.

### 3.1 Cost

At8 pairs,32 query cells,300 sources,2 raw heads:153,600 student score values per episode (FP32 about0.59 MiB). Q/K selected-row gathers and autograd add cost, but no LxL all-episode score matrix is built. Compute each pair or a small batch inside selected layer24's checkpoint function and return only a scalar sum. Teacher descriptors/relations persist only for the active microbatch.

## 4. Gradient pathways

Teacher probabilities are detached. Student q_t and k_tau remain in the graph; this directly supervises selected address variables and adapters, and indirectly their earlier hidden/conv contributors. Original nav loss continues through all native heads. Values/gates are not separately geometry-aligned but can change through navigation and shared inputs.

Hard top-k admission/eviction is absent. No custom surrogate ranking gradient is required. Pair sampling is fixed metadata; it is not trained by the loss.

## 5. Quality control before the single full A1 run

Inspect descriptors/relations on a TRAIN calibration panel and a separate held-out diagnostic panel: entropy, similarity spread, permutation correctness and visualized patch relations. No claim of physical matching accuracy without true correspondence evidence. Uniform/high-entropy rows are logged; they are not automatically relabeled “no match”. If extraction is degenerate/nonfinite or all relations are numerically uniform, stop A1 before production and report a target failure. Do not switch to joint-two-view teacher or DINO without approval because that changes the planned experiment.

Use the same 8 delayed pairs as G3/R1; do not condition selected pairs on validation outcome. Navigation trains on all actions even if an episode has no valid delayed pair.

## 6. Tasks

### A1.1: soft relation loss
**Files:** `models/geoaddress_loss.py`; `tests/vln/test_geoaddress_relations.py`.
**Interface:** `relation_loss_sum(q_selected, k_source, teacher_probs, pair_index) -> Tensor[]`.
- [ ] Test exact KL=0 when distributions match and nonzero gradients otherwise.
- [ ] Finite distributions, teacher stop-gradient, wrong shapes rejected, 32-query/all300-source indexing.
- [ ] Reordering source cells together in student keys/teacher probabilities preserves loss; permuting only one changes it.

### A1.2: actual native address capture
**Files:** `models/gdn_research.py`, `models/research_forward.py`.
- [ ] Selected q/k taken after adapters, before FLA normalizes; normalize once for probe computations.
- [ ] Contrast with an intentionally detached old-key test; earlier source gradients must exist in the correct fixture.
- [ ] Current/future gold-action perturbation cannot change earlier selected Q/K in a causal forward.
- [ ] Return scalar from checkpoint function, no persistent tensor hooks.

### A1.3: paired run and serving
- [ ] A0/A1 initial native+adapter parameter hashes match; only training-only objective metadata differs.
- [ ] Global pair denominator, ramp and optimizer grouping tests pass.
- [ ] Serving A1 loads the same architecture class as A0 and never loads teacher targets.
- [ ] Qualify longest resource/save/reload then launch A1 independently.

## 7. Evidence to report

Primary A1–A0 comparison: paired SR/SPL and runtime under identical deployment budget. Also report native selected-head relation KL, old-key address behavior and frozen recall diagnostics when available. Improving descriptor matching alone is insufficient. Do not add a post hoc anti-interference term and call the model the same A1. No exact physical-address or “memory bottleneck solved” claim from this screen.
