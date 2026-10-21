# Task: Run diagnostics D-a … D-g without navigation-model parameter updates to explain why G3 + M5 failed, then write the report

You are a senior research engineer working in the SimpleMemVLN repository (GitHub `anhdao69/SimpleMemVLN`) on the local evaluation server (1× NVIDIA RTX PRO 6000 Blackwell, 96 GB; Habitat 0.2.4 simulator environment and Python 3.12 policy environment as documented in `EVALUATION_SETUP.md`).

**Goal:** determine *why* adding the G3 recurrent-state supervision to the M5 episodic memory lowered navigation success instead of adding to it. You will:
1. run seven diagnostics on already-trained checkpoints **without updating navigation-model or original auxiliary-head parameters**: the original pre-registered diagnostics D-a … D-f, plus exploratory gradient diagnostic D-g;
2. decide which of five hypotheses (R1–R5) the evidence supports, and separately assess the exploratory objective-interference evidence from D-g;
3. deliver a verified report with machine-readable results.

**No training of any navigation model is allowed in this task.** Small linear probes and ridge regressions fitted on frozen features are allowed. D-g permits gradient computation on existing navigation and auxiliary-head parameters, but **no optimizer steps, parameter updates, or new auxiliary-head training**.

Read this entire prompt before writing code. Where this prompt and the repository disagree, trust the repository code, stop, and record the discrepancy in the report. Do not invent a workaround that changes what a diagnostic measures.

---

## 1. Background you must understand first

### 1.1 Model and memory

- **Backbone:** Qwen3.5-4B hybrid, 32 decoder layers.
  - Full attention at layers 3, 7, 11, 15, 19, 23, 27, 31.
  - Every other layer is Gated DeltaNet (GDN).
- **GDN state:** each GDN layer keeps a recurrent state per value head, S ∈ R^{128×128}. There are 32 value heads. Keys come from 16 key heads repeated ×2 (`repeat_interleave`); keys are L2-normalized; the kernel is FLA `chunk_gated_delta_rule`.
- **Delta-rule update per token** (see `src/qwen_vl/models/committed_write.py`):
  - S̄ = exp(g)·S_prev
  - e = v − S̄ᵀk
  - S = S̄ + β·k·eᵀ
- **Frames:** each observation contributes 300 visual tokens, forming a 15×20 grid.
- **Window8 (G3, G1):** attention sees the instruction prefix plus the 8 most recent observation/action groups (lags 0–7). GDN state persists for the whole episode.
- **M5 (`episodic_window_v1`, R4/B1200, score `committed_mean_l2_v1`):** each attention layer l sees:
  - the prefix;
  - the 4 most recent groups (lags 0–3);
  - up to 1,200 older visual K/V rows chosen by a running top-B over a write-time priority.

  The priority is computed at the GDN layer immediately below layer l (pairs 2→3, 6→7, …, 30→31) and equals mean over heads of ‖β·(v − S̄ᵀk)‖, scored in FP32 in a no-grad side pass.
  - Code: `src/qwen_vl/stream/episodic.py` (`build_layer_plan`, `RunningTopB`, `plan_summary`), `src/qwen_vl/models/episodic_forward.py`, `src/qwen_vl/research/episodic_contract.py`.

### 1.2 G3: training-only supervision

- Location: GDN layer 24.
- **Probe rows:** read-only rows (k = e₁, v = 0, β = 0, g = 0) are inserted after anchor positions. The anchor is the last observation token of step t, computed by `observation_anchors`.
- **Read:** each probe reads S_tᵀq for a learned query.
- **Decoder target:** the decoder predicts the frozen VGGT-Omega layer-23 descriptors of frame s = t − lag, pooled into 5 regions (4 quadrants + global; `region_targets`).
- **Lags:** 8–32 only, hard-coded in `sample_pairs` in `src/qwen_vl/research/vggt_g3.py`. Quotas: 3 pairs in 8–15, 3 in 16–23, 2 in 24–32. At most 8 pairs per episode.
- **Loss:** cosine loss, λ ramped to 0.1.
- **Serving:** probes and heads are removed; nothing remains at inference.
- **Code:** `src/qwen_vl/models/vggt_g3_forward.py`, `src/qwen_vl/models/vggt_g3_heads.py`, `src/qwen_vl/research/vggt_g3_teacher.py`.

### 1.3 Results to explain

All runs: R2R val_unseen, 1,839 episodes, same local evaluation host. All use the same one-epoch recipe and seed 429 unless noted.

| Checkpoint | Memory | G3 loss | Successes | SR | Teacher-forced action loss (epoch) |
|---|---|---|---|---|---|
| G3 | Window8 | layer 24 | 956 | 51.98 | — |
| M5_E1 | R4/B1200 | none | 950 | 51.66 | 0.12658 |
| G3_M5 | R4/B1200 | layer 24 | 917 | 49.86 | 0.12690 |
| G3_M5_L8_L16_L24 (triple) | R4/B1200 | layers 8, 16, 24 | 916 | 49.81 | 0.12599 |
| G1 (reference) | Window8 | current-frame alignment (no effect) | 914 | 49.70 | — |

- Under Window8, G3 adds about +2.5 SR. Under M5, G3 subtracts about 1.8 SR.
- Teacher-forced action loss is unchanged, so the gap appears only in closed loop.
- A prior code audit found **no implementation bug**: recipe identity, probe math, loss normalization, data and teacher were all verified. Do not re-audit those unless a parity gate fails.

### 1.4 Hypotheses to test

- **R1 — Lag hole.** M5 keeps only lags 0–3 exact, but G3 supervises only lags 8–32. Lags 4–7 are neither exact nor supervised. G3's pressure on lags 8–32 weakens retention of lags 4–7 in the state.
- **R2 — Bank→state relay shortcut.** Six attention layers sit below the probe (3, 7, 11, 15, 19, 23). Under M5 these layers can see bank tokens of lag-8–32 frames. G3's loss can therefore be satisfied by attention retrieving source-frame tokens from the bank and re-writing them into S₂₄ near the anchor, instead of by long-term retention. Under Window8, no attention layer sees lag ≥ 8, so this route does not exist.
- **R3 — Score flattening.** G3's gradient retrains the parameters that compute M5's priorities: in_proj_qkv, β, g and conv of the scorer GDN layers 2–22, all below layer 24. This compresses the score spread, so selection drifts toward random. Matched random selection (M5c) costs about 4 SR.
- **R4 — Redundancy (secondary).** The bank duplicates what the G3-trained state already holds. The model attends to the bank less, but still pays the R4 cost.
- **R5 — Probe-induced train/serve selection drift.** Training-time hidden states include inserted probe rows, which shift FLA chunk boundaries. These perturbations flip near-tie bank selections that serving (no probes) does not reproduce.
  - Prior evidence on 17-frame fixtures: mean plan overlap with vs. without probes was 0.99994 on R2R and 1.0 on RxR. **R5 is expected to be negligible; D-f checks full-length episodes.**
- **Exploratory extension — objective interference (D-g).** Measure whether navigation and weighted auxiliary gradients conflict or whether auxiliary gradients dominate shared backbone updates. This is separate from the pre-registered R1–R5 decisions; final-checkpoint gradients alone do not establish the cause of the training regression.

---

## 2. Fixed inputs

### 2.1 Code

- Create branch `diag/g3m5-failure-20261019` from `origin/c2_vggt_omega_g3_m5_layers` at commit `c5d5422`. Its executable sources equal scientific source `f4d1c2f`.
- All new code goes under `scripts/diag/` (and `src/qwen_vl/diag/` if you need importable helpers).
- **Additive only.** Do not edit any existing file under `src/`, `configs/`, `train/` or `scripts/vln/`.
- Instrument with wrappers, forward hooks, or copied functions in your own files. Never monkeypatch native Transformers modules in place.

### 2.2 Checkpoints

Serving exports. Reuse the local, hash-verified copies listed in `evaluation_reports/evidence/LOCAL_R2R_FINAL_20261019.json` (fields `root`, `weights_sha256`). Re-verify each weight SHA-256 before use.

| ID | Local evaluation root | Weight SHA-256 (verify) |
|---|---|---|
| M5_E1 | `evaluation/m5_e1_epoch1_20261019` | `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6` |
| G3_M5 | `evaluation/g3_m5_epoch1_20261017` | `4e2d2095325fa779dcdece278f52213013d6889e72705ff860ab2eba1b13194f` |
| TRIPLE | `evaluation/g3_m5_l8_l16_l24_epoch1_20261018` | read from evidence JSON |
| G3 | `evaluation/g3_epoch1_20261018` | `a71be6b5291b6acddbeb389318befaded5818c5b1438d5d44cf351c64248333b` |
| G1 | download `anhdao69/Streaming_VGGT` folder `G1/epoch-1` | `a620ab17b1c0cbec6f7c5f95d67177a4b078ae53e83ea08c6696c5c4797a63ae` |

- Use **G1** as the Window8 reference without effective auxiliary supervision.
- **Do not use A0.** Its serving model needs GeoAddress adapters implemented only on branch `v2_vggt_omega_a1`.

**Training checkpoints with G3 heads** (optional but valuable). For G3 and G3_M5, locate the completed `checkpoint-3852` recovery checkpoints. These contain `research_aux.state_head.*` and their run directories are referenced in `reports/vggt_five/G3/implementation_report.md`, `reports/vggt_five/G3_M5/implementation_report.md` and their evidence JSON.
- If they are reachable, load the head weights.
- Verify that the backbone tensors equal the serving export **bitwise** before using them.
- If they are not reachable, mark every head-based metric "not available". Do not substitute a re-trained head. **D-g requires the original trained heads for each measured model; dummy or newly fitted heads are not allowed for D-g.**

### 2.3 Data panel (build once, shared by D-a, D-b, D-c, D-d, D-f; D-g uses a fixed subset)

- **Source split:** R2R VLN-CE val_unseen. The dataset gzip SHA-256 must be `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`. Use the evaluation scenes.
- **Selection:**
  - **240 episodes**, stratified by scene (11 scenes, proportional, at least 8 per scene) and by ground-truth path length tertile.
  - Prefer episodes with at least 34 ground-truth actions so that lags up to 32 exist.
  - Fixed RNG seed 20261019. Save the list as `panel_ids.json` with episode IDs, instruction IDs, scenes and lengths.
- **Expert trajectories (teacher forcing):**
  1. Check whether rendered val_unseen trajectory frames already exist in the same layout as training (`trajectory_data/R2R/val_unseen/<episode_id>/step_XXX_<ACTION>.png`). If they do, use them.
  2. Otherwise render them in the Habitat environment:
     - Use ground-truth actions from `R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz` if that file exists and its actions use 15° turns and 0.25 m forward steps.
     - If not, generate actions with Habitat's `ShortestPathFollower` along the reference path to the goal, with goal radius matching the training data convention.
     - Use the evaluation sensor configuration: 640×480 RGB, HFOV 79°, same agent height.
  3. **Rendering parity gate:** render 3 *training* episodes with your renderer and compare against the existing training frames. Mean absolute pixel difference must be ≤ 1/255 per channel. Report the value. If it fails, stop and report.
- **Manifest:** convert to the training manifest schema with `scripts/vln/prepare_r2r.py` (`--split val_unseen --frames-root …`), so `validate_episode` and `EpisodeSerializer.encode_episode` accept it unchanged.
- **D-e only:** a separate 600-episode closed-loop panel, described in its section.
- **D-g only:** preselect **32 episodes** from the shared 240-episode panel, balanced across scenes and trajectory-length tertiles, using seed 20261019. Save `dg_panel_ids.json` before any gradient measurement, and use exactly the same episodes for G3_M5 and G3.

### 2.4 VGGT targets

- Compute G3 targets for every frame of the panel exactly as in training:
  - teacher `OnlineG3Teacher` from `src/qwen_vl/research/vggt_g3_teacher.py`;
  - `spec` = the `research` block of `configs/vggt/g3_omega_h4.yaml`;
  - frozen VGGT-Omega-1B-512, FP32, single view, layer 23, no positional term, 5 regions × 2048.
- The teacher identity check in the constructor must pass. If teacher weights are not on this host, copy them and verify the identity hashes in the config. Never bypass the identity check.
- Cache the targets as FP16 `[frames, 5, 2048]` together with their identity.
- **D-g exception:** use detached FP32 teacher targets as in the original training loss. Recompute them from the verified teacher if only the FP16 cache is available; do not describe upcast FP16 targets as exact original FP32 targets.

---

## 3. Common infrastructure and parity gates (all must pass before any diagnostic)

### 3.1 Loading

- Load each checkpoint with its stored navigation configuration through `qwen_vl.train.vln_runtime.load_checkpoint`.
- Use `model.eval()`, BF16 weights and `torch.inference_mode()` for navigation-model forward-only measurements.
- **D-g exception:** keep `model.eval()`, but run the measured forward under `torch.enable_grad()` outside any active `torch.inference_mode()` context. Enable gradients on the parameters measured in D-g while preserving the original frozen vision/merger and detached teacher/query-context paths. No optimizer steps or parameter updates are permitted.
- Separate diagnostic-probe fitting in D-a remains allowed; it must not update the navigation model or the original auxiliary heads.

### 3.2 Offline teacher-forced forward

- Run `serializer.encode_episode(episode)` → `model(**data, episodic_diagnostics=diag)` for M5-family models, and the native Window8 path for G3/G1.
- This is the same computation used in training and in the export checks (see `scripts/vln/check_g3_m5_export.py::snapshot`).

### 3.3 Gate G-1: instrumentation is numerically transparent

For 2 panel episodes per model, logits with all hooks installed must equal logits without hooks **bitwise**.

### 3.4 Gate G-2: checkpoint parity with the frozen evaluation source

For each checkpoint, offline logits on the 128-step replay fixture from its evaluation root must match logits produced by that checkpoint's own frozen evaluation source (recorded in the evidence JSON) within the unchanged qualification tolerance: atol 0.75, rtol 0.03, RMS ≤ 0.15.
- Report RMS, max abs, and argmax flips.
- For M5-family models also report bank-selection overlap.
- Do not relax tolerances. If a checkpoint fails, exclude it and say so.

### 3.5 GDN state reconstruction at anchors

For GDN layers L ∈ {8, 16, 22, 24, 26}:
- Capture the decoder layer-L **input** hidden state for every token with a forward pre-hook on `text_model.layers[L]`.
- Recompute native GDN quantities with `committed_write.prepare_gdn_inputs`. This gives FP32 L2-normalized keys, values, g and β, and carries the conv tail.
- Scan the exact recurrence in serving segmentation order: prefix, then for each step obs_t, then resp_t. Use the internal 64-token blocks of `CommittedWriteScorer`.
- After each observation segment, snapshot S (FP32 `[32,128,128]`). This is the state at anchor_t, i.e. the state after the last observation token of step t, before the action.
- Keep, for every visual token i of every frame, its keys k̂_i, values v_i and β_i at layer L, FP16 on CPU is fine.

**Gate G-3:** on 2 episodes, compare the reconstructed state against native FLA. Either take the recurrent state from a streaming `StreamSession` run with forced expert actions, or run the native kernel with `output_final_state=True` on the same prefix. Relative Frobenius error must be ≤ 1% at every checked anchor (the same gate the M5 scorer passed: about 0.19%). Report the value.

---

## 4. Diagnostics

**Shared notation:**
- lag L = t − s, for anchor step t and source frame s.
- Regions r ∈ {Q1, Q2, Q3, Q4, Global}: rows 0:7 / 7:15 × cols 0:10 / 10:20 of the 15×20 grid. Global covers all 300 tokens.
- Report every lag-resolved metric for single lags 0–12, then buckets 13–16, 17–24, 25–32.
- Pre-defined lag groups for decisions: **A = 1–3, B = 4–7, C = 8–32.**

### D-a: What the state still holds, by lag (tests R1)

**Models:** M5_E1, G3_M5, TRIPLE, G3, G1. **Primary layer:** 24. Also report 22 and 26, and 8 and 16 for TRIPLE.

**Metric 1 — native self-recall error (primary, no learned parts).**
- For each pair (t, s) with 0 ≤ L ≤ 32 and each visual token i of frame s:

  ε_i(t) = mean_h ‖v_{i,h} − S_{t,h}ᵀ k̂_{i,h'}‖² / (‖v_{i,h}‖² + 1e-6), where h' = h // 2 is the key head.

- Aggregate per (t, s, region) as the mean over tokens in the region.
- Also compute ε at the anchor of the source step itself (L = 0, post-write) and report **forgetting** F(L) = ε(L) − ε(0) per pair.
- To bound cost, subsample anchors at every 2nd step and lags {1,…,12, 14, 16, 20, 24, 28, 32}. Document this.

**Metric 2 — decodability of the VGGT target from the state (learned linear probe).**
- **Features:** a fixed random Gaussian projection P ∈ R^{(32·128·128)×4096} (seed 0, shared across models) applied to vec(S_t). Implement it blockwise per head to avoid materializing P.
- **Model:** for each lag group A, B, C, fit a ridge regression from Pᵀvec(S_t) and a 16-dim lag embedding (sin/cos) to the 5×2048 target of frame t − L.
- **Split:** 2-fold **by scene** (no scene appears in both train and test). Choose λ by inner 3-fold CV on training scenes. Standardize features with training-fold statistics only.
- **Report:**
  - (i) mean cosine(pred, target) over 5 regions;
  - (ii) **within-episode retrieval accuracy**: rank the true frame s among all frames of the same episode by cosine with the prediction; report top-1 accuracy and MRR. This controls for the shared scene prior.
- **Current-view control:** the same ridge, but with features from the *current* frame's pooled values (mean of v_i over frame t's tokens, per region, at layer 24) instead of the state. **Memory-specific decodability = state probe − control.**

**Metric 3 — G3 head readout (only if training checkpoints are available; G3 and G3_M5).**
- Run the trained `NativeStateHead` exactly as in training: the probe path through `native_state_decoder` with real anchors and the 5-region queries.
- Report cosine to the target by lag, including lags outside training (1–7) as an out-of-range check.

**Prediction under R1:**
- G3_M5 − M5_E1: forgetting/ε **higher** in lag group B, and lower in group C (or decodability lower in B, higher in C).
- G3 − G1: lower ε / higher decodability in C.

### D-b: Is G3_M5's recall carried by the bank? (tests R2)

**Models:** G3_M5 (primary), TRIPLE, M5_E1 (control). G3 serves as the no-bank reference where noted.

**Global bank-off.**
- Re-run the offline forward with banks **removed at attention layers ≤ 23** (3, 7, 11, 15, 19, 23).
- Keep R4 recent groups everywhere and keep banks at 27 and 31.
- Implement this in a copied `episodic_text_forward` that replaces those layers' `LayerSelectionPlan`s with B0 plans (same `recent_groups`, empty token IDs, valid digest).
- Unit-test that B0 plans on all layers reproduce recent-only visibility.
- Recompute D-a Metrics 1–3 at layer 24.
- Also report action-distribution KL(original ‖ bank-off) at the decision positions, as the size of the perturbation.

**Targeted vs. matched-control ablation (decisive test).**
- Per episode, choose 4 anchors and, per anchor, 2 sources s in lag group C whose tokens are present in at least one bank at layers ≤ 23 at step t.
- **Targeted condition:** remove only frame s's tokens from the banks of layers ≤ 23 for query group t.
- **Control condition:** remove the same number of bank tokens from a different frame s' in the same lag group with a similar bank share.
- **Measure** the change in:
  - the G3-head cosine for (t, s) (if heads are available);
  - Metric-2 decodability for (t, s);
  - Metric-1 ε for frame s at anchor t.

  Report targeted − control.
- To compute only what you need, run the forward up to and including anchor t.

**Prediction under R2:**
- In G3_M5, targeted − control is clearly negative for head readout and decodability: the bank carries the recalled information.
- Metric 1 changes little, because self-recall is keyed by the source tokens' own keys and relayed content is stored under other keys.
- The bank-off decodability drop in lag group C is larger for G3_M5 than for M5_E1 (difference-in-differences, with lag group A as the within-model control).

### D-c: Where attention looks (tests R2; secondary evidence for R4)

**Models:** M5_E1, G3_M5, TRIPLE. G3 and G1 as Window8 reference.

- Recompute attention probabilities explicitly for a subset of queries: the anchor token and 32 seeded random visual tokens of the current group t, at every step.
- Wrap the project's attention function. For the episodic path, see the `episodic_recent_groups` / plan branch in `src/qwen_vl/stream/window_attention.py`. Capture post-RoPE Q and K and the visible key index set of the plan.
- Compute softmax(QKᵀ·scaling) over exactly the keys the kernel sees, in FP32. Respect GQA head mapping and the model's attention gating. Report mass before any output gate.
- **Gate:** on 2 episodes, the attention output reconstructed from your probabilities times V must match the kernel output for the captured queries within BF16 tolerance (relative error ≤ 1e-2). Report it.
- **Categories:** prefix; recent groups by lag 0, 1, 2, 3; bank tokens by source-lag bucket 4–7, 8–15, 16–32, 33+.
- **Report per attention layer:**
  - total mass per category;
  - mass per token per category (normalizes for bank composition).
- **Prediction under R2:** G3_M5 places more mass than M5_E1 on bank tokens from lags 8–32 at layers 3–23.
- **Prediction under R4:** G3_M5 places less total mass on the bank overall, especially at layers 27 and 31.

### D-d: Score spread and selection behavior (tests R3)

**Models:** M5_E1, G3_M5, TRIPLE. Optionally the older M5 (2-epoch schedule) as a second score-based reference.

- From `episodic_diagnostics` (keys `visual_scores` `[G,300]` per attention layer, and `plans`) on the panel, compute **per (episode, scorer layer)**:
  - coefficient of variation of scores among each admission's candidate set (current bank plus the group being admitted);
  - **normalized cutoff margin** = `plan.cutoff_margins` / std of that candidate set's scores (scale-free);
  - Gini coefficient of the selected scores;
  - `plan_summary` fields: distinct source groups, fraction from steps 0–2, mean age, turnover;
  - spatial histogram of selected tokens on the 15×20 grid, and the border share (columns 0–2 and 17–19);
  - concentration: maximum share of the bank coming from a single source frame.
- **Cross-model on identical inputs (per step, per layer):**
  - Jaccard of the selected ID sets, M5_E1 vs G3_M5;
  - Spearman correlation of the two models' scores for the same tokens.
- **Bridge to recall:** Spearman correlation between each token's write-time score and its Metric-1 ε at eviction time (the step its group leaves the recent window), per model.
- **Prediction under R3:** lower normalized margins and lower CV in G3_M5 at most scorer layers 2–22, but not at 26/30 (above the probe). Score–utility structure differs from M5_E1.

### D-e: Closed-loop selector swap (tests R3; behavioral)

**Models:** M5_E1 and G3_M5.

- **Panel:** **600** R2R val_unseen episodes, stratified by scene proportionally, seed 20261019. Save `de_panel_ids.json`. This may overlap the 240-episode panel.
- **Condition:** serve each model with `episodic_score: random_hash_v1`, `episodic_selector_seed: 429`, keeping R4/B1200 and everything else unchanged.
- **Implementation:**
  - Add an explicit diagnostic override in your own launcher. The frozen qualified evaluator must still refuse a changed navigation contract.
  - Label all outputs `diagnostic=true, qualified=false`.
  - Use each model's frozen evaluation source and the same simulator configuration (seed 42, cap 500, same dataset, scenes and scene paths). Record episode UIDs: the random hash depends on the UID, which contains the absolute scene path.
- **Baseline (score selector):** take the existing full-run journals restricted to the same 600 episode IDs. Verify IDs, UIDs and instructions match.
- **Determinism check:** re-run the original selector on 60 of the 600 episodes per model. Report the agreement rate of outcomes.
- **Report per model:** SR, SPL, OS, NE, forced STOPs, ΔSR(random − score) with exact McNemar p-values, and the interaction ΔSR(G3_M5) − ΔSR(M5_E1) with a scene-cluster bootstrap 95% CI.
- **Prediction under R3:** M5_E1 drops by ≥ 2 SR; G3_M5 changes little; interaction > 0.

### D-f: Probe-induced selection drift on full-length episodes (tests R5)

**Models:** G3_M5 and TRIPLE.

- For each panel episode, compute plans twice on identical inputs:
  - (a) the normal offline forward;
  - (b) the forward with G3 probe rows inserted at the training pairs: `sample_pairs(collection_trajectory_key(episode), len(steps), 429)`, anchors from `observation_anchors`, zero targets, as in `check_g3_m5_export.py::snapshot`.
- **Heads:** use training heads if available. Otherwise use a freshly initialized `NativeStateHead` / `build_state_heads` (seed 429).
  - Real-token outputs must not depend on probe query values. **Verify** that two different dummy heads give bitwise-identical real-token logits and plans before relying on dummy heads. If they do not, use only training heads.
- **Report:** per layer and per step, selected-ID overlap; logit RMS and max abs; argmax flips. Break down by episode length (R2R long tail) and by scorer layer (layers above vs. below the probe).
- **Decision:** mean overlap ≥ 0.995 → R5 negligible. Mean overlap < 0.98 in any layer → R5 material.

### D-g: Navigation–auxiliary gradient interaction (exploratory)

**Models:** G3_M5 (primary) and G3 (comparison). Use only their original trained auxiliary query/decoder heads, with backbone weights verified bitwise against the corresponding serving export. If a required trained head is unavailable, mark that model's D-g unavailable; do not initialize or fit a replacement.

- **Panel:** the 32 episodes preselected in `dg_panel_ids.json`. Use identical teacher-forced images, instructions, actions and source/read sampling across the two checkpoints. Save and hash the panel before measuring gradients.
- **Execution:** keep `model.eval()` and use `torch.enable_grad()`. Use the actual differentiable native recurrence, not the no-grad state-reconstruction helper from Section 3.5 or saved inference-mode tensors as a substitute. Do not run an optimizer, update parameters, or train new heads. Use ordinary detached FP32 teacher targets.
- **Fixed visibility:** for each G3_M5 episode, create one immutable set of M5 selection plans and reuse those exact plans for both gradient calculations. Keep scorer/selector computation detached and preserve the existing detach on current native-Q context. G3 uses its unchanged Window8 visibility.
- **Losses:** compute navigation and weighted auxiliary gradients separately, preserving the original per-action and per-pair normalization and the completed checkpoint's auxiliary coefficient. Verify the coefficient from the saved recipe/schedule; the expected completed-run value is 0.1. Record the actual denominators and coefficient.

  \[
  g_{\mathrm{nav}}^{(\ell)}
  =\nabla_{\theta_\ell}\mathcal L_{\mathrm{nav}},
  \qquad
  g_{\mathrm{aux}}^{(\ell)}
  =\nabla_{\theta_\ell}\bigl(\lambda\mathcal L_{\mathrm{aux}}\bigr).
  \]

- **Report per episode and decoder layer:** gradient cosine, both gradient norms, and

  \[
  \frac{\|g_{\mathrm{aux}}^{(\ell)}\|}
  {\|g_{\mathrm{nav}}^{(\ell)}\|+\epsilon}.
  \]

  Report cosine as unavailable when either gradient norm is zero; distinguish an unused gradient path from a numerical failure. Do not average over undefined cosines.
- **Global clipping diagnostic:** separately report the auxiliary head's contribution to the combined global gradient norm and the clipping factor implied by the original training threshold. Combine navigation and weighted auxiliary gradients on shared parameters before computing the global norm, count tied parameters only once, and do not actually clip or update parameters. Label per-episode clipping estimates as per-episode; do not present them as measurements of the original distributed accumulation window.
- **Safety checks:** verify finite gradients and unchanged model/head parameters before versus after D-g. Save the parameter-integrity evidence. Release each episode's graph/gradients before proceeding to avoid accumulation and unnecessary memory retention.
- **Interpretation:** consistently more negative gradient cosine or greater auxiliary-gradient dominance in G3_M5 than G3 supports an objective-interference hypothesis. Final-checkpoint gradients alone do **not** establish the cause of the training regression. Do not add a new hard pass/fail threshold or silently change the R1–R5 decision table.
- **Outputs:** save per-episode/per-layer measurements, paired cross-model summaries, uncertainty estimates, loss-normalization metadata, clipping diagnostics and parameter-integrity checks in `results.json` and the report.

---

## 5. Statistics (apply everywhere)

- The unit of resampling is the **episode**, with scene-stratified cluster bootstrap (2,000 resamples) for 95% CIs.
- Cross-model comparisons are **paired** on the same episodes and inputs.
- The primary contrasts (Section 6) are pre-registered. Everything else is exploratory and must be labeled as such.
- Report effect sizes with CIs, not only p-values.
- Never select episodes, layers or lags after seeing results. If you change a definition, keep the original result and report both.
- Report n (episodes, pairs, tokens) for every number.

---

## 6. Pre-registered decision table (fill it in exactly)

| Hypothesis | Primary contrast(s) | "Supported" if | "Refuted" if |
|---|---|---|---|
| R1 lag hole | D-a Metric 1 at layer 24: Δ = G3_M5 − M5_E1 in mean forgetting F, lag group B (4–7) and group C (8–32) | Δ_B > 0 and Δ_C < 0, both 95% CIs excluding 0 (Metric 2 must agree in sign) | Δ_B ≤ 0 with CI excluding positive values |
| R2 relay shortcut | D-b targeted − control in G3_M5 (head readout or Metric 2, lag group C); D-c per-token bank mass, lag 8–32, layers 3–23, G3_M5 − M5_E1 | targeted − control < 0 (CI excludes 0) **and** D-c difference > 0 (CI excludes 0) | targeted − control CI includes 0 **and** D-c difference ≤ 0 |
| R3 score flattening | D-d normalized cutoff margin, G3_M5 − M5_E1, scorer layers 2–22; D-e interaction | margin lower at ≥ 5 of 6 layers (CIs excluding 0) **and** D-e interaction > 0 (point estimate ≥ +1.5 SR) | margins not lower **and** D-e interaction ≤ 0 |
| R4 redundancy (secondary) | D-c total bank mass (all layers), G3_M5 − M5_E1 | lower, CI excludes 0 | not lower |
| R5 probe drift | D-f mean selected-ID overlap | < 0.98 in any layer | ≥ 0.995 in all layers |

Anything between "supported" and "refuted" is **inconclusive**. Say so explicitly. Multiple hypotheses can be supported at the same time.

**D-g is exploratory:** summarize its objective-interference evidence separately, with effect sizes and uncertainty. Do not use it to silently redefine the pre-registered R1–R5 decision thresholds. If trained heads are unavailable, report D-g as unavailable rather than negative evidence.

---

## 7. Deliverables

Create `reports/diagnostics/g3m5_failure_20261019/` on the diagnostic branch containing:

1. **`REPORT.md`** with this structure:
   1. **Executive summary:** a filled decision table (R1–R5: Supported / Refuted / Inconclusive, each with the key number and CI), a separate exploratory D-g finding or unavailability statement, a single-sentence answer to "why did G3 + M5 fail?", and the recommended next action (see 7.2).
   2. **Provenance:** branch and commit, every checkpoint with weight SHA-256, teacher identity, dataset/config hashes, environment (Python, PyTorch, Transformers, FLA, Habitat), GPU, panel files and their hashes.
   3. **Parity gates G-1, G-2, G-3, and the D-c / D-f gates:** values and pass/fail. Include D-g trained-head identity, finite-gradient and unchanged-parameter checks.
   4. **One section per diagnostic D-a … D-g:** exact definition as implemented (any deviation from this prompt stated first), tables, figures, n, CIs, the prediction, and the observed outcome.
   5. **Synthesis:** how the diagnostics fit together. State which mechanisms explain the size of the drop and which do not. Distinguish teacher-forced evidence (D-a … D-d, D-f), exploratory gradient evidence (D-g), and closed-loop evidence (D-e).
   6. **Limitations:** e.g. teacher-forced expert trajectories vs. closed-loop distribution, the linear-probe ceiling, a single training seed, the dummy-head assumption, and the limits of inferring training dynamics from final-checkpoint gradients.
   7. **Reproduction:** exact commands.
2. **`results.json`:** every number in the report, keyed by diagnostic, model, layer and lag.
3. **Per-pair / per-episode CSVs** (Parquet is fine) for D-a, D-b, D-c, D-d and D-e, plus per-episode/per-layer gradient measurements for D-g.
4. **Figures (PNG, labeled axes and units):**
   - D-a: ε / forgetting vs. lag, all models, layer 24; decodability vs. lag; retrieval top-1 vs. lag.
   - D-b: bank-off and targeted/control deltas.
   - D-c: attention mass by category and layer.
   - D-d: margin and CV per scorer layer; spatial heatmaps.
   - D-e: bar chart with CIs.
   - D-f: overlap by layer and step.
   - D-g: layerwise gradient cosine and auxiliary/navigation gradient-norm ratio, with uncertainty and G3_M5 versus G3 comparisons.
5. **Code:** all scripts under `scripts/diag/`, a `README.md` with run order, and unit tests for:
   - B0 plan construction;
   - state reconstruction (gate G-3);
   - attention reconstruction;
   - dummy-head invariance;
   - D-g loss/gradient decomposition, identical selection plans, zero/unused-gradient handling, and unchanged parameters.

Commit and push to `diag/g3m5-failure-20261019` only.

### 7.1 Do not draw conclusions the data cannot support

- Teacher-forced metrics do not prove closed-loop causes.
- A linear probe failing does not prove information is absent.
- Single-seed checkpoints do not establish general effects.
- D-g measures gradients at the inspected checkpoint; it does not reconstruct the original training dynamics or prove a cause of the closed-loop regression.
- Say this where relevant.

### 7.2 Recommended-action mapping (state which applies)

- **R1 and/or R2 supported** → a visibility-aware G3 is indicated: supervise lags ≥ R (4–28 for R4) and down-weight pairs whose source tokens are in banks below the probe layer.
- **R3 supported** → selection must be protected or changed: recall-gap selection, a frozen scorer, or no G3 gradient into scorer layers.
- **R4 only** → the memories are redundant. Keep G3 with Window8 and do not combine.
- **Nothing supported and D-g inconclusive or unavailable** → treat the drop as plausibly within closed-loop and seed noise. Recommend a second seed of M5_E1 and G3_M5 before any redesign.
- **D-g indicates objective interference** → recommend a subsequent controlled loss-weight/gradient-isolation experiment, not an automatic redesign. This task must not perform that training experiment; state that final-checkpoint gradient evidence is suggestive, not a confirmed root cause.

---

## 8. Execution order, budget and stop rules

1. Panel and rendering gate, VGGT targets, loading, gates G-1 to G-3. Estimate: ≤ 6 h.
2. D-d and D-f. These are cheap and reuse the offline forwards.
3. D-a, then D-b, then D-c.
4. D-g on its fixed 32-episode subset, before D-e. Run gradient diagnostics separately from concurrent closed-loop GPU workers; release graphs between episodes and verify parameters remain unchanged. If required trained heads are unavailable, record the blocker and continue with the remaining diagnostics.
5. D-e. Closed-loop: 2 models × 600 episodes plus 2 × 60 determinism episodes. Run 4 workers per GPU as in prior evaluations, after its launcher passes a 2-episode smoke and D-g has completed or been marked unavailable.
6. Report.

**Stop and report, rather than improvise, if:**
- any parity gate fails;
- the rendering gate fails;
- the teacher identity check fails;
- a checkpoint hash mismatches;
- D-g produces nonfinite gradients, mismatched plans between its gradient calculations, or any parameter mutation;
- a diagnostic would require changing frozen source files or evaluation tolerances.

**Write a short status note** in the report folder after each step, with what finished, numbers so far and blockers, so partial progress is never lost.

---

## 9. Must not happen

- No training or fine-tuning of navigation weights; no new auxiliary-head training. D-g may compute gradients only, with no optimizer step, parameter update, or actual gradient-clipping mutation.
- No edits to existing source, configs, frozen evaluation sources or evaluation journals. Never overwrite existing `evaluation/` directories; write new outputs under a new root such as `evaluation/diag_g3m5_20261019/`.
- No relaxing of replay or qualification tolerances. Diagnostic closed-loop runs are labeled unqualified.
- No VGGT or any teacher in any *policy* forward. The teacher only produces targets.
- No post-hoc changes to panels, lags, layers or decision thresholds without reporting the original pre-registered result alongside.
- No claims beyond the evidence. Every number in `REPORT.md` must be traceable to `results.json`.