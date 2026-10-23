# M5 learned GDN rankers (Phase B, L series) — labels, rankers, qualification and matched training

Report date: **2026-10-23 (UTC)**. Plan: [`plans/M5_GDN_S_L_Technical_Plan.md`](../../../plans/M5_GDN_S_L_Technical_Plan.md)
(revision 2, Sections 9–12, 14–16). Branch: `m5_gdn_l_sweep` (from `m5_gdn_s_sweep` `a2d075f`).
Sealed sources (detached worktrees on the cluster):

| Role | Commit | Worktree |
|---|---|---|
| Label collection | `9099f5dadee194a36b68000725cba78291524c7b` | `m5-gdn-l-label-source-9099f5d` |
| Rankers, contract, serving, campaign | `022e526b279b6d7af0f0c5d861429a0d59a7f7e5` | `m5-gdn-l-source-022e526` |
| Holdout report (diagnostic only) | `e42f297` | `m5-gdn-l-report-e42f297` |

Campaign root: `/mnt/data/vmo-ai-task/anhdh35/m5-gdn-l-sweep-20261023`.

**No navigation result is claimed.** Offline ranking quality below is measured against exact local
attention-output utilities on held-out training scenes; it is evidence that the learned priority is far
better aligned with that label than any fixed S score, not that it improves navigation. Closed-loop
evaluation of the trained L arms (R2R/RxR) has not started.

---

## 1. Status at a glance

| Lane | Allocation | Arms in order | State at report time (2026-10-23 23:05Z) |
|---|---|---|---|
| B | Slurm job **4925**, worker-2, 4×H100, 768 GiB | L3 → L1 | Shared qualification **passed** (22:41Z); **L3 production training since 22:50:11Z**; checkpoint 10 audited |
| A | Slurm job **5248**, worker-2, 4×H100, 480 GiB | L2 → L0 | Waiting (as designed) for S lane A to reach `lane-complete` (S7 at update ~1,800/3,852); then starts automatically |

- Labels: 1,024 episodes, **119.8 M exact labels**, 0 missing (Section 3).
- Rankers (frozen, holdout capture@96, S0 = 0.243):

  | Arm | L3 | L2 | L1 | L0 |
  |---|---:|---:|---:|---:|
  | Capture@96 | **0.765** | 0.755 | 0.707 | 0.652 |

  Every L arm beats S0 and every other fixed GDN score on 9 unseen scenes, with scene-bootstrap intervals far
  from zero (Section 4).
- Every L arm passes real-model serving replay on M5_E1. Recovery passes. L3's first updates use exactly
  M5_E1's data order, LR schedule and non-memory config.
- **Cost.** L arms recompute GDN features in no_grad side passes. L3 is ~15.3 s/update vs ~13.7 s for S arms,
  about **16.5 h** per run. Rough completion estimates (not guarantees):

  | Lane | Arm | Estimated completion (UTC) |
  |---|---|---|
  | B | L3 | Oct 24 ~15:30 |
  | B | L1 | Oct 25 ~07:00 |
  | A | L2 | Oct 24 ~22:00 |
  | A | L0 | Oct 25 ~14:00 |

- Each finished arm is published automatically to the public Hub repo
  **[`anhdao69/M5_L`](https://huggingface.co/anhdao69/M5_L)**, one folder per arm (Section 8).

## 2. Request and scope

The user asked (2026-10-23) to implement and run Phase B — learned rankers L0–L3 — with the goal of
beating the S series, to train the L arms on both 4×H100 lanes in sequence (lane B now, lane A after S7), and
to push the weights to `anhdao69/M5_L`. Decisions taken:

- **Matched training exactly like the S arms.** Each L arm reuses M5_E1's recipe (fresh Qwen3.5-4B, joint
  R2R + English RxR15deg, seed 429, M5_E1's data order, 3,852 updates); only the memory section differs. S0 is
  M5_E1 and is not retrained. The ranker is frozen (plan 11.7).
- **Primary target `shadow_fixed_query_loo_near16_mean_v2`** for every L model; tail and remaining
  aggregates are stored. The optional remaining-target refit of ≤ 2 architectures was not run.
- **Controls are offline only** (L0_raw, L0_S0res, H_linear, H_mlp_matched), as the plan specifies.
- **L4–L6 were not started.** O1 combined-removal, O2 future-oracle replay and O3 action-CE diagnostics were
  not run (Section 9).
- **Hub:** public repository `anhdao69/M5_L` (created 2026-10-23, rev `d28c7fd`), one folder per arm,
  mirroring `anhdao69/M5_S`.

## 3. Utility labels (plan 9–10, Task 5)

**Label.** For a visual K/V row *i* of source observation *s*, attention layer *l* and a later decision *u*:
the gated attention output at the pre-action anchor `observation_ends[u] − 1` is recomputed in FP64 from the
module's own Q/K/V (q_norm/k_norm, partial RoPE, GQA 16→4, sigmoid output gate) over the exact visible set
(prefix + bank + recent R=4 window). The label is
`‖W_O concat_h(z_h ⊙ Δo_h)‖ / √2560`, with `Δo_h = a_hi (v_hi − o_h^(−i))` in the stable masked form.
Fallbacks when `a > 0.5` are an exact masked recompute.

- Resident rows are removed from the reference context.
- Non-resident rows replace one utility-independent `displaced_row` of a full B=1200 bank (shadow context).

**Logging.** The frozen M5_E1 runs teacher-forced training episodes under a per-episode logger mixture:
random_hash_v1 seeds 7001 (330 episodes) and 7002 (365), and M5_E1's own S0 memory (329). Passive hooks never
alter execution (tested bitwise).

**Horizons and sampling.** For each source:
- M = T − a and N = min(16, M).
- At most 4 near and 4 tail query offsets, drawn from a local SHA-256 RNG.
- Raw near/tail means are support-weighted into remaining; log1p(Y/κ) is applied only after aggregation, and
  missing is never zero.
- 24 rows per source (2 per 5×5 tile of the 15×20 grid).
- Dense quality subset: 64 episodes × 4 spanning sources × 300 rows × every query × context seeds 429/430.

**Partitions.** Scene-level, created before any label. The S calibration episodes are excluded from the
holdout.

| Partition | Scenes | R2R | RxR | Rows |
|---|---:|---:|---:|---:|
| ranker_fit | 43 | 360 | 360 | 1,480,488 |
| ranker_dev | 9 | 76 | 76 | 296,400 |
| ranker_holdout | 9 | 76 | 76 | 325,968 |

**Collection.** A 256-episode pilot ran first (quality and throughput gate), then all 1,024 episodes. Totals:
- 29.3 M tokens and **119,759,808** exact labels;
- 3.0 GPU-hours on 4 H100s (0.37 GPU-s per 1k tokens), 139 GB.

| Check | Result |
|---|---|
| Missing labels (requested query × row × layer) | **0** in all 1,024 episodes; dense labels finite exactly where requested |
| Reconstruction vs the module's own output at every anchor | max relative error **0.78 %** (BF16 model vs FP64 recompute) |
| Same check with the bank ablated (wrong context must be detected) | detected in ≥ **95.5 %** of queries in every episode-layer (1st percentile 100 %) |
| Near-one stable fallbacks | 36 (exact masked recompute) |
| Zero or non-finite near16 aggregates | none of 16.8 M complete values; median 2.1e-6, p99 2.1e-4, max 0.058 |

Tests (CPU, independent oracles):
- brute-force per-head LOO;
- resident LOO equals rerunning the module without the row;
- shadow displacement;
- near-one fallback;
- GQA cancellation;
- horizon supports for M = 0…137, exact-enumeration unbiasedness, and log-after-aggregation;
- hooks are passive, and reconstruction equals the module output on a tiny real Qwen3.5 under C_RANDOM and S0.

## 4. Rankers (plan 11–12, Task 6)

**Features.** gdn_raw_v2 is extracted by one shared `GdnFeatureExtractor` (same segmentation and carried state
as the S scorers), used identically by the label collector, offline training and serving.
- F0_v1 = log β, log(−g), log‖v‖.
- F1_v2 = F0_v1 + log‖e‖, log‖S̄ᵀk‖, cosine error.
- Inputs are flattened in GDN-layer, head, channel order and standardized on the fit partition (std floor
  1e-4). No age.

**Protocol (identical for every model).**
- Target, loss and weighting: near16 target, Huber δ=1 on log1p(Y/κ_l), κ_l = median positive fit target;
  equal weight per domain and per episode.
- Optimization: AdamW with LR ∈ {3e-4, 1e-3}, wd 1e-2, batch 4096, ≤ 20 epochs, clip 1, seed 429; one predictor
  per attention layer.
- Model selection: early stopping (patience 3) on dev **capture@96**, tie-break dev loss; the LR is chosen per
  arm by mean dev capture.

capture@96 is the raw utility kept by an episode's top-96 scored rows divided by the best possible 96 rows,
where 96 = 4 frames × 24 sampled rows. It is the sample-relative analogue of the B=1200 budget, macro-averaged
over episodes, then domains, then layers.

**Holdout results** (9 unseen scenes, 152 episodes; never used for fitting, early stopping or LR choice).
Δ vs S0 comes with a scene-cluster bootstrap (5,000 draws) 95 % interval.

| Scorer | Parameters | Holdout capture@96 | Spearman | Δ capture vs S0 [95 % CI] |
|---|---:|---:|---:|---:|
| **L3** (F1_v2, 3 GDN layers, MLP) | 295,944 | **0.765** | **0.786** | **+0.523** [0.513, 0.531] |
| L2 (F0, 3 GDN layers, MLP) | 148,488 | 0.755 | 0.780 | +0.512 [0.505, 0.519] |
| L1 (F0, 1 GDN layer, MLP) | 50,184 | 0.707 | 0.733 | +0.464 [0.455, 0.472] |
| L0 (F0, 1 GDN layer, linear) | 776 | 0.652 | 0.695 | +0.409 [0.402, 0.415] |
| L0_S0res (S0 + zero-init linear) | 776 | 0.646 | 0.687 | +0.403 [0.396, 0.410] |
| L0_raw (raw β, g, ‖v‖) | 776 | 0.638 | 0.651 | +0.395 [0.388, 0.401] |
| H_linear (attention-input hidden, 2560) | 20,488 | 0.807 | 0.824 | +0.564 [0.554, 0.573] |
| H_mlp_matched (fixed seed-0 projection → 576, L3 head) | 295,944 | 0.802 | 0.809 | +0.560 [0.549, 0.569] |
| S2 β‖v‖ | — | 0.251 | 0.238 | +0.008 [0.005, 0.012] |
| **S0** (M5) | — | **0.243** | 0.223 | — |
| S1 ‖e‖ | — | 0.165 | 0.084 | −0.078 |
| S5 mean cosine error | — | 0.149 | 0.036 | −0.094 |
| S3 ‖v‖ | — | 0.138 | 0.041 | −0.105 |
| S4 min cosine error | — | 0.133 | −0.014 | −0.109 |
| random | — | 0.117 | 0.001 | −0.126 |
| S9 (1−β)‖e‖ | — | 0.093 | −0.171 | −0.150 |
| *position prior* (fit mean log-utility per raster position) | 300 | 0.438 | 0.400 | +0.195 [0.185, 0.207] |
| *newest source* (recency) | — | 0.217 | 0.103 | −0.026 |

Per layer, L3's holdout capture ranges from 0.64 (layer 23) to 0.93 (layer 3), and every layer is far above
S0. L3 holds up across domains, e.g. layer 3: R2R 0.923, RxR 0.846. Chosen LRs: L0 3e-4; L1, L2, L3 1e-3.
Best epochs range from 4 to 20 (20 is the predeclared cap). κ_l is between 1.1e-6 and 7.9e-6.

Interpretation:
- **The learned GDN priorities are far better aligned with the exact local utility than every fixed GDN
  score.** The fixed scores span 0.09–0.25 capture; L0–L3 span 0.65–0.77. Capacity ordering is monotone,
  L3 ≥ L2 > L1 > L0. The extra F1_v2 residual channels (L3 vs L2) add +0.010.
- **Specificity.**
  - A static raster-position prior already captures 0.438: some image positions are systematically more
    useful. Every L arm exceeds it by a wide margin, so GDN features carry information beyond position.
  - The hidden-state controls are stronger still (0.80–0.81 vs 0.77). As plan 12.2 notes, this weakens any
    claim that only GDN features make utility accessible. The L arms remain the cheaper GDN-only mechanism,
    and the plan's closed-loop question is about them.
- Offline capture does not by itself predict navigation. S2 is slightly above S0 offline, but its R2R
  val_unseen SR (50.35 %) is below M5_E1 (51.66 %).

**Pilot.** The pilot (256 episodes, dev only) gave the same ordering (L3 0.758, L2 0.752, L1 0.704, L0 0.659,
S0 0.255). It was used only to validate the pipeline; the final rankers were refit on all 1,024 episodes from
the sealed source.

## 5. Contract, offline training path and serving

- **Registry.** `LEARNED_REGISTRY` (L0–L3) is separate from the S `ARM_REGISTRY`, so every S arm, its parser
  path and its serialized dict are unchanged. The real-model check: sweep S0 equals legacy S0 exactly in
  serving replay.
- **Embedded artifact.** The memory section embeds the frozen artifact: normalizers plus weights as exact
  little-endian FP32 base64, 16 KB (L0) to 1.6 MB (L3), with its canonical SHA-256.
  - Parsing validates schema, window, widths, layer coverage, shapes and finiteness, and fails closed. There is
    no fallback to S0 or random.
  - `navigation.json` of every checkpoint is therefore self-contained.
- **Offline training.** `OfflineRankerFeatures` runs no_grad side passes on the detached input of every GDN
  layer in each attention layer's window, outside the checkpointed call. It scores 300-row frames with the
  FP64 frozen ranker before the window's last layer and builds the same immutable plan as the token S arms.
- **Serving.** `GdnSweepSessionState.feature_append` carries extractor state across prefix, observation and
  response tokens, scores each observation's 300 rows once its window is complete, and releases the features.
- **Tests (CPU).** On a tiny 8-layer Qwen3.5 ([GDN×3, attention]×2), for every L arm:
  - offline training equals incremental serving (logits, bank IDs, priorities);
  - plans ignore the future; gradients never reach the ranker; checkpointing is exact;
  - parameter counts match the plan;
  - tampered, foreign-schema or incomplete artifacts are refused;
  - the zero-initialized residual reproduces S0's order;
  - fit constants never see dev rows;
  - aggregates match the reference implementation; scene leakage is refused;
  - each L arm's resolved config equals M5_E1's except memory and loader fields.

  Full CPU suite at `022e526`: **270 passed**, 3 skipped (GPU-only). The campaign's cpu-suite gate, run on the
  cluster with GPUs visible (memory_sweep plus episodic contract/selector/attention/forward/committed-write
  tests), had **273 passed**.

## 6. Ranker input drift (plan 11.7)

The rankers are fitted on M5_E1 features; matched training starts from the base model and the ranker is frozen.
On identical histories (C_RANDOM banks, selector seed 7001), the frozen L-ranker scores computed from other
checkpoints' features were compared against those from M5_E1. Each cell gives score Spearman / static top-B bank
overlap (B = 1200 over frames older than R = 4), averaged over the 8 attention layers.

Qualification gate: 24 ranker_dev episodes, final artifacts.

| Features from | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| M5_E1 (reference, self-check) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| base Qwen3.5-4B (training start) | 0.539 / 0.421 | 0.527 / 0.402 | 0.532 / 0.416 | 0.502 / 0.403 |

Trial with the pilot artifacts on 4 dev episodes:

| Features from | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| S1 final (trained from base, same recipe) | 0.816 / 0.698 | 0.793 / 0.685 | 0.803 / 0.703 | 0.804 / 0.701 |
| S3 final (trained from base, same recipe) | 0.822 / 0.701 | 0.798 / 0.684 | 0.806 / 0.697 | 0.805 / 0.697 |

One epoch of M5 training moves the GDN feature distribution substantially. Models trained from base on this
recipe end much closer to M5_E1 than the base is, so the L arms should see largely M5_E1-like features by the
end of training, but not identical ones. This is a stated limitation of the frozen-ranker design. The trained
L arms' drift will be measured after training. Per the plan, any relabel/refit would be a separately named
second-stage experiment.

## 7. Qualification (shared gates, lane B, sealed source `022e526`)

The lane B controller (Slurm step 4925.180) ran the gates in order once the label step exited. Each gate's
result is in `shared/gates.json`; QUALIFIED.json binds them to the source commit and the four artifact SHAs.

| Gate | Passed (UTC) | Result |
|---|---|---|
| labels | 22:09:18 | 1,024 episodes complete, 0 missing, recon ≤ 0.78 %, ablation detection ≥ 0.955, source/selection/M5_E1 SHAs bound |
| dataset | 22:16:09 | 2,102,856 rows; scene and trajectory partitions disjoint; rows/manifest SHAs recorded |
| rankers | 22:26:08 | 4 GPU shards; L0–L3 parameter counts equal the plan; fixed-score holdout baselines recorded |
| arm-configs | 22:26:16 | `arm_L{0..3}.yaml` embed the artifacts; KV guard 4,784 for every arm |
| drift | 22:31:01 | base vs M5_E1 (Section 6) |
| cpu-suite | 22:31:54 | 273 passed |
| provenance | 22:32:17 | base model revision, M5_E1 weights/publication, manifest and counts, transformers pin, native forward, baseline ancestry: all true |
| serving-replay | 22:34:01 | below |
| recovery | 22:41:06 | L3 interrupted at update 2 and resumed vs uninterrupted reference: equal losses, weights relative L2 2.5e-8 (S campaign: 2.5e-8) |

**Serving replay on M5_E1** (fixed expert histories). The offline training path is compared with incremental
serving using the same weights and episodes. Every L arm passes on both domains:
- argmax agreement 1.0 and decoded actions identical;
- maximum logit error 0.50–0.66, RMS 0.034–0.053;
- selected-bank overlap ≥ 0.966 (mean ≈ 0.989).

Sweep S0 is bit-identical to legacy S0 in every metric. Its RxR maximum-abs failure (1.98) is the inherited one
documented in the S report and is not a learned-arm effect. Plan-ID mismatches (447–448/512) are the same
BF16 near-tie divergences seen for legacy S0.

The L arms keep only **31–42 %** of S0's bank: they select substantially different memory. Their action NLL on
M5_E1 (R2R 0.121–0.125, RxR 0.315–0.319) is slightly above S0 (0.113 / 0.300). That is expected, because
M5_E1 was trained with S0 banks, and it is not a matched comparison.

**Thread smoke (L3).** OMP 4 vs 8 gave identical losses (0.43771) and grad norms. Peak reserved memory was
73.45 GiB (S arms: 73.02), process RSS 58.9 GiB. 8 threads were chosen.

**Production audit (L3, checkpoint 10; `evidence/L3_production_audit.json`):**
- exposure order identical to M5_E1 over 288 exposures (updates 0–35);
- LR schedule identical;
- losses and grad norms finite and close to M5_E1/S1 (update 10: 0.4065 vs 0.4093 / 0.4080);
- embedded ranker == fitted artifact == run identity (`a8bd49da…`);
- the only non-memory config differences from M5_E1 are the S campaign's loader fields
  (`dataloader_num_workers` 8→4, `dataloader_prefetch_factor` 2→1).

Rank-0 compute per update is 15.9 s vs 12.5 s for S1 on the same updates: forward 3.71 s vs 2.19 s, from the
24 vs 8 GDN feature side passes.

## 8. Campaign operation and publication

- **Controllers.**
  - Lane B: `lanes/run_lane_B.sh`, step 4925.180.
  - Lane A: `lanes/run_lane_A.sh`, step 5248.28. It waits until S lane A (`5248.27`) writes `lane-complete`,
    then trains L2 and then L0 under the shared QUALIFIED.json. It does not requalify.
  - State, heartbeat and history live in `lanes/<L>/`. `STOP_AFTER_CURRENT` and `SKIP_<ARM>` work as in the
    S campaign.
- **Runs.** `runs/<ARM>_seed429_e1/` holds the run identity: source commit, memory SHA, ranker SHA and fit
  report SHA. A resume is refused if any of these changes.
- **Publication.** The watcher (`m5-gdn-l-publication-20261023`, login-1) publishes each arm when its training
  is complete and verified. Each arm folder holds:
  - final weights, `navigation.json` with the embedded ranker, and the artifact;
  - fit report, drift report and label provenance;
  - a README with that arm's holdout capture and Spearman vs S0–S3 and random.

  The publisher refuses to upload if the embedded ranker differs from the fitted artifact.

## 9. Deviations, limitations and what is not done

- capture@96 is a **sample-relative** top-budget metric over 24 sampled rows per source, not a complete
  top-1200 over all rows. Complete 300-row pools exist only in the dense subset.
- Not run:
  - O1 combined-removal error (joint deletion of the discarded set);
  - O2 future-oracle fixed-history replay;
  - O3 action-CE/argmax diagnostics;
  - the ≤ 2-architecture remaining-target refit;
  - pairwise-ranking objectives;
  - L4–L6.
- Both hidden-state controls beat the GDN rankers offline (Section 4).
- The ranker is frozen during matched training from base; its inputs drift (Section 6).
- BF16 serving replay shows the inherited near-tie plan divergences of every arm, including legacy S0 (Section 7).
- No navigation success has been measured for any L arm.

## 10. Evidence index (copied from the cluster, `evidence/`)

| File | Content |
|---|---|
| `campaign.json` | campaign identity, lanes, assets |
| `labels/selection.json`, `labels/LABEL_SOURCE.json` | episode selection (partitions, loggers, quotas); label source commit |
| `shared/labels_verified.json` | label verification summary |
| `rankers/L{0..3}_artifact.json` | the frozen rankers exactly as embedded in training |
| `rankers/fit_report.json` | per-arm, per-layer, per-LR fit histories, dev/holdout metrics, fixed baselines, controls |
| `reports/holdout_report.json` | holdout capture/Spearman of every scorer with scene-bootstrap Δ vs S0 (`ranker_report.py`, `e42f297`) |
| `rankers/drift_base_vs_m5_e1.json` | drift gate |
| `shared/gates.json`, `shared/QUALIFIED.json`, `shared/provenance.json` | qualification |
| `shared/replay/shard{0..3}.json` | serving replay |
| `shared/recovery/comparison.json` | recovery |
| `lanes/B/threads.json` | L3 thread smoke |
| `evidence/L3_production_audit.json`, `evidence/audit_arm.py` | L3 production audit and the script |
| `configs/runtime.yaml`, `shared/diagnostic-runtime.yaml` | runtime profiles |
