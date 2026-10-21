# Full-panel capture review

Reviewed `measure_panel.py` and `recall.py` against active D-a/D-c/D-d/D-f requirements. CPU verification covers the state orientation/repeated key-head mapping, sampled lags, shared Gaussian projection, gating prerequisites and raw denominators. No GPU measurement was started by this review.

Fixed in `measure_panel.py`:

- Require the instrumentation gate's input manifest SHA-256 to equal the current manifest, and its declared selected UIDs to equal measured gate-row UIDs. Previously equal episode UIDs alone could admit evidence from different manifest contents.
- Match `load_checkpoint`'s safetensors-first weight selection when both weight files exist.
- Record research recipe hash, teacher index hash, projection identity, lag/anchor sampling and eviction timing. Mark D-d inapplicable for Window8 models.
- Save per-layer recall pair/region counts and token denominators, D-c query/block counts, and D-d step counts with each episode. Quadrants contain 70,70,80,80 tokens; Global contains the same 300 tokens and must not be added to quadrant counts as independent observations.

Confirmed by inspection and existing focused CPU tests:

- M5 input capture includes all eight scorer layers plus D-a layers. Bridge errors use source+4 anchors when a source leaves R4, pair each token with its original write score, and retain explicit correlation token counts.
- D-a retains layers22/24/26, plus8/16 for TRIPLE. Reconstruction is post-observation/pre-action. Recall normalizes squared errors per value head, repeats key heads contiguously, then averages heads; forgetting subtracts the same source's lag-zero error.
- The requested even-anchor and sparse-lag schedule is used. The Gaussian4096 projection resets the CPU seed0 generator per call and generates identical head blocks for all models. Current-view features retain five regional pooled value vectors.
- D-c wraps the same full forward used for state capture; original logits and native selection plans are reused as D-f baseline with the identical encoded inputs. D-f runs two dummy seeds and stops on failed invariance.

Scope limits:

- This runner captures D-a Metric1 and features for Metric2. It does not itself fit the scene-disjoint ridge probes, retrieval tests, matched current-view control, confidence intervals or D-b interventions. Those remain downstream work.
- D-d cross-model Jaccard and same-token score comparisons require the saved per-model captures and a downstream paired analysis. D-f layer/length summaries and D-c paired category contrasts likewise require downstream analysis.
- Original-head metrics remain unavailable without original trained-head identity evidence; this runner does not substitute dummy heads for D-a/D-b trained-head readouts.
- This review is not a substitute for the queued two-panel all-hook G-1/G-3/D-c tests on every model. Full-panel execution remains gated on those results and the complete teacher cache.

Verification: `/storage/anhdh35/SimpleMemVLN/.venv/bin/python -m unittest scripts.diag.test_measure_panel scripts.diag.test_recall -v` — 7 tests passed.

## Resource and integrity review, 2026-10-20

Read-only inspection found no confirmed measurement memory blocker. The frozen
panel contains 13,847 frames, with a maximum of 119 frames per episode, below the
128-frame fixture previously exercised. Eleven BF16 width2560 captured layer
inputs occupy about 2 GiB of CPU memory at 40,000 tokens. Captures are popped as
each layer is processed. A reconstructed 119-frame record is about 659 MiB;
stacking briefly duplicates that storage. Reconstruction with `check_fla=False`
retains no entire prepared-prefix GPU history. Projection stages one 256 MiB
Gaussian head block; attention intermediates are bounded per step. Full-panel
attention retains numeric masses, IDs and parity results; raw QKV/hash provenance
is retained in the separate two-episode instrumentation gates.

Native state records across the 17 requested model/layer combinations total
approximately 1.24 TiB before other artifacts. Review-time resources were
176 GiB available host memory and 14 TiB free disk. These are estimates, not
full-panel peak-memory measurements. Temporary D-f heads restore in `finally`;
parameter versions are checked per episode and checkpoint SHA-256 at completion.

Added an explicit finite-logits guard immediately after the measurement forward,
before logits persistence, D-f baseline use, or state reconstruction. It rejects
NaN and either infinity without replacing or changing valid tensors or dtype.
D-b already rejects nonfinite copied-baseline logits through `bitwise_logits`
and nonfinite global-ablation logits through `logit_kl`, before saving logits or
reconstructing global state; targeted prefix forwards produce no logits. New
CPU regression coverage exercises all three nonfinite values in those paths.

Ridge fitting is a runtime concern, not a demonstrated memory failure. Registered
pair totals are A=20,004, B=25,072, C=54,548. Inner-training sizes range from
4,529–8,221 (A), 5,652–10,328 (B), and 12,063–22,702 (C). For the current-view
control p=20,496, the FP64 Gram occupies 3.13 GiB and each RHS/weight matrix
1.56 GiB. Copies, factorization and training arrays raise peak use to tens of
GiB. The 660 control solves alone require approximately 7.57 PFLOP under dense
Cholesky plus multiple-RHS operation counts, excluding Gram construction and
predictions; elapsed time depends on actual CPU throughput. Exact dual solves
when n<p and spectral reuse across lambda candidates are possible future
optimizations, requiring numerical and CV-selection parity checks. No ridge
objective, scaler, fold, tolerance, or implementation was changed in this review.
