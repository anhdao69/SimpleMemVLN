# G3 + M5: online native-state supervision with episodic visual memory

## Specification (2026-10-16)

Start from `origin/c2_vggt_omega_g3` b42cfe5 in isolated branch
`c2_vggt_omega_g3_m5`. Reference `origin/c2_ablations` 1b93100;
M5 scientific source a340e131, G3 scientific source 22593831.
The new run starts from original Qwen3.5-4B, not either trained checkpoint.

Keep M5 exactly: four recent complete groups including current, 1,200 older
visual K/V tokens per full-attention layer, detached committed-mean-L2 write
priorities, no per-source cap, seed 429, original logical positions, original
canonical action text/history, no step lane. Native recurrent state carries
the entire episode. No truncation and no changed scoring/selection backend.

Add G3 at native GDN decoder 24 only. Online frozen FP32 VGGT-Omega supplies
unpositioned LN2048 quadrant/global targets for at most eight sampled pairs
with lags 8..32. The existing training-only learned query and decoder MLPs
read native state via identity-update probes after real observation anchors.
Remove probe rows before native norm, residual, MLP and every later layer.
No PCA, preextracted features, teacher-conditioned queries or inference head.
Keep sampling, target identity, loss normalization and teacher unchanged.

One joint R2R + English RxR15-degree epoch: 30,815 episodes, global batch 8
(4 GPUs x 1 episode x accumulation 2), 3,852 updates, 116 warmup updates,
cosine-to-0.1 decay; backbone LR 5e-6, head LR 1e-4, auxiliary weight 0.1
ramped over 771 updates. Full BPTT, frozen vision. Start with G3 CPUAdam,
ZeRO2 CPU optimizer offload, pinned activation arena, four CPU threads/rank,
eight loader workers/rank, BF16 staged pixels, vision/teacher batches 8.
Benchmark attention batches without changing the scientific recipe.

Empty-support episodes must execute the same differentiable head/packing
graph as supported episodes, with zero-sized tensors and no synthetic
examples. Preserve historical G3 behavior and teacher identity by enabling
this only for the combined forward.

Serving is exactly ordinary M5 after removal of `research_aux.*`; export
must preserve M5 contract and all backbone tensors. Auxiliary gradients can
also reach earlier episodic attention: this is supervision of native state
in an episodic model, not proof that its information came only from recurrent
memory. M5's historical epoch-1 checkpoint used half a two-epoch schedule;
this experiment deliberately uses G3's true one-epoch schedule. Improvements
over either model require navigation evaluation, not only training smoke tests.

## Execution plan

1. Compare reference code/config/reports/tests; establish CPU baseline. Add
   strict combined config tests, then implement explicit G3_M5 dispatch and
   checkpoint-local native-state supervision inside the existing M5 loop.
2. Test empty/support gradient participation, causality, checkpoint parity,
   M5 selection/serving preservation and export. Run existing CPU suite.
3. Add source-bound qualification and worker-2-only durable queue/launcher.
   Validate checkpoint inventory and successful predecessor completion; no
   start on cancelled/failed R1 or a partial checkpoint. Auto-resume own run.
4. With server-side R1 resume safeguard, pause R1 at a complete checkpoint
   only if still running and GPU tests are ready. Qualify actual four-H100
   mixed R2R/RxR, empty-support ZeRO2, longest episodes, save/resume and export.
   Compare bounded runtime settings, select measured safe configuration.
5. Resume R1 and verify progress, or recognize completed R1. Queue combined
   epoch only with a source-bound passing qualification. Independent review,
   detailed report, push isolated branch. Report measured runtime/memory and
   limits; never promise mathematical 100% or navigation improvements.

## Progress ledger

- Reference inspection complete; M5 scorer/selector/config match between
  reference and G3 base. No edits to shared teacher or sampling sources.
- SSH restored 03:51 UTC; R1 still active, latest checkpoint 3300. Not paused.
- Ruling: implement inline under user's explicit authorization; no redundant
  design-approval round. Final independent review follows executing-plans.
- Shared interfaces: G3_M5 config -> dataset/teacher/head dispatch -> episodic
  forward -> existing trainer reduction/export -> qualified launcher.
