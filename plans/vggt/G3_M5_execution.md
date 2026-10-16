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
- Task 1–3 core implementation complete in53ace67. Targeted regression suite:
  33 passed. Local full suite:331 passed,34 skipped,4 failures due unavailable
  Slurm, DeepSpeed CPUAdam, Omega source and base processor assets. These
  server-dependent tests remain part of H100 qualification.
- Independent whole-branch review found no model-integration defect. Fixed
  empty-support fixture ordering (length sorting invalidates row rotation)
  and partial supervisor JSON reads; added exposure-slot and queue tests.
  Review fixes:4 tests passed. The selected-layer/head readiness-order CPU
  regression also passes for empty and supported episodes.
- R1 safely saved3362. First handoff did not start GPU tests because its
  supervisor result was not visible immediately after step exit; R1 resumed
  as step4925.85. Added bounded NFS publication wait to the owned handoff.

- Qualification complete on source9031d75:377 full-suite tests passed, all15
  gates passed. Both maximum-length datasets, both real empty-support slots,
  exact export parity, distributed reduction and fresh-process recovery passed.
- Attention16/32/64 were effectively tied on the mixed fixture; retain32.
  Representative warm intervals:R2R9.7277s/update,RxR19.7602s/update.
  Initial joint-epoch budget18–22h including checkpoint overhead.
- Longest RxR627 actions/199,735tokens passed at67.87GiB allocated GPU,
  73.65GiB reserved GPU and429.12GiB host working set; zero new OOMs.
- R1 resumed from3373 under originalsourcebcb14ea as4925.91. Combined queue
  PID3739763 waits for R1's successful3852 checkpoint onworker-2/4925.
  Production root:g3-m5-omega-worker2-20261016/production.
- Full evidence and limitations:reports/vggt_five/G3_M5/implementation_report.md
  andevidence/h100_qualification.json. No navigation improvement is claimed
  before evaluation; historical M5 uses a different LR denominator.
