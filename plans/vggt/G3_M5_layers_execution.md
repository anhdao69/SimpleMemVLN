# G3+M5 supervision-layer ablations

## Binding specification

Implement the user's zero-based L8, L16, and L8+L16+L24 ablations from
`c2_vggt_omega_g3_m5` (295e85e). L24 remains the unchanged baseline.
Use independent native-state readouts, shared original causal anchors, lags,
and online frozen VGGT-Omega regional targets. Average the per-layer auxiliary
numerators before the existing global pair normalization; lambda maximum is 0.1.
Preserve M5 selection, full BPTT, original pretrained backbone, seed 429,
R2R+English RxR15deg manifest, global batch 8, one epoch/3852 updates,
116 warmup, cosine schedule, and optimizer/runtime settings.

## Design

An explicit research.student_layers list enables independent ModuleDict heads.
Legacy student_layer: 24 retains its exact checkpoint namespace and behavior.
Each head uses the existing deterministic seed without consuming global RNG;
heads have identical initial values but independent parameter storage. Validate
sorted unique native GDN indices against the actual decoder configuration.
Checkpoint closures capture both layer and head. Read probes are removed before
later layers and M5 scoring. Serving export removes all research heads and
research configuration and preserves the native inference architecture.

All work occurs on c2_vggt_omega_g3_m5_layers in an isolated sibling worktree.
Remote runs use isolated source/output directories, exclusively worker-2.
Existing jobs and checkpoints are never stopped or modified. A worker-resident
controller owns L8 -> L16 -> L8+L16+L24, advances only after verified successful
full-epoch recovery checkpoints, and stops on error. Restart resumes the current
arm; completed arms are verified and skipped. Source/config identities bind
qualification and queue execution.

## Task 1: Configurable heads and causal forward

Write failing tests for invalid layer contracts, independent RNG-neutral heads,
legacy L24 parity, multi-layer mean gradients, checkpoint recomputation,
empty-pair gradients, causal reads, unchanged M5 plans and navigation outputs.
Implement configuration, model routing, independent heads, and mean reduction.
Verify targeted existing and new CPU tests, then commit.

## Task 2: Reproducible profiles, recovery, export, and queue

Add three configurations identical to baseline except selected layers. Extend
profiling and export/recovery checks for all heads. Implement a sequential
worker controller with locking, atomic state, explicit child exit handling,
completion gates, fail-closed source identity, and resume. Test failures,
completion gating, restart, sequence order, and per-layer export/recovery.
Verify CPU suite, then commit.

## Task 3: H100 qualification and launch

Use idle allocation 4925 on worker-2 only after rechecking jobs. Test real
four-rank training, heterogeneous/empty support, both datasets, longest R2R and
RxR episodes, multi-update memory stability, checkpoint save/resume, and serving
export. Record measured timings, GPU/host memory, exact source and configs.
Qualify all arms before queue launch. Obtain one fresh whole-branch review,
resolve important findings with regression tests, and launch sequential training.
Verify initial full-training progress and durable queue state. Publish code and
an implementation/evidence report; distinguish pending training from completion.

## Review focus

Check closure late binding, per-head ZeRO readiness, normalization under unequal
pair counts, zero-pair graphs, global RNG invariance, checkpoint identity,
probe removal, no serving auxiliary dependency, memory accumulation on longest
episodes, and controller survival independent of login-node watcher processes.

## Execution ledger

- Initial review: baseline clean/current; new isolated worktree created.
- Design ruling: identical seed per independent head controls initialization
  across arms; parameter storage is separate. No teacher or sampling changes.
- Pre-flight: Task 1 head naming is consumed by Task 2 export/recovery and Task 3
  qualification; preserve legacy state_head and explicitly validate state_heads.
- Pre-flight: Task 2 queue consumes Task 3 qualification records; launch is gated
  until records match exact source/config and every required check passes.
- Baseline verification: 12 targeted existing tests passed (11.27 s).
- Task 1: implemented and verified; configurable-head tests failed before the
  implementation, then causal/checkpoint/gradient and baseline tests passed
  (21 tests, 15.76 s). Head mapping requires canonical string layer keys.
- Task 2: export/config and queue tests failed before implementation. Combined
  targeted suite passed: 31 tests (18.33 s). Qualification scripts now require
  three complete longest-episode updates for each arm and per-head recovery.
- Queue design: one worker-resident process launches all three local children;
  each arm has separate source/output, auto-resume, and successful-checkpoint
  gating. A cancelled allocation still requires starting the controller in a
  replacement allocation; its saved weights and queue position are recoverable.
- Fresh whole-branch reviewer: no critical scientific issue; one important
  completion-skip identity gap. Added run/source binding and explicit final
  checkpoint layer identity verification; regression reproduced the missing
  guard before implementation.
- Deferred minor: interrupted qualification profiles use fixed directories.
  Preserve a failed experiment directory by renaming it with an attempt suffix
  before restarting that qualification; successful gates are retained only
  when the source manifest is identical. Production resume is unaffected.
- Review hardware items remain gated by actual H100 qualification, not inferred
  from CPU tests. Slurm allocation cancellation requires controller relaunch;
  worker-resident orchestration does not claim survival of scheduler cancellation.
- Frozen reviewed source: f4d1c2f216e68a960a5a691d21cbf1f08ea4a901, pushed
  to origin/c2_vggt_omega_g3_m5_layers. Three detached cluster worktrees:
  g3-m5-layers-worker2-20261017/source-f4d1c2f-{l8,l16,l8_l16_l24}.
- Final qualification is running in owned Slurm step 4925.100. All 403 cluster
  tests and three-head NCCL normalization passed. Triple-head empty-support
  tests passed for both accumulation slots; remaining hardware gates pending.
- A separate CPU-only launch gate was submitted inside allocation 4925. It
  waits for the final qualification seal and for all other compute steps to
  exit, then execs the worker-resident production queue. Six operational safety
  cases passed (failed/missing qualification, active competing step, ready).
  Existing interactive shell .0 and unrelated jobs are untouched.
