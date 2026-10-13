# G3 online Omega execution plan

Spec: G3_native_state.md, common_infrastructure.md, teacher_single_view.md; online override in the G3 spec has precedence. Base c2_vggt_omega_g1@67b206b; isolated branch c2_vggt_omega_g3.

## Design decisions
- Reuse G1 teacher preprocessing, frozen FP32 resident inference, source/weight provenance, CPUAdam ZeRO2, full-episode checkpoint/offload, attention batch32, vision batch8, workers8/rank, four CPU threads. Requalify actual G3 memory/time.
- No PCA, offline features or added positional target. Regional targets are means of no-affine LN2048 pooled tokens. Only unique source images used by selected pairs are computed online (at most8), reducing teacher work without changing targets.
- Preserve original G3 decoder hidden1024; widen output256 to2048. This is the student-side learned dimension conversion, not a learned teacher compressor. G1's full spatial projector is unsuitable for native state-head inputs.
- Deterministic collected-RGB-path identity inherits G1's documented physical-alias limitation. Pair ranking/quotas and lag bounds follow the shared plan exactly.
- Query context is detached native normalized post-convolution q. Probe rows are injected only into native layer24 prepared GDN inputs, with effective beta=g=0. Native convolution, other layers and navigation labels retain original sequence length.
- Auxiliary count is pairs (not frames). Maintain separate global action/pair denominators across ranks and accumulation; keep navigation loss_sum untouched. Empty pair episodes train navigation and participate in auxiliary parameter reduction with differentiable zero.
- Fresh base, one joint epoch3852 updates, LR warmup116, auxiliary lambda .1 ramp771, headLR1e-4, backbone5e-6. Full recovery checkpoints every100, first10; immutable source launch after G1 completion.

## Task 1: Pure sampling, head and probe mathematics
Write failing tests, then implement deterministic delayed pairs, anchors, regional targets, detached query network and MLP decoder, immutable probe insertion. Verify FP64 independent recurrence reads/state/real-output/gradient invariance, repeated anchors, empty support, no future gradient and RNG isolation.
Expected: targeted CPU tests pass; actual FLA shape32x128 qualification follows Task3.

## Task 2: Native training integration
Integrate pure layer24 GDN forward inside one decoder checkpoint closure. Reuse native parameter names, prep and one production FLA call, restore real rows before native gated norm/out/residual/MLP. Wire loader/online teacher/global pair support/head LR/recovery/export. Preserve G1 behavior.
Expected: unit/regression tests pass; no teacher or training-only head in serving export.

## Task 3: Four-H100 qualification and speed
After successful G1 completion, run FLA numerical/gradient tests and complete R2R/RxR smoke training, shortest/empty support, mixed and longest627-frame199735-token episodes with max40 probes. Verify checkpoint on/off, full-state fresh-process resume and teacher-free serving export. Benchmark qualified G1 runtime first; test limited alternatives only where measurements justify them.
Expected: finite navigation/aux losses, nonzero earlier-write gradients, no OOM, bounded RAM, successful recovery/export. Report measured speed rather than guarantee global optimality or navigation quality.

## Task 4: Review, report, push and launch
Fresh independent whole-branch review, resolve important findings, final regression, commit/push new branch. Archive immutable source and launch full one-epoch joint run on4H100 after G1 success. Verify schedule, startup updates and early complete recovery checkpoint. Update report and plans with evidence.
Expected: G3 production active and resumable; estimated duration uses measured G3 timing.

## Review focus
Probe no-write/no-decay and no conv advancement; normalization exactly once with native additive epsilon; no future context/teacher leakage to query; FP32 state orientation; correct global pair normalization including empty ranks/GAS; checkpoint closure stability; serving auxiliary removal; full longest episodes and immutable resumable launch.

## Execution ledger
- Preflight: existing G1 aux_frame_count is a frame count; G3 needs an explicit auxiliary pair count without changing navigation denominators. Existing teacher adds position unconditionally; G3 requires an explicit target mode with identity binding.
- Ruling: preserve G3 decoder1024 hidden rather than copy G1 projector4096 hidden; the user requested online/noPCA adaptation, and native state input differs from G1 residual visual tokens. Scientific comparisons remain method-level, not single-factor delay ablations.
- Ruling: teacher compute only unique selected source frames, maximum8 rather than G1's16; single-view independence makes unneeded frames irrelevant.
- Status: implementation starting; no G3 test or training success claimed.

- Task1: targeted CPU sampling/recurrence/head tests passed; real FLA32x128 and native checkpoint tests passed onH100. Probe output/reference relativeL2 .00416; real output .00287; state .00332; native input-gradient errors <=.00453. Probe-free native equality exact. BF16 insertion is numerical, not bitwise parity.
- Task2: first complete-episode four-rank training smoke passed3 updates,24 exposures,192 pairs;17.016s/update and39.809GiB active GPU. Export restriction test observed RED for G3 then fixed to namespace-specific G1/G3 export;10CPU tests passed.
- G1 completed3852 updates, exit0, supervisorPASS, no new OOM events. G3 GPU tests now use the freed allocation4925 onworker-2.
- Historical failed diagnostic mixed-initial used a missing fixture path (omitted fixtures/); no training updates. Corrected path and added early fixture existence validation.

- Task3: longestRxR3updates passed, all24exposures with8pairs;74.572s/update,67.874GiB activeGPU. Teacherbatch4 mixed16.905s vsbatch8 17.016s (0.7% noise-scale difference); retainbatch8, no claim of exhaustive optimum.
- Independent whole-branch review: noCritical/Important/actionableMinor findings. Remaining evidence gates are recovery/export/fullsuite and immutable launch.

- Recovery: fresh-process checkpoint2->3 passed on4ranks. Head bitwise equal; full-state relativeL2 9.4106e-9, max1.1921e-7; loss exactly .4165976941585541 in both; all22 recovery files validated. All512/4096/1024/2048 query/decoder biases changed at nonzero weight.
- Ruling: three optimizer updates from base do not establish canonical greedy action competence. Original export audit failed on the bounded response-limit error, while source/export produced identical16 tokens and identical score SHA2568da9553977cefdbbd25b0cda605fec69c1760843bf1fd2c76985a6e4d8666267. Audit now records this known quality failure and requires exact source/export/reset outcomes and scores; unrelated runtime errors remain fatal. Unit test RED(import missing helper)->GREEN. Replay tolerances remain unchanged. Cost: this qualification certifies serving equivalence, not navigation readiness.
- Operational diagnostic failure: a one-GPU4-CPU srun omitted ntasks and inherited multiple tasks, causing duplicate model loads/CUDA OOM. It was separate from all qualified training runs; corrected ntasks=1 diagnosis completed. All further launches explicitly request one Slurm task (torchrun alone creates4GPU ranks). No training implementation was changed for this launch mistake.

- Task3 complete: eight4GPU diagnostic runs passed;366regressiontests passed,118warnings,48.30s,no skips. Final export/replay bothdatasets passed with unchanged tolerances; source/export/reset exact including R2R bounded policy failure. Real globalpairDDP/GAS gradientmaxerror2.3842e-7; actualOmega regionaltargets independently verified exact.
- Final source verification:57src files match longest-episode qualified source exactly. Production launch pending.
