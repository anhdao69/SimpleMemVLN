# Two-H100 episodic training qualification and launch report

Status: implementation checks passed; one-epoch production jobs submitted. Extended testing was curtailed at the user’s explicit request to start training promptly. Scheduler state is recorded below.

## Scope and source

Requested: optimize V0, M1 and M2 on two H100 GPUs, compare alternative runtime settings, train each independently for one epoch, push the finished code, and then release interactive allocation 4876. Other user jobs are protected. This is a three-arm subset of the larger research plan, not completion of the full Phase-1 evaluation campaign.

Upstream `c2_episodic` was checked against GitHub at `a06c7896dd4ca8df79d20a49ea537f8b2019631b`. Work is isolated on `c2_h2_speed`; the original working directories and existing training jobs were not modified. Runtime and safety implementation commit: `e789a13` (the report itself may be a later documentation commit). Production uses immutable source archives of the validated implementation commit.

Data: 30,815 complete joint R2R + English RxR15 episodes, 3,128,624 action steps; manifest SHA-256 `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`. Every arm independently initializes from original Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Benchmark weights are never production initialization.

## Does the implementation follow the plan?

The implemented Phase-1 mechanisms match the intended causal design:

- V0 retains eight complete recent observation/feedback groups and no episodic bank.
- M1 retains nine complete recent groups and no episodic bank. This is a near-budget control, not exactly the same token count as M2.
- M2 retains eight recent groups plus at most 300 old visual K/V rows per attention layer. Old framing and action tokens are excluded from the bank. Selection is layer-local, fixed at write time, uncapped by source, and uses committed-write magnitude from the preceding GDN layer.
- The score is the mean across value heads of the norm of beta times the prediction residual after state decay. It is not beta alone and does not multiply beta twice.
- Selection runs without gradients, outside checkpointed recomputation. The selected attention plan is immutable during backward, while the selected K/V rows remain part of full-episode differentiable attention. The native GDN policy computation remains differentiable.
- Original logical positions and multimodal rotary positions are retained. Cache admission/eviction occurs at group boundaries, with bounded recent groups and bank storage.
- Training uses complete episodes and full BPTT, the existing action labels/history/resolution, the global action normalization, and the existing AdamW implementation.

There are documented deviations and outstanding scientific gates, so “100% exactly identical to every sentence of the plan” would be incorrect:

1. D1 resolves the plan's inconsistent epoch/update table: train to epoch one/update 3,852 while keeping the 7,704-update, 232-warmup schedule. The saved state permits continuation of the same two-epoch trajectory on the same topology.
2. D2 uses a separate no-gradient projection/convolution side pass to compute scores instead of changing the dispatched native GDN kernel to emit them. This costs extra work but preserves the policy kernel and checkpointing path.
3. D3 uses the versioned FP32 block-64 helper with explicit segmentation. This is mathematically the committed-write rule; finite-precision helper and native-kernel states are not bitwise identical.
4. D4 accumulates overlapping K/V gradients in FP32, consistently for all arms.
5. Offline and incremental M2 scores and selected row sets are not exactly identical. Prior real 96-step audit showed approximately 98.2% mean set overlap, with many nonidentical sets. Current replay measurements are reported separately; overlap is not proof of unchanged navigation behavior.
6. These three requested runs omit M2c, which the full research plan requires to distinguish score-specific benefits from generic old-context access. M3 and later Phase-2 comparisons are not implemented or authorized by these three runs.
7. Trained-checkpoint serving parity and closed-loop Habitat SR/SPL remain subsequent evaluation gates. Base-weight replay and training smoke tests cannot establish navigation quality. The replay forces canonical response tokens: 100% decoded fixture-label agreement checks that forced path, not unconstrained policy agreement; raw response-token argmax disagreement is reported separately.

The serving promotion diagnostic list also grows on CPU with stream length; the bounded-cache claim applies to resident K/V, recent groups, bank rows and scoring state, not every diagnostic record. This is not a training GPU-memory leak.

## Speed experiments and preserved protocol

All experiments use two H100 80GB HBM3 GPUs in allocation 4876 on worker-1, in sequence. The environment was reused without installing or changing shared packages. Per rank: one episode per microbatch; accumulation four; global batch eight. ZeRO-2 offloads optimizer state to CPU. Activation offload uses the bounded pinned arena above 65,536 tokens. Pixel staging is BF16 and loader pinning remains disabled.

Initial screening uses four optimizer updates over the same eight complete 79-observation episodes, four from each dataset. Steady time is mean elapsed time between complete update endpoints after the first update, including between-update overhead.

| Configuration | Threads/rank | Workers/rank | Heap trim/update | Attention batch | Vision microbatch | Seconds/update |
|---|---:|---:|---|---:|---:|---:|
| Baseline | 2 | 2 | yes | 8 | 4 | 25.77 |
| Threads 4 | 4 | 2 | yes | 8 | 4 | 23.09 |
| Threads 8 | 8 | 2 | yes | 8 | 4 | 21.26 |
| No heap trim | 4 | 2 | no | 8 | 4 | 22.38 |
| Four workers | 4 | 4 | no | 8 | 4 | 20.94 |
| Larger batches | 4 | 4 | no | 16 | 8 | 22.21 |
| Largest batches | 4 | 4 | no | 32 | 16 | 21.79 |
| Combined workers + threads | 8 | 4 | no | 8 | 4 | 19.64 |
| Combined larger batches + threads | 8 | 4 | no | 32 | 16 | 20.45 |

All nine short trials passed. Selected candidate: eight OMP/MKL threads per rank, four persistent loader workers per rank with prefetch two, no per-update heap trim, original attention batch eight and vision microbatch four. Initial improvement is approximately 24%; this short-fixture number is not an epoch ETA or a guarantee of maximum possible speed. Full-manifest measurements and long-episode qualification are reported below. External node contention, filesystem caching, kernel compilation and episode length can affect timings. A warmed baseline rerun is included because novel shapes incur substantial first-use compilation costs. Resolved final H2 configuration was checked to equal the selected external benchmark YAML exactly.


### Full-manifest 20-update measurements

All four runs passed, with identical per-rank episode/action-exposure SHA-256. Each processed 160 complete episodes. The optional warmed-baseline rerun was canceled at the user’s request to avoid further testing delay. The 59.175-second baseline includes first-use compilation and must not be used to claim a production speedup. The short-fixture comparison supports approximately 24% lower update time; a full-epoch improvement has not been established.

| Run | Seconds/update | Peak reserved GPU GiB | Sampled peak host working set GiB |
|---|---:|---:|---:|
| representative_baseline | 59.175 | 56.91 | 197.07 |
| representative_v0 | 22.617 | 56.91 | 205.31 |
| representative_m1 | 24.342 | 56.91 | 201.75 |
| representative_m2 | 25.218 | 56.91 | 200.43 |

## Safety defects fixed

The original helper checker printed state errors but did not include helper finiteness/error in its aggregate PASS. It now rejects nonfinite comparisons and relative error above the previously declared independent-audit 1% budget. The existing logit replay budgets remain unchanged: absolute 0.75, relative 0.03, RMS 0.15. Failed evidence is retained.

The supervisor originally inspected OOM kills but could miss recoverable OOM events or transiently observed event counters. It now remembers maxima over sampled events and rejects any new OOM or OOM kill, terminating only the command's owned process group when necessary.

The numerical checker now accepts the selected runtime profile and applies it to both episodic and feature-disabled legacy comparisons. It records actual Git source revision, tracked-source cleanliness and runtime SHA-256. Campaign initialization and resnapshot require passing evidence on the clean requested revision. Reports are immutable files named by content hash; each launch stores its own reference, so later source snapshots cannot replace evidence for an older queued/resumed attempt. Preflight verifies source and runtime hashes and evidence contents.

Cluster partition, CPU and thread overrides are recorded and propagated. Automatic Hub publication can be disabled explicitly. These runs use the `main` partition, only the H2 profile, no sibling races and no automatic ablation submissions. Each job also runs its own longest-episode startup gate before production training.

Targeted regression tests and independent review found and closed checker/runtime provenance, resnapshot evidence association, and placement propagation defects. A second focused review confirmed expired-group-only visual admission, serving promotion before opening the new group, the committed-delta formula, text/prefix state updates, checkpoint-plan immutability and gradients to original selected K/V rows; it found no new scientific correctness blocker. Scientific core files are unchanged from upstream a06c789. The committed revision passed the full GPU-capable suite: **324 passed, 0 skipped, 115 warnings**, in 27.12 seconds with warmed compilation caches. Final resource/numerical results are recorded below when complete.


### Numerical qualification on the committed H2 runtime

All 18 R2R/RxR replays passed (V0/M1/M2, 64/96/128 observations). Feature-disabled V0 exactly matched legacy Window8 logits and loss. Real helper/native-state relative errors were 0.204296% and 0.235266%, below the predeclared 1% engineering budget. The 520-step streams peaked at 2,946 resident KV rows for M2 (guard 5,420) and 2,646 for V0 (guard 5,120).

| Arm | Largest logit error | Largest RMS error | Lowest raw-token argmax agreement |
|---|---:|---:|---:|
| V0 | 0.640625 | 0.041750 | 98.953% |
| M1 | 0.750000 | 0.041219 | 99.217% |
| M2 | 0.750000 | 0.041996 | 99.303% |

M2 exact selected-ID sets differed in 4,074/4,608 layer/step comparisons. Mean overlap per replay ranged from 98.073% to 98.371%; worst single-step overlap was 93.333%. These results confirm the previously documented sensitivity; they do not justify claiming exact train/serve policy equivalence. Full evidence is in `numerical.json`.

## Final checks and production launch

All 12 ordinary H2 resource gates passed: V0 and M2 on short mixed data, longest R2R, just below activation offload, just above offload, longest RxR, and mixed episode lengths. The longest fixture contains about 199,735 tokens per episode. Peak reserved GPU memory was 72.9414 GiB, below the 78-GiB hard ceiling and 76-GiB preferred target. These runs used the 460.8-GiB host working-set guard. Zero OOM events were observed in the completed gates.

V0 also passed stressed full-state checkpoint save and fresh-process resume from checkpoint 2 through update 3. Its checkpoint contained 18 files totaling 70,585,285,937 bytes (65.74 GiB). Save/reload host working-set peaks were 274.05/310.21 GiB, with zero OOMs. The resumed loss was 0.348683 versus 0.349048 in the uninterrupted run; BF16 training is not bitwise reproducible, and exact identical trajectories are not claimed. The real reload completed the correct next update.

M2 completed its full checkpoint-save gate; its reload was intentionally interrupted when the user requested less testing and immediate production scheduling. Dedicated M1 recovery, the optional warmed-baseline comparison, and additional long/replay repetitions for M2c/M5/M5c were canceled. These are uncompleted checks, not passing results. The shared recovery implementation and contract/retention/ownership regression tests passed; explicit real save/reload was demonstrated for V0. All production arms use that recovery path and retain full optimizer state.

The original V0 job 4877 was submitted with the broad `full` qualification mode. It passed 11 gates on worker-0 but had no production updates, training output directory, or lease. At the user’s request to start full training promptly, it was canceled and replaced with V0 4880 / attempt a2 using the normal own-arm longest-episode startup gate. Historical qualification evidence is retained, and no production training progress was lost. M1/M2 holds and temporary dependencies were removed. Each production job starts from original base weights and trains the full joint manifest after its own startup gate passes.

| Arm | Active Slurm ID | Attempt | GPUs | Host RAM | Epochs |
|---|---:|---|---:|---:|---:|
| V0 | 4880 | P1-V0-s429-H2-a2 | 2 H100 | 512 GiB | 1 |
| M1 | 4878 | P1-M1-s429-H2-a1 | 2 H100 | 512 GiB | 1 |
| M2 | 4879 | P1-M2-s429-H2-a1 | 2 H100 | 512 GiB | 1 |

All request 30 CPUs, eight OMP/MKL threads per rank, four loader workers per rank, and the measured H2 settings. The campaign root is `/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011`; source revision is e789a13. Hub uploading is disabled. Startup test updates do not count toward the 3,852 production updates. Representative update times suggest approximately 24–27 hours of training per arm, plus checkpointing, initialization and queue time; this is a short-sample estimate, not an observed epoch duration.

## Cancellation and recovery procedure

All three production attempts use the same full-state recovery mechanism. Each save includes model weights, the DeepSpeed optimizer shards (including optimizer/scheduler state), Trainer step/epoch state, per-rank RNG state, navigation/data contracts, and processor/configuration metadata. A completion marker is written only after all ranks finish saving and required files exist. Automatic resume selects the newest valid completed checkpoint; a partially written newer checkpoint is quarantined. The two newest completed rolling saves and epoch-boundary saves are retained on shared `/mnt`, independently of the Slurm allocation.

The first recovery save is update 50, subsequent saves are every 250 updates, and epoch one saves at update 3,852. An abrupt cancellation resumes from the latest completed save, not from the exact killed instruction. With representative speeds, 250 updates are approximately 1.6–1.8 hours; actual duration depends on episode lengths and I/O. A cancellation before the first complete save requires starting again. Recovery does not mean automatic scheduler resubmission.

For a planned interruption **after training starts**, request a checkpoint and orderly stop using `scancel --batch --signal=USR1 JOB_ID`. This sends the signal to the batch Python coordinator, which writes its job-specific checkpoint request. Both ranks honor it at the next optimizer boundary, save, and stop. Wait for the job to exit and for `RECOVERY_COMPLETE.json`; do not immediately follow the signal with a hard cancellation. The early-stop lifecycle records interruption rather than epoch completion, so a nonzero final job status is expected. The configured 30-minute pre-time-limit warning uses this same mechanism. Sending USR1 during preflight/qualification is not a training checkpoint request.

For an already canceled job, resubmit its **existing attempt directory**, not a newly created arm attempt. Keep the immutable source, manifest, arm, H2 profile, two ranks, and batch accumulation unchanged. The coordinator checks that its previous owner is no longer live, reruns allocation qualification, and passes `--resume_from_checkpoint auto`. Trainer restores optimizer/scheduler/RNG/step state and skips previously consumed batches. BF16 computation is not claimed bitwise reproducible across independent executions. Do not resume a two-GPU checkpoint using four GPUs.

Campaign root: `/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011`.

| Arm | Initial Slurm ID | Existing attempt directory relative to campaign |
|---|---:|---|
| V0 | 4880 | `runs/P1-V0-s429/P1-V0-s429-H2-a2` |
| M1 | 4878 | `runs/P1-M1-s429/P1-M1-s429-H2-a1` |
| M2 | 4879 | `runs/P1-M2-s429/P1-M2-s429-H2-a1` |

Resubmission commands are generated beside this report in `resume_commands.md`; they preserve the launch identity and resource request. Never execute them while the previous job is live. An epoch-one-complete attempt exits without training a second epoch. Incomplete-only saves are rejected instead of silently training from base.




## Remaining-variant smoke results

The final short checks used the exact committed H2 runtime, two GPUs, and one complete optimizer update over the short mixed R2R/RxR fixture. They exercise forward, backward, gradient accumulation and optimizer execution; a single update is not a convergence or steady-state-speed measurement. Together with the V0/M1/M2 20-update training measurements, all six existing variants completed a training check. The extended three-arm longest-episode and train/serve replay repetitions were deliberately omitted under the final user instruction.

| Arm | Result | Total wall time including initialization | Peak GPU reserved GiB | Peak host working set GiB |
|---|---|---:|---:|---:|
| M2C | PASS | 63.12 s | 30.04 | 129.84 |
| M5 | PASS | 58.44 s | 30.04 | 134.60 |
| M5C | PASS | 55.96 s | 30.04 | 125.92 |

All three checks reported zero new OOM and OOM-kill events. Evidence: `quick-remaining.json`. The superseded extended runner/release scripts remain archived for reproducibility and are not evidence of completed tests.
