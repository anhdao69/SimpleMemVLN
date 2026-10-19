# G3 with constant geometric weight0.1 and seed430

The user's replacement for the unstarted G1-future run is original G3 trained with `L_nav + 0.1 * L_geo` from optimizer update0, using a different seed. The new branch is `c2_vggt_omega_g3_fixed01_seed430`, based on latest original G3 `034f2eb`; its training code is identical to the earlier qualified2259383 before these changes. G3+A1 continues its existing epoch first. G1-future's implementation and results are retained, but it is removed from the active sequence.

## Scientific changes

| Setting | Original G3 | Replacement G3 |
|---|---|---|
| Auxiliary weight | `0.1 * min(1,u/771)` | `0.1` at every update, including0 |
| Data/training seed |429|430|
| G3 frame-pair sample seed |429|430|
| Readout initialization seed |fixed429|explicit430|

All other resolved configuration fields are identical. G3 native decoder24 GDN state reads, source/read lag8–32, five regional cosine targets, training-only native-state head, frozen online VGGT-Omega, original teacher identity, and full BPTT are preserved. There is no PCA, future-conditioned teacher, M5 memory, or A1 adapter. Student inference behavior and serving namespaces are unchanged.

One complete joint R2R + English RxR15deg epoch starts independently from the original Qwen3.5-4B snapshot851bf6e, not from G3+A1 weights. The corpus is30815episodes/3128624actions. The schedule remains3852updates,116warmup updates, cosine decay to0.1of peak LR; backboneLR5e-6,headLR1e-4,AdamWbetas.9/.95,weight_decay.01,clip1. Four H100s with one complete episode/rank and accumulation2 give global batch8. Native Window8 text/canonical action history, optimized loader settings, CPUAdam/ZeRO2 offload and pinned activation arena are unchanged.

`ramp_updates=0` is the explicit constant-weight convention, accepted only for G3. Original G3's default771 and all G1 ramp validation remain unchanged. `head_seed` is optional and defaults to429 for existing G3 configs; the constructor uses its existing isolated CPU generator. The teacher/source sampling code is unchanged, and its identity hashes remain valid. Seed/ramp fields are saved in navigation/resume contracts, so incompatible checkpoints cannot silently override this recipe.

The loss still normalizes navigation by global action count and geometry by global sampled-pair count across all four ranks and both accumulation slots. The coefficient applies to that globally normalized geometric loss; no additional division by accumulation or number of regions is introduced. Empty auxiliary support remains a differentiable zero.

## Tests and safe replacement

The new CPU tests first produced8intended failures and1pass against unchanged code. After implementation,28tests passed and1CUDA-only test skipped on the login CPU. Coverage includes fixed weights at0/1/770/771/2623/3852, original ramp behavior, invalid configuration, the actual Trainer loss/gradient normalization, isolated deterministic head seeding, exact resolved-recipe differences, and queue order/restart/failure/completion checks.

G3+A1 was paused through its native checkpoint-and-stop mechanism at step2649. Its old queue rejects an incomplete epoch and therefore cannot start G1-future on this controlled early exit. A server-side replacement script waits for the old queue lock, archives the handled stop flag, then starts the qualification/replacement step4925.139 on worker-2. Allocation4925 and its interactive shell remain intact.

Before launch, the new source was frozen read-only at `/mnt/data/vmo-ai-task/anhdh35/g3-fixed01-seed430-worker2-20261019/source`, commitb3a3ff4. The driver runs the full GPU suite and a two-update mixed-extreme smoke using the actual production config, including the longest627-frame RxR episode. Acceptance requires finite losses/gradients, fixed0.1 weights on all16exposures, zeroOOM, complete four-rank checkpoint2 and nonzero trained readout biases. The short smoke has a diagnostic two-update horizon; the full run retains the declared3852-update schedule.

If qualification fails, the server-side fallback resumes only G3+A1 from its latest complete checkpoint. An unqualified G3 cannot launch, and G1-future is not restored. After successful qualification, the replacement persistent controller resumes the exact original G3+A1 source/config/state first, then starts G3 in a separate fresh output directory. Transition requires a complete3852-step epoch checkpoint, matching contract, full final model and finite final forward reference. Queue identity, source/data hashes, locks, heartbeat, recovery saves and auto-resume protections are retained.

GPU acceptance, resumed G3+A1 progress and final queue identity are recorded below. As with the earlier queue, this is a sequential process inside the existing interactive Slurm allocation, not a separate pending batch allocation. It survives SSH disconnection; canceling allocation4925 stops both stages and requires resuming the saved queue in a new four-H100 worker-2 allocation.

## Acceptance and active sequence

The full worker-2 GPU suite passed381tests with118dependency warnings in49.64s, no skips. The two-update mixed-extreme smoke passed:16complete episode exposures, max627frames/199735tokens, every recorded auxiliary weight0.1, finite losses0.48249042and gradients32.8011/32.8006, no CUDA/cgroup OOM or memory-limit events. Peak GPU reserved memory74.38GiB and peak host working memory381.47GiB. Checkpoint2 contains22validated recovery files. Readout bias updates are nonzero in all four linear layers (511,4096,1024,2048elements), proving optimization occurred beyond weight decay. The first update uses the existing LR warmup's zero LR; the next update changes the readout. This smoke validates execution, not convergence or navigation performance.

G3+A1 checkpoint2649 passed inventory validation with22files, world_size4, max_steps3852 and epoch0.6876947040. At2026-10-19 03:53:45UTC, all four ranks had resumed through2654, with40new unique exposures, no overlap with pre-pause episodes, finite navigation/state/relation losses, and zero CUDA OOM. The one-epoch3852/116schedule gate passed again. G3+A1's weights, optimizer, scheduler, RNG, source, sampling alias and recipe were not changed.

The active controller is in Slurm step4925.139 (`g3a1_then_g3_fixed`), worker-2. Its queue root is `/mnt/data/vmo-ai-task/anhdh35/g3-fixed01-seed430-worker2-20261019/queue`, with stage order `g3a1` then `g3_fixed`. The new G3 full output directory is `/mnt/data/vmo-ai-task/anhdh35/g3-fixed01-seed430-worker2-20261019/production`. G3 has not started full training yet; it starts automatically after G3+A1's complete epoch checkpoint/export is verified. The completed smoke is a separate disposable run.

The superseded G1-future controller exited without launching its second stage. Its launcher is archived as `launch_queue.superseded.sh`; the old launcher now exits with an explanatory message. `SUPERSEDED.json` points to the new queue. The original G1-future source and qualification are retained for possible later use.

The new queue has a restart-only launcher at `<queue-root>/launch_queue.sh`; it skips completed stages and resumes incomplete ones using the existing complete-checkpoint selection and locks. To restart after an allocation change, use `G3_ALLOCATION_ID=<new-four-H100-worker-2-job>` with that launcher under `nohup`; do not rerun the one-time qualification/replacement driver. No manual action is required for the normal G3+A1-to-G3 transition in the current allocation.

Evidence and exact operations are stored alongside this report. The immutable trained/queued source remainsb3a3ff4; subsequent branch commits only add the report and evidence, so publishing them does not modify live training.
