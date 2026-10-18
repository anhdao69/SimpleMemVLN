# Plain M5 with the G3 one-epoch decay schedule

Verified at **2026-10-18 04:28:24 UTC**: four-H100 qualification passed, and
the triple-layer predecessor resumed successfully from checkpoint **2227**.
All four ranks reached **2228/3852**, with zero CUDA OOMs. The worker-resident
controller is running in **Slurm step 4925.127 on worker-2**, with
`status=running-predecessor` and `m5_qualified=true`. Full plain-M5 training
will start automatically after that predecessor completes successfully.
M5 full-epoch training has not started yet.

The user requested a matched-schedule plain M5 control on October 18. This
preserves native M5 memory and original Qwen3.5-4B initialization. It removes
the G3 research section, teacher and auxiliary heads; it does not initialize
from a trained G3/L8 checkpoint. The resolved configuration is tested for exact
equality with the optimized G3+M5 configuration after removing `research`.

- Joint R2R + English RxR15deg: 30,815 episodes.
- Original Qwen snapshot: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
- Seed 429; four H100s; one episode/rank and GAS2; global batch eight.
- One epoch, 3,852 optimizer updates, 116 warmup updates.
- `cosine_with_min_lr`, final LR 10% of peak; backbone LR 5e-6,
  weight decay .01, Adam betas .9/.95, gradient clip 1.
- Full BPTT, canonical text action history, frozen vision encoder;
  M5 R4/B1200 committed-mean-L2 memory and unchanged causal selection.
- Same optimized CPUAdam/ZeRO2 CPU offload, pinned activation arena,
  attention batch32, vision batch8, eight persistent workers/rank,
  prefetch2 and BF16 staged pixels. No teacher or auxiliary loss.

Scientific source is unchanged at f4d1c2f216e68a960a5a691d21cbf1f08ea4a901,
in a separate detached worktree. Operational code and runtime configuration
live outside the frozen source tree and are separately hashed in the qualification
contract. The complete training-source seal and manifest SHA256 are rechecked.
The actual Transformers scheduler API was tested at updates 0, 116 and 3,852.

Campaign root: `/mnt/data/vmo-ai-task/anhdh35/m5-one-epoch-worker2-20261018`.
Production directory: `production/m5_joint_e1`. The old M5, layer-ablation
outputs and user workspaces are not reused or overwritten.

The controller requests the existing full-BPTT checkpoint-and-stop callback,
waits for the previous compute step to exit, consumes the request flag, and
runs M5 qualification. It then restarts the existing triple-layer controller,
which resumes the exact model/optimizer/scheduler/RNG checkpoint. After the
triple-layer subprocess exits, full epoch completion and layer identity are
checked before M5 can launch. A qualification failure still triggers resuming
the triple-layer predecessor, but blocks M5. A failed/incomplete predecessor
also blocks M5. Existing L16 remains stopped.

Qualification uses the existing baseline mode of `profile_vggt_g1` with M5
memory: three mixed R2R/RxR updates with saves, fresh-process checkpoint-2 to
update-3 recovery, complete tensor/loss comparison, and two repeated longest
RxR updates. These short tests override the optimizer-update limit only for
diagnostics. The full production launcher has no such override and retains
the 3,852/116 runtime schedule guard. Smoke checkpoints never initialize the
production run. Every diagnostic resolved configuration must match the plain
M5 production configuration. No auxiliary tensor or teacher identity is allowed.

Five queue tests cover normal order, qualification failure with predecessor
restart, failed predecessor, zero-exit incomplete predecessor, and competing
compute steps. Shell syntax, Python compilation and exact configuration/schedule
checks passed. No unconditional guarantee against future infrastructure failure
is claimed.

## Hardware qualification and recovery

All diagnostics used the exact plain-M5 resolved production configuration,
with only the explicitly short diagnostic update limit/fixture substituted.
There was no teacher identity, no auxiliary loss pair, and no research-auxiliary
tensor in the serving weights.

| Test | Result | Peak GPU reserved | Host working-set peak |
|---|---|---:|---:|
| Mixed R2R/RxR, 3 updates and full recovery saves | PASS | 46.29 GiB | 358.40 GiB |
| Fresh-process resume checkpoint 2 to update 3 | PASS | 38.28 GiB | 350.03 GiB |
| Longest RxR, 627 actions, 2 repeated updates | PASS | 73.02 GiB | 387.51 GiB |

Every supervisor reported zero new OOM/OOM-kill events and no host safety-guard
trigger. The longest test reserved 73.02 GiB of the 79.19 GiB visible device
memory; peak allocated tensors were 58.33 GiB. Host accounting includes a large
reclaimable file cache: total cgroup usage reached 768 GiB and generated reclaim
limit events, while the measured working set remained below 388 GiB. This is
not a claim that total cgroup usage remained below its limit.

Checkpoint recovery compared 5,174,964,736 tensor elements. Relative L2 was
**2.54103e-8**, maximum absolute difference **2.38419e-7**, and final logged
loss was **0.48531851172447205** in both runs. Weights were numerically close,
not bitwise identical. All three tested recovery checkpoints passed the complete
four-rank checkpoint validator.

The longest-repeat update interval was **84.01 s**. Mixed-reference intervals
averaged **97.45 s**, including a full recovery save every update; they are not
a representative full-dataset training-rate estimate. The resumed test contains
only one new update and has no steady-state interval. These diagnostics verify
the reused optimized setup; this task did not conduct a new speed search or
establish a full-epoch ETA/navigation result.

Evidence: [qualification_and_resumed_predecessor.json](evidence/qualification_and_resumed_predecessor.json)
contains the complete configuration/source/operation-file hashes, measured
results, recovery comparison, checkpoint pause record, live Slurm steps,
fresh heartbeat and resumed per-rank progress. The original triple supervisor
also printed `SCHEDULE_GATE_PASS (3852, 116)` on all four ranks. The consumed
checkpoint-stop request is archived outside the training directory, so it
cannot immediately stop the resumed run again.

## Unattended handoff

Allocation 4925 has no time limit. The controller's live process owns the
resumed triple subprocess and will launch M5 itself after validating the
predecessor's successful epoch completion. No additional command or SSH session
is needed for that handoff. L16 remains stopped. Production saves at update 10,
every 100 updates and the epoch boundary, with automatic checkpoint resume on
relaunch into the same identity-bound output directory.

The outer queue/heartbeat are in the M5 campaign root; the nested original
triple queue remains in the layer campaign's `production` directory. Use the
outer controller for future queue recovery; running only the old triple helper
would omit M5. Do not start another controller while step 4925.127 is alive.

The worker-resident controller and heartbeat do not depend on SSH remaining
connected. The Slurm allocation must remain alive. If it is cancelled, saved
checkpoints remain recoverable; a replacement worker-2 allocation and controller
restart are required. The current controller intentionally asserts allocation
4925. Replacing it requires a deliberate allocation-identity update and
revalidation of the operation-file qualification hashes; editing a deployed
operation silently invalidates its qualification seal.
