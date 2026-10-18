# G3+M5 native-state supervision-layer ablations

## Latest status — plain M5 queued after triple, October 18

At the user's request, triple-layer training saved a complete checkpoint at
**2227** and paused briefly for plain-M5 qualification. All M5 qualification
gates passed, including mixed updates, fresh-process recovery and repeated
longest RxR episodes. Triple resumed and reached **2228/3852 on every rank**
at 2026-10-18 04:28:24 UTC, with zero CUDA OOMs.

The active outer controller is **4925.127 on worker-2**, rooted at
`/mnt/data/vmo-ai-task/anhdh35/m5-one-epoch-worker2-20261018`. It resumes the
existing triple controller, waits for its successful full epoch, then
automatically trains plain M5 from the original base with G3's one-epoch
3,852-update/116-warmup schedule. L16 remains stopped. The new M5 output is
separate; no scientific source changed. See the
[M5 report and evidence](../M5_E1/implementation_report.md).

The recovery commands below describe earlier controllers. Use the new outer
controller when recovering the current triple-then-M5 sequence; restarting
only the old triple helper does not queue M5.

## Historical status — user-directed queue change, October 17

L8 completed its one joint epoch (3,852 updates) in approximately **14h25m**,
with mean wall time **13.4564 s/update**, action-weighted training loss
**0.12704034**, and zero GPU/host OOM events.
Published serving weights: [G3_M5_L8/epoch-1](https://huggingface.co/anhdao69/Streaming_VGGT/tree/main/G3_M5_L8/epoch-1).
Immutable revision: `792a0c959d8e6be8aee20e9cfa560c4f4b5a6b9b`.
All **19 files** passed hash/size verification; **120 existing non-attributes
files** remained unchanged. The only automatic metadata addition was the LFS
rule for the new folder's tokenizer. The CPU export strictly reloaded the
navigation architecture and preserved all **724 serving tensors bit-for-bit**.
No new GPU replay of the completed model was run while L16 occupied the GPUs;
the earlier GPU serving qualification is explicitly labeled as such.

After verified upload, the user explicitly requested stopping L16 and running
the triple-layer arm next. L16 stopped after update **35**; its latest complete
recovery is **checkpoint-10**. Updates 11–35 would be replayed on resume.
No L16 checkpoint was deleted. The original queue exited after the intentional
stop; its blocked/FAIL record reflects that user cancellation, not a training
numerical failure. `production/l16/USER_STOPPED.json` records the reason and
preserved checkpoint. The interactive allocation and shell remain available.

The replacement controller now runs **L8+L16+L24 only**, in **4925.124** on
worker-2. At 2026-10-17 19:52:07 UTC, every rank had completed update **5/3852**,
with finite loss and no OOMs. It starts from original pretrained Qwen weights,
not L8 or stopped L16, and uses layers [8,16,24], seed 429, lambda 0.1 and the
same one-epoch schedule. L16 is explicitly omitted, not marked complete.

Operational records: `evidence/publication_and_user_override_20261017.json`,
`evidence/triple_user_start_20261017.json`; executable recipes in
`operations/publish_l8.py` and `operations/run_selected_arm.py`. The deployed
controller is at the campaign root. It reuses the original source seal,
per-arm identity, launcher, recovery and completion checks; scientific source
remains unchanged at f4d1c2f. Do not restart the original three-arm queue if
L16 is meant to remain stopped. To recover the triple run in allocation 4925,
use the same srun/environment command below but execute
`python "$ROOT/run_selected_arm.py"`; the helper's allocation assertion must
be updated deliberately if allocation 4925 is replaced. It resumes the triple
arm from its latest complete checkpoint and leaves L16 stopped.

## Initial launch verification

Verified at 2026-10-17 05:17:55 UTC.

The implementation is on `c2_vggt_omega_g3_m5_layers`. All **403 cluster tests**,
**29 source-bound qualification gates**, and the supplemental cross-arm
initialization audit passed. The automatic sequential queue is running on
**worker-2**, Slurm step **4925.101**, inside allocation 4925.
L8 has reached optimizer update **12 / 3852** and has a
complete four-rank recovery checkpoint at update 10. L16 and L8+L16+L24 follow
in that order, only after their predecessor successfully completes its epoch.
The three full training runs are not all finished; full-epoch navigation
accuracy and ablation conclusions are not yet available.

Scientific source: `f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`. Later commits
contain reports/operational evidence, not changes to qualified training code.
The starting branch was `c2_vggt_omega_g3_m5` at `295e85e`; the completed L24
baseline's scientific implementation was `9031d75`. Existing branches, outputs,
and training jobs were not modified. Only our disposable preliminary
qualification step was replaced after the queue review fix.

## Controlled experiment

| Arm | Zero-based native GDN layers | Independent readout heads | Auxiliary parameters |
|---|---|---:|---:|
| Existing baseline | 24 | 1 | 10,526,368 |
| L8 | 8 | 1 | 10,526,368 |
| L16 | 16 | 1 | 10,526,368 |
| L8+L16+L24 | 8, 16, 24 | 3 | 31,579,104 |

The three new runtime YAMLs differ from `g3_m5_omega_h4.yaml` only in the
supervised-layer field; this exact equivalence is tested. Every head uses the
original architecture and seed 429, with independent parameter storage.
Identical head initialization controls the comparison; it is not parameter
sharing. Head construction preserves global CPU RNG and does not draw CUDA RNG.

Every full run starts independently from the original Qwen3.5-4B snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. No run initializes from the trained
L24 baseline, a diagnostic checkpoint, or another ablation's final weights.
The existing canonical text actions and action history are preserved.

M5 is unchanged: recent groups R=4, episodic token budget B=1200,
`committed_mean_l2_v1` scores, selector seed 429, no per-source cap, and the same
causal selection, old-token handling and bounded serving cache. The online
frozen FP32 VGGT-Omega teacher, preprocessing, sampling, targets and full BPTT
are unchanged. There is no PCA or offline feature extraction.

## Data and complete one-epoch schedule

- Joint R2R + English RxR-CE 15deg: 30,815 episodes, comprising 10,819 R2R and
  19,996 RxR episodes; 3,128,624 original action states.
- Manifest: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl`.
- Rechecked manifest SHA256:
  `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.
- Four H100 GPUs, one episode per rank per microbatch, GAS=2: global batch eight
  episodes, 3,852 optimizer updates, and one repeated tail slot.
- Seed 429; 116 warmup updates; **one-epoch** `cosine_with_min_lr`, minimum LR
  ratio 0.1. The one-epoch policy overlay replaces the older two-epoch defaults.
- Backbone LR 5e-6, readout LR 1e-4, weight decay 0.01, Adam betas (0.9,0.95),
  global gradient clipping 1.0.
- Lambda maximum 0.1, ramped over the original 771 updates.
- BF16 student; frozen visual encoder; complete episodes and full BPTT;
  non-reentrant decoder checkpointing; no cache during training.
- ZeRO-2 CPU optimizer offload and DeepSpeed CPUAdam, four OMP/MKL threads/rank;
  pinned activation arena above 65,536 tokens; maximum context 262,144.
- Eight persistent loader workers/rank, prefetch two, no loader pinning;
  BF16 staged pixels, vision batch eight, attention batch 32.
- Online resident FP32 Omega, teacher batch eight. Weight SHA256:
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.

## Implementation and causal gradient path

`research.student_layers` is a sorted, unique, nonempty list of zero-based
native GDN indices. Empty/duplicate/unsorted/noninteger, attention and
out-of-range selections fail validation; the actual backbone layer types are
checked again during model construction. The list is exclusive to G3_M5 and
cannot accompany the legacy scalar field.

Legacy `student_layer: 24` retains the old `research_aux.state_head` namespace.
Explicit lists create `research_aux.state_heads.<index>` using the unchanged
NativeStateHead: a 4160→512→4096 query MLP and 4096→1024→2048 readout decoder,
with the original five learned region embeddings and lag encoding.

Each selected decoder uses the existing G3 read-only FLA probes at the original
causal observation anchors. Queries use that layer's normalized native query
context, lag and region; teacher targets never enter queries or navigation.
The original query-context detach is retained. Gradients still flow through
the state reads, recurrent writes, earlier backbone computation and readout
parameters. Probe rows are removed before later decoders or M5 scoring.

Checkpoint closures bind **both decoder and readout head**, avoiding loop
late binding. Immutable M5 selection plans remain outside recomputation.
Each selected head follows the same uniform empty-support graph, including
zero-pair ranks, so ZeRO readiness remains consistent.

Teacher targets are computed once per episode and reused across layers.
The original maximum-eight pair sampling, lags 8–32, collected-RGB trajectory
identity, teacher patch offset 17, layer-23 descriptors, unpositioned 2048-D
regional targets and five-region loss are unchanged.

## Loss normalization

For N selected layers, the forward returns the mean of their auxiliary
numerators. For the triple arm this is `(L8 + L16 + L24) / 3`.
The original trainer then applies `world_size * lambda(update) / global_pairs`,
where `global_pairs` counts sampled pairs across all ranks and both accumulation
slots. Pair counts are **not multiplied by the number of layers**.
`lambda(update) = 0.1 * min(1, update / 771)`.

This keeps the total auxiliary weight at 0.1 rather than increasing it to 0.3.
All-empty updates retain connected zero gradients. Independent CPU references
verify the multi-layer backbone/head gradients against the mean of separate
single-layer runs; real NCCL/DDP tests verify heterogeneous and all-empty
normalization with three heads and GAS=2.

## Tests actually executed

The frozen reviewed source passed all **403 tests/vln tests** on worker-2
(62.65 seconds, no skips). Before deployment, targeted local suites passed,
including the 34-test regression suite after the queue identity fix. The broad
Mac run had 364 passes and 35 CUDA skips, with four environment failures caused
by absent Slurm, DeepSpeed, Omega and the cluster base-model path. Those same
tests passed in the complete cluster environment.

The 29 qualification gates comprise the full CPU/CUDA suite, the real
three-head distributed reduction test, and nine gates for **each** arm:

1. Three updates with empty/nonempty support mixed in accumulation slot 1.
2. Three updates with mixed support in slot 0 after actual length sorting.
3. Six complete mixed R2R/RxR updates (48 episode exposures).
4. Three longest-RxR updates: 627 actions / approximately 199,735 tokens,
   repeated across every rank and both accumulation slots.
5. Three longest-R2R updates: 183 actions, similarly repeated.
6. Uninterrupted three-update reference with full recovery saves.
7. Fresh-process checkpoint-2 → update-3 continuation and save.
8. Per-head/backbone recovery comparison, including nonzero bias updates.
9. Strict serving export, both-dataset offline/streaming/greedy replay,
   reset/retry and M5 cache/selection checks.

The longest tests cover six maximum-length episodes per GPU across three
optimizer updates, including repeated activation-arena reuse. Checkpoint
recovery is separately tested on the complete mixed fixture. CUDA tests use
the actual pinned FLA kernels; CPU recurrence tests are not substituted for
hardware tests. All gates are bound to hashes of src/scripts/configs/train/tests.

The supplemental checkpoint-1 audit verifies learning rate zero, bitwise-equal
backbone tensors across all three arms, identical initial values for every
readout head, identical data contracts, and identical first-update episode
order and auxiliary support counts on all four ranks. Two early live-reader
attempts rejected checkpoint-1 file-size inventories while diagnostic saving
was still in progress. After the files stabilized, the unchanged strict
validator and tensor audit passed; no checkpoint repair or validation bypass
was used. Run this supplemental audit after the reference save finishes.

## Measured warm update times

These are diagnostic complete-episode fixtures, not estimates that every
production update takes the same time. Startup and checkpoint writes are
excluded from the warm intervals below.

| Arm | Mixed fixture s/update | Longest R2R s/update | Longest RxR s/update |
|---|---:|---:|---:|
| L8 | 18.8299 | 23.9888 | 82.8307 |
| L16 | 18.6822 | 24.0741 | 80.6622 |
| L8+L16+L24 | 19.0662 | 24.4795 | 82.7879 |

Historical L24 G3+M5 on the same mixed fixture averaged 18.5025 s/update.
The triple arm adds about 3.05% in this bounded comparison. The original L24
full epoch took about 14h22m; approximately 14–16 hours per new epoch is a
planning range, not a completion promise. Actual timing depends on episode
lengths and shared-storage checkpoint latency.

## Longest-RxR memory measurements

All rows use three repeated maximum-length updates. GPU capacities are
79.189 GiB. Host limits are 768 GiB; the working-set guard is 691.2 GiB.

| Arm | Allocated GPU GiB | Reserved GPU GiB | Sampled device use GiB | Host working set GiB | New OOM / OOM-kill |
|---|---:|---:|---:|---:|---:|
| L8 | 67.874 | 76.055 | 78.288 | 502.57 | 0 / 0 |
| L16 | 67.874 | 76.055 | 78.288 | 496.57 | 0 / 0 |
| L8+L16+L24 | 67.913 | 76.055 | 78.288 | 479.44 | 0 / 0 |

File cache is included in total cgroup usage but excluded when calculating
reclaimable working-set headroom. Total usage sometimes reached the cgroup
limit and caused reclaim/limit events; there were no OOM or OOM-kill events.
For example, the triple stress recorded 24,827 limit events, a 479.44-GiB peak
working set, and zero OOM events. Resource guards remain enabled in production.
These tests establish observed safety for the audited inputs/runtime; they
cannot guarantee safety after input, software or resource changes.

## Recovery results

| Arm | Global relative weight L2 error | All readout heads bitwise equal | Final logged loss difference |
|---|---:|---|---:|
| L8 | 2.506e-08 | True | 0.000e+00 |
| L16 | 2.492e-08 | True | 0.000e+00 |
| L8+L16+L24 | 2.503e-08 | True | 0.000e+00 |

Predeclared BF16 gates were global relative L2 <=1e-5, each head <=1e-4 and
loss difference <=1e-4. Every head also had nonzero query/decoder bias updates
between steps 2 and 3, demonstrating learning beyond weight decay. The saved
inventory includes model, optimizer, scheduler and per-rank RNG state.

## Serving consistency

All configured training heads are validated before export, then removed along
with research configuration. Strict reload uses the original native policy
architecture and unchanged M5 memory configuration, with no teacher dependency.
Source-versus-export offline, streamed, greedy outputs and selections are exact.

| Arm | Dataset | Exact source/export | Probe/native RMS | Offline/streaming RMS | Peak resident KV tokens |
|---|---|---|---:|---:|---:|
| L8 | r2rce | True | 0.02649 | 0.06198 | 2539 |
| L8 | rxrce | True | 0.01697 | 0.04845 | 2579 |
| L16 | r2rce | True | 0.02009 | 0.06194 | 2539 |
| L16 | rxrce | True | 0.01356 | 0.04898 | 2579 |
| L8+L16+L24 | r2rce | True | 0.02400 | 0.06588 | 2539 |
| L8+L16+L24 | rxrce | True | 0.01758 | 0.05328 | 2579 |

Training-probe/native and offline/streaming comparisons preserve the existing
G3 BF16 tolerances: atol 0.75 plus rtol 0.03, with RMS <=0.15. The absolute
maximum alone is not the tolerance criterion. Numerical M5 selection overlap
is recorded in each export artifact; probe insertion is not claimed to be
bitwise neutral under BF16 kernels. The replay uses 17 frames from each dataset
and validates bounded KV residency (guard 4784), R4 recent groups, greedy
reset/retry and source/export identity. It is not a navigation quality metric.

## Queue, isolation and restart

Campaign root:
`/mnt/data/vmo-ai-task/anhdh35/g3-m5-layers-worker2-20261017`.
There are three separate detached worktrees:
`source-f4d1c2f-l8`, `source-f4d1c2f-l16`, `source-f4d1c2f-l8_l16_l24`.
Outputs are respectively `production/l8`, `production/l16`, and
`production/l8_l16_l24`. The L24 baseline and unrelated outputs are untouched.

A worker-resident launch gate waited for the qualification seal and for other
compute steps to finish, then execed the sequential production controller.
The controller launches one local training process at a time. It advances only
after successful child exit, supervisor PASS, a validated four-rank checkpoint
at step 3852, and a completed epoch. Failed or incomplete runs block successors.
A completed checkpoint must also have the expected supervised-layer identity.
Each output directory is bound to its arm and qualified source/config hashes;
existing unbound or differently bound directories are refused.

The queue and individual launchers use locks. Status and heartbeat files are
atomic. The sequence is owned by a worker-2 Slurm step, not a login polling
watcher, and continued through intermittent SSH/VPN outages during qualification.
Cancelling the allocation or losing the node still stops its processes.
Restarting the same controller under a replacement worker-2 allocation resumes
the current arm from its latest complete checkpoint and verifies/skips finished
arms. The allocation must remain available for unattended sequential training.

Production saves at update 10, every 100 updates, and the epoch boundary. It
retains two rolling recoveries plus epoch checkpoints in each run's own folder.
A real-process controller test forced L16 to fail, verified that the last arm
never started, then restarted and observed L8 skipped, L16 retried and the final
arm executed. A further restart launched no completed arm again.

Restart command, after confirming the replacement allocation is idle:

```bash
ROOT=/mnt/data/vmo-ai-task/anhdh35/g3-m5-layers-worker2-20261017
CODE="$ROOT/source-f4d1c2f-l8"
srun --jobid=4925 --overlap --ntasks=1 --gpus=4 --cpus-per-task=60 \
  --nodelist=worker-2 --job-name=g3-layers-sequence \
  env G1_CODE_DIR="$CODE" \
  G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash "$CODE/train/g1_omega_env.sh" python -m scripts.vln.queue_g3_layers \
  --root "$ROOT/production" \
  --seal "$ROOT/qualification-f4d1c2f/qualification_pass.json" \
  --sources "$ROOT/source-f4d1c2f-l8" "$ROOT/source-f4d1c2f-l16" \
            "$ROOT/source-f4d1c2f-l8_l16_l24"
```

Replace job 4925 only if using a replacement allocation. Do not launch a second
controller while the first is alive. Monitor `production/queue_state.json`,
`production/heartbeat.json`, per-arm supervisor logs and
`production/<arm>/train/profile_rank*.jsonl`.

## Review and evidence

An independent whole-branch review found no critical scientific issue. Its
important finding—completed arms could be skipped without checking layer/source
identity—was fixed with failing-then-passing regression tests before qualification.
One minor diagnostic limitation remains: after an interrupted qualification
profile, preserve/rename its failed fixed-name directory before retrying that
gate. Successful qualification gates may be reused only with identical source
hashes. Production checkpoint resume is unaffected.

Implementation choices: identical seed per independently stored head controls
initialization across arms; supporting arbitrary qualified GDN indices does not
imply hardware qualification of indices beyond the three requested arms.

`evidence/` contains per-arm mixed/longest measurements, recovery comparisons,
serving checks, the complete qualification seal, initialization audit and the
verified production-start record. Detailed per-rank exposures, losses, memory,
source manifests and checkpoint inventories remain under the campaign root.
No full-epoch evaluation results or guaranteed finish time are claimed here.
