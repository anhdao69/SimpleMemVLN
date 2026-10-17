# G3+M5 native-state supervision-layer ablations

Status: implementation complete; H100 qualification and independent review in
progress. No production run has been declared qualified or complete yet.

## Scientific comparison

The reference is the completed L24 G3+M5 baseline, branch
`c2_vggt_omega_g3_m5`, baseline code commit 9031d75 and final baseline
report commit 295e85e. This branch implements three independent runs: L8, L16,
and L8+L16+L24. All indices are zero based. These are native GDN layers,
not the attention layers or M5's scoring layers.

The only research change is where the native state is supervised. Each layer
has its own original NativeStateHead (10,526,368 parameters), initialized with
seed 429 in isolated CPU RNG scope. Heads have identical starting tensors but
separate storage; constructing additional heads does not advance global RNG.
The original pretrained Qwen3.5-4B snapshot is used independently for every arm.
No arm starts from the baseline's trained checkpoint or another ablation.

M5 retains recent groups R=4, token budget B=1200, committed_mean_l2_v1 scores,
selector seed 429, no per-source cap, and its existing causal selection and
serving cache. Online frozen FP32 VGGT-Omega, preprocessing, unpositioned
regional targets, original pair sampling, anchors, lags, and target reuse are
unchanged. There is no PCA, offline extraction, truncation, or detached BPTT.

## Loss and implementation

Explicit `research.student_layers` enables `research_aux.state_heads.<index>`.
The legacy scalar `student_layer: 24` keeps the old `research_aux.state_head`
namespace. Empty, duplicated, unsorted, noninteger, attention, and out-of-range
layer selections are rejected. Model construction also checks actual decoder
layer types.

At each selected decoder, the existing native state read uses read-only FLA
probes at the original causal anchors. Probe rows are removed before passing
hidden states onward. The checkpoint closure binds both the decoder and its
readout head; M5 plan construction stays outside checkpoint recomputation.
Query context remains detached exactly as in G3, while the recurrent state,
state writes, readout and backbone receive the existing full-BPTT gradients.

For N selected layers, forward returns sum(layer numerator)/N. The existing
trainer then multiplies by world_size * auxiliary_weight / global_pair_count,
where the count covers all ranks and both accumulation slots. The denominator
counts original sampled pairs, not pairs times layers. Thus N=3 implements
(L8+L16+L24)/3 and preserves lambda maximum 0.1 with the original 771-update ramp.
Every selected head participates in the uniform zero-pair graph.

## Identical training recipe

- One joint R2R + English RxR15deg epoch; 30,815 original episodes.
- Manifest SHA256: 5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933.
- Four H100 GPUs, one episode per GPU per microbatch, two accumulation slots:
  eight episodes per optimizer update, 3,852 updates with one repeated tail slot.
- Seed 429; 116 warmup updates; one-epoch cosine_with_min_lr schedule, minimum
  ratio 0.1; backbone LR 5e-6, auxiliary LR 1e-4, weight decay 0.01.
- ZeRO-2 CPU optimizer offload, DeepSpeed CPUAdam, full BPTT and pinned activation
  arena; eight loader workers/rank, prefetch two, BF16 staged images, vision batch
  eight and attention batch 32. Online Omega stays resident, batch eight.
- Base snapshot: Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
- Online Omega weight SHA256:
  c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934.

The three runtime YAMLs are tested for exact equality with the baseline after
replacing only the supervised-layer field. The original baseline YAML is intact.

## Tests and qualification gates

Existing targeted baseline: 12 passed in 11.27 seconds. Initial implementation
causal/gradient/checkpoint suite: 21 passed in 15.76 seconds. Combined targeted
configuration, export, queue and baseline suite: 31 passed in 18.33 seconds.
After canonical-key validation, focused new/export/queue suite: 28 passed in
8.30 seconds. These are CPU tests with an independent recurrence reference;
they do not substitute for real FLA/ZeRO testing.

The CUDA qualification runs the entire tests/vln suite and a real four-rank
NCCL/DDP reduction check with three heads, heterogeneous and all-empty support.
For EACH arm it requires mixed support in both actual accumulation slots,
six mixed complete-episode updates, three repeated longest-RxR updates, three
repeated longest-R2R updates, fresh-process checkpoint-2 to checkpoint-3 recovery
against an uninterrupted three-update reference, and serving export/replay.
The longest RxR episode has 627 actions and approximately 199,735 tokens;
three updates process six such complete episodes per GPU.

Recovery compares every readout head individually, including nonzero bias
updates, not just aggregate parameter movement. Original BF16 gates are retained:
relative weight L2 <=1e-5 globally and <=1e-4 per head, final loss difference
<=1e-4. Serving export validates all configured head namespaces, strips them,
strictly reloads the native policy, and checks exact source/export offline,
streaming, greedy and selection equality, plus existing replay numerical gates.

## Queue and isolation

Remote campaign: `/mnt/data/vmo-ai-task/anhdh35/g3-m5-layers-worker2-20261017`.
There are separate detached source worktrees for all three arms. A single
worker-resident controller runs L8, then L16, then all three layers. It waits for
child exit and verifies a complete step-3852 recovery checkpoint, epoch and
supervisor success before advancing. Failures block successors. A queue lock
prevents duplicate controllers; each run also has a launcher lock. Status is
atomic and a worker heartbeat records controller liveness. Qualification binds
all executable/config/test source hashes, and each launch rechecks the seal.

The controller is owned by one Slurm step on worker-2, not a login-node polling
watcher. Cancelling the allocation still stops its step: restarting the
controller in a replacement worker-2 allocation resumes the current arm from
its latest complete checkpoint, and verifies/skips finished arms. The queue
cannot promise continued execution after allocation cancellation or node loss.
No existing job is cancelled, and no existing output directory is reused.

## Measurements and production results

Pending actual H100 qualification. Baseline full L24 training took about
14h22m, averaging 13.39 seconds/update; this is a reference, not a measured ETA
for these new ablations. No validation metrics for the new models exist yet.

## Review and environment checks

A fresh reviewer found no critical issue and one important operational issue:
completed-arm skipping initially checked epoch completion without layer/source
identity. Two failing regression tests demonstrated the missing guard. The
fixed queue binds each run directory to its arm, source path and qualified file
hashes before launch or skip, refuses unbound existing outputs, and checks final
checkpoint supervised-layer identity. The resulting targeted suite passed all
34 tests in 16.23 seconds.

The complete suite on the Mac produced 364 passes, 35 CUDA skips and four
missing-environment failures: no Slurm `squeue`, no DeepSpeed, no Omega package,
and no cluster `VLN_MODEL_PATH`. The preliminary cluster checkout passed all
401 tests and the three-head distributed reduction test before qualification
was restarted on the reviewed commit f4d1c2f. Those preliminary results are
not substituted into the final source-bound qualification seal.

Deferred minor from review: a failed/interrupted qualification profile leaves
its fixed experiment directory. Preserve that failed directory under an attempt
suffix before rerunning the same gate; successful gates may be reused only with
an identical source manifest. Production auto-resume is a separate path and
is not affected by this diagnostic-directory limitation.

A separate real-process queue exercise launched disposable child processes,
forced L16 to exit 17, and verified that the third arm did not start. A fresh
controller process then skipped completed L8, retried L16, and ran the final
arm. Another fresh restart launched nothing. Launch sequence was
`l8, l16(failed), l16(retry), l8_l16_l24`; both restart controllers exited zero.
This tests process-level ordering/failure/restart, not cluster cancellation.

The current joint manifest was rehashed on the cluster and exactly matched the
baseline SHA256 above. Final reviewed source f4d1c2f passed all 403 cluster tests
in 62.65 seconds and its three-head distributed reduction gate passed. Further
per-arm qualification remains pending at the time of this record.

## Controller launch / recovery command

After the qualification seal exists and all other GPU steps have finished,
run the following inside the same idle four-H100 worker-2 allocation (replace
the allocation number only if it has changed):

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

Use the same production root and source paths when recovering. Do not launch
another controller while the first is alive. `production/queue_state.json`
records the current arm, `heartbeat.json` records worker liveness, and each
arm retains complete recovery checkpoints with model, optimizer, scheduler,
per-rank RNG and resume contract. The first production save is update 10,
followed by every 100 updates and the epoch boundary; bounded older recovery
pruning applies only within that arm's own output directory.

## Automatic launch gate

The deployment recipe in `operations/launch_after_qualification.py` is a
CPU-only waiting process in a separate Slurm step of the same allocation.
It waits for the successful source-bound seal, the qualification step to exit,
and any unrelated compute step to finish before replacing itself with the
production controller. A failed qualification or an exited qualifying step
without a seal blocks training. The queue then independently verifies the
seal and run identities. Six safety cases passed using
`python3 reports/vggt_five/G3_M5_layers/operations/check_launch_gate.py`.
This gate allows qualification-to-training handoff without another SSH session.

The automatic launch gate is confirmed active as Slurm step **4925.101** on
worker-2, waiting on qualification step **4925.100**. It has no training child
while qualification is pending. The existing interactive shell remains .0.

Initial three-head mixed fixture measurement: six complete optimizer updates,
48 episode exposures, all four ranks, no OOM or RAM/disk guard event. Five warm
intervals were 19.1599, 19.2868, 19.0838, 18.9248 and 18.8759 seconds; mean
**19.0662 s/update**. The prior L24 baseline's same-fixture mean was 18.5025 s,
so this bounded measurement is about **3.05% slower**, not three times slower.
Peak allocated/reserved GPU memory: **39.8478 / 50.2012 GiB**; sampled host
working set: **330.96 GiB**. This fixture timing is not a full-epoch ETA.

Three-head longest-RxR stress passed **three complete updates**, with all four
ranks processing the 627-action episode twice per update. Warm intervals:
84.2468 and 81.3290 seconds (mean 82.7879). Peak allocated/reserved GPU memory
was **67.9133 / 76.0547 GiB**; sampled maximum device use was **80,167 MiB**
(78.288 GiB) on a 79.189-GiB device. Host working-set peak was **479.44 GiB**.
Total cgroup usage, which includes reclaimable file cache, reached the 768-GiB
limit and recorded 24,827 reclaim/limit events; there were **zero OOM and
zero OOM-kill events**, and the working-set guard did not trigger. This is
measured headroom, not a guarantee for changed inputs or resource conditions.

All nine three-head qualification gates passed. Fresh-process recovery produced
**bitwise-identical weights for each of L8, L16 and L24's readout heads** and
identical final logged loss (0.4841836094856262). Backbone/global relative L2
error was 2.503e-8, maximum absolute difference 2.384e-7, comfortably inside
the predeclared 1e-5 gate. Every head had nonzero query and decoder bias updates;
268,005 head tensor elements changed from step 2 to step 3. Export and replay
passed for both datasets. These are qualification results, not full-epoch
navigation evaluation metrics.
