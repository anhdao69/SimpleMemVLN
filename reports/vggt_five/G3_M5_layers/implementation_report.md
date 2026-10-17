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
