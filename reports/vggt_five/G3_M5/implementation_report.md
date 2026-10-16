# G3 + M5 implementation and training qualification

Status: implementation and all 15 H100 qualification gates complete. The
combined epoch is queued behind R1 in allocation 4925 on worker-2. R1 was
safely restored from checkpoint 3,373. Production requires the complete
source-bound qualification and a successfully completed R1 epoch; it has
not yet started at this report snapshot (2026-10-16, about 04:50 UTC).

## Reference audit and experimental question

The isolated branch `c2_vggt_omega_g3_m5` starts from latest fetched
`origin/c2_vggt_omega_g3` b42cfe5. The M5 reference is latest fetched
`origin/c2_ablations` 1b93100. Original user worktrees, untracked plans and
running R1 source are preserved. G3's production scientific source is
22593831c80b742ee77b478ce7cb76fb44bc328b; M5's is
a340e131b63c9ab7846690acdb369599462076c4.

The latest completed R2R unseen evaluation covers 1,839 episodes. G3 achieved
51.984774% SR /48.095941% SPL; M5 achieved51.005982% SR /47.438938% SPL.
These are promising individual systems, not evidence that their combination
improves either score. The full RxR result was not present in the inspected
report. M5c remains incomplete, so the specific value of its importance
scorer over random selection is not established by that table.

Inspected sources include G3's native-state and online-Omega plans,
implementation report and evidence, the episodic plan/ablation report,
scientific and runtime YAMLs, model/teacher/dataset/trainer paths, selection
and serving code, numerical tests, checkpoint/export utilities and launchers.
The M5 scorer, selector and arm configuration already exist unchanged in G3's
base. No broad merge from the ablation branch was necessary.

| Property | Trained M5 reference | Trained G3 reference | New G3_M5 |
|---|---|---|---|
| Initialization | Original Qwen3.5-4B | Original Qwen3.5-4B | Original Qwen3.5-4B |
| Recent groups including current |4|8|4|
| Older visual K/V budget/layer |1,200|0|1,200|
| Old-token priority |Committed-mean-L2|None|Unchanged M5|
| Recurrent state |Native GDN|Native GDN + training probes|Native GDN + training probes|
| Supervised layer |None|Decoder24 GDN|Decoder24 GDN|
| Online teacher |None|Frozen VGGT-Omega|Same frozen VGGT-Omega|
| Action text/history |Canonical text, appended|Canonical text, appended|Canonical text, appended|
| Step lane/PCA |Neither|Neither|Neither|
| Epoch-1 LR denominator |7,704 updates|3,852 updates|3,852 updates|
| Warmup |232 updates|116 updates|116 updates|
| Epoch-1 exposure |One joint pass|One joint pass|One joint pass|

The historical M5 epoch-1 checkpoint is halfway through a two-epoch schedule.
This new experiment intentionally uses the G3 one-epoch schedule requested by
the user. Consequently, a comparison against historical M5 changes both
supervision and schedule. A matched-schedule M5 control would be needed to
isolate the supervision's causal contribution. This task does not silently
launch that additional training run.

## Combined architecture

Full attention at layers3,7,11,15,19,23,27,31 keeps the instruction prefix,
four recent complete observation/response groups including the current
observation, and up to1,200 selected older visual K/V rows. Only original
visual rows enter the bank. Scores are detached FP32 committed-write residual
norms, mean over heads, computed at the preceding GDN layers
2,6,10,14,18,22,26,30 with the original block64 backend and segmentation.
Logical positions, eviction-before-observation, deterministic tie breaking,
seed429, no per-source cap and the4,784-token structural KV guard remain M5.
K/V gradients remain connected across the complete episode. The detached
score is not a differentiable selector, exactly as in M5.

At decoder24 the existing G3 state read is added inside the same episodic
forward loop. The layer performs native projections, convolution, gates and
head repetition. For each sampled pair it inserts five post-write read rows
inside the FLA recurrence only. Their effective decay and write gates are
literal zero, keys are unit vectors, values are zero. Queries come from the
learned head, using detached normalized native query context plus region and
lag embeddings. The teacher target never enters the query.

All added rows are removed before native gated RMS normalization, output
projection, residual addition and decoder MLP. No later scorer or attention
layer sees probe rows. Decoder24 is not an M5 scoring layer. Decoder26's
scorer sees ordinary real-token hidden states after decoder24, as it should.
The selected decoder's checkpoint closure binds the layer explicitly; M5's
selection plans are computed outside checkpoint recomputation and remain
immutable. No hook-based global state or recurrent-state snapshots are used.

M5's old bank may retain tokens from a G3 source observation even when its
lag exceeds the recent window. Therefore auxiliary supervision is a request
for geometric information in native state in the presence of episodic
attention. It does not prove that this information survived exclusively in
recurrent memory. Auxiliary gradients legitimately reach upstream episodic
attention and selected old K/V rows.

## Teacher, targets and objective

The resident, frozen FP32 teacher is VGGT-Omega-1B-512. The original G3 teacher
adapter, region construction, sampling and teacher identity are unchanged.
Weights SHA256 is
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
The640x480 RGB input becomes448x592 under the existing balanced512 loader;
layer23 output has17 special tokens followed by28x37 patches. Pooling gives
15x20 cells, normalized over2048 channels with epsilon1e-6. Four quadrants
and an area-weighted global mean give five2048-dimensional targets. No
positional term, PCA, cached features or preextraction is added.

Sampling uses collected RGB trajectory identity, not instruction text, with
lags8..32 and bin quotas3/3/2. At most eight pairs and eight distinct source
images are used per episode. Repeated observation anchors are allowed. The
anchor is immediately before the first supervised response token. The
query MLP is4160→512→4096, reshaped to32x128; the read decoder is
4096→1024→2048 after per-head FP32 RMS normalization. Together the existing
head has10,526,368 parameters. Both MLPs are training-only.

The cosine distance is averaged across the five regions, summed over valid
pairs, then normalized by the actual global pair count over all ranks and
both accumulation microsteps. Navigation retains its separate global action
normalization. Auxiliary maximum weight0.1 ramps over771 optimizer updates.
Backbone LR5e-6, auxiliary LR1e-4, weight decay0.01. Teacher and query context
are detached; native state and upstream policy receive auxiliary gradients.

## Empty-support safety

Episodes with eight or fewer observations have no valid lagged pairs. They
still train navigation. A separate sum-of-parameters zero loss can give
ZeRO2 a different gradient readiness order, which previously caused R1 to
stall. The combined path instead uses the same query, tensor-packing and
decoder graph with zero-sized tensors. There are no fake probe rows, targets
or examples. Every head parameter participates with zero gradient when
appropriate. Historical G3 retains its original default path, and the
shared sampling/probe helper is unchanged so teacher identity is preserved.

CPU tests compare parameter-readiness order for supported and empty cases,
checkpointed/noncheckpointed gradients, causal outputs and M5 selection
plans. Actual four-rank ZeRO2 qualification uses the original failing R1
batch and a second fixture containing five complete empty-support episodes
plus three supported episodes. Real per-rank exposure counts must show
heterogeneity in each requested accumulation slot. Simply rotating manifest
rows does not work because the sampler sorts by encoded length; independent
review caught and corrected that test-design error.

## Training recipe and performance policy

Joint R2R + English RxR15-degree manifest:30,815 episodes,3,128,624 actions;
R2R10,819 /631,244 and RxR19,996 /2,497,380. Four ranks, one whole episode per
rank, accumulation2 gives global8. The one repeated tail episode fills30,816
slots. Maximum audited sequence199,735 tokens /627 actions, within262,144.
There is no truncation, shorter training substitute or dropped long episode.

The production schedule is exactly3,852 updates with116 warmup updates,
`cosine_with_min_lr` ending at0.1 of peak LR. This is configured through the
existing G3 one-epoch policy; production ScheduleGuard checks the actual
Trainer denominator and warmup. Diagnostic update limits are not production
schedule evidence and their weights never initialize the full run.

Start from G3's optimized runtime: ZeRO2 CPU optimizer offload with vectorized
CPUAdam, pinned activation arena for sequences>=65,536 tokens, full decoder
checkpointing, frozen visual backbone/merger, BF16 staged student pixels,
vision and teacher batches8, four OMP/MKL threads/rank, eight persistent data
workers/rank, prefetch2, no per-update malloc trim. Attention batch32 is the
initial setting, with16/64 comparisons on identical real episodes. All
measurements include complete episode training and teacher work; warm
fixture estimates are distinguished from production wall time.

The existing allocation is4925 onworker-2, four H10080GBs,60CPUs and768GiB RAM.
Every GPU command explicitly uses one Slurm task; torchrun creates four
ranks. No other worker is permitted. No allocation is cancelled.

## Qualification, recovery and serving

Qualification covers the pinned FLA numerical tests, actual online teacher,
combined CPU invariants, both empty-support slots, mixed R2R/RxR episodes,
longest episodes of both datasets, fresh-process checkpoint2→3 recovery,
real distributed pair normalization and export/replay. Performance profiles
record source and fixture hashes, per-rank exposures, losses, gradients,
allocated/reserved GPU memory, host working set and OOM counters.

Recovery stores model plus auxiliary head, optimizer/scheduler and all-rank
RNG/sampler state, with a validated file inventory. The production launcher
saves at10 then every100 updates, retaining two rolling recoveries plus
epoch boundaries. Same-run `auto` resume selects only complete checkpoints.
Serving export removes exactly `research_aux.*` and the research config;
all backbone tensors and the complete M5 memory metadata remain. A serving
export is not a resumable training checkpoint.

Export audit requires exact source/export offline logits, forced-history
streaming logits, greedy outcomes/scores and selected-token IDs. It checks
reset/retry idempotence and bounded M5 KV residency. Offline-versus-streaming
BF16 tolerance remains atol0.75, rtol0.03 and RMS<=0.15, not a standalone
0.75 maximum-error cap. Dynamic M5 priorities may choose slightly different
near-tied rows online; overlap is recorded instead of hiding it. The audit
also compares the full probe-expanded model against ordinary M5 forwarding.
A three-update base policy need not yet generate a valid canonical action;
that quality failure is recorded and must be identical across source/export.

## Operational handoff and limitations

The durable login-side controller requests an R1 checkpoint and waits for a
complete four-rank inventory, successful supervisor report and idle GPU
allocation. An independent server-side watchdog resumes R1 if the controller
exits unexpectedly or exceeds its bounded qualification window. The first
handoff savedR1 at3362 but saw the Slurm step disappear before NFS exposed its
supervisor report; it started no GPU test and safely resumed R1 as4925.85.
The supervisor report subsequently confirmedPASS and zero OOM events. A
bounded publication wait now handles that filesystem ordering.

The combined production queue requires a successful complete R1
checkpoint3852 with epoch>=0.999 and matching topology. It then validates
its own qualification source hashes before using the same allocation. A
failed or partially saved R1 does not trigger training. Locks and immutable
source snapshots prevent duplicate launch and live-code drift. GPU test
results and review establish bounded engineering confidence, not a proof of
100% correctness or navigation improvement. Full evaluation remains needed.

The actual completed G3 checkpoint3852/navigation.json was inspected on the
server and confirms the stated one-epoch schedule, head settings and optimized
runtime. The current manifest byte SHA256 is
`5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.
Its path-independent fingerprint is
`f11fcceac19724e586f1b0a97828c388f619a88ee14d513bd8498856e9c2b0a9`,
exactly matching the M5 reference report. That verifies identical ordered
UIDs, instructions, actions, image basenames and validity flags under the
reference fingerprint. It does not hash image pixel contents across clusters.


## Measured runtime selection

On the same complete mixed fixture (eight episodes, 24 exposures across
three optimizer updates), attention microbatches 16, 32 and 64 gave warm
update means of 18.4615, 18.5025 and 18.5858 seconds respectively. Individual
intervals were [18.4680, 18.4550], [18.6895, 18.3155] and
[18.1680, 19.0035] seconds. These tiny differences are not convincing evidence
of a speedup. Production retains attention batch 32, so the full stress,
recovery and representative timing qualification uses the exact production
runtime. This is a bounded comparison of three settings on the inherited
optimized G3 recipe, not a claim of a global performance optimum.

The mixed run peaked at 39.81 GiB allocated and 49.95 GiB reserved GPU
memory, with a sampled host working set of 293.59 GiB. These are diagnostic
fixture measurements, not the longest-episode bound or a joint-epoch ETA.

## Independent review and environment corrections

Independent implementation review found no blocking model integration issue.
It identified two verification/operations issues: rotating fixture rows did
not force empty support into the first accumulation slot under length sorting,
and a partially published supervisor JSON must be treated as pending. Both
were corrected with regression tests. The follow-up review approved these
fixes and the added full-model probe/no-probe audit, with no new blocking
findings.

A second handoff saved R1 at step 3,372. The archived source initially lacked
Git metadata, causing four legacy helper-gate tests that call `git rev-parse`
to fail; the other 373 tests passed. No numerical model test failed. The
safeguard resumed R1. Adding local Git metadata to the archive preserved every
scientific file hash. The third handoff saved step 3,373; its complete H100
suite passed all 377 tests (118 warnings, 55.04 seconds), with no skips.
The qualified original source is recorded separately as commit `9031d75`;
the archive's internal snapshot commit is not its GitHub source identity.


## Long-episode memory qualification

The longest RxR fixture repeats the actual 627-action / 199,735-token episode
across all eight global-batch slots, so all four ranks process a maximum
length episode in both accumulation slots. It completed forward, backward and
an optimizer update with finite loss/gradient norm and zero new OOM, OOM-kill
or cgroup limit events. Peak allocated/reserved GPU memory was
67.8741 / 73.6523 GiB; sampled NVIDIA device usage reached 77,707 MiB
(75.89 GiB, including allocations outside the PyTorch allocator). The device
reports 79.19 GiB total. The sampled host working set was 429.12 GiB;
total cgroup usage including clean cache was 653.87 GiB against 768 GiB.
Each rank's pinned activation arena was 30.48 GiB. Rank-0 profiled update
compute was 82.51 seconds; full process startup/run/teardown took 156.96 seconds.

The longest R2R fixture repeats the actual 183-action episode similarly.
Peak allocated/reserved GPU memory was 37.0117 / 46.9531 GiB, and sampled
host working set 260.81 GiB. This gate also passed with zero new OOM events.

These stress gates use profile-only mode and do not write checkpoints during
the maximum-length update. Checkpoint save/resume is independently tested on
the complete mixed fixture. They establish a measured safe forward/backward
configuration for the audited manifest, not a guarantee against future input,
software or resource changes. Production retains the RAM and disk guards.


## Checkpoint storage cost

The mixed reference run with a recovery save on every update passed and
produced complete four-rank checkpoint inventories, about 65.89 GiB per
checkpoint. End-to-end update intervals were 96.0870 and 95.9226 seconds,
versus 18.6895 and 18.3155 seconds without saves on the same fixture. The
rough incremental shared-storage overhead is therefore 77.5 seconds per
save in this measurement. It includes serialization, distributed inventory
validation and storage contention, not only raw byte throughput. Production
saves at update 10 and every 100 updates rather than every update; roughly
40 saves would add about 0.9 hours at this measured rate, with additional
uncertainty for final export and changing filesystem load. Rolling checkpoint
retention and disk guards are inherited from the existing training system.


## Reproduction, monitoring and restart

Scientific source snapshot on the cluster:
`/mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/source-9031d75`.
Its `SOURCE_COMMIT` identifies the original GitHub commit; later report-only
commits need not replace or alter that qualified snapshot.

Qualification artifacts are in the campaign's `attempt3/qualification`:
`qualification_pass.json` binds all 15 successful gates to the exact hashes
of Python, YAML, JSON and shell files in src/scripts/configs/train/tests.
The production queue checks those hashes both when it starts waiting and
immediately before dispatch. `qualification.json`, per-experiment
`result.json`, recovery comparison and serving-export audit retain detailed
machine-readable evidence. A copy is included with this report.

Production queue root is the campaign's `production`; full training output
is `production/g3_m5_joint_e1/train`. Queue status is `queue_state.json` and
its persistent watcher log is `queue.log`. Per-attempt supervisor logs are
preserved under `g3_m5_joint_e1/supervisor-*`. Profile JSONL files in the
training directory give actual completed-update times and memory. The R1
predecessor is
`/mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015/attempt2/r1_joint_e1`.

The exact queue command (run detached on the login node, with its own log) is:

```bash
export PYTHONPATH=/mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/source-9031d75/src:/mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/source-9031d75
/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv/bin/python -m scripts.vln.queue_g3_m5 \
  --source /mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/source-9031d75 \
  --root /mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/production \
  --allocation 4925 \
  --predecessor /mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015/attempt2/r1_joint_e1 \
  --seal /mnt/data/vmo-ai-task/anhdh35/g3-m5-omega-worker2-20261016/attempt3/qualification/qualification_pass.json
```

The SSH endpoint can connect to either login-0 or login-1. The watcher
runs on login-0, so checking its PID on login-1 is not a liveness test;
verify its advancing shared queue heartbeat or inspect login-0. Do not
start a duplicate watcher while the recorded process is alive. If the
allocation is cancelled, acquire a new four-H100 allocation with 768 GiB on
worker-2 and rerun this command with the new allocation ID, retaining the same
source, qualification seal and output root. The launcher uses complete-checkpoint
`auto` resume; never substitute a serving-only export. A failed R1 predecessor
blocks the queue and must be investigated/resumed before dispatch. This is a
login-side dependency watcher inside an existing interactive allocation,
not a separate pending `sbatch` job with a new allocation. It waits for R1
training to finish rather than waiting for interactive allocation 4925 to end.


## Fresh-process recovery result

The reference trained through update 3 with complete checkpoints; a fresh
four-rank process restored checkpoint 2 and continued to update 3. Both
reported final loss 0.4834437370300293. All 10,526,368 auxiliary-head elements
were bitwise identical after continuation. Across all floating model elements,
relative L2 difference was 2.50647e-8 and maximum absolute difference
2.38419e-7 (56,178 changed elements out of 5,185,491,104). Thus full-model
continuation is numerically equivalent within the predefined gate, not
bitwise identical. The gate limits remain global relative L2 <=1e-5,
head relative L2 <=1e-4 and final-loss difference <=1e-4.

The reference head actually learned between updates 2 and 3: 131,045 elements
changed, including all query/decoder biases (512, 4,096, 1,024 and 2,048
entries). This excludes the explanation that head movement came solely from
weight decay. The resumed complete checkpoint inventory contains 22 files
and the expected four-rank topology. This verifies the inherited recovery
path with the new auxiliary state and M5 memory configuration.


## Export and distributed-loss evidence

The serving export passed on 17-frame forced-history audits from both
datasets, including exact source/export offline logits, streaming logits,
selected token IDs, greedy scores/outcomes and reset/retry results. Both
short-checkpoint greedy snapshots produced a valid TURN_RIGHT action; this
is interface evidence, not navigation-quality evaluation. The exported
model contains ordinary M5 memory metadata and no auxiliary head.

| Dataset | Probe/no-probe RMS | Probe max abs | Probe selection mean/min overlap | Online/offline RMS | Online/offline max abs | Online/offline selection mean/min overlap | Peak resident KV |
|---|---:|---:|---|---:|---:|---|---:|
| R2R |0.013918|0.125|0.999944 /0.9975|0.057615|0.53125|0.994447 /0.978333|2,539|
| RxR |0.010067|0.125|1.0 /1.0|0.052878|0.640625|0.993470 /0.975|2,579|

Probe-expanded navigation and ordinary M5 forwarding pass the existing
BF16 tolerance; they are not asserted bitwise identical. Small BF16 differences can affect near-tied dynamic M5 rankings;
overlap, rather than exact equality, is therefore reported between different
forward execution modes. Individual ranking differences were not attributed
to a specific arithmetic operation. Within the same mode, the
source/export comparison is exact. All resident groups remained the latest
four, with the older bank bounded at 1,200 and no hidden full-history archive.

The real four-rank distributed pair-reduction check also passed. The
qualification's mixed-empty fixtures separately verify the actual ZeRO2
backward path with heterogeneous valid-pair counts in each accumulation
slot; both completed three updates without hanging or dropping episodes.


## Final qualification and timing results

All 15 required gates returned zero on the exact source commit `9031d75`:
full CPU/CUDA suite (377 passed, zero skips), mixed empty support in both
accumulation slots, mixed complete episodes, attention batch 16 and 64,
longest RxR and R2R, checkpoint reference and fresh-process resume, recovery
comparison, serving export, distributed pair reduction, and representative
R2R/RxR timings. The source-bound pass seal was independently checked against
the local scientific files before handoff. Documentation-only changes do
not change the qualified executable tree.

Each representative fixture contains 16 complete episodes at evenly sampled
length quantiles. Eight updates give 64 episode exposures; timing uses the
six end-to-end intervals for updates 3–8, excluding the first two updates.
These intervals use the slowest rank's completion timestamp and include
online teacher, student forward/backward, CPU optimizer and inter-update
work. They do not include checkpoint saves. Rank-0 compute time alone would
underestimate distributed wall time, particularly for RxR.

| Dataset | Fixture action lengths | Mean seconds/update | Interval range | Approx. dataset-only compute pass |
|---|---|---:|---:|---:|
| R2R |31,36,40,43,46,49,51,54,57,60,63,67,71,77,84,99|9.7277|6.7618–12.6435|3.65 h|
| RxR15° |22,40,53,65,76,85,97,107,119,131,145,158,174,193,224,283|19.7602|9.8751–29.7145|13.72 h|

The additive compute estimate is 17.37 hours, plus roughly 0.9 hours of
checkpoint overhead at the measured shared-storage rate. A practical initial
budget is **18–22 hours after the combined run starts**. This is an estimate
from small warm fixtures, not a measured full epoch: joint length grouping,
new image I/O, long tails, checkpoint contention and system load can change
it. Refresh from actual production updates once the new run is underway.
R2R and RxR profiles peaked at 26.69 /38.60 GiB GPU allocated and
281.40 /351.27 GiB sampled host working set, respectively.

The controller resumed original R1 source `bcb14ea` as Slurm step `4925.91`
after successful qualification. The new persistent queue watcher is PID
3739763 on login-0 and waits specifically for R1's successful complete checkpoint 3,852.
Its launch source is the immutable qualified `source-9031d75`; no GPU work
from the combined full run overlaps R1. Allocation 4925 remains intact.
At the pre-pause rate (mean 11.52 seconds over updates 3,260–3,360), R1's
479 remaining updates imply around 1.5 hours of compute plus checkpointing;
allow roughly 1.5–2 hours after resume before the combined run begins.
