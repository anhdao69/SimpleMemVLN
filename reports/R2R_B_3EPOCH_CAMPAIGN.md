# R2R option-B three-epoch campaign

Approved sequence: FullContext+B, then independently initialized Window8+B.
Both start from Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, use global batch 8
(four H100s × one full episode × GAS2), LR 5e-6, three epochs,
4,059 optimizer steps and 122 warmup steps. Save all epoch checkpoints
at steps 1353, 2706 and 4059 with local optimizer states. Epochs 1–2
are intermediate points in the same three-epoch cosine schedule.

## Safety and execution

The Slurm templates follow StageVLN-v2: partition `main`, one node,
four GPUs, 24 CPUs, 256 GiB memory. Main training has a 48-hour limit;
preflight has a two-hour limit. Existing interactive job 4443 is not used,
changed or canceled. Four loader workers per rank, `OMP_NUM_THREADS=2`,
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are explicit.

Only the preflight is submitted initially. It verifies the immutable
source snapshot and manifest hashes, disk (at least 600 GiB free),
four H100s, the full test suite, actual four-rank three-epoch checkpoint
retention/reload, and FullContext+B short/median/p95/longest profiles.
The profile runs two global-batch-8 updates, including optimizer-state
allocation. All representatives must be consumed and all four ranks
must report peak reserved memory at most 78 GiB. Any failure stops
submission of full training. CPU-only test success is not a GPU gate pass.

After preflight success, `submit_campaign.py` submits:

1. FullContext training.
2. Window8 training, `afterok` FullContext **training**.
3. FullContext upload, independently `afterok` FullContext training.
4. Window8 upload, independently `afterok` Window8 training.

An exclusive submission journal prevents accidental duplicate submission.
Partial submission IDs are retained for manual reconciliation; do not
remove the journal and rerun blindly. Failed dependencies are canceled,
not converted into unconditional training.

The actual Trainer schedule is checked at `on_train_begin`; a mismatch
exits nonzero before the first update and prevents the dependent run.
Completion also requires exactly three epochs, 4,059 steps and all
four optimizer-rank files in each epoch checkpoint. Disk is rechecked
at each job start (Window8 requires at least 300 GiB remaining).

## Reports

Each training output contains `reports/first-50-updates.json` and
`reports/epoch-{1,2,3}.json`, also emitted into `train.log` as
`CAMPAIGN_REPORT`. Reports contain action-weighted loss, first/last loss,
per-update loss/timing, slowest-rank update compute time, wall time per
update, allocated/reserved memory maxima, and mean sampled GPU utilization.
The latter comes from the allocated devices sampled every five seconds.
Update compute timers exclude loader prefetch; wall intervals include gaps
between updates but exclude the subsequent epoch checkpoint write.

Reports cannot exist until queued jobs actually run. No Habitat performance
or convergence claim is implied by successful training or finite loss.

## Publication

Public repositories:

- `anhdao69/SimpleMemVLN-R2R-FullContext-B`
- `anhdao69/SimpleMemVLN-R2R-Window8-B`

Each contains `epoch-1/`, `epoch-2/`, `epoch-3/` model/processor/navigation
artifacts, configuration, training metrics, model card and SHA256 manifest.
The explicit allowlist excludes dataset images, optimizer states,
credentials and unrelated files. These wrapper checkpoints require the
SimpleMemVLN loader; they are not plain AutoModel checkpoints.

Upload jobs request no GPU and retry upload/verification five times.
Every uploaded file, including the checksum manifest, is downloaded at
the immutable Hub commit revision and hashed locally before
`UPLOAD_VERIFIED.json` is written. A failed upload cannot block Window8.
Hash verification uses one temporary file cache at a time, separate from
the base-model cache. Hub credentials live outside the source checkout
in a private directory; no credential appears in job arguments or source.

Model cards state the recipe, exact source commit, serializer
`vln_append_only_chat_v1`, **not yet evaluated in Habitat**, the independent
initializations and mid-schedule status of epochs 1–2. Capture-before-action
alignment remains an explicitly unverified data assumption.
