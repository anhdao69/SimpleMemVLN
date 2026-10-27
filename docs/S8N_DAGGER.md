# S8N recovery collection: R2R and RxR 15°

This collector generates **Janus-style executed recovery trajectories** using the
original S8N checkpoint. It does not implement the different VLN-CE objective
that assigns the oracle action to every learner-visited state. Raw records retain
both labels so that distinction remains auditable.

The supported entry points are `python -m s8n_dagger.run`,
`python -m s8n_dagger.campaign`, `python -m s8n_dagger.validate`, and
`python -m s8n_dagger.merge`. Run them from the repository root. The earlier
`qwen_vl.dagger.collect_r2r` experiment is retained for comparison; it is not the
collector used by this campaign.

## Executed actions are the memory history

1. Capture the current simulator RGB before acting and advance reached reference
   waypoints. Query the shortest-path expert for the current waypoint.
2. During learner control, run the unchanged S8N greedy policy. Save a transaction
   snapshot immediately after observation ingestion, before response decoding.
3. If Janus requests immediate expert takeover, restore that snapshot. Append the
   executed expert's canonical action tokens, EOS, and separator through S8N's
   normal response path. The discarded learner response must not remain in the
   recurrent state, convolution cache, attention cache, or episodic scorer.
4. During ongoing recovery, ingest each new RGB and append the expert action
   directly. The final observation and actual STOP are also ingested.
5. Execute the committed action in Habitat. Save this same action as the training
   target paired with the pre-action RGB.

The policy adapter preserves exact generated tokens when the learner action is
executed. An expert override uses canonical tokens. The independent GPU check
compares full session-state fingerprints: unchanged learner vs original session,
override vs direct expert forcing, subsequent continuation, and episode reset.

## Collection rules

| Item | Rule |
|---|---|
| Teacher | Original S8N, epoch 1, seed 429, R4/B1200 |
| Learner mixture | Janus beta=0: greedy until recovery is needed |
| Takeover | Premature STOP at distance ≥3 m, or disagreement ratio >0.8, or >12 disagreements |
| Ratio denominator | `max(1, int(reference_action_count / (waypoint_count - 1)))` |
| Waypoint advancement | Ends recovery; preserves accumulated disagreements |
| Takeover transition | Immediate; resets disagreement counter |
| Intermediate/final radius | 1.8 m / 0.25 m |
| STOP near goal | Expert replaces it until the final waypoint is reached |
| Saved trajectory | Actual expert STOP, distance <0.5 m, and recovered efficiency <0.93 or any efficiency <0.85 |
| Efficiency | Initial geodesic distance / max(initial distance, executed translation distance) |
| Actions | Forward 0.25 m, left/right 15°, STOP |
| RGB | 640×480, HFOV 79°, lossless PNG |
| Maximum actions | 675, including the final STOP |
| Truncation/errors | Retain diagnostic record; never fabricate a terminal training label |
| Inputs | R2R train; RxR 15° English-guide train only |

The conservative context bound `2048 + 675 × 384 = 261248` fits the original
262144-token limit. This collection cap intentionally differs from benchmark
evaluation's 500 steps. RxR reference trajectories reach 627 actions; an
R2R-derived 183-action save limit would remove valid RxR examples.

## Dependencies and site configuration

Use separate model and Habitat environments. The model requires the project's
pinned Python 3.12, Torch 2.10/cu129, Transformers 5.11, FlashAttention 2.8.3, and
FLA 0.5.2 environment. The simulator uses Habitat-Lab/Sim 0.2.4 and the extensions
under `docs/evaluation_setup`. Consult that directory for simulator installation.

`source` must point to an S8N-compatible source tree. Production uses the frozen
source that produced the original S8N evaluation; it must stay unchanged during
collection. The old local `streaming` checkout lacks this episodic implementation.
The new `dagger_s8n` branch is based on the compatible `dagger_s8n_test` revision.

The original evaluation source can be reproduced in a separate checkout with
`git worktree add --detach /checkpoints/original_s8n/source a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7`.
The collector binds the actual imported Python files by SHA256; a revision name
alone is not its provenance check.

Example site JSON (all paths are absolute and local to that server):

```json
{
  "repo": "/campaign/code",
  "root": "/campaign/run",
  "source": "/checkpoints/original_s8n/source",
  "checkpoint": "/checkpoints/original_s8n/checkpoint/S8N",
  "base": "/checkpoints/qwen-base",
  "model_python": "/project/.venv/bin/python",
  "sim_python": "/envs/habitat/bin/python",
  "scenes": "/datasets/scene_datasets"
}
```

Place labels under `root/assets/r2r/{train.json.gz,train_gt.json.gz}` and
`root/assets/rxr15/{train_guide.json.gz,train_guide_gt.json.gz}`. R2R labels are the
preprocessed v1-3 data linked in the JanusVLN README. RxR labels are from
[DVLe/RxR_CE_15_deg_Trajectory_data](https://huggingface.co/datasets/DVLe/RxR_CE_15_deg_Trajectory_data/tree/eadf29156a05b89c4411778d10906be627254ce3).
Both `.glb` and `.navmesh` scene assets are required.

Each `root/shards/<dataset>/<NNN>.json` contains an ordered JSON list of episode
IDs. A host assignment is a list of `{"dataset":"r2r","shard":0}` objects.
Partition each complete training set once, sorting by scene and ID; divide it
into contiguous chunks. Allocate each chunk to exactly one host, balancing
summed reference-action counts by host capacity. Preserve the resulting lists
and their audit instead of regenerating them during resume. The current campaign
uses 60 chunks per dataset and verifies exact coverage of all 30,815 IDs.

## Validation and launch

Before a full run, verify checkpoint/data hashes, all scene files, selected GT
replays, the GPU transaction gate, and real collector rollouts for **both**
datasets on **both** servers. Include a long RxR episode. CPU tests alone cannot
verify CUDA recurrent-state behavior or simulator action conventions.

```bash
PYTHONPATH=/checkpoints/original_s8n/source/src:$PWD \
  /project/.venv/bin/python -m pytest tests/dagger/test_s8n_*.py -q

/project/.venv/bin/python -m s8n_dagger.run \
  --site /campaign/site.json --dataset rxr15 \
  --episodes /campaign/rxr15_smoke.json --out /campaign/run/smoke/rxr15 --keep-rgb

PYTHONPATH=/checkpoints/original_s8n/source/src:$PWD \
  /project/.venv/bin/python -m s8n_dagger.validate /campaign/run/smoke/rxr15

PYTHONPATH=/checkpoints/original_s8n/source/src:$PWD \
  /project/.venv/bin/python -m s8n_dagger.campaign \
  --site /campaign/site.json --assignment /campaign/local_assignment.json --workers 3
```

Use a durable process manager or a detached session and redirect stdout/stderr to
a campaign log. Preserve the controller PID. Stop the controller with SIGTERM;
it stops its own worker process groups. Restart using the same command, site
configuration, source, data and assignment. Existing committed episode records
are reused; interrupted staging directories remain evidence. Contract changes
require a fresh output directory. Do not edit the frozen deployment in place.

## Outputs and subsequent training

Each shard has an immutable `contract.json`, model `runtime.json`, atomic
`episodes/<id>/record.json`, accepted PNGs and `training_episode.json`, plus
`training.jsonl`, `status.json` and final `validation.json`. Rejected attempts
retain their action/oracle/proposal/token/pose audit but normally discard RGB.
`campaign_status.json` reports progress and failures; it is not proof of a
performance improvement. Completed shards are independently validated before
entering the host manifest.

Before combining hosts, copy accepted RGB and metadata to accessible storage.
Use the merge tool's explicit path maps for relocated absolute paths. It checks
content provenance instead of requiring identical server paths and rejects
overlapping attempts, mixed protocols, incomplete shards and corrupted exports.
See `python -m s8n_dagger.merge --help` for its CLI.

The merge intentionally rejects a shard containing a failed attempt, even if
that attempt was rejected for training. Its purpose is to require investigation
of collection failures before publication. A normal capped/rejected trajectory
is different from a simulator/model failure and remains valid audit evidence.

The optional original-oracle/recovery mixture is a training candidate, not a
claim of an optimal ratio. It preserves the original number of episodes and
each dataset's share. Start with a controlled candidate, retain an oracle-only
baseline, and compare R2R and RxR validation SR/SPL and failure modes using the
same seeds and evaluation settings. A successful collection or larger recovery
set does not establish improved navigation performance. No training or
evaluation-result claim is part of the collection launch.

Mixture admission also requires the original and recovery `dataset_version` to
match. If the original manifest uses a named version and recovery uses the
verified source-file SHA256, supply an explicitly audited equivalence via
`--source-versions versions.json`, for example
`{"r2r":{"oracle":"<original-version>","recovery":"<verified-episode-file-sha256>"}}`.
The tool checks both sides and records that mapping. Matching episode IDs or
instruction text alone never establishes that two trajectory versions agree.
