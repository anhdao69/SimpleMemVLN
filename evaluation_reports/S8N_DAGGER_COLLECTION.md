# S8N DAgger implementation and branch review — 2026-10-27

## Source and model audit

Compared the latest fetched `dagger_s8n_test` revision
`d8c408f007299ad7eab8b74cbf0f74ffff493e10` with the original Janus recovery
collector, VLN-CE's DAgger label semantics, and the evaluated S8N streaming path.
The branch was checked out separately so the user's existing report changes and
other workloads stayed intact.

The teacher is the original S8N epoch-1 checkpoint, not S8N_E2, LR1E5 or Muon:

- Weight SHA256: `c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b`.
- Navigation SHA256: `12e94ae65e40d30422e1b0f1894c91788f9441c24a781b4a957a471d94ecd15b`.
- Evaluated source: `/data14t/anhdh35/m5_s8n_epoch1/source`.
- Original training: 10,819 R2R + 19,996 English-guide RxR15 episodes;
  3,128,624 actions; one epoch / 3,852 optimizer updates; effective batch 8;
  learning rate 5e-6; Adam; frozen vision/merger and full text BPTT.
- Original full R2R val_unseen: 964/1,839 successes, SR 52.419793%,
  SPL 48.679311%, no evaluation errors. These are baseline results, not results
  from newly collected data.

S8N keeps four recent observation groups and 1,200 older visual tokens. Its
eight GDN/attention layer pairs use normalized FP32 net-state-drift priorities
with frozen calibrated floors. Native recurrent and convolution state carry
forward even when attention KV entries are evicted. Response tokens affect
future recurrent/scorer state. Overriding only the simulator action after
`StreamSession.observe` would therefore leave the wrong history in memory.

## Comparison and decisions

| Behavior | `dagger_s8n_test` | New collector |
|---|---|---|
| Non-STOP disagreement takeover | Starts on next decision | Immediate, as in Janus; verified rollback |
| Disagreements across waypoint advance | Preserved | Preserved |
| Expert response in memory | Forced canonical response | Forced canonical response, including final STOP observation |
| STOP handling | First-token interception | Validate complete greedy response, then override transactionally |
| Datasets | R2R only | R2R and English-guide RxR15 |
| Rollout cap | 500 | 675 within original context bound |
| Saved length cap | 183 by default | No separate 183-action cap |
| Clean detour filter | Excluded unless flag enabled | Janus efficiency filter |
| Export supervision | Executed recovery actions | Executed recovery actions, with separate raw oracle labels |
| Cross-host provenance | Merge compares literal checkpoint paths | Compare content hashes; explicit path relocation |

The new collector follows the user's explicit choice, **“Match Janus recovery
behavior.”** It does not silently substitute conventional per-state oracle-label
DAgger. Both approaches are valid research protocols, but the existing S8N
episode schema uses one action for both history and target; changing the latter
objective would require a separate training-path change.

The old branch's first-token STOP shortcut has a parser edge case: a valid
whitespace-prefixed STOP can evade interception, while a malformed response
beginning with the canonical STOP token can be intercepted before full parsing.
No frequency estimate is claimed. Complete-response validation avoids that
ambiguity in this implementation.

During comparison, we found and corrected an error in our initial prototype:
it reset the disagreement counter on waypoint advancement. A regression test
failed before the fix and passes afterward. The early local protocol-v1 queue
was stopped. Its 14 completed attempts and 6 accepted episodes remain diagnostic
data and are excluded from protocol-v2 production. Existing contracts and data
were not rewritten to disguise the change.

## Data and simulator evidence

R2R labels follow the [preprocessed v1-3 dataset linked by JanusVLN](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view).
Local episode SHA256 is `340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea`;
GT SHA256 is `b63fa10e5c57d1b80b241c49bdeddc3d481bbe54b41ca0d59f9584c10da7bd5d`.

RxR episode/GT files exactly match the Hugging Face LFS hashes at revision
`eadf29156a05b89c4411778d10906be627254ce3`:

- Episodes: `7821f140ed8ce9221678a1125b56e767454673cdb28a08ba8afaffc2309db11f`.
- GT: `e8a985e9061232abfa10701608cb13b9ff2ca36bb31f97a5a362af0bd74de846`.

All 122 scene files (61 GLBs and 61 navmeshes) have matching hashes on both
servers. Missing remote scenes were copied into a campaign-owned scene tree.
Six selected GT and expert replays passed on each host, including the RxR
trajectory with 627 reference actions. The labels use 0.25 m translations and
15° turns. Validation and test episodes are excluded from collection.

The final assignment audit covers all 30,815 training IDs exactly once across
120 chunks. Allocation by reference-action workload is 13,174 episodes
(51 chunks, 1,338,136 reference actions) locally and 17,641 episodes (69 chunks,
1,790,488 actions) remotely. Planned concurrency is three local and four remote
workers. The initial 3:5 allocation was rebalanced before any v2 production
launch because other remote GPU jobs increased memory use. Unrelated workloads
are not stopped.

## Evidence and operational status

Initial GPU transaction checks passed bitwise on six local and ten remote real
frames. The remote check includes episodic-bank overflow. These checks cover
pure learner equivalence, expert override vs direct forcing, continuation and
reset; measured peak allocated memory was about 8.77 GiB. The protocol-v2 counter
correction does not change the policy adapter. New gate reports additionally
bind source, checkpoint, images and runtime with hashes.

The reference branch's DAgger CPU suite passed 44 tests. The final new suite
passes 69 tests (113 including the retained experiment's tests). A targeted
integration run covering serialization, streaming, training contracts, Habitat
evaluation and GDN math passes 230 tests.

The full integrated `tests/` run passes **754 tests**, with **23 skipped** and
**2 deselected**. The exclusions are existing tests requiring unavailable
`squeue` and the external `vggt_omega` module. The unmodified baseline run had
three failures from these two missing dependencies and an unset
`VLN_MODEL_PATH`; configuring the available base-model path resolved the third.
This is not a claim that every optional test/environment was exercised. Raw
output is preserved in `janus_v2/branch_tests.log`.

Real protocol-v1 smoke rollouts on both hosts completed three R2R and three RxR
episodes without failures; their data are diagnostic only. Fresh v2 GPU gates
passed ten frames on **each** host, including bank overflow. Final v2 rollout
results and launch state are recorded below after verification.

The production snapshot differs from the smoke snapshot only in the offline
`merge.py` source-version guard. It now requires equal original/recovery
`dataset_version` or an explicitly declared and checked equivalence. All files
used by collection, model serving, memory, the simulator and queue are identical;
`verification_code_equivalence.json` records that comparison. The postprocessing
guard has dedicated passing CPU regression tests. Frozen snapshots are retained
separately and their contracts are not edited in place.

Campaign parent directories:

- Local: `/data14t/anhdh35/s8n_dagger_20261027`.
- Remote: `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/dagger/s8n_20261027`.
- Corrected production: the `janus_v2/` child of each parent.
- Original v1 outputs remain in each parent's `smoke/` and local `full/`.

The protocol-v2 outputs must remain separate from the initial pilot. See
[the runbook](../docs/S8N_DAGGER.md) for rules, commands, resume contracts and
training-use constraints. Collection correctness is supported by these tests
and rollouts; improved SR/SPL requires a subsequent controlled training study.

## Final protocol-v2 gates and launch

All twelve real smoke rollouts passed (six per host), with zero failures. Both
hosts matched exactly on every action, committed/proposed response token,
pre-action RGB hash, waypoint and pose for all **939 steps per host**:

| Dataset | Episode | Actions | Accepted | Final distance (m) |
|---|---:|---:|---|---:|
| R2R | 2267 | 30 | No: efficiency filter | 0.22808 |
| R2R | 1760 | 72 | Yes | 0.22367 |
| R2R | 6967 | 219 | Yes | 0.03273 |
| RxR15 | 14029 | 5 | No: efficiency filter | 0.08120 |
| RxR15 | 31138 | 104 | Yes | 0.04986 |
| RxR15 | 31151 | 509 | Yes | 0.10071 |

The strict merger validated local R2R plus relocated remote RxR data: six
attempts, four accepted training episodes. This exercised real content hashes,
absolute-path relocation, PNG identity, executed response history, pose/distance
metrics and schema validation. Smoke outputs are excluded from production
manifests.

Full collection was launched using frozen `janus_v2/production_code/` on both
hosts. Local controller PID: **168044**, three workers. Remote controller PID:
**1722838**, four workers. Each host records `launch.json`,
`campaign_status.json`, shard logs under `queue/`, and outputs under `full/`.
Use `production_local.json` / `production_remote.json` for the respective
resume command in the runbook. The job is a first pass over all 30,815 training
episodes; its launch is not a completed-dataset or improved-model claim.

The implementation branch is `dagger_s8n`, based on the reviewed `d8c408f`
revision. The original dirty `streaming` workspace remains untouched apart from
the new task files. The branch's package files match both production snapshots.
Evidence: [consolidated verification](evidence/S8N_DAGGER_20261027_VERIFICATION.json),
[local GPU gate](evidence/S8N_DAGGER_20261027_LOCAL_POLICY.json), and
[remote GPU gate](evidence/S8N_DAGGER_20261027_REMOTE_POLICY.json).
