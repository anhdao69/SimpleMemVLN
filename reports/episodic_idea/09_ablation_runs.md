# 09 — Remaining arms (M1, M2c, M5, M5c) on the H4/H2 cluster, 2026-10-12

Status: **qualified and submitted, 2026-10-12 12:2x EDT** (see "Submission" below). Not a navigation result.

## Request and scope

User instruction (2026-10-12): V0 and M2 (and, per its report, M1) are training on the other
server (branch `c2_h2_speed`, campaign `fddde5b1…`, H2 jobs 4880/4878/4879). On this cluster:
debug the other arms of the plan and submit them as 4H100/2H100 queue alternatives, from the
latest GitHub code. Asked which arms, the user chose **M1 + M2c + M5 + M5c**.

* **Phase-2 override.** M5/M5c are Phase-2 arms. `08_phase2_decision.md` keeps the gate closed
  until E1 evaluations of V0/M1/M2/M2c exist. The user explicitly chose to run M5/M5c now.
  This is recorded as a **user override of the evidence gate**, not as a gate outcome: the
  ledger entry and every launch manifest carry `phase2_reason`. No Phase-2 conclusion may cite
  the gate as satisfied, and the 08 outcome (A–D) must still be recorded from the E1 results.
* **M1 duplication.** The other server's report lists M1 (job 4878) as running there. The user
  chose to train M1 here as well. If both complete, they are two independent M1 runs, possibly
  on different world sizes and clusters, not a seed replicate. Report one as primary
  (state which) and do not average them silently.
* V0/M2 are **not** trained here (no duplicate of the other server's runs).

## Starting point and code changes (branch `c2_ablations`)

`c2_ablations` branches from GitHub `c2_h2_speed` @ `2430fd2` (latest; built on `c2_episodic`
`a06c789`). `c2_h2_speed` changed only engineering files relative to `c2_episodic`; the science
modules (`src/`) are byte-identical. It adds the H2 speed profile (8 OMP threads, 4 loader workers
per rank, no per-update heap trim), the helper-error gate (1%), recoverable-OOM supervision, and
immutable numerical evidence bound to source commit + runtime-profile hash.

Production commit **`a340e13`** adds, with tests:

1. **Defect: the H4 sibling of a race could never pass preflight.** The campaign held a single
   numerical report whose `runtime_sha256` is one profile's runtime file; each attempt's
   preflight compares it with its *own* runtime, so in an H4/H2 race the H4 attempt fails
   (`numerical qualification runtime differs`). Fix: `init`/`resnapshot` take one passing
   report per runtime profile (`numerical_reports`, keyed by runtime SHA-256). `submit` gives
   each attempt the report that matches its profile, or refuses. Old single-report campaigns
   still resolve. Duplicate reports for one runtime are rejected.
2. `--cpus` / `--omp` accept per-profile values (`H4=32,H2=30`). A plain integer still applies to
   all profiles, which is the form the winner's ablation submission uses.
3. Phase-2 run IDs `P2-M5-s429` (m5) and `P2-M5c-s429` (m5c). They require `--allow-phase2` **and**
   `--phase2-reason`, recorded verbatim in the launch manifest and ledger.
4. `check_episodic.py` covers m5/m5c replays, and `--stream-arms` selects the dry-stream arms.
5. `scripts/vln/manifest_fingerprint.py` gives a path-independent manifest fingerprint. Byte SHA
   differs across clusters because records hold absolute `rgb_path`s. Here:
   `fingerprint_v1 = f11fcceac19724e586f1b0a97828c388f619a88ee14d513bd8498856e9c2b0a9`,
   30,815 episodes, 3,128,624 steps (same counts as the other server). **Matching across servers
   is not yet verified:** run the script on the other server's manifest
   (`r2r_rxr15_train_20260929.jsonl`, SHA `5deb425d…`) and compare. Only equal fingerprints
   imply identical episodes in identical order.

Tests: `tests/vln/test_episodic_profile_evidence.py` (6 new), plus updated launch and provenance
tests. 31 campaign/submission/job/evidence tests pass (CPU). Full `tests/vln` suite at `a340e13` on H100 with `VLN_MODEL_PATH` set to the pinned snapshot:
**330 passed, 0 skipped** (`quals/interactive_898977/pytest_a340e13_pinned.log`). Without
`VLN_MODEL_PATH`, 12 tokenizer/processor tests skip (318 passed).

## Numerical evidence (GPU, interactive allocation 898977, evc102)

`check_episodic.py` at clean `a340e13` for arms v0 m1 m2 m2c m5 m5c, replays at 64/128 observations
on R2R and RxR, and dry streams for m2/m5/v0. One run per runtime profile:
`quals/interactive_898977/numerical_a340e13_{h4,h2}.json`. **Both PASS**, with identical
numbers (the runtime profile does not affect inference).

| Check | Result |
|---|---|
| A: feature-disabled V0 vs legacy Window8 | logits bitwise equal, loss equal |
| B: helper vs FLA final-state relative error | 0.204% (R2R), 0.235% (RxR); budget 1%; finite |
| C: 24 replays (6 arms × 2 datasets × 64/128) | all pass, atol 0.75 / rtol 0.03 / RMS ≤ 0.15 (unchanged) |
| C: largest logit error / RMS | 0.75 / 0.0450 (m5, R2R-64 RMS) |
| C: lowest raw-token argmax agreement | 98.69% (m1 and m5, R2R-128) |
| C: M2 offline-vs-online selected-set overlap | mean 98.05–98.36%, worst step 93.3% |
| C: M5 overlap | mean 98.27–98.62%, worst step 94.75% |
| C: M2c/M5c overlap | 100% (activation-independent priority) |
| E: 520-step dry stream, peak resident KV / guard | M2 2,946/5,420; M5 2,574/4,784; V0 2,646/5,120 |

The 96-observation R2R fixture failure for B300 arms (04) still stands: it was not re-run and
is not overridden. The protocol default is 64/128, as before.

## Resource gates (same allocation; 768 GiB cgroup)

Complete real episodes, three optimizer updates per gate. Every rank, update and exposure was
verified, with zero new OOM/OOM-kill events. H2 gates used the production guard of 460.8 GiB
(0.9 × 512). H4 gates used a 691.2 GiB guard so the actual host need could be measured. Gates
ran sequentially, because concurrent gates share one cgroup and would trip each other's guard.

| Profile | Gate | Arm | Pass | GPU reserved peak GiB | Host working set GiB | Wall s |
|---|---|---|---|---:|---:|---:|
| H2 | longest RxR (199,735 tok) | M5 | ✓ | 72.94 | 259.9 | 669 |
| H2 | longest RxR | M5c | ✓ | 72.94 | 256.3 | 464 |
| H2 | mixed tail | M5 | ✓ | 72.95 | 214.1 | 404 |
| H2 | mixed tail | M5c | ✓ | 72.94 | 220.7 | 302 |
| H4 | longest RxR | M1 | ✓ | 76.96 | 222.0 | 276 |
| H4 | longest RxR | M2c | ✓ | 76.96 | 210.6 | — |
| H4 | longest RxR | M5 | ✓ | 76.96 | 212.3 | — |
| H4 | longest RxR | M5c | ✓ | 76.96 | 233.8 | — |
| H4 | mixed tail | M5 | ✓ | 76.91 | 181.0 | — |
| H4 | above offload | M5 | ✓ | 46.18 | 88.8 | — |

H4 per-rank peak allocated memory on the longest episode is 72.85–72.94 GiB, matching the 05
estimate of ≈73. Reserved memory is 76.96 GiB, under the 78 GiB hard ceiling but above the
76 GiB preferred target. **H4 is tight; H2 (72.9 GiB) is the safe sibling of every race.** The
longest-RxR fixture is the actual longest manifest episode, and every job repeats this gate on
its own allocation before training. H4 longest-episode updates take 60–67 s, versus about 127 s
on H2.

Host: H4 peaks at ≈222 GiB and H2 at ≈260 GiB, against a production request of 512 G (guard
460.8 GiB), so ≥1.7× headroom. The other server measured H2 save/reload peaks of 274/310 GiB.
Checkpoint save/fresh-process reload is arm-independent code. It was proven for H4 V0 at `e28bd01`
(05; save + fresh-process resume) and for H2 V0 on the other server, and was not repeated here.

Evidence: `quals/interactive_898977/{H2_phase2,H4_ablations,H4_m5_extra}/<profile>/<gate>/gate.json`
(supervisor host samples, per-rank `profile_rank*.jsonl`, exposures) and the `*.log` files.
Losses for updates 1–3 are finite and nearly equal across updates. Update 1 has LR 0 in warmup,
so this is not a learning signal.

## Submission (campaign `c42b737b…`, source `a340e13`)

`resnapshot` replaced the campaign source `e28bd01` → `a340e13` and stored both runtime reports
under `numerical/<sha>.json`. The earlier V0 attempts keep their recorded source. Every attempt
runs preflight (GPU idle, source hashes, numerical evidence for its own runtime, disk) and then
its own **longest-RxR startup gate** on its allocation, before claiming the run's lease
atomically. The single-winner protocol is unchanged (07). The loser exits `DUPLICATE_STANDBY`, or
the winner cancels its recorded pending/running sibling after 50 healthy updates and a verified
recovery checkpoint. There is no cross-world-size resume. No automatic follow-on submissions
(`--ablations` not used).

| Run | Arm | R | B | Score | H4 job (attempt) | H2 job (attempt) |
|---|---|---:|---:|---|---|---|
| `P1-M2c-s429` | M2c | 8 | 300 | random_hash_v1 | 899877 (`P1-M2c-s429-H4-a1`) | 899878 (`P1-M2c-s429-H2-a1`) |
| `P1-M1-s429` | M1 | 9 | 0 | none | 899881 (`P1-M1-s429-H4-a1`) | 899882 (`P1-M1-s429-H2-a1`) |
| `P2-M5-s429` | M5 | 4 | 1200 | committed_mean_l2_v1 | 899883 (`P2-M5-s429-H4-a1`) | 899884 (`P2-M5-s429-H2-a1`) |
| `P2-M5c-s429` | M5c | 4 | 1200 | random_hash_v1 | 899885 (`P2-M5c-s429-H4-a1`) | 899886 (`P2-M5c-s429-H2-a1`) |

Resources: H4 = 4 GPUs, 32 CPUs, OMP/MKL 4, 512G, 3 days. H2 = 2 GPUs, 30 CPUs, OMP/MKL 8,
512G, 5 days. Partition `highgpu`. H2 uses the `c2_h2_speed` profile (4 workers/rank, no heap
trim, CPU optimizer offload). H4 keeps the profile qualified in 05 (2 workers/rank, heap trim,
GPU optimizer state). The H4 profile was not speed-tuned; any H4 speed change needs its own
measurement.

Each run trains epoch 1 only: update 3,852 of the unchanged 7,704-update schedule, with recovery
saves at 50 and every 250. It then uploads `export-e1/` + `checkpoint-3852/` + `campaign/` to
`anhdao69/Streaming_episodic/<run>/<attempt>/` (public). Epoch 2 resumes only on the same
profile/world size.

### Matched-control caveats (must accompany any comparison)

* V0/M2 (and M1) on the other server use H2 world size 2, GAS 4, global batch 8. Here, a run
  that wins on H4 uses world size 4, GAS 2, the same global batch 8 and the same global action
  normalization. `EpisodeLengthSampler` (seed 429 + epoch) fixes the length-sorted 64-episode
  groups and their order independently of world size. Only the within-block rank assignment
  draws `randperm(world_size)`, and an update covers 8 consecutive group members (world × GAS),
  so **the per-update episode set is the same on H2 and H4**, given an identical manifest
  (fingerprint above, still to be compared). Rank assignment and reduction order differ, so
  the runs are not bitwise comparable. Record the winning profile per run.
* Source differs by engineering-only commits (`e789a13` there, `a340e13` here; `src/` identical).
  Both servers' runtime profiles reproduce the same numerical evidence values.

## Expected timing

At about 60–67 s/update on the longest episodes (H4) and the other server's 22–25 s/update
representative H2 rate, one epoch is roughly 24–27 h on H2. H4 should be faster, but the
production rate is not yet measured here. Queue wait is set by the scheduler. These are
estimates, not observations.

## Update 2026-10-12 12:50 EDT — M2c moved into interactive allocation 898977

At the user's request, to reduce queued submissions to three races, P1-M2c-s429 trains inside
the user's interactive allocation 898977. That allocation is 4 × H100, 16 CPUs and 768 G on
evc102, and ends 2026-10-13 05:55:47.

* Pending 899877/899878 were cancelled after a live identity check (job name, comment, PENDING)
  and recorded as `superseded_before_start`. They were never allocated.
* New attempt `P1-M2c-s429-H4-a2` (H4 profile, OMP 2, 2 workers/rank, `cpus=16`, `mem=768G`)
  was created through `submit --dry-run` and recorded in the ledger as job 898977. It runs the
  unmodified lifecycle (`episodic_candidate.sbatch` → `episodic_job`) inside the allocation:
  preflight PASS; startup longest-RxR gate PASS (76.34 GiB reserved, 225.7 GiB host, 0 OOM);
  lease CLAIMED; training from base weights.
* **Measured production rate: 23.9 s/update** (updates 5–32, rank 0). Peak reserved 66.75 GiB,
  0 CUDA OOMs. Epoch 1 therefore needs ≈25 h, more than the allocation's remaining ≈17 h. The
  earlier 10–13 h H4 forecast (05) was wrong for this CPU budget.
* An interactive allocation sends no USR1. `deadline_stop.sh` therefore writes the job-scoped
  `REQUEST_CHECKPOINT_AND_STOP` flag (content `898977`) at 05:10, unless checkpoint-3852 is
  already complete. The run then saves full state and stops at the next optimizer boundary.
* Continuation **899912**: same attempt directory, same H4 world size, same source, `afterany:898977`,
  4 GPUs / 16 CPUs / 768G / 3 days. It refuses to run while 898977 is live, re-qualifies its
  node, and resumes from the newest complete checkpoint (`--resume_from_checkpoint auto`). At the
  epoch-1 boundary it uploads to `P1-M2c-s429/P1-M2c-s429-H4-a2/`. This is not a duplicate:
  it is the same logical run and attempt.

Remaining queued races: M1 899881/899882, M5 899883/899884, M5c 899885/899886.
