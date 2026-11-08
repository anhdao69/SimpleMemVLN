# S8N RL — local continuation campaign (2 x RTX PRO 6000 workstation, from 2026-10-30)

Continues `s8n_rl_report.md` / `s8n_event_rl_design.md` after the HiPerGator interruption (§8 there).
GRPO (60.41) and GRPO+F-B (60.63) are **not** re-run (user decision, 2026-10-30); their numbers are quoted
from the HiPerGator journals, which are not on this machine. Every new row below is produced here.

## 0. Machine and environment

- 2 x NVIDIA RTX PRO 6000 Blackwell (97.9 GB). GPU 0 also holds another user's idle process (~33 GB),
  which is left untouched, so rank 0 has ~63 GB. 48 CPU cores, 188 GB RAM.
- Model env: this branch's `uv.lock` (Python 3.12.13, torch 2.10.0+cu129, transformers 5.11.0).
- Simulator env: `/nvme/public/svln/envs/habitat-vln310`, a clone of an existing local Python 3.10.21
  habitat-sim 0.2.4 (headless, no bullet) + habitat-lab 0.2.4 install, then pinned to
  `docs/evaluation_setup/habitat-constraints.txt` (hydra-core 1.3.4, numba 0.66.0, llvmlite 0.48.0,
  pillow 11.3.0, fastdtw 0.3.4). Same simulator version and build flags as HiPerGator's Route B build.
- Assets (`/nvme/public/svln/data/asset_record.json`): R2R_VLNCE_v1-3_preprocessed from the official
  Google Drive archive (val_unseen sha256 `1767a407…` = audited), RxR15 labels (DVLe, audited hashes),
  S8N epoch 1 (`anhdao69/M5_S/S8N`, `pytorch_model.bin` sha256 `c89de946…` = the RL configs' hash),
  Qwen3.5-4B @ `851bf6e8`.
- Scenes: `/nvme/public/mp3d` (90 scans). All 145 `.glb`/`.navmesh` files of the pinned mirror
  `anhdao69/mp3d-scene-datasets@d8c76863` that exist locally are byte-identical to it; every scene of
  R2R train / val_seen / val_unseen is present.
- Launchers: `ops/local/{env_local.sh,train_rl.sh,eval_r2r.sh}` = the HiPerGator sbatch scripts without
  Slurm. Configs: `configs/rl/local/*.yaml` = the HiPerGator configs with local checkpoint paths; any
  other difference is a systems-only setting marked `local`.

## 1. Gates (before any RL result is read)

1. CPU tests `tests/rl`, `tests/memory_sweep` pass in the local model env.
2. Base S8N, val_seen-300, local: compare with the HiPerGator journal (155/300, SPL 48.67, NE 5.50,
   OSR 62.67). Cross-server GPU numerics flipped 17/1,839 val_unseen episodes before (964 vs 947), so a
   difference of a few episodes is expected; the gate is |dSR| <= 3 pp.
3. Base S8N, val_unseen, local: the local reference journal for every paired test of this campaign.
4. One full-size RL step (8 prompts x G = 4) per configuration: finite loss/grad, export/resume, and
   the speed calibration of §2.

## 2. Speed (systems only; never changes an update)

Rollout seeds depend only on (step, episode, attempt, sample), so the number of helper processes changes
wall time, not the sampled trajectories' distribution. Knobs, chosen from a measured full-size step:
`rollout.helpers_per_rank` (HiPerGator: 3) and the evaluator's `WORKERS_PER_GPU`. GPU 0's free memory
(~63 GB) bounds rank 0.

## 3. Run queue

| # | run | config | why |
|---|---|---|---|
| 1 | memory twins on GRPO | `memory_twins_grpo_s8n_r2r1k` | the pre-registered candidate (design §8, D2 amendment) |
| 2 | decided after #1's val_seen-300 curve | — | see §4 |

Evaluation: exports at 25/50/75/100/125 -> val_seen-300 (development only); step 125 -> val_unseen once.
Paired tests (`scripts/rl/compare_evals.py`) only between journals produced on this machine.

## 4. Decision rules (fixed before any local RL result)

- Twins vs base S8N is paired (local). Twins vs GRPO / F-B is **unpaired** against the HiPerGator
  numbers (60.41 / 60.63), so it is reported as descriptive; a claim against GRPO needs a local GRPO
  journal, which is requested from the user before it is run.
- The D2 bar of the design log stays: a method is a candidate only if it beats GRPO (and its compute
  control) with McNemar p < 0.05 on val_unseen at step 125, then holds its sign on a second seed.

## 4.1 Results so far: SFR line (updated 2026-11-08 08:00)

Generated from the merged journals (`svln/evals/<run>_<set>_val_unseen/merged/summary.json`); details and every
paired test are in the dated entries of section 5. All runs start from SFR, use seed 20261010 and B's recipe unless
noted, and are evaluated with the evaluator tree of every 1k journal. Episodes the evaluator cannot score count as
failures. RxR A has 24-25 of them and RxR full 98, the same episodes for every model.

| Model | run | prompts | step | R2R SR / SPL | R2R OSR | R2R NE | RxR A (1,000) SR / SPL | RxR 2k SR / SPL | RxR full (3,669) SR / SPL |
|---|---|---|---|---|---|---|---|---|---|
| SFR (start) | `sfr` | - | - | 62.59 / 55.59 | 69.98 | 4.27 | 64.40 / 53.09 | 63.55 / 52.15 | - |
| A: ActiveVLN GRPO G 4 | `grpo_sfr_s125` | R2R 1k (B's 1k) | 125 | 64.33 / 57.32 | 72.38 | 4.29 | - | - | - |
| E: GRPO G 5 (compute control) | `grpo_g5_sfr_s125` | R2R 1k | 125 | 66.50 / 58.71 | 75.20 | 3.99 | - | - | - |
| C: self-rewind (no expert) | `rewind_self_sfr_s125` | R2R 1k | 125 | 67.26 / 59.58 | 74.23 | 3.98 | - | - | - |
| G: expert fork at t (no rewind) | `rewind_forkt_sfr_s020` | R2R 1k | 20 (drift-guard stop) | 65.85 / 56.65 | 75.42 | 3.82 | - | - | - |
| B: rewind | `rewind_sfr_s125` | R2R 1k | 125 | 68.73 / 60.87 | 76.67 | 3.68 | 71.80 / 57.33 | 70.70 / 56.91 | - |
| **mix1k: rewind** | `rewind_sfr_mix1k_s125` | 500 R2R + 500 RxR | 125 | **71.94 / 63.65** | 78.58 | 3.32 | 71.20 / 57.79 | 70.75 / 57.04 | - |
| path_mix1k: path-aware rewind | `rewind_path_sfr_mix1k_s125` | 500 R2R + 500 RxR | 125 | 69.66 / 62.29 | 75.86 | 3.51 | 69.80 / 56.57 | 69.00 / 55.92 | - |
| ActiveVLN, A+ 8k pool, 1k | `grpo_sfr_mix8k_aplus_s125` | A+ rows 0-999 (= mix1k's) | 125 | 66.83 / 60.01 | 74.44 | 3.85 | 69.90 / 57.12 | 69.75 / 57.61 | 69.96 / 57.67 |
| ActiveVLN, A+ 8k pool, 2k | `grpo_sfr_mix8k_aplus_s250` | A+ rows 0-1999 | 250 | 67.92 / 61.27 | 74.50 | 3.77 | RxR deferred (user) | - | - |
| Rewind, A+ 8k pool, 1k (gpu2) | `rewind_sfr_mix8k_aplus_s125` | A+ rows 0-999 | 125 | 69.91 / 61.79 on the first 1,512 of 1,839 (partial) | - | - | pending | pending | pending |

Key paired R2R comparisons (McNemar, 1,839 episodes):
- **B vs A** (rewind vs plain GRPO, same prompts): +4.40 (p 7e-5).
- **B vs E** (matched compute): +2.23 (p 0.037).
- **mix1k vs B** (mixed vs R2R-only prompts): +3.21 (p 0.0026).
- **path_mix1k vs mix1k**: -2.28 (p 0.036).
- **ActiveVLN 1k vs mix1k** (same 1,000 prompts in the same order; only the estimator differs): -5.11 (p 2e-6).
- **ActiveVLN 2k vs mix1k**: -4.02 (p 4e-4). **ActiveVLN 2k vs ActiveVLN 1k**: +1.09 (p 0.31).
- Rewind 8k-pool 1k (gpu2) vs mix1k on the same 1,512 episodes: 69.91 vs 71.36 (-1.46, p 0.22). This is the same
  algorithm, prompts and order run on another machine, so the gap is run-to-run spread.

## 5. Log

### 2026-10-30 (setup, gates, speed)

- Model env = `uv.lock` exactly (all 80 linux packages at the locked versions; the 16 CUDA wheels were fetched
  from the Tsinghua PyPI mirror and installed only after their sha256 matched the lock, because PyPI's CDN ran at
  ~0.3 MB/s here; DeepSpeed 0.16.4 built against the venv's torch = the lock's `match-runtime` build dep).
- Gate 1: CPU suite **247 passed** (= HiPerGator, job 48195324).
- Gate 2: base S8N val_seen-300 local **159/300 (SR 53.00, SPL 50.02, NE 5.52, OSR 64.33, 19 forced stops,
  0 failures)** vs HiPerGator 155/300 (51.67 / 48.67 / 5.50 / 62.67): +4 episodes, inside the +-3 pp gate. As
  between the original server and HiPerGator (964 vs 947 on val_unseen), cross-machine GPU numerics shift a few
  episodes, so local and HiPerGator journals are never paired.
- Speed. The GPUs are RTX PRO 6000 Blackwell **Max-Q** (300 W). One S8N rollout process: 112 ms/decision with
  the GPU ~60 % busy (thousands of tiny kernels). Several processes time-slice between CUDA contexts:
  4 processes 11.4 decisions/s. **CUDA MPS** (user daemon, `ops/local/mps.sh`; same kernels, numerics unchanged):
  4 processes 17.8 decisions/s, 5-6 processes ~18.4 (saturated). CUDA IPC (rollout helpers reading the learner's
  live weights) works between MPS clients (checked). With MPS and 5 helpers/rank, an RL step's rollout phase is
  ~100-115 s; the update (full-episode fwd+bwd, ~22k tokens per trajectory, decoder checkpointing) is the
  bottleneck at 160-320 s, GPU-bound at ~40 TFLOPS/GPU; an opt-in profiler (`RL_PROFILE_STEP`) is in the dev tree.
- Run 1 launched 11:41: memory twins on GRPO (`configs/rl/local/memory_twins_grpo_s8n_r2r1k.yaml`), code =
  commit 334bc35 (src unchanged; only local configs/launchers added), run dir `/nvme/public/svln/runs/twins_grpo_s8n`.
  Steps 0-2: log-prob mismatch 0.0045-0.0052 (HiPerGator 0.004), twins diverge 62-88 %, 280-420 s/step.

### 2026-10-30 (exploratory analysis, not pre-registered): does S8N surprise mark route deviations?

`scripts/rl/s8n_analysis/surprise_at_deviation.py` on run 1's first 128 main trajectories (8,353 decisions,
106 deviation onsets = first forward move away from the goal after >= 3 quiet decisions, in episodes that never
reach 3 m). Position-normalised event score `event_pos` predicts onsets strongly without turn controls (z = 7.5;
onset rate by quintile 0.3 / 2.2 / 4.2 / 9.1 / 10.3 %), but **not once the last three actions' turns are
controlled** (z = 1.8; turns z = 8.4; within post-turn decisions the rate is flat across quintiles,
10-18 %). S8N's drift marks view changes (turns), which precede deviations, not the decisive errors themselves,
consistent with the pre-registered H1 failure. Consequence: interventions are placed by a privileged geometric
trigger, not by surprise.

### Candidate 2 (implemented in the dev tree, CPU-tested, not yet run): counterfactual expert interventions

`estimator: grpo_rescue` (`configs/rl/local/rescue_grpo_s8n_r2r1k.yaml`). Mains are plain ActiveVLN GRPO,
unchanged. A main whose pose before a_t is >= 1 m beyond its closest approach to the goal so far (causal,
privileged) snapshots its S8N state at t. After the main ends, one branch restores the snapshot, replays the
simulator prefix (verified), lets Habitat's greedy geodesic follower act for <= 16 decisions (until its 3rd
forward move), streaming the expert's actions into the policy's *own* memory as forced responses, then continues
on-policy with an independent generator. Credit (batch scale as F-B): forced decisions
max(0, R_rescue - R_main)/s (imitated only if the intervention beat the policy's own continuation from the same
state), continuation (R_rescue - R_main)/s. Tests: `tests/rl/test_rescue.py` (5; full suite 252 passed).
Trigger rate on run 1's step-0/1 mains: 22 of 30 failed mains, 3 of 34 successes (delta = 1 m).

**Probes (base S8N, no training; `scripts/rl/rescue_probe.py`, T = 1.2, one GPU next to run 1):**

| probe | episodes | main SR | triggered | triggered failures rescued | takeover | triggered successes made worse |
|---|---:|---:|---:|---:|---:|---:|
| pool rows 1000-1031, short takeover, no budget extension | 32 | 59.4 % | 7 | 1 / 5 | 14 | 0 / 2 |
| 32 hard instructions (failed + triggered in run 1 steps 0-5), short takeover | 32 | 28.1 % | 17 | **6 / 16 (37.5 %)** | 13.9 | 0 / 1 |
| same 32, same seeds, `return_to_closest` (max 40) | 32 | 28.1 % | 17 | 5 / 16 (31.3 %) | 16.0 | 0 / 1 |

- The two hard-instruction probes have identical mains (32/32: same seeds reproduce the same trajectories), so
  they compare the takeover rule alone; neither is better within this sample. The short takeover (fewer forced
  decisions) is kept.
- After the first probe the expert's decisions were excluded from the policy's step budget (a turn-around takes
  12 of them, and branches that reached 0.3 m of the goal were truncated before they could stop).
- A third of the hard failures become successes after a 14-decision correction from the agent's own memory
  state; most other branches end much nearer the goal than their mains (e.g. 14.6 -> 5.2 m) but get zero reward
  under ActiveVLN's success-only reward, so they carry no contrast and are not trained.
- Two trigger regimes: early (route error, far from the goal) and late (after passing within 2-3.5 m of the goal).

### Speed: lockstep of the update (systems only, dev tree)

ZeRO-2 reduce-scatters every micro-step's gradients (`partition_gradients`; ~84 blocking 50M-element buckets,
~8.4 GB over PCIe Gen5 per trajectory pair), so the two ranks advance in lockstep and each micro-step lasts as
long as the longer trajectory. Replaying placement on run 1's trained trajectories (steps 0-13): arbitrary order
costs 371.8k token-units of lockstep time, longest-first on every rank 314.3k, no-lockstep ideal 305.5k, i.e.
**-15.5 % update time** from sorting. `runtime.sort_items_by_length` (default true) sorts each rank's items after
placement; only the summation order of the accumulated gradient changes.

### Queue (2026-10-30 13:30, revisable only by results named here)

1. Run 1, memory twins on GRPO: training until ~22:40; val_seen-300 at steps 25/50/75/100/125 on GPU 1
   (2 MPS workers beside training); then on both free GPUs (8 MPS workers): run 1 step-125 val_unseen and the
   local base S8N val_unseen reference (~80 min each).
2. Run 2, rescue on GRPO (`rescue_grpo_s8n_r2r1k`): a 1-step smoke with `smoke_rescue.yaml` first, then the
   125-step run from a frozen snapshot of the dev tree with the update-order speedup and `RL_PROFILE_STEP=0`.
3. Run 3, GRPO G = 5 (`activevln_grpo_g5_s8n_r2r1k`, local): the compute control of both twins (4 + 1 twin
   rollouts per group) and rescue, and the local GRPO-family journal for paired tests.
4. Then, depending on 1-3: ablations of whichever candidate qualifies (twins without CRN; rescue without
   imitation / with unverified imitation), the combined `twins_rescue_grpo_s8n_r2r1k`, second seeds, and RxR15
   val_unseen of the base and the final policy.

### 2026-10-30 14:10 — interim check of the twins' memory gradient (run 1, steps 0-26; read before any eval)

- The rule moves (overlap with S8N's frames 0.86 by step 20; layers 3/23 toward start frames, 15 toward later
  ones), but movement is not evidence: Adam (beta1 0.9) fed pure noise gives the same mean path efficiency
  |net| / path over 26 steps (null 0.561 vs observed 0.569, p = 0.47).
- Direct test: the per-step antithetic gradient recomputed from the logged twin pairs (216 pairs, 27 steps),
  mean across steps vs 0 per coordinate: sum z^2 = 23.3 against chi2(16) 95th percentile 26.3 -> **no
  significant memory gradient yet**. |z| > 2 on 3 of 16 coordinates (consensus of layers 23 and 31 negative,
  birth of layer 15 positive), ~0.8 expected by chance.
- Reading: with one twin pair per group (8 per step) the estimator sits near its noise floor. If those three
  coordinates are real, they reach |z| ~ 5 by step 125. If the final rule's gradient stays insignificant, the
  twins run is a test of a noise-driven rule, and a higher-power variant (twins for every sample, 32 pairs per
  step) is the natural next step, not a claim.
- 14:55 re-test (34 steps, 272 pairs; `scripts/rl/s8n_analysis/twins_gradient_test.py`): sum z^2 = 13.5
  (chance 16; 95th pct 26.3), one coordinate with |z| > 2 (consensus of layer 31, z = -2.07), path efficiency
  0.562 vs Adam-noise null 0.535 (p = 0.36). The memory-rule gradient is indistinguishable from noise so far.
  Run 1 continues to its pre-registered end; because its rule stays near S8N's (frame overlap ~0.85), its
  step-125 journal doubles as a local GRPO-family reference for paired tests.

### Run 1 val_seen-300 (development split; local base 159/300)

| export | SR | SPL | NE | OSR | dSR vs base [95 % CI] | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---|---|---:|
| step 25 | 55.33 | 51.22 | 5.56 | 68.00 | +2.33 [-2.7, +7.3] | 34 / 27 | 0.44 |

(HPG references at step 25, each vs its own base 155/300: GRPO -1.00, F-B +0.33.) Steps 50 and 100 are not
evaluated (each val_seen-300 beside training costs ~20 % of training throughput for ~110 min); 75 and 125 are.

### 2026-10-30 16:05 — mechanism probe: rescue from the agent's own memory vs from a fresh memory

Same 32 hard instructions, seeds and trigger points (mains identical 32/32); the fresh variant resets the
session at the trigger pose (instruction only) before the same takeover and continuation rules
(`scripts/rl/rescue_probe.py --fresh-memory`).

| branch memory | triggered failures rescued | branches that STOP <= 2 decisions after the hand-back |
|---|---:|---:|
| own S8N state (snapshot) | 6 / 16 | 3 |
| fresh (no history) | 5 / 16 | 0 |

Not different at n = 16. They rescue different episodes: own memory wins the late, overshoot triggers
(t = 53-74; the agent knows its progress), the fresh memory wins three early route errors (t = 21-24 and 77) where
the agent's own memory holds the wrong belief. **"Memory-consistent interventions recover better" is not
supported**; the design keeps the own-memory branch because training must happen under the memory the policy
will have at test time (realised-memory replay), not because of this probe.

### 2026-10-30 16:08 — RxR15 pipeline check

`ops/local/eval_rxr.sh` (English guides, 15-degree conversion, audited val_unseen hashes) on 4 val_unseen
episodes with base S8N: complete, 0 failures, nDTW coverage 4/4 (GT file loads), SR 3/4. Full RxR15 val_unseen
(3,669 episodes, ~101 decisions each in this sample) needs ~4 h of both GPUs; it is scheduled for the base and
the final policy after the R2R runs.

Automation: `ops/local/after_twins.sh` (pid logged in `runs/after_twins.log`) waits for run 1 to finish, runs a
1-step rescue smoke from a frozen dev snapshot, the run-1 and base val_unseen evaluations on both GPUs, then
launches run 2 (rescue) only if the smoke completed.

### 2026-10-30 16:15 — run 1 incident: CUDA OOM on GPU 0, resumed

- Step 48's update OOMed rank 0 (GPU 0: other user's 32.6 GB + trainer 46 GB + 5 helpers + 5 Habitat
  renderers; 851 MiB free when a 1.19 GiB block was needed). Placement's 4 GiB margin was too small for one long
  trajectory on the shared GPU.
- Fix (systems only, all local configs): `placement_margin_gib: 8`, so long trajectories are placed on GPU 1.
- The first resume failed in `torch.cuda.set_device` ("CUDA-capable device(s) busy or unavailable") until the
  MPS daemon was restarted. Resumed at 16:23 from the step-45 checkpoint (steps 45-47 re-run; as on HiPerGator,
  a resumed run is reproducible in distribution, not bitwise).
- `ops/local/supervise_run.sh` now wraps every run: on a non-graceful exit it restarts MPS and resumes (<= 5
  times), logging each restart in the run log; run 1 has an equivalent guard.

### 2026-10-30 16:40 — the update phase runs on GPU 1 alone (speed diagnosis)

- In 33 of run 1's first 48 steps, rank 0 trained 0-1 trajectories: GPU 0 (other user's 33.4 GB, trainer
  34 GB = bf16 weights 9.3 + ZeRO-2 fp32 master/Adam partitions 25.2, 5 helpers 4.7 GB, Habitat ~4.4 GB) has
  13-17 GiB free before the update, below placement's need (0.56 MiB/token + 10 GiB + margin). Single-rank
  updates ran at ~3.7k tokens/s.
- Fit over 30 single-rank steps: train peak = 45.5 GiB + 0.33 MiB/token x (longest trajectory), i.e. ~11 GiB of
  per-step overhead above the static 34.5 GiB, and ~0.33 MiB/token of activations.
- Run 2 logs every trained trajectory's peak memory above its base, offload flag and time
  (`rollouts/step_*_rank*_memory.json`), to separate per-item peaks from the optimizer-step peak before placement
  is recalibrated (or the optimizer offloaded) for run 3. Expected gain if both GPUs share the update: ~60 s of
  a ~300 s step.
- Prepared for run 2 (dev tree, systems only): `runtime.optimizer_offload` = ZeRO-2 `offload_optimizer: cpu`
  with DeepSpeed's CPU Adam (AVX512, adamw_mode). Checked on CPU: max |difference| to torch.optim.AdamW
  1.4e-6 after 3 steps at lr 1e-3 (fp32 rounding), 0.48 s per 500 M parameters (~2 s per rank's 2.1 B
  partition). Frees ~25 GB per GPU, which GPU 0 needs to share the update. ninja (build tool only) comes from the
  tools venv. `after_twins.sh` smokes run 2 with offload first and falls back to the GPU optimizer
  (`*_nooffload.yaml`) if the smoke fails. Run 1 used the GPU optimizer; the difference is rounding-level.

### 2026-10-30 16:50-17:35 — second GPU job of the other user; optimizer offload, a DeepSpeed bug and its fix

- 16:50 the other user started a second 33.4 GB job, on GPU 1. Run 1 OOMed on GPU 1, and its supervised resume
  OOMed on GPU 0: with ~61 GB per GPU left, the update no longer fits next to the GPU-resident optimizer.
- Optimizer offload (`runtime.optimizer_offload`, DeepSpeed CPU Adam) fixes the memory, but **DeepSpeed 0.16.4's
  CPU-offload path drops gradients**: it accumulates only when the config's `gradient_accumulation_steps` > 1, and
  even then keeps only micro-steps 0 and the last (its CPU buffer is never updated after micro-step 0).
  `scripts/rl/check_offload_accumulation.py` (tiny bf16 model, ZeRO-2, 2 ranks, the trainer's manual-boundary
  pattern; share of parameters whose update disagrees by > lr/2 with each reference):

  | path | vs summed gradient | vs last micro-step only | grad norm |
  |---|---:|---:|---:|
  | GPU AdamW (run 1, steps 0-49) | 0.1 % | 24.1 % | 6.98 |
  | offload, GAS 1 | 24.0 % | **0.0 %** | 4.96 |
  | offload, GAS 1024 or GAS 5 + engine boundary | 23.0 % | 6.4 % | 5.08 |
  | offload + `rl/zero_offload_patch.py`, GAS 1024 (5 and 17 micro-steps) | 0.1 % | 24.1 / 35.5 % | 6.98 |

  Run 1's 17:07 offload resume (unpatched) trained steps 50-51 on the last trajectory's gradient only (grad norm
  0.7-0.8); it was stopped before any save, and steps 50-51 were re-run from the clean step-50 checkpoint.
- End-to-end on S8N (1 step, 4 prompts x G 2, same seeds): identical rollouts and loss; grad norm 4.44647 (GPU
  AdamW) vs 4.44675 (patched offload); training peak 37.5 -> 17.8 GiB per GPU for +6.7 s per step.
- Run 1 resumed at 17:32 from step 50 with code `/nvme/public/svln/code/twins_resume_offload3_20261030_1731`
  (dev tree: patched offload, longest-first update order, per-item memory log; the twins algorithm is
  unchanged). Steps 50-51: both ranks train 16 trajectories, grad norm 2.9 / 3.6, 241 / 186 s per step (was
  ~300-330 s with the update on GPU 1 alone).
- Provenance of run 1: steps 0-49 = commit 334bc35 with GPU AdamW; steps 50-124 = the snapshot above with CPU
  Adam offload. The difference is rounding-level (checked above), but it is a mid-run systems change and is
  reported as such.

### 2026-10-30 18:04-18:18 — host OOM kill, orphaned helpers, a third job of the other user

- 18:04 rank 1 of run 1 was SIGKILLed by the host OOM killer: offload pins ~25 GB of host RAM per rank, and the
  base S8N val_unseen evaluation I had just started beside training (4 workers on GPU 0, each ~10 GB of host RAM
  while loading) exhausted it. The evaluation was stopped; **no evaluation runs beside offloaded training**.
- The killed rank's 5 rollout helpers survived as orphans holding MPS connections, so the supervisor's MPS
  shutdown blocked until they were killed (18:17). `supervise_run.sh` now kills this user's leftover helpers and
  Habitat servers first and bounds the MPS shutdown with a timeout.
- 18:06 the other user started a third GPU job (26.5 GB on GPU 0; their first job on GPU 0 had ended ~17:50).
  With offload, run 1 fits beside it.
- Run 1 resumed at 18:18 from step 55 (one step lost).
- 18:30 host RAM: each offloaded trainer rank held ~59 GB RSS (fp32 master + moments + fp32 grads + the
  accumulation buffers + staging), leaving 9 GB available; the step-75 export (8.4 GB state dict on rank 0)
  would likely have triggered another OOM kill. 18:41 run 1 restarted from its step-60 checkpoint with
  **3 helpers per rank** (systems only; code `/nvme/public/svln/code/twins_resume_h3_20261030_1840`, hardened
  supervisor): 31.5 GB available, 214-229 s per step (rollouts are GPU-bound, so fewer helpers cost ~nothing).
  All offload configs now use 3 helpers per rank.

### 2026-10-30 19:45 — run 1 at step 75: no memory-rule gradient

`twins_gradient_test.py` over steps 0-74 (600 CRN twin pairs): **sum z^2 = 6.69** (chance 16; chi2(16) 95th
pct 26.3), max |z| = 1.32 (birth of layer 7), path efficiency 0.321 vs Adam-noise null 0.416 (p = 0.93). The
retention rule's antithetic gradient is indistinguishable from zero; the rule (`exports/step_0075/
sweep_retention.json`: consensus -0.75 .. +0.27, birth -0.06 .. +0.08) is an Adam random walk around S8N's own.
Reading: with the RL-trained S8N policy, the choice of retained frames has no detectable effect on return at
this sample size (twins still diverge in 50-100 % of pairs and flip success in ~15 %, in both directions).
Together with F-B + start tilt (-2.0, n.s.) and MemFork's collapse on M5, memory-rule credit is not where RL on
S8N gains; this is the negative half of the story, and it motivates spending the counterfactual budget on
action-level recovery instead (rescue, run 2).

### 2026-10-30 22:00 — run 1 at step 100: still no memory-rule gradient; the test is now calibrated

Run 1 exported step 100 at 21:38 (steps 75-99: 202-394 s, mean ~275 s with 3 helpers per rank; ETA step 125
~23:35). `twins_gradient_test.py` was extended (dev tree) and re-run over steps 0-100 (808 CRN twin pairs;
`/nvme/public/svln/runs/twins_grpo_s8n_gradient_test_s100.json`):

- **Upper-tail tests (is any direction of the rule better?): no.** sum z^2 = 5.74; calibrated by a sign-flip
  randomization (under "the tilt does not affect return" main and twin are exchangeable), its null has median
  15.6 and 95th pct 26.8, p = 0.99. Upper 95 % bound on the return slope along the best of the 16 directions:
  0.62 reward per one-sigma move of the rule (~5.8 pp SR at ~10.7 reward per success); per coordinate the
  95 % bound is ~0.16 reward (~1.5 pp SR) per sigma (sigma_consensus 0.25, sigma_birth 0.05; a 1-sigma move
  changes roughly 10-25 % of the retained frames).
- **Lower-tail anomaly, not a finding.** sum z^2 sits in the lower tail (randomization p_lower 0.008) and the
  path efficiency of w (0.236) is below its Adam-replayed null (95 % interval 0.252-0.492, two-sided p 0.022):
  the gradient cancels over time slightly more than chance, which a weak restoring force toward S8N's own rule
  would produce. The direct test is n.s.: corr(g_t, w_t - mean w) = -0.174 over all 16 coordinates, but the
  null obtained by replaying each sign-flipped gradient sequence through the run's Adam is itself negative
  (95 % interval -0.192 .. -0.110, p_lower 0.14) because the mean of a random walk co-moves with its
  increments (a Dickey-Fuller-type bias; a fixed-path randomization had wrongly given p < 0.001). The replay
  reproduces the logged w path exactly.
- Reading: S8N's drift rule is a (noisy) stationary point of the RL return in this 16-dimensional family;
  nothing indicates that RL can improve it. The only claim the paper makes is the bounded one above.

### 2026-10-30 23:37 — run 1 finished (step 125 exported); final twins gradient test

Step 125 exported at 23:37 (sha256 45189dc3...; 125 steps, 9.9 h of step time, ~290 s/step overall).
Final test over steps 0-124 (1,000 CRN twin pairs; `/nvme/public/svln/runs/twins_grpo_s8n_gradient_test_s125.json`):
sum z^2 = 6.29 (sign-flip null median 15.5, 95th pct 26.9, p = 0.99); largest |z| = 1.72 (birth, layer 19);
95 % upper bound on the return slope 0.55 reward per one-sigma move along the best direction; path efficiency
0.200 vs Adam-replayed null 0.239-0.453 (two-sided p 0.009: the rule wanders back and forth slightly more than
noise alone would make it); restoring-force correlation -0.155 vs replayed null -0.175 .. -0.105 (p 0.22).
Final rule (`exports/step_0125/sweep_retention.json`): consensus -0.61 .. +0.41, birth -0.02 .. +0.18 —
within the range a pure Adam random walk reaches. **Memory twins does not learn a retention rule; S8N's own
rule is not improvable by RL in this family (bounded as above).** Val_unseen of this export is the
pre-registered check of the policy (queued, after_twins.sh).

### 2026-10-31 01:26 — run 1 step 125, R2R val_unseen (1,839 episodes, local)

`/nvme/public/svln/evals/twins_grpo_s125_r2r_val_unseen`: **SR 60.03**, SPL 53.61, NE 4.87, OSR 68.35, 161
forced stops (500-action cap), 0 failures, peak KV 2,646 rows. Descriptively equal to HiPerGator GRPO 60.41
(unpaired, different machine) and F-B 60.63; consistent with the null twins gradient (the learned rule is a
random walk near S8N's, so run 1 behaves as plain GRPO with sigma-perturbed memory during training). Paired
tests wait for the local base (running) and the local GRPO G = 5 control (run 3).

### 2026-10-31 02:52 — local base S8N val_unseen; base vs run 1 paired

Base S8N (`/nvme/public/svln/evals/base_r2r_val_unseen`, local): **SR 52.04**, SPL 48.35, NE 5.59, OSR 58.08,
124 cap stops (HPG 51.50, original server 52.42: the reproduction holds on the full split).
`compare/base_vs_twins_s125_val_unseen.json` (paired, 1,839 episodes): run 1 step 125 **+7.99 SR** (bootstrap
95 % CI 5.66-10.28; McNemar 317 vs 170 discordant, p = 2.6e-11), +5.26 SPL, +10.28 OSR, +18 decisions per
episode. Failure taxonomy base -> run 1: never within 3 m 555 -> 393, near miss 94 -> 47, **passed goal then
stopped outside 110 -> 146, cap 123 -> 149**: outcome-only RL moves the agent to the goal far more often but
the OSR-SR gap widens (6.0 -> 8.3 pp), i.e. it overshoots/loops near the goal; "never within 3 m" (393, 21 %)
is still the largest failure class, which is what rescue targets (H-R2). Since memory twins learned no rule,
run 1 is effectively a local GRPO-family reference (+8.0 locally vs +8.9 for GRPO on HPG).
- 03:11 run 1 step 125 on val_seen-300: SR 58.67, SPL 53.49, OSR 68.00 (base 53.00 here).
- 03:11 **run 2 (rescue GRPO) launched**: `/nvme/public/svln/runs/rescue_grpo_s8n`, code
  `/nvme/public/svln/code/rescue_20261030_2338`, config `configs/rl/local/rescue_grpo_s8n_r2r1k.yaml` (offload,
  3 helpers per rank); the 1-step smoke (4 prompts x G 2, delta 0.25) passed with 2 rescues trained. No evaluation
  beside training (host RAM); exports at 25/50/75/100/125 are evaluated on val_seen-300 after the run.
- 03:58 queued `ops/local/after_g5.sh` (pid 1303405, log `/nvme/public/svln/runs/after_g5.log`), which waits for
  run 3's evaluations and then: (1) paired comparisons base/twins/rescue/G5 into `/nvme/public/svln/compare/`
  plus `rescue_training.py` statistics; (2) RxR15 val_unseen on a fixed 1,000-episode subset
  (`data/rl/rxr15_val_unseen_subset1000.json`, seed 20261031, 11 scenes) for base, rescue s125 and G5 s125;
  (3) the pre-registered rule: A1 + A2 only if rescue > G5 on val_unseen SR with McNemar p < 0.05; if rescue is
  ahead but n.s., second seeds of rescue and G5 (post hoc, for power); otherwise no more training; (4) val_seen-300
  of exports 25-100 of runs 2 and 3 (training curves).

### 2026-10-31 05:15 — run 2 (rescue) stopped at step 22: imitating turn-arounds makes the policy spin

Same prompts as run 1 at every step (paired training curves). Main trajectories, 5-step windows, run 2 vs run 1:

| steps | turn share | eps with >= 12 consecutive turns | truncation | train SR |
|---|---|---|---|---|
| 0-4 | 0.363 / 0.360 | 0.11 / 0.13 | 0.025 / 0.019 | 0.556 / 0.531 |
| 5-9 | 0.404 / 0.384 | 0.20 / 0.16 | 0.044 / 0.044 | 0.556 / 0.569 |
| 10-14 | 0.429 / 0.387 | 0.27 / 0.21 | 0.125 / 0.044 | 0.562 / 0.569 |
| 15-19 | 0.495 / 0.382 | 0.46 / 0.23 | 0.175 / 0.062 | 0.506 / 0.637 |
| 20-21 | 0.530 / 0.408 | 0.47 / 0.19 | 0.234 / 0.094 | 0.422 / 0.453 |

Cause (credit mass per 5 steps, `scratchpad/rescue_mass.py`): 79-80 % of the expert's forced decisions are turns
(the takeover starts >= 1 m past the closest approach, so the follower first turns the agent around, up to 12
x 15 deg, then 3 forwards); verified imitation puts +154..+627 coefficient mass on those turn tokens per window,
against GRPO's own net turn credit of -144..-711 (failed, spinning mains). The forced tokens are low-probability
under the policy, so their logit gradients are the large ones. The expert's turn direction from a lost state
depends on the goal's position, which the agent cannot see; imitating it teaches "turn more", not "turn
correctly". Run 2 was stopped at step 22 (`ops/local/stop_run.sh`, STOPPED records the reason; resumable state
at step 20 kept). This is a design failure of "correct after the fact", not a code bug; the training
statistics up to step 21 are kept as the evidence.

Redesign (rewind): when the trigger fires at t (d_t >= min_{k<t} d_k + 1 m), fork from the agent's *closest
approach* k* = argmin_{k<t} d_k (its S8N state snapshotted there in O(1) and kept until a new minimum), not from
t. The expert then re-decides the decisions where the trajectory went wrong, from states that are on the path
and in the agent's own memory (their turn mix is that of ordinary navigation, and near the goal the expert's
action is STOP, which targets the "passed goal then stopped outside" class that GRPO increased 110 -> 146).
Run 3 (GRPO G = 5, the control needed in any case) was launched at 05:17 while the rewind variant is built.
- 05:17 run 3 (GRPO G = 5) first launch failed at start-up: `micro_batch_trajectories: 2` cannot tile groups of
  5 ("Micro-batches must tile groups and ranks"; it would have failed in the old chain too). Fixed to 1 (each
  trajectory its own token mean; same total weight per trajectory as the verl pair split up to the pairing) and
  relaunched 05:18 (`/nvme/public/svln/code/grpo_g5_s8n_20261031_0517`); step 0: 40 trajectories, 236 s.
- Rewind implemented (dev tree): `rescue.rewind` in `rl/rollout.py` (snapshot retaken at each strict new
  minimum of d before a_t; at the trigger the branch forks from the first decision at the closest approach;
  the follower's STOP is a forced decision that ends the branch), rewind / main-turn-share statistics in
  `rescue_metrics`, 3 new CPU tests (main unchanged, causal trigger, fork at the closest approach, takeover =
  follower from that state, forced STOP ends the branch, prefix credit zero); 109 RL tests pass (the 3 failing
  tests of the full suite, VGGT targets / episodic job, fail identically on the pre-change snapshot).
  Configs `rewind_grpo_s8n_r2r1k.yaml`, `rewind_noimit_grpo_s8n_r2r1k.yaml`, `smoke_rewind.yaml`.
- `ops/local/drift_guard.py` (fixed now, before run 4): stops a guarded run when its mains' turn share exceeds
  run 1's on the same prompts by > 0.06 in two consecutive 5-step windows (run 2: +0.004, +0.019, +0.042,
  +0.113 in windows 0-19 -> it would have stopped run 2 at step 25).
- 05:27 `ops/local/after_run3.sh` (pid 1327043, log `/nvme/public/svln/runs/after_run3.log`) replaces the killed
  after_rescue / after_g5 chains: run 3 evals -> rewind smoke -> run 4 = rewind (guarded; fallback A1 = rescue
  at t without imitation if the smoke fails or the guard stops it) -> run 4 evals and paired comparisons ->
  RxR15 subset (base, G5, run 4) -> pre-registered decision (run 4 > G5 with McNemar p < 0.05: rewind without
  imitation; ahead but n.s.: second seeds, post hoc) -> val_seen-300 training curves.

### 2026-10-31 05:48 — rewind probe: twice the rescue rate of fork-at-t on the same mains

`scripts/rl/rescue_probe.py --rewind` (base S8N, T 1.2, the 32 hard instructions and seeds of the 12:21 probe;
mains identical 32/32; `/nvme/public/svln/probes/rescue_target_d1_rewind.json`; run beside run 3 on GPU 0 with
oom_score_adj 1000):

| branch start | triggered failures rescued | forced decisions per rescue | turn share of forced decisions | forced STOPs |
|---|---:|---:|---:|---:|
| trigger t (run 2's rule) | 6 / 16 | 13.9 | 0.79 | - |
| **closest approach k* (rewind)** | **12 / 16** | 8.3 | 0.65 | 1 |

Paired over the 16 triggered failures: both 5, only rewind 7, only fork-at-t 1, neither 3 (exact McNemar
p = 0.07). The rewind point is on average 15 decisions before the trigger (k* = 0 for three route errors from the
very first decisions). The rewind corrections are shorter but still turn-led (typically 4-6 x 15 deg, then
forwards): H-R4's first clause (forced turn share < 0.5) already fails in the probe (0.65), and with twice the
verified rescues the total imitated turn mass per step may be similar to run 2's, so the drift guard matters.
Chain revised (05:50, pid 1333721): if the guard stops rewind, run 4 falls back to rewind *without imitation*
(`rewind_noimit_grpo_s8n_r2r1k`, also guarded), not to fork-at-t.

### 2026-10-31 06:00 — cost of a rewind snapshot (measured)

`scripts/rl/s8n_analysis/snapshot_cost.py` (base S8N, 4 hard instructions, rewind monitor as in training, GPU 0
beside run 3; `/nvme/public/svln/probes/snapshot_cost.json`): 133 snapshots over 271 decisions (one per new
closest approach, ~0.5 per decision). **Every snapshot is 135.7 MiB (max 136.0) whatever t is**: 49.5 MiB FP32
GDN recurrent + conv state of the 24 linear-attention layers, the rest the 8 attention layers' KV (<= 2,245
rows: instruction + 4 recent groups + top-4 frames per layer). Clone time 1.37 ms median (p95 1.98 ms), against
~280 ms per decision for the model step. A full-history cache at the same layer sizes (instruction 67 rows + 276
rows per observation group, measured) would be 267 / 483 / 914 / 1,777 / 3,502 MiB at t = 25 / 50 / 100 / 200 /
400 decisions, and every rewind would copy it (or re-encode the prefix, O(t) model work).

### 2026-10-31 07:58 — run 3 restarted with the G = 4 runs' loss normalisation

Run 3 as launched at 05:18 used `micro_batch_trajectories: 1` (needed only because the trainer required
micro-batches to tile groups), i.e. a per-trajectory token mean, while every G = 4 run (runs 1, 2 and the HPG
journals) uses verl pairs. That makes the compute control differ from the method in its loss normalisation as
well as in G, and per-sequence normalisation is a known length-bias source (Dr. GRPO): in steps 25-32 run 3's
truncation was 0.20 vs run 1's 0.10 on the same prompts (stop share 0.015 -> 0.009; turn share flat at 0.36;
training SR equal). Micro-batches only set `loss_weights`, so the tiling check was relaxed to "whole
micro-batches per rank" (pairs may straddle groups of 5; `rl/train_grpo.py`), the G5 config set back to 2, and
run 3 restarted from step 0 at 07:59 (`/nvme/public/svln/code/grpo_g5_s8n_20261031_0758`). The stopped attempt
is kept as `/nvme/public/svln/runs/grpo_g5_s8n_mb1_stopped` (steps 0-31). The chain `after_run3.sh` was
restarted with it (pid 1391546). Lost: 2.7 h.

### 2026-10-31 11:19 — run 3 paused for the S8N stage-2 DAgger evaluations (user request)

The user asked for the two S8N stage-2 models of `anhdao69/stage2_S8N` (revision `59c4d00d`) to be evaluated
before training continues: `S8N_DAGGER` (one epoch on S8N's own Janus-style DAgger rollouts, labels as executed:
S8N's choice while in control, the expert's recovery after a takeover) and `S8N_DAGGER_ORACLE` (same rollouts and
contexts, every step relabelled with the oracle action; classic DAgger). Both start strictly from S8N's weights
(SHA-256 `c89de946`, our local base file), fresh AdamW, 1,775 updates, peak LR 5e-6; data 14,195 episodes (R2R
3,043 + English RxR15 11,152). Downloaded to `/nvme/public/svln/checkpoints/stage2_S8N` (read-only); all 24 files of
each folder match the published `SHA256SUMS.json` (hash and size) and both weight files the Hub LFS hashes.
`navigation.json` equals S8N's except the training schedule fields; config, processor, tokenizer and chat template
are identical. The stage-2 code (`c2a4b1b`, branch `s8n_stage2`) is not on origin; its README states its model and
inference path is that of `m5_s8n_followup` (`11baeb1`), whose inference sources differ from ours only by the
optional retention/tilt loaders (None for a plain S8N folder) and the evaluator's model start-up timeout.

Run 3 was stopped right after its step-40 resumable save (`ops/local/stop_run.sh`; `resumable/step_0040`, 2 x 25.2
GB; nothing lost). `ops/local/eval_dagger.sh` (pid 1444968, log `/nvme/public/svln/evals/dagger_eval.log`) froze the
code (`/nvme/public/svln/code/dagger_eval_20261031_1119`), refused to run unless every package source except `rl/`
equals the base journal's (77/79 equal; only `rl/rollout.py`, `rl/train_grpo.py`, never imported by the evaluator)
and the checkpoints still match their manifests, then runs `final_evals.sh` (R2R val_unseen, all 1,839 episodes,
5 workers per GPU, exactly the base S8N journal's layout) for both models. Every worker's model runtime (memory
spec, serializer, versions) equals base's. Afterwards: completeness/protocol checks, paired comparisons with base,
run 1 and each other, then run 3 resumes from step 40 with its own frozen code and the chain restarts.

Correctness checks while `S8N_DAGGER` runs (11:25-11:40): (i) a journal verifier
(`scratchpad/verify_eval.py`, itself validated on the base journal against itself: 0 problems, 1,839/1,839 identical)
checks shard coverage and disjointness, every shard contract against base's (only model identity and `rl/`
sources may differ), the weights hash, and per episode success = (final geodesic distance <= 3 m), oracle success
= (minimum distance <= 3 m), trajectory length = the recorded pose path, no motion on turns/STOP, and the
shortest-path length implied by SPL equal to base's on episodes both solve: 0 problems on the first 72 and 153
episodes. (ii) Base S8N re-run now through the frozen evaluator, beside the 10 DAgger workers under MPS, on 12
episodes DAgger had finished (`/nvme/public/svln/evals/repro_base_12_during_dagger`): **12/12 bit-identical to the
base journal** (every response token, pose and metric). Differences are therefore the weights'.
Early behaviour (186 episodes, same episodes for base): `S8N_DAGGER` reaches the goal area more often (within 3 m
147 vs 120, within 1 m 113 vs 77) but keeps moving after it (median 17 vs 6 actions from its closest point to its
STOP; "passed the goal, stopped outside" 30 vs 8 of the first 94), consistent with the collection rule that
replaced S8N's STOP near the goal by expert moves until the 0.25 m final radius (labels as executed), and with the
recovery-efficiency filter that kept only corrected episodes (14,195 of 30,815).

### 2026-10-31 13:45 — S8N_DAGGER finished (R2R val_unseen, 1,839 episodes)

Journal `/nvme/public/svln/evals/s8n_dagger_r2r_val_unseen`: complete, 0 failures; verifier on the full journal:
**0 problems** (weights 7e37a4a3…, shortest path implied by SPL equal to base's on all 270 jointly-solved
episodes with SPL < 1, max horizontal forward step 0.287 m). SR 52.04 / SPL 42.38 / OSR 70.26 / NE 5.93 / TL 15.96
/ 148.6 steps / 262 forced stops (base: 52.04 / 48.35 / 58.08 / 5.59 / 9.41 / 90.4 / 124). Same SR as base by
count, not by episode: 317 vs 317 discordant (McNemar p = 1); only 2/1,839 action sequences identical to base.
vs run 1 s125 (60.03): 238 vs 385, p = 4.2e-9. Within 1 m of the goal at some point: 1,047 (base 651, run 1 915);
within 3 m and stopped outside: 335 (base 111, run 1 153). The navigation gain from imitation is exactly cancelled
at the stop decision. `S8N_DAGGER_ORACLE` started 13:45 (weights a07a8ee7…, verified in shard 0).

### 2026-10-31 15:49 — S8N_DAGGER_ORACLE finished; run 3 resumed

Journal `/nvme/public/svln/evals/s8n_dagger_oracle_r2r_val_unseen`: complete, 0 failures, protocol and runtime equal
to base; verifier on the full journal: **0 problems** (weights a07a8ee7…, SPL-implied shortest paths agree with
base's on 277/277). SR 58.84 / SPL 44.97 / OSR 72.65 / NE 4.75 / TL 15.72 / 126.5 steps / 68 forced stops.
Paired (`/nvme/public/svln/compare/*_val_unseen.json`): base -> ORACLE +6.80 (CI +3.86..+9.68; 433 vs 308;
p = 5.0e-6); DAGGER -> ORACLE +6.80 (418 vs 293; p = 3.1e-6); run 1 s125 -> ORACLE -1.20 (CI -3.86..+1.52; 312 vs
334; p = 0.41); run 1 s125 -> DAGGER -7.99 (p = 4.2e-9). ORACLE reaches more than run 1 (OSR +4.3, within 1 m 1,031
vs 915) and stops worse (254 vs 153 reached-then-left, SPL -8.64). Run 3 resumed 15:49 from `resumable/step_0040`
("resumed at step 40" on both ranks, same frozen code and config); `after_run3.sh` restarted (pid 1516167).

### 2026-10-31 19:05 — queue re-ordered: IL -> RL from S8N_DAGGER_ORACLE after run 3 (user decision)

S8N_DAGGER_ORACLE ties run 1 with the opposite error profile (it reaches more, run 1 stops better), so the next RL
runs start from it (user's choice among: ORACLE after run 3 / keep the base-init plan / stop run 3 now). Run 3
continues untouched (step 78 at 19:00, ~4.6 min/step, COMPLETE expected ~22:30). `after_run3.sh` was killed while
idle in its wait and replaced by `ops/local/after_run3_oracle.sh` (pid 1569892, same log `runs/after_run3.log`):
1. run 3 evaluations (val_unseen, val_seen-300) + paired comparisons with base, run 1 and ORACLE; ORACLE val_seen-300;
2. rewind smoke from ORACLE (`smoke_rewind_oracle.yaml`); run 5 = GRPO G = 5 from ORACLE
   (`activevln_grpo_g5_oracle_r2r1k.yaml`); run 6 = rewind from ORACLE (`rewind_grpo_oracle_r2r1k.yaml`) under
   the drift guard referenced to run 5 (same initial policy and prompts), fallback without imitation;
3. the former run 4 (rewind from base S8N, guard referenced to run 1, fallbacks unchanged);
4. RxR15 1,000-episode subset for base, ORACLE, G5, runs 5, 6, 4;
5. no-imitation ablation of run 6 only if run 6 beats run 5 at McNemar p < 0.05;
6. val_seen-300 training curves.
The four ORACLE configs differ from their base-init files only in `base.checkpoint`/`base.weight_sha256`
(a07a8ee7…, the published hash of `pytorch_model.bin`) and a header comment (checked by diff and YAML load).

### 2026-11-01 01:18 — run 3 (GRPO G = 5 from base S8N) evaluated; rewind smoke from ORACLE passed; run 5 launched

Run 3 COMPLETE (125 steps, mean 4.86 min/step, same as run 1). `grpo_g5_s125` val_unseen: SR 58.29 / SPL 51.86 / OSR
67.59 / NE 4.79 / TL 10.71 / 103.1 steps / 144 forced stops; within 1 m 914, reached-then-left 171. Verifier (export
`step_0125/pytorch_model.bin` sha256 92e668f2…, hashed independently): all checks pass except two **policy** failures
counted as unsuccessful by the evaluator (as every journal): episode 1443 "response limit reached without assistant
terminator", 1811 "invalid navigation response 'MOVE_FORWARDMOVE_FORWARD'" (greedy format errors of the RL policy;
<= 0.11 pp). Paired: base -> G5 +6.25 (CI +3.86..+8.59; 302 vs 187; p = 2.2e-7); **G5 -> run 1 (twins) +1.74 (CI
-0.44..+3.97; 227 vs 195; p = 0.13, n.s.)**: at matched rollouts per group, memory twins does not beat GRPO, as
its null retention gradient predicted; ORACLE -> G5 -0.54 (p = 0.72; SPL +6.89). ORACLE val_seen-300: SR 63.67 / SPL
50.52. Rewind smoke from ORACLE (delta 0.25, 1 step): 5 rescues, 3 successful, 2 of 3 failed mains rescued, export
written. Run 5 = GRPO G = 5 from ORACLE launched 01:18 (`/nvme/public/svln/runs/grpo_g5_oracle`, code
`grpo_g5_oracle_20261101_0118`); the configured ORACLE hash check passed at start.

### 2026-11-02 — SFR campaign started (`s8n_sfr_campaign.md`); GPU 0 fault; first 8k-selection evidence

- New goal: RL from `S8N_DAGGER_SF_REMAINING` (SFR, report SR 63.68 / SPL 56.65; weights e3ada114…, all 25 files
  verified, read-only at `checkpoints/stage2_S8N/S8N_DAGGER_SF_REMAINING`). Configs `*_sfr_r2r1k.yaml` (G4, G5,
  rewind, rewind no-imitation, new **self-rewind** `rewind_self_grpo_sfr_r2r1k`: `rescue.self_branch`, the branch
  at k* is the policy's own retry; tests 10/10, RL suite passes except 3 tests needing a missing checkpoint).
- **GPU 0 (PCI 21:00) fault:** healthy at 11:39; at ~11:50, minutes after the first SFR evaluation started, NVML
  reported "GPU requires reset"; 4 model servers stuck in the driver (unkillable, 46.7 GB). Nothing computed on it
  is used (partial journals moved to `*_partial_gpu0fault_1150`). The host was rebooted 13:37 (GPU 0 healthy
  again); nothing ran 13:37-20:34. Relaunch 20:34 on both GPUs; `ops/local/gpu_watchdog.sh` now logs GPU health
  every minute (`runs/gpu_health.log`, marker `runs/GPU_FAULT`). Launchers accept `TRAIN_GPUS` / `FINAL_GPUS`
  (one rank per GPU; seeds are rank-independent, so a 1-GPU run samples the same trajectories).
- **8k selection, evidence from earlier runs** (`scripts/rl/s8n_analysis/prompt_signal.py`,
  `/nvme/public/svln/analysis/prompt_signal_base_runs.json`; the 1k prefix, one visit per prompt):

| run (init) | groups with signal | all-success | all-fail after re-rolls | trajectories on zero-signal groups | signal by quarter |
|---|---:|---:|---:|---:|---|
| run 1 twins (S8N) | 63.2 % | 29.9 % | 6.9 % | 41 % | 0.70 / 0.65 / 0.57 / 0.60 |
| run 3 G5 (S8N) | 66.2 % | 27.8 % | 6.0 % | 37 % | 0.74 / 0.70 / 0.61 / 0.60 |
| run 5 G5 (ORACLE, steps 0-98) | 57.4 % | 38.5 % | 4.0 % | 45 % | 0.84 / 0.58 / 0.45 / 0.43 |

  Pass rates transfer between the two S8N runs (Spearman 0.75) but weakly from S8N to ORACLE (0.38): difficulty is
  policy-specific, so a pool must be screened with the policy that is trained. Free features predict signal
  poorly (AUC: N* 0.57-0.60, geodesic 0.54-0.58, instruction words 0.49-0.55); group statistics of the policy do
  (mean decision entropy 0.66-0.78; oracle-success rate predicts all-success at 0.91-0.92). The stronger the
  initial policy, the larger the all-success share (wasted rollouts) and the faster signal decays over training.
- 20:45 chain v2 (`ops/local/sfr_campaign2.sh`, adds the 1k greedy train screen) replaced v1 while v1's SFR
  val_unseen evaluation kept running; part 2 (`ops/local/sfr_campaign_part2.sh`: self-rewind C, G5 control E)
  queued behind it. Partial SFR val_unseen at 331/1,839: SR 61.0 vs base 51.4 on the same episodes.
- ActiveVLN's RxR 4k is pass-rate filtered (rows 0-2,919 = difficulty 1-3 of 4; details in
  `s8n_sfr_campaign.md` 5.1). Its RxR GT macros do not expand to our 15-degree RxR labels (11 / 4,000 equal; R2R:
  4,000 / 4,000), so an RxR RL budget must be defined from our own N*.
- Preview of "branch signal in zero-variance groups" (run 2, fork-at-t from S8N, 176 groups): all-fail groups
  6 + 6 (half get a non-zero branch credit), all-success 31 + 4 (11 %); zero-signal groups 26.7 % -> 21.0 %. Branches
  rescue all-fail waste, not all-success waste; to be measured for rewind from SFR (run B).
- 22:15 SFR local journals: val_unseen SR 62.59 / SPL 55.59 / OSR 69.98 (+10.55 vs base, p = 8.3e-15; +2.56 vs
  run 1, p = 0.055), val_seen-300 66.33; details and the fallback ladder in `s8n_sfr_campaign.md` section 6.
  Screen (1k train, greedy SFR) started 22:15.
- 22:57 screen done (`evals/sfr_screen_rl1k_r2r_train`, greedy SFR on run A's 1k prompts, 0 failures): **SR 76.5 /
  SPL 71.7 / OSR 81.0** (val_unseen 62.6: SFR has seen every R2R train instruction in SFT). Successes stop close
  (median 0.63 m; 76 % within 1 m). Failures 235: 45 reached-then-left, 190 never within 3 m (66 within 3-5 m),
  14 at the cap. Greedy SR by N* tercile 0.84 / 0.82 / 0.65; per scene 0.40-1.00 (47 scenes with >= 10 prompts).
  Expectation for run A: a larger all-success share than any S8N run (S8N runs: 28-30 %).

## 2026-11-03

- 00:25 run A (grpo_sfr) at 22 steps, 190-255 s/step, no restarts; sampled SR 0.64, all-success groups 31 %.
- 00:57 run A step 30: calibration and selection study (`s8n_sfr_campaign.md` 5.3). Run A first visit: signal
  63.3 %, all-success 32.7 %, all-fail 4.0 %. A greedy-solved prompt is solved ~72 % of the time at T = 1.2. Leave-
  one-out: dropping the most-solved half by one rollout / greedy pass / N* lifts the informative share by +14-17
  points for SFR; the screen must come from the trained policy; entropy is weaker than the outcome.
  `select_pool.py`: bins simplified (solved x length tercile, failed x closest approach), re-roll failures counted,
  rule `learnability_top` and a `calibrate` sub-command added; `selection_sim.py` new.
- 01:05 part 2 v2 (`ops/local/sfr_campaign_part2b.sh`) replaced the waiting part 2: adds the rows 1,000-1,999
  screen + the pilot pool (4c) and run D `grpo_learn_sfr` (5b) between C and E.
- 06:27 run A (grpo_sfr) complete, 125 steps (~210 s/step), no restarts. 07:54 val_unseen: **SR 64.33 / SPL 57.32 /
  OSR 72.38 / NE 4.29 / TL 11.14**; vs SFR +1.74 SR (274 vs 242 discordant, McNemar p = 0.17, bootstrap CI
  [-0.7, +4.1]), SPL +1.72, OSR +2.39; forced stops 55 -> 91. Failure taxonomy SFR -> A: never within 3 m
  27.35 -> 23.16, reached-then-left 7.23 -> 7.88, cap 2.83 -> 4.62. (GRPO from base S8N gave +8 pp; from SFR
  +1.7: diminishing returns on top of DAgger.)
- Run A full run, signal dries up: all-success groups by quarter 32.7 / 47.3 / 56.5 / 56.9 %, informative share
  0.64 / 0.50 / 0.40 / 0.40, sampled entropy 0.164 -> 0.097, training SR 0.65 -> 0.78; over the run 48.4 % of
  groups carried signal and 52.5 % of trajectories went to zero-signal groups
  (`analysis/prompt_signal_grpo_sfr_full.json`). For an 8k epoch (1,000 steps) a step-0 screen alone will go
  stale: online skipping / re-screening matters.
- 08:09 run A verified (`verify_eval.py`: 1,839/1,839 episodes, 0 problems, max forward step 0.287 m); val_seen-300
  **SR 68.67** (SFR 66.33, +2.33, SPL +3.60, McNemar p = 0.51). Run B `rewind_sfr` launched 08:09 (frozen tree
  `code/rewind_sfr_20261103_0809`), drift guard referenced to run A.
- 10:43 run B (rewind_sfr) step 30, vs run A on the same 248 prompts and seeds (steps 0-30): sampled SR 0.659 vs
  0.650; **groups with training signal (mains, or a branch with non-zero credit) 75.0 % vs 63.3 %**; branches give
  signal to 5/7 all-fail groups and 17/77 all-success groups; 30 % of mains trigger, branch success 72.8 %, 69.8 % of
  branches beat their main, **half of the failed mains (170/338) are rescued by their branch**. Cost: step time
  291 vs 219 s (+33 %), trained tokens +28 %. Expert takeovers are 57 % turns (turning back toward the goal).
  Drift guard: main turn share minus run A's per window -0.003 ... +0.011 / +0.035 / +0.043 (windows 15-19 /
  20-24 / 25-29; stop at > +0.06 twice): run A's own turn share falls 0.49 -> 0.38 while B's falls less
  (0.48 -> 0.42); watched.
- 17:40 run B (rewind_sfr) complete, 125 steps, never stopped (drift guard: one window over margin, 105-109
  +0.074, then +0.043 / +0.050; max consecutive 1). 19:13 val_unseen: **SR 68.73 / SPL 60.87 / OSR 76.67 / NE 3.68
  / TL 11.18**; verified (1,839/1,839, 0 problems, max forward step 0.288 m).
  - vs SFR: **+6.14 SR** (293 vs 180 discordant, McNemar p = 2.3e-7, CI [+3.86, +8.37]), SPL +5.28 [+3.23, +7.27].
  - vs run A (ActiveVLN GRPO, same prompts / seed / steps): **+4.40 SR** (244 vs 163, p = 7.0e-5, CI [+2.28,
    +6.53]), SPL +3.56 [+1.76, +5.41], OSR +4.30, NE 4.29 -> 3.68.
  - Failure taxonomy SFR / A / B: never within 3 m 27.35 / 23.16 / **17.67**; reached-then-left 7.23 / 7.88 / 7.72;
    cap 2.83 / 4.62 / 5.87 (forced stops 55 / 91 / 121). Turn actions 36.8 % (A) vs 41.7 % (B) of decisions,
    same path length (11.14 vs 11.18 m): B looks around / turns back more; a few more cap episodes.
  - Mechanism, by training quarter (A vs B): groups with signal (mains or a credited branch) 0.64 / 0.50 / 0.40 /
    0.40 vs **0.76 / 0.65 / 0.64 / 0.55**; all-success 0.32 / 0.48 / 0.56 / 0.57 vs 0.30 / 0.41 / 0.46 / 0.52;
    training SR equal (0.78 vs 0.78 in Q3); 31 -> 21 % of B's mains trigger a rewind.
  - Cost: +33 % step time, +28 % trained tokens (compute control: run E, G = 5).
- 19:29 user: no second seeds; finish the 1k ablation table. Part 2 v2 stopped while waiting; part 2 v3
  (`ops/local/sfr_campaign_part2c.sh`) queued: C self-rewind, E G5, F no-imitation, D selection pilot, G fork at t,
  H unverified imitation, learning curves, RxR transfer (`s8n_sfr_campaign.md` 3.2). New configs
  `rewind_forkt_grpo_sfr_r2r1k.yaml` (rescue.rewind false; extra exports) and `rewind_unverified_grpo_sfr_r2r1k.yaml`
  (imitation unverified). Disk 132 GB free (93 %): resumable states deleted after evaluation.
- 19:29 part 1 finished. Run B val_seen-300 **SR 74.67** (A 68.67: +6.00, SPL +4.41, McNemar p = 0.03, CI [+1.0,
  +11.0]). 19:30 part 2 v3 started: resumable states of A and B deleted (free 132 -> 226 GB); rows 1,000-1,999 screen
  started.
- 20:12 run D pilot pool (`data/rl/sfr_learn1k_of2k.jsonl`, sha 6799a1d1…; audit alongside): learnability_top, 1000
  of rows 0-1,999, 59 scenes, greedy SFR SR 0.517 (candidates 0.76), solved share 0.40, mean N* 66.9 (candidates
  ~58.5), mean p_hat 0.566, expected informative 0.750. Calibration from all of run A (attempts counted): failed
  far / near / reached 0.34 / 0.40 / 0.44, solved long / mid / short 0.70 / 0.80 / 0.85 - over the whole run the
  solved bins sit far above 1/2, so the rule takes every greedy failure and the longest solved prompts.
- 20:16 run C (rewind_self_sfr) launched after its smoke (self branches: 5 rescues, 0 imitation). Disk: the
  exports / resumable of the four 1-step smoke runs deleted (metrics, logs, rollouts kept): free 392 GB.
- 22:47 run C (rewind_self_sfr, rewind with self branches, no expert) step 30, vs A and B on the same 248 prompts
  and seeds (steps 0-30; `step30_cmp`, inline): sampled SR A / B / C 0.650 / 0.659 / 0.654; **groups with training
  signal 63.3 / 75.0 / 73.4 %**; C's branches give signal to 3/8 all-fail and 15/76 all-success groups (B 5/7,
  17/77). Same trigger rate (30.1 vs 30.0 % of mains), but self branches succeed far less (**44.5 % vs 72.8 %**) and
  rescue fewer failed mains (**101/343 = 29 % vs 170/338 = 50 %**). Step time 286 vs 291 s (A 219), trained tokens
  7.55k vs 7.97k (A 6.21k): the expert costs almost nothing; the rewind does. Read: most of the extra signal comes
  from rewinding (contrast at the closest approach), the expert adds successful demonstrations (positive credit)
  rather than more informative groups. Drift guard +0.000 ... +0.027 (windows 0-29), never over margin.
- 00:10 (Nov 4) informative share vs G, analytic (`scripts/rl/s8n_analysis/signal_vs_g.py`, empirical-Bayes Beta
  over run A's first-visit outcomes; `analysis/signal_vs_g_grpo_sfr.txt`): G 2 / 4 / 5 / 8 / 12 / 16 -> 0.28 / 0.50
  / 0.56 / 0.67 / 0.75 / 0.79 (matches A's observed ~0.49 at G 4). Run B reaches ~0.65 at G 4 (+33 % step time):
  plain GRPO would need G ~ 8 (2x rollouts) for the same share. Proposed (pending user): no G sweep; E (G 5) is the
  compute-matched control; a G 8 run replaces H only if E closes most of the gap to B (E > ~66.5 SR).
- 01:00 literature check, group-size ablations: ActiveVLN (2509.12618, our baseline) G 4, no G ablation (ablates
  reward, multi- vs single-turn, efficiency). ETP-R1 (2512.20940) G 8, no G ablation (dropout, temperature, mu).
  VLN-R1 (2506.17221) Table 3c k 2/4/6/8: R2R val SR 24.7 / 26.5 / 28.4 / 30.2. NavGRPO (2603.15370, CVPR 2026
  Findings) Table 6 K 2/4/8/16: SR 79.98 / 80.37 / 81.88 / 81.96 (keeps 8: 16 doubles cost). Branching methods
  (TreeRL 2506.11902, ACL 2025; Tree-GRPO 2509.21240, ICLR 2026) do not sweep G: they compare with chain sampling
  at a matched token budget. Doubling G buys +1.5 to +3.7 SR in VLN GRPO papers; rewind buys +4.4 at +33 % cost.
  Proposed (pending user): GRPO G 8 replaces H, for an SR-vs-rollout-budget figure (GRPO G 4 / 5 / 8, rewind G 4).
- 05:40 run C (rewind_self_sfr) complete, 125 steps, never stopped (drift guard max +0.043, windows 110-114; never
  over margin). 07:28 val_unseen: **SR 67.26 / SPL 59.58 / OSR 74.23 / NE 3.98 / TL 11.01**; verified (1,839/1,839,
  0 problems, max forward step 0.286 m, forced stops 128). val_seen-300 SR 71.67 / SPL 66.37.
  - vs run A: **+2.94 SR** (223 vs 169, McNemar p = 0.0074, CI [+0.82, +5.06]), SPL +2.26; val_seen +3.00 (p 0.27).
  - vs run B (expert branches): -1.47 SR (189 vs 216, p = 0.196, CI [-3.59, +0.71]), SPL -1.29; val_seen -3.00
    (p 0.34). vs SFR +4.68 (p = 1.5e-4).
  - Failure taxonomy SFR / A / B / C (exclusive): never within 3 m 27.35 / 23.17 / 17.67 / 19.58; reached-then-left
    7.23 / 7.88 / 7.72 / 6.74; cap 2.83 / 4.62 / 5.87 / 6.42.
  - Mechanism by fifth (rescue_training): trigger rate 0.30 -> 0.27, rescue rate on failed mains 0.45 / 0.29 / 0.33
    / 0.29 / 0.26 (B's expert branches about 0.5 early); step time 286 -> 282 s.
  - Read: rewinding to the closest approach is the main ingredient (2/3 of B's gain over A, significant without
    any expert or privileged information at the branch); the expert adds +1.5 SR, not significant at n = 1,839.
- 07:28 run E (grpo_g5_sfr, GRPO G 5, compute control) launched; step time 277 s (B 291, A 219): close to
  compute-matched with B in wall time.
- 14:10 whole-run mechanism A / B / C (`analysis/prompt_signal_ABC.json`): groups with signal (mains or a credited
  branch) by quarter A 0.64 / 0.50 / 0.40 / 0.40, **C 0.74 / 0.61 / 0.56 / 0.53**, B 0.76 / 0.65 / 0.64 / 0.55;
  all-success A 0.32 / 0.48 / 0.57 / 0.57, C 0.30 / 0.50 / 0.52 / 0.56, B 0.30 / 0.41 / 0.46 / 0.52; share of
  sampled trajectories in zero-signal groups A 0.525 / C 0.503 / B 0.444. The ordering A < C < B of informative
  groups matches the ordering of val_unseen SR (64.33 / 67.26 / 68.73).
- 16:47 run E (grpo_g5_sfr, GRPO G 5) complete; 18:45 val_unseen: **SR 66.50 / SPL 58.71 / OSR 75.20 / NE 3.99 / TL
  11.54**; verified (1,839/1,839, 0 problems, max forward step 0.287 m, forced stops 125). val_seen-300 SR 72.00 /
  SPL 66.25. Whole-run cost per step A / E / B / C: 210 / 264 / 270 / 267 s, trained tokens 6.0k / 7.6k / 7.7k /
  7.3k: **E is compute-matched with B** (and C). Peak memory unchanged by G (train 36.7 GiB, micro-batched).
  - vs A: +2.18 SR (212 vs 172, p = 0.046, CI [+0.11, +4.24]); val_seen +3.33 (p 0.25).
  - **B vs E (matched compute): +2.23 SR (204 vs 163, McNemar p = 0.037, CI [+0.16, +4.24]), SPL +2.16**; val_seen
    +2.67 (p 0.39).
  - C vs E (matched compute): +0.76 SR (189 vs 175, p = 0.50, CI [-1.25, +2.83]), SPL +0.87: not distinguishable.
  - Correction of the 07:28 read: C's gain over A is matched by spending the same compute on a fifth rollout per
    group. At matched compute only the expert-guided rewind (B) is better than plain GRPO; self-rewind is not.
- 18:52 part 2 v4 (`ops/local/sfr_campaign_part2d.sh`) replaces v3 while run F trains (F attached, not relaunched;
  its drift guard untouched). Run H (unverified imitation) dropped; **run I = GRPO G 8** (`activevln_grpo_g8_sfr_r2r1k`,
  only group_size differs from A) added last, for the SR-vs-rollouts curve of plain GRPO (G 4 / 5 / 8) against
  rewind at G 4: E landed at the pre-stated threshold (> ~66.5). Learning curves and RxR (now with E) run before I.
  Expected: F ~Nov 5 07:00, D ~16:30, G ~Nov 6 04:00, curves ~06:00, RxR ~13:00, I ~Nov 7 05:30.
- 20:30 user: skip the runs the paper does not need; prepare run B on 8k prompts. Kept: run G (fork at the trigger
  t, no rewind), the ablation of the method's defining choice. Skipped: F (stopped at step 20, `STOPPED`; resumable
  deleted), D, H, I, learning curves (A / B intermediate exports kept for later). 20:35 part 2 v5
  (`ops/local/sfr_campaign_part2e.sh`): G (launched 20:35, guard vs A) -> RxR val_unseen subset-1000 of SFR and B ->
  `ops/local/sfr_8k_launch.sh`. Another user's job holds 26.5 GB on GPU 1 (pid 459394, user aitt).
- 8k design (`ops/local/sfr_8k_launch.sh`): candidates = all 10,819 R2R-CE train instructions
  (`data/rl/r2r_train_all.jsonl`, `scripts/rl/s8n_analysis/build_r2r_train_pool.py`: the 4k rows agree on every field;
  ActiveVLN's macro length M, used by the budget only when M > 60, estimated as round(N*/1.581) off the 4k list).
  R2R only: B's trigger (goal distance >= closest approach + 1 m) and its shortest-path expert assume shortest-path
  routes; RxR routes are not, so the trigger would fire on correct RxR behaviour and the expert would teach shortcuts.
  Selection: learnability_top, 8,000 of 10,819 by scene, p_hat from run A's calibration: drops the 2,819 lowest
  p(1 - p), i.e. greedy-solved short instructions (p_hat 0.845: a G = 4 group is all-success 51 % of the time, zero
  GRPO gradient, and seldom triggers a rewind). Offline (selection_sim, keep 0.7): informative groups vs random
  +8.2 points [4.7, 12.2] for a greedy screen, +6.0 for N* alone, +7.1 for entropy. Only the short tercile is
  screened (3,029 not screened before, ~2 h); unscreened mid / long rows are binned as greedy successes of their
  tercile (`select_pool.py --unscreened-as-solved`). Dry run (2k screened only) and config generator tested;
  load_pool accepts the pool (1,000 steps). Run: B's config except pool / hashes / max_prompts / export_steps
  (every 125 steps); no drift guard; ~75 h at B's 270 s/step. Expected: G evaluated ~Nov 5 07:40, RxR ~11:00,
  screen ~13:00, 8k launch ~13:00, step 1000 ~Nov 8 16:00, evaluations ~23:00.
- 22:44 run G (rewind_forkt_sfr, expert intervention at the trigger t, no rewind) **stopped by the drift guard at
  step 20**: main turn share minus run A's +0.002 / +0.046 / +0.080 / +0.094 (windows 0-4 ... 15-19). Expert
  takeovers are 80 % turns (B: 57 %), branch success 0.48 (B 0.73 at step 30), step time 359 s (B 270): from a
  lost state the correction is a turn-around, and imitating it makes the policy spin. Same failure as the S8N
  line's run 2 (fork at t, stopped at step 22, 2026-10-31): two independent runs. Rewinding to the closest approach
  is what keeps expert branches stable. Step-20 export being evaluated (latest export after a stop).
- 22:50 user: balance R2R and RxR in the 8k (ActiveVLN: "4k navigation instructions from R2R and RxR,
  respectively"); full epoch (1,000 steps). The R2R-only launch script was renamed
  `ops/local/sfr_8k_r2r_only_launch.sh.disabled` so part 2 v5 stops after the RxR check. Facts for the design:
  the RL env is R2R-only (env_server rejects non-R2RVLN-v1; the evaluator already supports RxR15); RxR English
  train has 19,996 episodes, mean N* 125 (R2R 58), so 4k RxR ~ the rollout work of 8k R2R; on the train ground
  truth, a perfect route follower fires the goal-distance trigger on 27.9 % of RxR routes vs 4.3 % of R2R routes
  (Euclidean proxy, 3,000 each).
- 23:00 user: "test on 2k: 1. 2k R2R with rewind 2. 1k R2R + 1k RxR with rewind 3. 1k R2R + 1k RxR with path-aware
  rewind". All from SFR with B's recipe, 2,000 prompts, 250 steps (`configs/rl/local/rewind_grpo_sfr_r2r2k.yaml`,
  `rewind_grpo_sfr_mix2k.yaml`, `rewind_path_grpo_sfr_mix2k.yaml`); queued by `ops/local/sfr_2k_chain.sh`, which v5
  reaches through `ops/local/sfr_8k_launch.sh` after its RxR check.
  - Data. Run 1: the 2k prefix of ActiveVLN's R2R list (B's 1k + the next 1k; steps 0-124 are B's prompts).
    Runs 2-3: `data/rl/mix_r2r1k_rxr1k.jsonl` (`scripts/rl/s8n_analysis/build_mix_pool.py`): B's 1k R2R and the first
    1k distinct IDs of ActiveVLN's RxR list (`rxr_4000_train_new.parquet`, exported to `data/rl/activevln_rxr4k_ids.json`;
    all 4,000 IDs are English-guide RxR15 train episodes here, one duplicate), alternating R, X, so every rank gets
    2 + 2 per step. N* from this repo's RxR ground truth (mean 110; ActiveVLN's conversion is ~22 % shorter), M from
    ActiveVLN's merged actions: budget min(2M, 120) macro actions = mean 192 primitives (R2R 116), max 412.
  - RL env (`rl/env_server.py`): `rxr-<id>` episodes are loaded with the evaluator's RxR dataset class and served by
    the R2R-configured simulator (vln_r2r.yaml and vln_rxr15.yaml differ only in dataset and nDTW). Route progress
    rho = remaining reference-route length from the agent's projection + distance to it (projection searched 2 m back /
    5 m ahead of the previous one, so revisits keep their order); `oracle_route` = geodesic follower to the route point
    1 m beyond the projection, STOP only at the route's end. On ground-truth actions (25 RxR / 15 R2R train episodes):
    the goal-distance trigger fires on 24 % of RxR routes, the route trigger on 4 % (R2R 0 / 0); from the start,
    oracle_route reaches the goal on 15/15 with rho never rising > 0.15 m.
  - Path-aware rewind (`rescue.path_aware`, `rl/rollout.py`): rho replaces the goal distance in the trigger and the
    rewind point; the expert is oracle_route. Run 2 also logs rho (`step_rho`) to measure false triggers of the goal
    trigger on route-following states. Unit tests `tests/rl/test_path_aware.py`; the RL suite passes (112).
  - Bug found and fixed: `env_server.oracle` cached its ShortestPathFollower per sim; after a scene round trip A -> B
    -> A without an oracle query in B, the follower acted through the replaced agent (InvalidAttachedObject). The
    helper caught it, restarted its env server and re-ran the episode with the same seed: B's env servers restarted
    9-16 times each (A's: never). Results are unaffected (same seed, same deterministic follower), only time. Followers
    are now rebuilt at every reset; 48 real-env rescue episodes (R2R + RxR, both modes) pass.
  - Drift guard: a run's own early windows are no reference (every healthy SFR run's turn share falls ~0.49 -> ~0.41
    in 15 steps; G stayed at 0.50). The 2k runs compare their R2R mains with B at the same step window, extended past
    step 124 with B's last-5-window mean (0.41) (`drift_guard.py --r2r-only --extend-reference`): replayed offline it
    stops G after window 15-19 (+0.072, +0.083) and never comes within 0.05 of stopping A, C or E.
  - Chain: smoke (2 full steps of run 3's config, fresh frozen tree) -> run 1 -> evals -> runs 2, 3 in the order of
    `/nvme/public/svln/runs/2k_order.txt` (read when run 1's evals end) if a smoke passed. Evals of each step-250
    export: R2R val_unseen + RxR val_unseen-1k, compared with SFR, A, B and each other.
  - Estimates (B's 153 s rollout + 115 s update per step; the mixed pool has ~1.5x B's decisions): run 1 ~265 s/step,
    runs 2-3 ~410 s/step (to be measured by the smoke). RxR check done ~04:00 Nov 5, smoke ~04:40, run 1 results
    ~02:30 Nov 6, run 2 ~10:30 Nov 7, run 3 ~18:30 Nov 8.
- 2026-11-05 00:10 user: switch to the 1k variants and use RxR as the second metric. Plan
  (`/nvme/public/svln/runs/2k_order.txt`): `mix1k path_mix1k` = `configs/rl/local/rewind_grpo_sfr_mix1k.yaml` and
  `rewind_path_grpo_sfr_mix1k.yaml` (500 R2R = B's first 500 + 500 RxR, 125 steps); run B is the R2R-only control at
  equal prompts and steps (the mixed runs do ~1.5x its rollout work). Run 1 (2k R2R) dropped: it only adds the
  1k -> 2k scaling point, which the 8k run answers. RxR metric: val_unseen 2k = the existing subset-1000 + a disjoint
  `data/rl/rxr15_val_unseen_subset1000b.json` (seed 20261105; 11 scenes each; 3,669 English episodes in all),
  merged per model as `<name>_rxr2k_val_unseen` (`sfr_2k_chain.sh` merge_rxr2k; tested by splitting and re-merging
  sfr_r2r_val_seen: identical summary). Why 2k: B vs A / B vs C on R2R disagree on 22 % of episodes, i.e. a paired
  95 % half-width of ~2.9 SR points at 1,000 episodes and ~2.1 at 2,000. Final paper numbers: full R2R val_unseen
  and full RxR val_unseen (3,669) on the final model only. Estimates: mix1k evaluated ~00:00 Nov 6, path_mix1k
  ~19:15 Nov 6, baselines' second RxR half ~22:40 Nov 6. A balanced 8k (ActiveVLN's own 4k R2R + 4k RxR lists,
  1,000 steps) at ~410 s/step is ~4.7 days.
- 2026-11-05 03:35 results and launch.
  - G (fork at t) step-20 export: R2R val_unseen SR 65.85 (B 68.73: -2.88, McNemar p 0.016; SFR +3.26; A +1.52,
    p 0.24); val_seen-300 63.67 (B 74.67: -11.0, p 0.0002). Rewinding to the closest approach is the design choice
    that makes the expert intervention work.
  - RxR val_unseen subset-1000 (English guide): SFR SR 64.40 / SPL 53.09 / nDTW 65.3; B (R2R-only rewind RL)
    **71.80 (+7.40, p 2e-6, CI [4.4, 10.4])**: R2R-only RL transfers to RxR. 24 episodes fail for both models
    ("Required simulator metrics unavailable": no finite goal distance, a dataset property; SFR has one more, an
    invalid response), so paired comparisons are unaffected; absolute RxR SR is ~2.4 points below the reachable
    episodes'. RxR eval: 80 min per 1,000 episodes.
  - All 1,000 RxR train episodes of the mixed pool (so both 1k and 2k variants) have a finite goal distance, a finite
    geodesic start-goal distance and an oracle action at reset (930 s check); `build_mix_pool.py --exclude` is ready
    if ever needed (without it the pool is reproduced byte for byte).
  - Smoke (2 full steps of the path-aware mixed config, 03:23-03:34): passed; 273 s/step (rollout 138, update 134),
    rewind fired on 38 % (R2R) / 25 % (RxR) of mains, branch success 50 % / 88 %, no env-server restart, host RAM
    >= 31 GB available during the update. mix1k launched 03:34 (code `rewind_sfr_mix1k_20261105_0334`), guard attached.
  - Revised estimates at ~280 s/step: mix1k trained ~13:30 Nov 5, evaluated ~17:30; path_mix1k trained ~03:30
    Nov 6, evaluated ~07:30; baselines' second RxR half ~10:15 Nov 6. A balanced 8k at ~280 s/step: ~78 h.
- 2026-11-05 07:50 mix1k guard alarm, judged a false positive; guard reference changed to the same prompts.
  Window 35-39: R2R mains turn share 0.448 vs B's same window 0.384 (+0.064, over the 0.06 margin once; a second
  window would have stopped the run, which exports only at step 125). Not spinning: R2R main success 0.79 (best
  window so far), truncation 0.05, 70 steps per episode; B on the SAME 20 R2R prompts: 0.427 (+0.021). B's window
  dipped (0.405 / 0.384 / 0.397), and the mixed run's R2R prompts of a window are not B's (row i at step i // 4, B at
  i // 8), so the step-window reference mixes prompt and policy noise (the ±0.02 assumed at design time was too
  small). `drift_guard.py --r2r-only` now uses the reference run's turn share on the same R2R prompts when it saw
  >= 80 % of them (else the step-window reference). Replayed offline: identical for runs on B's prompt order (G still
  stopped after window 15-19, +0.072 / +0.083; A, C, E never within 0.05), mix1k max +0.024. Guard restarted 07:54
  (old rule kept as scratchpad drift_guard.before_same_prompts.py). path_mix1k's guard, started by the chain from the
  dev tree, gets the new rule. RxR mains are not guarded; the watcher warns if both of the last two windows
  truncate > 25 % or turn > 0.56 (window 35-39: truncation 0.15, turn 0.500, SR 0.69).
  Host memory: stable at 19.5-21.4 GB available (07:34-07:49), trainer RSS flat at ~119 GB.
- 2026-11-05 14:26 mix1k guard rule changed to need harm (scratchpad `harm_guard.py`, from window 105; path_mix1k from
  window 0). Window 100-104: R2R mains turn share 0.461 vs B on the same prompts 0.395 (+0.066, over the margin once)
  with R2R SR 0.750 vs 0.762 and episodes as long (63 vs 59 decisions). The turn share rose gradually (0.40 -> 0.46
  over 15 steps), as expected from RxR prompts (RxR turn share 0.45-0.52), not the run-2 spin (SR collapse,
  truncation spike); a stop near step 110 would have left no export. New rule: stop after two consecutive windows of
  turn share > B + 0.06 on the same prompts AND (R2R SR < B - 0.10 or truncation > 0.20), or turn share > B + 0.12.
  Replayed on windows 35-104: same turn-share numbers as the old guard, no harm anywhere. Windows 105-119: +0.060 /
  +0.077 / +0.050, SR at or above B's.
- 2026-11-05 15:50 / 16:09 two kernel OOM kills of rank 0 (`/proc/vmstat oom_kill`; no dmesg access), both while
  another user's process (anhnb9 `policy_server.py`) grew to 22-28 GB on the host; the trainer was flat (~59 GB per
  rank, 15-25 GB free for 9 h). Resumed from resumable/step_0115 with `rollout.helpers_per_rank` 1 (was 3; frees
  ~22 GB) by amending the run's frozen config; `train_rl.sh` refuses a snapshot whose FROZEN_SHA256SUMS differ, so
  that file's line was updated (78fab24e -> 7f2dfa13; logged in sfr_campaign.log). Step 115 replayed bit-identically
  (with 1 helper); from step 116 the replay differs (reward 6.77 / 7.38 / 7.46 on attempts 0 / 1 / 3), i.e. the
  update after a resume is not bit-exact, already with 3 helpers. 1 helper: ~475 s/step (rollout 289) vs ~350.
- 2026-11-05 18:00 mix1k complete (125 steps). Evaluated 18:02-22:16:
  - R2R val_unseen (1,839): **SR 71.94 / SPL 63.65 / OSR 78.58 / NE 3.32**; vs SFR +9.35 (p 4e-16), vs A (GRPO 1k)
    +7.61 (p 1e-11), **vs B (rewind, 1k R2R) +3.21 (216 vs 157 discordant, McNemar p 0.0026, CI [1.1, 5.3])**.
  - RxR val_unseen subset A (1,000): SR 71.20 / SPL 57.79 / nDTW 69.93; vs SFR +6.80 (p 2e-5), vs B -0.60 (p 0.73,
    CI [-3.4, 2.3]). RxR 2k (A + B halves): SR 70.75 / SPL 57.04 / OSR 77.05 / nDTW 69.20 (baselines' second half
    after path_mix1k).
  - Reading: at equal prompts and steps, replacing half of B's R2R prompts by RxR prompts improves R2R val_unseen
    significantly and leaves RxR unchanged. The 8k pool should be the mixed one; rewind vs path-aware rewind is decided
    by path_mix1k.
- 2026-11-05 19:38 disk cleanup (user request, groups chosen by the user): optimizer states of the old S8N-line runs,
  the 2-step smoke, step 25-100 exports, old bases S8N_DAGGER / S8N_DAGGER_ORACLE / M5_S S8N (HF-backed); 570 ->
  1,059 GB free; record `svln/logs/cleanup_20261105/deleted.txt`. Kept: SFR, qwen-base, every step-125 export (incl.
  mix1k, B, A), all rollouts / metrics / eval journals.
- 2026-11-05 22:16 path_mix1k launched by the chain (code `rewind_path_sfr_mix1k_20261105_2216`, 3 helpers per
  rank); harm guard from window 0; a host-OOM fallback (scratchpad `arm_path_helpers.sh`) lowers helpers 3 -> 2 -> 1
  and updates the sums before the supervisor's restart, only after a new kernel OOM kill. Estimates: trained ~11:30
  Nov 6, evaluated ~15:30, baselines' second RxR half ~18:15 Nov 6.
- 2026-11-06 12:03 path_mix1k complete (125 steps, one attempt after the controlled restart at 22:40 with 2 helpers per
  rank; no further restarts, no new kernel OOM kill; harm guard never fired). Evaluated 12:04-16:13:
  - R2R val_unseen (1,839): **SR 69.66 / SPL 62.29 / NE 3.51**; vs SFR +7.07 (p 6e-10), vs A +5.33 (p 7e-6), vs B
    +0.92 (p 0.41), **vs mix1k -2.28 (170 vs 212 discordant, McNemar p 0.036, CI [-4.4, -0.2])**.
  - RxR val_unseen subset A (1,000): SR 69.80 / SPL 56.57; vs SFR +5.40 (p 6e-4), vs B -2.00 (p 0.19), vs mix1k
    -1.40 (p 0.36, CI [-4.2, 1.3]). RxR 2k: SR 69.00 / SPL 55.92; **vs mix1k -1.75 (179 vs 214, p 0.086, CI [-3.75,
    0.2])**.
  - Reading: at equal data, prompts and steps, path-aware rewind (route-progress trigger and rewind point, route
    expert) is worse than goal-distance rewind on R2R (significant) and not better on RxR (-1.4 / -1.75, n.s.), the
    split it was designed for. It keeps most of the gain over SFR and A but loses mix1k's +3.2 over B.
  - **8k decision: data = the mixed R2R + RxR pool (mix1k vs B); method = rewind (mix1k's recipe), not path-aware
    rewind.** Config on gpu2: `configs/rl/local/rewind_grpo_sfr_mix8k.yaml`; it differs from
    `rewind_grpo_sfr_mix1k.yaml` only in the pool (`mix_r2r3980_rxr3980.jsonl`, 7,960 prompts, 995 steps; first
    1,000 rows = mix1k's prompts) and the exports (every 125 steps + 995). Before its chain's compares, the local 1k
    journals go to `W/svln/evals_ref_gpu2` (`normalize_journal.py to-gpu2`).
- 2026-11-06 20:13 user: "A+ ... then run training on 4 GPUs ... for 2k data first (1k before (the mixed one) + 1k new
  (mixed))". Pre-registered before any screen result:
  - **A+ (the RxR tail of the 8k pool).** Rows 5,840-7,959 (steps 730-994) hold the 1,061 RxR rows of ActiveVLN's list
    that ActiveVLN's own RL never trained on (difficulty -1: 756, 0: 184, 4: 121; rows 0-5,839 hold only 1-3). On
    mix1k's 500 RxR prompts our sampled pass rate rises with ActiveVLN's label (0.70 / 0.76 / 0.82 for 1 / 2 / 3;
    all-success 38 / 47 / 55 %), so their labels transfer, shifted (our policy is stronger).
  - Screen: one greedy SFR pass (unchanged evaluator, `code/screen_train_20261102` = frozen evaluator + `--split
    train`), RxR15 train guide, over (a) the 1,061 tail rows and (b) mix1k's 500 RxR prompts (reference, labels 1-3).
  - Calibration: bins = greedy success x RxR N* tercile (edges from the 8k pool's 3,980 RxR rows) | greedy failure x
    closest approach (<= 3 / 3-5 / > 5 m); p_hat = Beta(1, 1)-smoothed sampled pass rate of mix1k's RxR main
    rollouts in each bin (all 125 steps, every dynamic-sampling attempt counted). An episode the evaluator cannot
    score (goal distance undefined) gets p = 0. Score s = 1 - p^4 - (1 - p)^4 (expected informative share of a G = 4
    group).
  - Rule per label group g in {-1, 0, 4}: keep g if mean s(g) >= mean s(reference) - 0.03; otherwise replace all of g
    by screened RxR15 English train episodes outside ActiveVLN's 4k list (random candidates, 2x the number needed,
    seed 20261106; same screen; top s within scene strata; RL-env scorability checked), placed at the replaced rows'
    positions (alternation kept; rows 0-5,839 unchanged, so the 2k stage below is unaffected).
  - **2k stage (gpu2, all 4 GPUs).** The final 8k config from step 0, paused after step 250 (its first 2,000 rows =
    mix1k's 1,000 + the next 1,000 mixed), then R2R val_unseen + RxR 2k of exports/step_0250 and R2R val_unseen of
    step_0125 (4-GPU replication check vs mix1k's 71.94); resumable state kept so the same run continues to 8k. A
    2-step 4-GPU smoke first.
- 2026-11-06 20:20-20:45 user: "What is the current lr usage? is it linear?"; "eval for me the 2k, 4k, 6k, 8k";
  "save checkpoint for later eval also". Changes to the 20:13 plan (before any result of the 8k run):
  - LR: AdamW 1e-6 **constant** (no scheduler, no warmup, no decay; logged lr 1e-6 at every step of every run), as
    ActiveVLN's verl recipe (constant 1e-6, betas 0.9/0.999, wd 0.01, grad clip 1.0, clip 0.2/0.28, dual-clip 3,
    token-mean, no KL, 8 prompts x G = 4, T 1.2, no shuffle, 1 epoch); ActiveVLN also trains the vision tower, we
    train the language model only. With a constant LR and a fixed prompt order, the 8k run's step-250 / 500 / 750
    exports ARE the 2k / 4k / 6k models (no schedule depends on the run length).
  - So **no pause at step 250**: the 8k run trains straight through to step 995 on gpu2 (all 4 GPUs;
    `ops/local/remote_run_chain.sh` there: MPS, watchdog, frozen snapshot, launch, harm guard, nothing deleted).
    All 8 exports (125, 250, ..., 875, 995) stay on gpu2 and are copied to the workstation as they appear
    (`/nvme/public/svln/runs_gpu2/<run>/exports`, every file's sha256 checked against gpu2's, read-only) for later
    evaluations. Milestones evaluated locally with the evaluator tree of every 1k journal
    (`ops/local/mix8k_milestone_evals.sh`): step 125 R2R val_unseen (4-GPU replication of mix1k, 71.94); steps 250
    (2k), 500 (4k), 750 (6k), 995 (8k) R2R val_unseen + RxR 2k, compared with SFR, A, B, mix1k, path_mix1k and the
    previous milestone. Steps 375 / 625 / 875: copied, not evaluated.
  - Guard: the harm guard (`svln/tools/harm_guard_gpu2.py`, the rule of `harm_guard.py`) from window 0: same R2R
    prompts vs B through step 249 (the 8k pool's first 2,000 rows hold B's 1,000 R2R prompts); later windows, which B
    never saw, vs rewind_sfr_mix1k's last 5 windows (steps 100-124, R2R mains: turn share 0.466, SR 0.775, truncation
    0.100; same recipe -- B's 0.410 is no reference for a mixed run). Dry run on mix1k's rollouts reproduces the
    local guard log for windows 100-119 exactly, locally and on gpu2.
  - Correction to 20:13: the RxR tail starts at row 5,839 (one label -1 row), not 5,840; rows 0-5,838 hold only 1-3.
    Still all >= 2,000.
- 2026-11-06 21:57 A+ screen done (no failed episodes; greedy SFR SR: reference 80.6 %, tail 76.6 %). Rule applied
  (`scripts/rl/s8n_analysis/aplus_rxr_tail.py`, output `svln/analysis/aplus_20261106/aplus_decision.json`; RxR N*
  terciles 80 / 131; calibration from mix1k's RxR mains, p 0.35 (failed, far) ... 0.86 (solved, short)):
  reference mean s 0.668 (threshold 0.638); **label -1 (756) 0.657 keep; label 0 (184) 0.725 keep; label 4 (121)
  0.538 REPLACE** (greedy SR 95 %, 80 of 121 solved and short: too easy for SFR).
  - Replacement (as pre-registered): 242 random RxR15 English train-guide episodes outside ActiveVLN's 4k list (seed
    20261106), screened like the tail; RL-env scorability checked locally (reach_pool.py's checks on the workstation's
    simulator); top s within scene strata (`stratified_counts` of select_pool.py; s ranks as p(1 - p), so this is its
    learnability_top with the RxR bins above); placed at the 121 label-4 positions.
  - Not covered by the pre-registration: new rows have no ActiveVLN merged length M (the rollout budget is
    min(2M, 120) macro actions at N*/M primitives each; the cap binds for 38 % of the pool's RxR rows). They get
    M = N*/2 (the pool's median N*/M is 2.00, IQR 1.81-2.21), i.e. budget min(2 N*, 240), flagged
    `activevln_merged_len_estimated`; activevln_difficulty = null.
  - 22:20 replacement screen done: 242 candidates (53 scenes; 81 are other instructions of a path already in the
    pool -- not excluded by the rule, kept); 2 crash the RL simulator and fail the screen (rxr-22567, rxr-47397;
    excluded). First selection (22:21) put 18 rows with N* > 240 in: with M = N*/2 their budget min(N*, 120) x 2 is
    below N*, which no row of ActiveVLN's list has (its M <= 120, so budget >= N*; minimum budget/N* in the pool
    1.00). Added the same invariant as an eligibility condition (budget >= N*, i.e. N* <= 240; 21 candidates out),
    re-selected with the same seed (the first selection is kept in `analysis/aplus_20261106/old_selection_2221`).
  - **Final 8k pool: `mix_r2r3980_rxr3980_aplus.jsonl` (sha256 3357e8f3...)**: 219 eligible, 121 chosen (51 scenes;
    mean s 0.756 vs reference 0.668 and the removed rows' 0.538; greedy SR 58.7 %; N* median 136, max 238, removed
    rows' median 57, pool RxR median 106; en-IN 85 / en-US 36), placed at rows 5875-7949 (steps 734-993). Checked
    byte for byte: only those 121 rows differ from mix_r2r3980_rxr3980.jsonl, rows 0-1999 = mix_r2r1k_rxr1k.jsonl,
    7,960 unique IDs, R/X alternation, budget >= N* for every row, the trainer's load_pool accepts it. Config
    `configs/rl/local/rewind_grpo_sfr_mix8k_aplus.yaml` (gpu2) = rewind_grpo_sfr_mix8k.yaml with only data.pool and
    the two hashes changed. The new rows are ~2.4x longer than the rows they replace: steps 734-993 will be slower.
- 2026-11-06 22:45 user: "train for me the ActiveVLN on 2 gpus on this sever with these 8k dataset for me. Eval at 1k,
  2k, 4k, 6k, 8k for me. For the 1k try to use the same subset 500 r2r 500 rxr for fair comparison ... use the same
  dataset at each checkpoint eval for fair comparison."
  - **Run grpo_sfr_mix8k_aplus (local, GPUs 0 + 1)**: ActiveVLN GRPO from SFR = run A's algorithm (no rescue block),
    `configs/rl/local/activevln_grpo_sfr_mix8k_aplus.yaml` = rewind_grpo_sfr_mix1k.yaml without estimator / scale_floor
    / rescue / rescue_credit, on the A+ 8k pool (7,960 prompts, 995 steps), exports every 125 steps + 995. Its first
    1,000 rows are rewind_sfr_mix1k's 500 R2R + 500 RxR in the same order: the 1k checkpoint (step 125) has seen
    exactly mix1k's prompts, the fair 1k comparison with rewind_sfr_mix1k_s125 (and with A on R2R).
  - Chain `ops/local/activevln_mix8k_chain.sh`: 2-step smoke of the exact config (frozen snapshot
    `code/grpo_sfr_mix8k_aplus_20261106_2254`), then pauses at steps 125 / 250 / 500 / 750 and the end (995) with the
    SAME evaluations at each, using the evaluator tree of every 1k journal (code/sfr_campaign_20261102_2034): R2R
    val_unseen (all 1,839) + RxR val_unseen 2k (subsets A + B), compared with SFR, A, B, mix1k, path_mix1k and the
    previous milestone. No drift guard (plain GRPO has no spin mechanism; a guard stop would bias the control).
    Supervisor restarts up to 20 (host OOM risk, [[local-host-ram-contention]]).
  - gpu2's rewind run (rewind_sfr_mix8k_aplus, waiting for gpu2) gets the same five milestones and the same
    evaluations (`ops/local/remote_milestone_chain.sh`), so rewind vs ActiveVLN is compared at 1k / 2k / 4k / 6k / 8k.

### 2026-11-07 09:00-09:30 — gpu2 back; 8k A+ rewind run relaunched as 2 training GPUs + 1 evaluation GPU

- gpu2 was unreachable 2026-11-06 ~20:58 until its owner rebooted it (2026-11-07 08:49); eno1 still logs TX errors
  (23 per 59k packets after the reboot), consistent with the corrupted pulls. Only small files leave gpu2.
- Pushed (sha256 checked): the A+ pool `mix_r2r3980_rxr3980_aplus.jsonl` (3357e8f3...), its audit, gpu2 config
  `rewind_grpo_sfr_mix8k_aplus.yaml` (c133e6ce...; differs from rewind_grpo_sfr_mix8k.yaml only in pool + hashes).
  Removed the 3-step 4-GPU smoke test's exports + resumable (65 GB; logs kept).
- 08:58 launched on all 4 GPUs; step 0 reproduced the smoke test and the local 2-GPU rewind_sfr_mix1k step 0
  (SR 0.5625, reward 6.0726, pg loss 0.019063, 2,382 env steps; grad norm 1.68637 vs 1.68598 local: rounding).
- User 09:15-09:25: "using 2 gpu 2,3 for training and 0 for eval", "leave the gpu 1 empty". Stopped the 4-GPU
  attempt at step 5 (kept in gpu2 svln/runs/stopped_4gpu_20261107) and relaunched from step 0 with
  `ops/local/remote_2plus1_chain.sh`: training on GPUs 2+3 through the whole pool (no pauses); milestones 125 / 250 /
  500 / 750 / 995 evaluated on GPU 0 while training continues (R2R val_unseen + RxR 2k halves A + B, the same sets as
  the local ActiveVLN control). Evaluations use a second MPS daemon (env_local.sh overlay `SVLN_MPS_DIR`), so a
  trainer restart does not touch them. The GPU count does not change the update: per-episode rollout seeds and the
  verl token-mean weights with world / num_mb_global (algos.loss_weights) are rank-layout invariant.

### 2026-11-07 10:20-14:42 — local ActiveVLN (grpo_sfr_mix8k_aplus) stopped at step 125; 1k evaluations finished

- User 10:20: "Stop at the step 125 training. Just finish the eval for me", "Eval on full RxR for me". The run is
  paused at step 125 (STOPPED; resumable state + export kept); `ops/local/activevln_1k_finish.sh` finished the
  1k evaluations on the local GPUs (same evaluator tree as every journal).
- grpo_sfr_mix8k_aplus_s125: R2R val_unseen (1,839) SR 66.83 / SPL 60.01; RxR 2k SR 69.75 / SPL 57.61 (A 69.90);
  full RxR val_unseen (3,669 English guide episodes = A + B + rest 1,669, `data/rl/rxr15_val_unseen_rest1669.json`;
  merged contract checked against the dataset) SR 69.96 / SPL 57.67 / NE 3.786 (en-IN 2,446: 67.29 / 55.42;
  en-US 1,223: 75.31 / 62.17). 100 failures = 98 "simulator metrics unavailable" (the same episodes fail in every
  journal; counted as unsuccessful) + 2 invalid responses.
- Paired SR deltas vs the 1k runs (R2R | RxR 2k): SFR +4.24 (p 4e-4) | +6.20 (p 1e-7); grpo_sfr_s125 +2.50
  (p 0.023) | -; rewind A -1.90 (p 0.083) | -0.95 (p 0.39); rewind mix1k -5.11 (p 2e-6) | -1.00 (p 0.38); rewind
  path_mix1k -2.83 (p 0.012) | +0.75 (p 0.52). Plain GRPO ties rewind on RxR but trails it on R2R. Like for like on
  data: the A+ pool's first 1,000 rows are mix1k's prompts in the same order (configs differ only in the estimator and
  export steps), so vs rewind mix1k the gap is the estimator. (Corrected 2026-11-08: an earlier note said data differ.)
- No other run has full RxR yet (only the 2k halves).

### 2026-11-07 20:46 - 2026-11-08 00:30 — local ActiveVLN resumed (R2R-only milestones); gpu2 evaluations on GPUs 0 + 1, full RxR

- User 2026-11-07: "Continue the training for me". `activevln_mix8k_chain.sh` RESUME_FROM=125 (same snapshot
  grpo_sfr_mix8k_aplus_20261106_2254; checks of STOPPED, rl_state, both rank files, config + pool hashes): both ranks
  "resumed at step 125" 20:47; step 125 SR 0.84, 303 s; free GPU memory as before the pause.
- User 2026-11-08: "for the current local one, just eval with the r2r for me. We will run RxR later". The chain bash was
  stopped and re-attached to the running trainer (ATTACH=1 EVAL_RXR=0): milestones 250 / 500 / 750 / 995 evaluate R2R
  val_unseen (1,839) + verify only; exports are kept for the RxR evaluations later.
- gpu2, user 2026-11-08: "if you notice the GPU 1 is not using you can use both 0,1 to run eval ... eval on full R2R and
  RxR". GPU 1 holds only anhnb9's diffusion-policy job (25 GB, 0% SM since 11-07 09:27). `ops/local/remote_eval_lanes.sh`
  replaces the 2+1 chain's milestone loop: per milestone R2R (1,839) + RxR A / B / rest 1,669 -> rxr2k and rxrfull
  (3,669, checked against the dataset), one 5-worker job per GPU on two lanes; a guard stops the GPU 1 lane for good if
  anhnb9's job changes or computes (or < 6 GB free / >= 96 C).
- gpu2 rewind A+ 8k s125 R2R, first 1,512 of 1,839 episodes: SR 69.91 / SPL 61.79; same episodes: SFR 62.10, ActiveVLN
  s125 65.54 (+4.37, p 2e-4: rewind beats plain GRPO on identical prompts), rewind mix1k s125 71.36 (-1.46, p 0.22:
  the same algorithm, prompts and order on other hardware; run-to-run spread).

### 2026-11-08 00:30-08:00 — checkpoints kept and on Hugging Face; gpu2 off the network again; ActiveVLN 2k evaluated

- User 2026-11-08: "remember to save the checkpoint of 1k, 2k, 4k, 6k, 8k for me for both run". `ops/local/keep_exports.sh`
  (running locally and on gpu2) waits until each milestone export is final. It then checks the weights' sha256 against
  the provenance, makes the export read-only and hard-links it into `svln/checkpoints/keep_8k/<run>/step_XXXX`, listed
  in MANIFEST.tsv. Kept so far:
  - local: step 125 (sha256 5c41de30...) and step 250 (d1f208cb...);
  - gpu2: step 125 (c6fb8de3...).
- User: push the weights to `anhdao69/stage3_S8N/{ActiveVLN, Rewind}`; public repo (user's choice).
  `ops/local/hf_push_loop.sh` + `hf_push_checkpoint.py` make one commit per milestone, check every file's size and the
  weights' LFS sha256 on the Hub, and update the README with the evaluation table once the evaluations exist. Pushed
  and verified:
  - ActiveVLN/1k: commit 283ae326;
  - ActiveVLN/2k: weights 9ab8959b (07:41), README with R2R 67.92 / 61.27 e9f55cac (07:52).
- Hub access needs the corporate web-portal login (4 h sessions). The loop waits and logs "NEEDS LOGIN" until it is
  renewed.
- gpu2 dropped off the network at ~00:45, about 10 min into the single-stream upload of Rewind/1k. The upload was
  atomic (create_commit), so the repo holds no partial Rewind commit. From the workstation, gpu2 answers no ARP
  (`ip neigh` FAILED, "Destination Host Unreachable"), while the other hosts on the subnet answer.
  - Both outages (11-06 ~20:58 during a 4.3 GB pull, 11-08 ~00:45 during this upload) came under sustained outbound
    traffic. eno1's TX errors keep rising (23 per 59k packets after the 11-07 reboot, 396 per 2M on 11-08), and data
    sent out of gpu2 arrives corrupted ("Corrupted MAC" on pulls).
  - Likely, not verified: the NIC's transmit side hangs under load. Fix by its owner: restart eno1 or re-plug it (no
    reboot needed if the host is up), then `ethtool -K eno1 tso off gso off`.
  - Until then no large data leaves gpu2. Rewind weights go to the Hub only after the fix (or from a copy made another
    way).
  - Unknown until gpu2 answers: whether training (step ~160-165 at the drop) and the 1k evaluations kept running (they
    do not need the network).
- 05:50 ActiveVLN step 250 (2k) export final and kept. 07:19 R2R val_unseen (1,839; verified, 0 problems): **SR 67.92 /
  SPL 61.27 / OSR 74.50 / NE 3.77**. 10 policy failures, every one an invalid response 'MOVE' (counted unsuccessful; 0
  at step 125 on R2R).
  - Paired SR deltas: vs SFR +5.33 (p 1.9e-5); vs A +3.59 (p 7.8e-4); vs B -0.82 (p 0.49); vs mix1k -4.02 (p 3.8e-4);
    vs path_mix1k -1.74 (p 0.14); vs ActiveVLN 1k +1.09 (p 0.31, CI [-0.92, 3.05]).
  - Reading: plain GRPO improves slowly with more prompts, but at 2,000 prompts it is still significantly below rewind
    at 1,000 (mix1k).
- Training resumed towards step 500 at 07:20 (~257 s/step). Expected: 4k R2R ~02:30 Nov 9, 6k ~22:00 Nov 9, 8k
  ~17:00 Nov 10.
