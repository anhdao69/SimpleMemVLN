# ActiveVLN GRPO baseline on SimpleMemVLN — technical report

Status: **COMPLETE for the 1K-subset ActiveVLN GRPO baseline** (training 125/125, full R2R val_unseen evaluation, §8). Run log: §7.1.

> **Scope decision (user, 2026-10-10 ~12:00 UTC).** The full 4K run was stopped at step 22 and replaced
> by a run **from scratch on a 1k subset** = the first 1,000 rows of ActiveVLN's fixed-order 4K list
> (exactly the instructions of ActiveVLN's first 125 steps; 60 of 61 scans; N\* mean 58.9 vs 58.2;
> SHA-256 `ba062de3…`), with maximum training speed requested, and **R2R only** (the RxR15 baseline
> evaluation is not needed now). Config: `configs/rl/activevln_grpo_r2r1k.yaml`; everything else is
> identical to the 4K configuration. The final step is 125; the result is therefore a 1k-instruction,
> one-epoch ActiveVLN-GRPO baseline, not a 500-step 4K reproduction.
Branch: `RL_baseline`. Specification: `plans/RL/activevln_baseline_spec.md`.
Reference analysis of the official code: `plans/RL/activevln_reference_analysis.md`.

---

## 1. Codebase audit (Phase 1)

### 1.1 Branches

| Branch | Relation | Content relevant to RL |
|---|---|---|
| `RL_baseline` (new) | created from `eval-window8-text` @ `f6b8108` (= `origin/streaming_text_dual`) | all RL work |
| `origin/streaming_text_dual` | identical to the start point | Window8 text + DualLane code, evaluator, publication |
| `origin/streaming_logits_dual` | 2 ahead / 5 behind | candidate-logit DualLane variants (not used) |
| `origin/streaming_logits`, `origin/streaming` | older | earlier policy families (not used) |
| `main` | 7 ahead / 39 behind | old trunk |

The working tree carried an **uncommitted** evaluator change from `eval-window8-text`
(`--model-startup-timeout`, needed because the RAID-backed `/data` is slow to read). The running
baseline evaluation records exactly this file (`eval/habitat_r2r.py` SHA-256 `f3e9ef88…`), so the RL
evaluation uses the same bytes. Every other `src/qwen_vl/**/*.py` file is byte-identical to the
hashes in the baseline `evaluation_contract.json`; the only additions are the new `rl/` modules,
which the evaluator never imports.

### 1.2 Policy under study (starting checkpoint)

`/data/anh/checkpoints/simplemem/joint-window8-text-b/epoch-2` = HF `anhdao69/SimpleMemVLN-R2R-RxR15deg-Window8-B`
@ `0a111c78…`, weights SHA-256 `4391b1f6…dbcf` (verified by the trainer at start).

- Backbone Qwen3.5-4B @ `851bf6e8…` (hybrid: 24 Gated-DeltaNet linear-attention layers + 8 full-attention
  layers), vision tower (0.33 B params) frozen incl. merger; language model 4.21 B params trainable
  (tied LM head).
- Serializer `vln_append_only_chat_v1`: system prompt + instruction prefix; per step one 640×480 RGB
  frame (300 visual tokens) in a user turn, then an assistant turn with **one primitive action as
  text** (`MOVE_FORWARD`/`TURN_LEFT`/`TURN_RIGHT`/`STOP` + `<|im_end|>`, 2–3 tokens) and a `\n`
  separator. Generated actions are appended (action history).
- Memory: **Window8** — the instruction prefix plus the 8 most recent observation/action groups are kept
  in full-attention KV; older groups are evicted, while the GDN recurrent state persists for the whole
  episode. The training forward (`nav_model.hidden` with a `step_plan`) reproduces the same windowed
  attention over the complete episode, with full BPTT through the recurrent state.
- Inference (`stream/session.py`): greedy full-vocabulary decoding, ≤16 tokens, invalid text/special
  token ⇒ failed episode.
- SFT recipe: 2 epochs on 30,815 R2R + RxR15 episodes (incl. all 10,819 R2R train episodes), LR 5e-6,
  DeepSpeed ZeRO-2 + CPU optimizer offload on 2×H100.

### 1.3 Evaluation protocol of record

`qwen_vl.eval.habitat_r2r`, Habitat-Lab/Sim 0.2.4, R2R-VLNCE v1-3 `val_unseen` (1,839 episodes,
SHA-256 `1767a407…`), config `docs/evaluation_setup/vln_r2r.yaml` (SHA-256 `dd20dd0e…`): RGB 640×480,
HFOV 79°, 0.25 m forward, 15° turns, success = STOP with geodesic distance < 3.0 m, 500-action cap with
forced STOP, seed 42, greedy decoding, failed episodes count as 0. Baseline job:
`/data/anh/simplemem-eval/results/r2r-full/epoch-2` (PIDs 4038161 + 4038174, GPU 0), followed by an
RxR15 run in the same campaign. These processes were never touched.

### 1.4 Host resources (shared machine)

2× RTX PRO 6000 Blackwell Max-Q (97.9 GB, 300 W) shared with other users' jobs (≈23–28 GB in use per
GPU before RL), 48 CPU cores, 62 GB RAM (≈27 GB available, swap full), `/data` 5 TB free (RAID resync in
progress → slow reads). No Slurm.

### 1.5 RL integration points

| Need | Where | Decision |
|---|---|---|
| Stochastic policy | `stream/session.py` greedy loop | `rl/sampling.py::SamplingSession.sample_step` – identical loop, tempered full-vocabulary sampling, same validity rules |
| Log-probs for training | `nav_model.hidden` + tied `lm_head` | `rl/episode.py::response_log_probs` – full-episode Window8 forward, logits/T at response positions |
| Episode tensors from rollouts | `EpisodeSerializer.encode_episode` (reads PNGs, canonical action ids) | `rl/episode.py::encode_rollout` – in-memory frames and the **sampled** tokens; bit-identical to `encode_episode` on canonical actions (test) |
| Simulator | evaluator config/assertions | `rl/env_server.py` (simulator Python) + `rl/env_client.py`; same config and assertions, split `train` |
| Optimizer / distribution | SFT DeepSpeed ZeRO-2 | `rl/train_grpo.py`, ZeRO-2 on GPU (no CPU offload: host RAM too small) |
| Export | `EpisodeTrainer._save` format | same format; `navigation.json` byte-identical to the base (asserted) → loads unchanged in the evaluator |

---

## 2. ActiveVLN reference (Phase 2)

Official repository `arvillion/ActiveVLN` @ **`3a0c63b00e4f42c828cc74c3554afce17641da60`** (2026-02-11),
cloned to `/data/anh/code/ActiveVLN`. Full analysis with file/line citations:
`plans/RL/activevln_reference_analysis.md`. Summary of the RL stage:

| Component | ActiveVLN |
|---|---|
| Policy | Qwen2.5-VL-3B (SFT `Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn`), multi-turn chat, all past frames in context, ≤3 comma-separated macro actions per turn (forward 25/50/75 cm, turn 15/30/45°, stop) |
| Simulator | Habitat 0.1.7, 640×480, HFOV 90°, 0.25 m / 15° primitives, success 3 m |
| Data | `data/r2r_4000_train.parquet` (4,000 R2R-VLNCE train episode ids), fixed order, 1 epoch = 500 steps |
| Rollout | 8 prompts × n = 4, temperature 1.2, top-p 1.0 |
| "Dynamic early stopping" | per-episode step budget `min(2·len(gt_actions), 120)` macro actions; exceeding it ends the episode with reward 0 |
| Dynamic sampling | an all-fail group is re-rolled as a whole, ≤2 extra rounds; the last round replaces it |
| Reward | `15·(1 − d/3)` if STOP within 3 m, else 0 (format error / budget ⇒ 0) |
| Advantage | GRPO: `(R − mean)/(std_unbiased + 1e-6)` per group, broadcast to policy tokens |
| Loss | PPO dual-clip (0.2 / 0.28, c = 3), KL 0, entropy 0, no reference model; `token-mean` per micro-batch of 2 trajectories, ÷ 4 accumulation, averaged over 4 GPUs; one mini-batch, one epoch ⇒ one on-policy update per step |
| Optimizer | AdamW 1e-6 constant, betas (0.9, 0.999), wd 0.01, grad clip 1.0, all parameters incl. vision |
| Released checkpoint | step 350 (`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`) |
| Published R2R val-unseen | SFT 38.5 SR / 33.7 SPL / 6.83 NE / 42.7 OSR → SFT+RL **50.1 SR / 43.7 SPL / 5.31 NE / 58.3 OSR** |
| Evaluation | vLLM sampling T = 0.2, top-p 0.8, ≤120 model calls, forced stop at 400 simulator steps or 25 steps without distance change, lenient parsing |

### 2.1 Dataset audit (G1)

`scripts/rl/audit_activevln_pool.py` → `data/rl/activevln_r2r4k.jsonl` + `data/rl/activevln_r2r4k_audit.json`.

- Parquet SHA-256 `f83cd5f6…d94a`; 4,000 rows, 4,000 unique ids, all `split=train`, `data_source=r2r`.
- **4,000 / 4,000** ids map to R2R_VLNCE_v1-3 `train`; 61 scans.
- ActiveVLN's merged GT strings expand to **exactly** our 15° primitive GT (`train_gt.json.gz`) for all 4,000
  episodes (unit confirmation: 25 cm / 15° primitives on both sides).
- N\* (our expert length incl. STOP): min 21 / median 56 / mean 58.2 / max 171.
- Separation: vs `val_unseen` — 0 shared scans, 0 shared trajectories (one generic instruction string,
  "Go down the stairs and stop.", occurs in a different scan); vs `val_seen` — 0 shared trajectories
  (scans are shared by definition of "seen").
- Pool file SHA-256 `a1f93b80…d4d1` (order = parquet order), asserted by the trainer.
- The 4K list is a subset of SimpleMem's SFT data (all R2R train episodes were SFT data), as it is for
  ActiveVLN's SFT model (R2R + EnvDrop SFT).

### 2.2 Environment equivalence (do not compare SR across protocols)

| Item | ActiveVLN | SimpleMem (training and evaluation here) |
|---|---|---|
| Habitat | 0.1.7 | 0.2.4 |
| HFOV | 90° | 79° |
| Resolution, step, turn, success radius | 640×480, 0.25 m, 15°, 3 m | same |
| Decoding at eval | sampling T=0.2/top-p 0.8, lenient parse | greedy, strict parse |
| Episode cap at eval | 400 sim steps / 120 calls / 25-step stagnation stop | 500 actions, forced STOP |

The two protocols are **not equivalent**; published ActiveVLN numbers are contextual only. All
SimpleMem comparisons below use the identical SimpleMem protocol of §1.3.

### 2.3 Scenes

The evaluation setup held only the 11 `val_unseen` scans. The 61 train scans were downloaded with
`scripts/rl/download_train_scenes.py` (record `/data/anh/rl_baseline/records/train-scene-mirror-verified.json`):
47 from the project mirror `anhdao69/mp3d-scene-datasets@d8c76863`, 14 missing ones from
`bingCS/mp3d_dataset@3fdee1c7`, accepted only after verifying that **all 145 glb/navmesh files the two
mirrors share are byte-identical** (LFS SHA-256). Every file is SHA-256-verified on disk. A GT-action
replay on train episodes (incl. a fallback-mirror scan) gives Habitat success with SPL 0.98–1.0.

---

## 3. Port decisions (faithful vs unavoidable differences)

| Aspect | ActiveVLN | This port | Why |
|---|---|---|---|
| Data, order, steps | 4K list, fixed order, 8 prompts/step, 500 steps | identical | — |
| Group size, temperature, top-p | 4, 1.2, 1.0 | identical (full-vocabulary token sampling) | — |
| Reward | 15·(1−d/3) on success | identical; success = Habitat `success` (strict d<3, asserted) | — |
| Format errors | episode ends, R=0 | identical: invalid text, foreign special token, or 16 tokens without `<|im_end|>` (the evaluator's rules) | — |
| Step budget | min(2·M, 120) macro actions, M = merged GT length | **ceil(min(2·M,120)·N\*/M)** primitive steps (42–221, mean 115) | one primitive per decision; exact unit conversion with each episode's own primitives-per-macro ratio |
| Dynamic sampling | ≤2 re-roll rounds for all-fail groups | identical | — |
| Advantage | GRPO, unbiased std, ε=1e-6 | identical | — |
| Policy loss | dual-clip PPO, ratio vs recomputed old log-probs | identical formula; old log-prob = detached current (one on-policy update ⇒ ratio ≡ 1) | verl recomputes old log-probs with the same weights before the single update |
| Aggregation | token-mean per micro-batch of 2 trajectories, mean over 16 micro-batches | **identical weights** (`algos.loss_weights`, exact-gradient test); micro-batches pair the trajectories of a group in order (verl pairs after length balancing) | batch=1 episodes in our model; pairing order is an implementation detail |
| Zero-advantage trajectories | forwarded, contribute 0 | not forwarded; counted in denominators | exact (KL = entropy = 0), saves compute |
| Optimizer | AdamW 1e-6 const, (0.9,0.999), wd 0.01 on all params, clip 1.0 | identical on trainable params | — |
| Trainable set | all incl. vision | language model only; vision tower + merger frozen | preserves the SFT architecture/recipe |
| Context | full multi-turn history | SimpleMem Window8 + GDN memory, training forward = streaming context | inherent to SimpleMem |
| Observations | 1 frame per turn, ~88 tokens | 1 frame per primitive step, 300 tokens | inherent to SimpleMem |
| Simulator | Habitat 0.1.7, HFOV 90 | Habitat 0.2.4, HFOV 79 (SimpleMem's SFT/eval contract) | train on the policy's own observation contract |
| Hardware | 4×L40S, FSDP2 + offload | 2× RTX PRO 6000 (shared), DeepSpeed ZeRO-2 | available resources |
| Non-finite gradient | step skipped | run aborts (resume from last checkpoint) | DeepSpeed bf16 ZeRO-2 has no skip hook; never observed |

---

## 4. Implementation

| File | Purpose |
|---|---|
| `src/qwen_vl/rl/env_server.py`, `env_client.py` | Habitat 0.2.4 R2R train-split JSON-lines server (evaluator config + assertions), lossless raw RGB |
| `src/qwen_vl/rl/sampling.py` | `SamplingSession` (tempered sampling twin of the greedy decoder) |
| `src/qwen_vl/rl/rollout.py` | episode loop, budget, reward, records |
| `src/qwen_vl/rl/episode.py` | rollout → training tensors; tempered log-probs from the training forward |
| `src/qwen_vl/rl/algos.py` | reward, budget conversion, GRPO advantage, dual-clip loss, verl micro-batch weights |
| `src/qwen_vl/rl/train_grpo.py` | colocated rollout + DeepSpeed ZeRO-2 training loop, dynamic sampling, metrics, exports, resumable checkpoints |
| `configs/rl/activevln_grpo_r2r4k.yaml` | the baseline configuration (every value annotated) |
| `scripts/rl/audit_activevln_pool.py`, `download_train_scenes.py`, `check_ds_accumulation.py`, `eval_export.sh` | data audit, scenes, distributed-gradient gate, evaluation launcher |
| `tests/rl/` | unit tests |

| `src/qwen_vl/rl/rollout_pool.py` | multi-process rollout helpers that read the learner's live weights through CUDA IPC |
| `src/qwen_vl/rl/checkpoint.py` | exact per-rank ZeRO-2 save/restore (memory-mapped) |
| `scripts/rl/launch_unit.sh`, `supervise.sh`, `memwatch.sh` | isolated `systemd-run --user` units, auto-resume supervisor, memory log |

---

## 5. Validation (Phase 3)

| Gate | Evidence | Result |
|---|---|---|
| Environment / simulator | `rl/env_server.py` on R2R `train`: GT-action replay of 3 episodes (incl. a fallback-mirror scan) → Habitat success, SPL 0.98–1.0; renders 480×640 on the requested GPU | pass |
| Rollout encoding | `tests/rl/test_rollout_encoding.py`: on canonical actions, `encode_rollout` equals `EpisodeSerializer.encode_episode` tensor-for-tensor (ids, pixels, grids, step plan, targets); unterminated final responses handled | pass |
| Sampling | `tests/rl/test_sampling.py`: 20k draws match softmax(ℓ/T) within 1%; returned log-prob/entropy exact; seed-reproducible | pass |
| Reward / budget / GRPO / loss | `tests/rl/test_algos.py`: reward edge cases (strict d<3), unbiased-std GRPO incl. zero-variance groups, budget conversion, verl dual-clip values, on-policy gradient = −A, **micro-batch token-mean weights reproduce verl's aggregation exactly** | pass |
| Distributed accumulation (G6b) | `scripts/rl/check_ds_accumulation.py` (2 ranks, unequal items, zero-loss padding, manual boundaries) | max rel. grad error 9.8e-8; exactly 1 optimizer step |
| Resume (exactness) | `scripts/rl/check_resume.py` (2 ranks, bf16 ZeRO-2): per-rank save → fresh engine → load → step | bitwise-identical params and fp32 masters |
| Rollout helpers | `scripts/rl/check_rollout_pool.py`: IPC helper vs learner, same seed → identical actions and log-probs; after an in-place weight change the helper samples from the new policy | pass |
| Log-prob parity rollout ↔ trainer | logged every step (`logp_mismatch_*`): streaming cached forward vs full-episode training forward | mean ≈0.004 nats/token, max ≈0.2 |
| End-to-end smoke | 2-rank runs (`/data/anh/rl_baseline/runs/smoke1`, `smoke2`): finite loss, grad-norm 2–4 before clipping, LR 1e-6, one update/step, exports written | pass |
| Parameter update | export vs base (memory-mapped diff): 0/297 vision tensors changed; LM tensors changed with relative change ≈1e-5 after 2 updates | pass |
| Export loads in the evaluator | 2-episode run of the unchanged evaluator on a smoke export | 2/2 complete, 0 invalid responses |
| Determinism | full-scale step 0 re-run after a restart: identical reward (6.803) and success (0.719) | pass |
| Resume (real model) | main run stopped after the step-3 checkpoint; the supervisor relaunched it with `--resume`; per-rank state memory-mapped and restored in ≈4.5 min; DeepSpeed step counter asserted equal to the RL step | pass |
| Activation offload | step 0 re-run with pageable host offload of layer inputs (≥16,384 tokens): identical grad-norm (1.94) and metrics | pass (gradients unchanged) |
| Resources | per rank: ≈43–48 GiB peak (train), 16 GiB (rollout); host RAM: trainer ≈3 GB anon + helpers ≈2.2 GB each; checkpoint 25 GB/rank | fits beside other users' jobs |

### 5.1 Engineering incidents (and fixes)

1. **Host OOM killed the tmux session.** Two ranks reading DeepSpeed's ZeRO-2 checkpoint (and, at
   init, ZeRO-2 staging all bf16 weights through host RAM) exceeded the 62 GB shared host. The kernel
   OOM killer then made systemd stop the whole tmux pane scope (Claude included). Fixes: every GPU job
   runs in its own `systemd-run --user` unit (MemoryHigh 36G / MemoryMax 50G, no swap; user lingering
   enabled); base weights load memory-mapped; DeepSpeed's resume replaced by an exact per-rank
   save/restore; the trainable parameters are split into 8 AdamW groups with identical
   hyperparameters so ZeRO-2 stages one ~0.5 B-parameter group at a time.
2. **Rollouts were launch-bound.** Batch-1 streaming decoding issues hundreds of small CUDA/Triton
   kernels per action from Python (~0.14–0.27 s/action on the shared GPUs). Threads serialize on the
   GIL (the trainer sat at 100 % of one core). Fix: 3 helper processes per GPU whose parameters are
   CUDA-IPC views of the learner's own tensors (exactness gate above). CUDA IPC needs the default
   allocator (`expandable_segments` uses `pidfd_getfd`, which this host forbids). Rollout phase:
   ≈800 s → ≈165–245 s per step.
3. **CUDA OOM on the shared GPUs.** Other users' jobs grew from ≈23 GB to ≈31 GB per GPU while our
   learner peaks at ≈55 GB (bf16 weights 9.7 + fp32 master 8.4 + Adam 16.8 + fp32 grads 8.4 GB +
   activations). Fixes: after the IPC-shared weights are allocated the learner switches the
   allocator to expandable segments (no fragmentation), long episodes keep each layer's checkpoint
   input in pageable host memory, and the diagnostic entropy is computed outside autograd.


---

## 6. Training configuration and commands

Configuration: `configs/rl/activevln_grpo_r2r4k.yaml` (every value annotated as ActiveVLN's or as a
"port:" difference). Summary: 500 RL steps = 1 epoch of the 4,000-instruction list in parquet order;
8 instructions × G = 4 per step; T = 1.2, top-p 1.0; per-episode budget ceil(min(2·M,120)·N\*/M)
primitive actions; dynamic sampling ≤ 2 re-roll rounds; reward 15·(1−d/3) on success; GRPO
(unbiased std, ε 1e-6); dual-clip PPO (0.2/0.28/3.0) with one on-policy update per step; verl
micro-batch token-mean (2 trajectories); AdamW 1e-6 constant, (0.9, 0.999), wd 0.01, clip 1.0, no
warmup, KL 0, entropy 0; language model trainable, vision frozen; bf16, DeepSpeed ZeRO-2, 2 ranks;
3 IPC rollout helpers per GPU; exports every 50 steps; resumable per-rank state every 50 steps.

Environment: `source /data/anh/simplemem-eval/env.sh` (EVAL_PYTHON = Habitat 0.2.4 / Python 3.10,
MODEL_PYTHON = `.venv`, Python 3.12.13, torch 2.10.0+cu129, transformers 5.11.0, DeepSpeed 0.16.4,
flash-attn 2.8.3, fla 0.5.2).

```bash
# data audit (needs pyarrow: isolated tools venv)
/data/anh/rl_baseline/tools-venv/bin/python scripts/rl/audit_activevln_pool.py \
  --parquet /data/anh/code/ActiveVLN/data/r2r_4000_train.parquet \
  --r2r-root $DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed \
  --out data/rl/activevln_r2r4k.jsonl --report data/rl/activevln_r2r4k_audit.json
# train scenes
.venv/bin/python scripts/rl/download_train_scenes.py --train-json $DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --scenes-dir $MP3D_SCENES --record /data/anh/rl_baseline/records/train-scene-mirror-verified.json
# gates
.venv/bin/python -m pytest -q tests/rl
.venv/bin/torchrun --nproc_per_node=2 scripts/rl/check_ds_accumulation.py
.venv/bin/torchrun --nproc_per_node=2 scripts/rl/check_resume.py /tmp/resume_toy
PYTORCH_ALLOC_CONF= .venv/bin/python scripts/rl/check_rollout_pool.py 1 /tmp/pool_check
# training (unattended; resumes automatically; evaluates steps 500 and 350 when done)
systemd-run --user --unit=rl-supervisor --collect -p WorkingDirectory=$PWD /bin/bash scripts/rl/supervise.sh
#   which launches:  scripts/rl/launch_unit.sh rl-grpo-r2r4k LOG /usr/bin/env PYTORCH_ALLOC_CONF= \
#     .venv/bin/torchrun --nproc_per_node=2 -m qwen_vl.rl.train_grpo \
#     --config configs/rl/activevln_grpo_r2r4k.yaml --output-dir /data/anh/rl_baseline/runs/grpo_r2r4k [--resume]
# evaluation of an export with the baseline's exact protocol
bash scripts/rl/eval_export.sh /data/anh/rl_baseline/runs/grpo_r2r4k/exports/step_0500 GPU OUT_DIR
python3 scripts/rl/compare_evals.py --base /data/anh/simplemem-eval/results/r2r-full/epoch-2 --new OUT_DIR --out CMP.json
```

Run artifacts: `/data/anh/rl_baseline/runs/grpo_r2r4k/` (`run_config.json` with git commit, versions,
DeepSpeed config; `metrics.jsonl` per step; `rollouts/step_*_rank*.jsonl` per trajectory;
`exports/step_XXXX/` evaluable checkpoints with `rl_provenance.json`; `resumable/`), logs in
`/data/anh/rl_baseline/logs/`.

---

## 7. Completed run: `grpo_r2r1k` (ActiveVLN GRPO, 1k R2R subset, from the SFT checkpoint)

| Item | Value |
|---|---|
| Run directory | `/data/anh/rl_baseline/runs/grpo_r2r1k` |
| Config | `configs/rl/activevln_grpo_r2r1k.yaml` (final state at commit `6855f8e`) |
| Started / finished | 2026-10-10 13:22 UTC / 2026-10-11 08:44:58 UTC (`COMPLETE`: global_steps 125/125) |
| Optimizer updates | 125 (one per RL step, asserted every step) |
| Trajectories | 4,000 trained-batch trajectories (125 × 32) + 868 discarded by dynamic sampling; 262,696 simulator actions |
| Final export | `exports/step_0125/` — weights SHA-256 `bc6b7d2ad6a8de4669734e84b95526833c6cacd4cbc1cf42077716f03ced4a03`, `navigation.json` byte-identical to the base, `rl_provenance.json` |
| Other exports | `exports/step_{0025,0050,0075,0100}/` |
| Resumable state | `resumable/step_0125/rank{0,1}.pt` (exact per-rank ZeRO-2 state) |
| Training curve | `/data/anh/rl_baseline/evals/grpo_r2r1k-step0125/training_summary.json` |

Training statistics (sampled rollouts at T = 1.2 on *new* train instructions each step, so these are
not a held-out learning curve; last record per step):

| Steps | Train SR | Train OSR | Train SPL | Mean reward | Mean actions | Truncated | All-fail groups | All-success groups | Entropy | Grad-norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–24 | 0.618 | 0.666 | 0.589 | 6.01 | 63.7 | 0.046 | 0.045 | 0.285 | 0.147 | 1.96 |
| 25–49 | 0.579 | 0.659 | 0.540 | 5.90 | 67.3 | 0.086 | 0.065 | 0.255 | 0.125 | 1.98 |
| 50–74 | 0.620 | 0.711 | 0.578 | 6.76 | 64.4 | 0.098 | 0.060 | 0.310 | 0.093 | 2.44 |
| 75–99 | 0.630 | 0.686 | 0.588 | 6.75 | 66.4 | 0.061 | 0.040 | 0.330 | 0.075 | 2.80 |
| 100–124 | 0.651 | 0.726 | 0.603 | 7.24 | 66.7 | 0.069 | 0.045 | 0.355 | 0.081 | 2.60 |

Rollout ↔ trainer log-prob mismatch stayed at ≈0.0035–0.004 nats/token; clip fraction 0 (one
on-policy update per step, as in ActiveVLN); no format errors were sampled.

### 7.1 Run log (all attempts, for reproducibility)

| Run | What happened | Outcome |
|---|---|---|
| `smoke1`, `smoke2` | 2-rank smoke tests (budget-capped and full budget), export/eval/resume gates | pass |
| `aborted_threads_grpo_r2r4k` | 4K run with rollout threads: GIL-bound | stopped, replaced by IPC helpers |
| `aborted_serial_grpo_r2r4k` | 4K run, serial rollouts: >20 min/step | stopped |
| `aborted_cachefix_grpo_r2r4k`, `aborted_oom1/2_grpo_r2r4k` | 4K run: idle-helper GPU cache; CUDA OOM on shared GPUs | fixed (cache release, expandable segments, entropy outside autograd) |
| `stopped_partial_grpo_r2r4k_step22` | 4K run healthy at step 22 (7.3 min/step) | stopped on user request (switch to 1k, from scratch) |
| `aborted_oomd_grpo_r2r1k` | systemd-oomd killed the user session (incl. this job) under host-memory pressure | fixed: smaller host footprint, no MemoryHigh throttling |
| `aborted_ram_grpo_r2r1k` | host RAM headroom too small with 3 helpers/GPU | restarted from scratch with 2 helpers/GPU |
| `grpo_r2r1k` (final) | interrupted 8 times by the shared host (1 CUDA OOM at step 26, 6 host-RAM OOM kills, 1 DeepSpeed 10-min NCCL timeout under host thrash) and resumed 13 times in total (8 automatic by the supervisor, 5 deliberate restarts to deploy fixes); fixes applied between resumes (memory-aware episode placement, allocator-cache-aware free memory, on-demand host offload with headroom guard, 4 h DeepSpeed group timeout, heap trimming, 1 helper/GPU) | **complete, 125/125** |

Every resume restarted from an exact per-rank checkpoint (10-step cadence from step 30 on) and
re-ran the interrupted steps from the beginning; re-run steps reproduce the original rollouts
(e.g. step 40: identical SR 0.469 / reward 4.95). The systems changes alter only where and how
gradients are computed (placement, offload, kernel allocator), not their values.

---

## 8. Evaluation and comparison (Phase 4)

Both checkpoints are evaluated with **the identical SimpleMem protocol** (§1.3): `qwen_vl.eval.habitat_r2r`,
Habitat 0.2.4, R2R-VLNCE v1-3 `val_unseen`, all **1,839** episodes, greedy decoding, 500-action cap,
seed 42. The RL evaluation contract matches the baseline's on dataset, Habitat config, split, seed,
protocol, cap, failure policy, transport, simulator, scenes and base assets, and on the full episode
set; the only differing source files are the new `rl/*.py` modules, which the evaluator does not
import. The RL export was evaluated in 4 disjoint shards (2 per GPU) of the same episode list and
merged (`scripts/rl/eval_shards.py`; merging the baseline journal this way reproduces its summary
exactly up to float summation order). Evidence: `plans/RL/results/`.

| Metric (val_unseen, 1,839 ep.) | SimpleMem SFT (epoch 2) | SimpleMem + ActiveVLN GRPO (1k, step 125) | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| **SR** | 51.55 | **56.88** | **+5.33** | +10.3 % |
| **SPL** | 47.64 | **50.45** | +2.81 | +5.9 % |
| **NE (m)** ↓ | 5.80 | **5.31** | −0.50 | −8.6 % |
| **OSR** | 59.27 | **64.87** | +5.60 | +9.4 % |
| TL (m) | 9.89 | 10.63 | +0.74 | +7.5 % |
| Mean actions / episode | 91.3 | 96.6 | +5.3 | +5.8 % |
| Forced STOP at 500 | 114 | 133 | +19 | |
| Invalid responses / failed episodes | 0 / 0 | 0 / 0 | | |
| OSR − SR | 7.72 | 7.99 | +0.27 | |

Statistics (paired, same episodes): success both 771, only SFT 177, only GRPO **275**, neither 616;
**exact McNemar p = 4.7 × 10⁻⁶**. Bootstrap 95 % CI (10,000 resamples): **ΔSR [+3.10, +7.56]**,
**ΔSPL [+0.84, +4.78]**. nDTW is not computed for R2R by this evaluator (undefined in both).

Checkpoints: SFT `/data/anh/checkpoints/simplemem/joint-window8-text-b/epoch-2` (SHA-256 `4391b1f6…`);
GRPO `/data/anh/rl_baseline/runs/grpo_r2r1k/exports/step_0125` (SHA-256 `bc6b7d2a…`).
Journals: `/data/anh/simplemem-eval/results/r2r-full/epoch-2/` and
`/data/anh/rl_baseline/evals/grpo_r2r1k-step0125/val_unseen/`.

### 8.1 Published ActiveVLN (context only — not comparable)

| | Backbone / memory | Simulator & eval protocol | RL data | R2R val_unseen SR / SPL / NE / OSR |
|---|---|---|---|---|
| ActiveVLN SFT → +RL (paper) | Qwen2.5-VL-3B, full multi-turn history, ≤3 macro actions/turn | Habitat 0.1.7, HFOV 90°, sampled decoding (T 0.2, top-p 0.8), 400-sim-step / 120-call cap, stagnation stop | 4K R2R, step 350 | 38.5 / 33.7 / 6.83 / 42.7 → **50.1 / 43.7 / 5.31 / 58.3** (+11.6 SR) |
| SimpleMem SFT → +ActiveVLN GRPO (this work) | Qwen3.5-4B, Window8 + GDN memory, 1 primitive/step | Habitat 0.2.4, HFOV 79°, greedy, 500-action cap | **1K** R2R prefix, step 125 | 51.55 / 47.64 / 5.80 / 59.27 → **56.88 / 50.45 / 5.31 / 64.87** (+5.33 SR) |

Different backbones, starting points, simulators, sensors and evaluation protocols: the absolute
numbers must not be compared. The relative pattern is consistent: outcome GRPO raises SR and OSR
substantially from an SFT policy; the gain is smaller here from a much stronger SFT start (51.6 vs
38.5 SR) with a quarter of the RL data and steps.

---

## 9. Analysis (Phase 5)

Per-episode analysis: `plans/RL/results/grpo_r2r1k_step125_failure_analysis.json`
(`scripts/rl/analyze_failures.py`).

**Where the gain comes from.**
- Gains appear in every length bin (GT actions): <40: 62.0→66.2; 40–60: 56.0→**63.5**; 60–80: 45.6→48.2;
  ≥80: 31.9→36.3. By start–goal geodesic distance: <7 m +6.3, 7–10 m +6.0, 10–13 m +2.1, ≥13 m +6.5 SR.
- Failure taxonomy (SFT → GRPO): never within 3 m 552 → **466**; near miss (stop 3–5 m) 88 → **54**;
  passed within 3 m then stopped outside 141 → 142; forced STOP at cap 110 → 131.
  Median stop distance 2.75 → **2.10 m**.
- Of the 275 newly solved episodes, the SFT policy had been far from the goal in 145, had passed
  the goal in 69, was a near miss in 35 and hit the cap in 26.

**What changed in behaviour.** The action mix shifted to forward motion (MOVE_FORWARD 72.8 → 78.1 %,
turns 26.1 → 20.9 %): the policy commits to a route more often, which removes many "never within
3 m" failures and makes trajectories slightly longer (TL +0.74 m, +5.3 actions). Sampling entropy
fell from 0.147 to ≈0.08 nats/token over 125 updates; rollout/trainer log-prob agreement stayed
at ≈0.004 nats/token.

**Weaknesses / limitations of outcome GRPO on this policy.**
1. **STOP timing is not fixed.** "Passed the goal but stopped outside 3 m" is unchanged (141 → 142)
   and OSR − SR did not shrink (7.7 → 8.0). The episode-level reward gives every action the same
   advantage, so the decisive STOP decision gets no specific credit — exactly the gap the
   analytic-STOP estimators of `plans/RL_technical.md` (§7) target.
2. **More cap hits.** Forced STOPs rose 114 → 133, concentrated in 60–80-action episodes (31 → 52):
   committing to forward motion sometimes produces long wandering. The RL budget (2× expert length)
   is far below the evaluator's 500-action cap, so these trajectories never appeared in training.
3. **Credit is coarse.** With continuous success-weighted rewards nearly every trajectory receives a
   non-zero advantage (28–32 of 32 trained per step), but the signal per action is one scalar per
   ~65-action episode; dynamic sampling discarded 868 of 4,868 rollouts (18 %).
4. **Entropy decreases steadily** (−45 % in 125 steps); a longer run (the full 4K, 500 steps) may
   need entropy monitoring to avoid collapse.
5. **Single seed, single scale.** One RL seed and one data size (1K); no SFT-continuation control with
   matched compute. The effect is highly significant against the SFT checkpoint, but part of it could
   come from on-policy exposure that a matched-cost DAgger or SFT control would also give.
6. **Training-set overlap.** The 1K RL instructions were also SFT data (as in ActiveVLN). Validation
   scenes are disjoint (0 shared scans/trajectories with val_unseen).

**Deviations from ActiveVLN that matter for interpretation** (§3): 1K prefix / 125 steps instead of
4K / 500 (user decision); per-primitive actions and budget conversion; vision tower frozen; Habitat
0.2.4/HFOV 79 training and evaluation; greedy evaluation. All algorithmic pieces (reward, GRPO
advantage, dual-clip loss, micro-batch token-mean, dynamic sampling, optimizer, LR, batch, group
size, temperature) are ActiveVLN's.

---

## 10. Research directions (ranked)

Ranking criteria: novelty (N), expected SR benefit on this policy (B), feasibility in this codebase
(F), experimental cost (C; lower is better). Evidence from this baseline is cited.

| Rank | Direction | N | B | F | C | Why now |
|---|---|---|---|---|---|---|
| 1 | **Analytic STOP credit** (exact counterfactual "stop here" value from geometry at every visited decision; `RL_technical.md` §7, verified unbiased by enumeration) | high | high | high (σ = p(STOP) at the first response token is already computed in `response_log_probs`) | low (same rollouts, G = 4) | the dominant residual failure (141 passed-goal failures, OSR−SR 8.0) is untouched by outcome GRPO |
| 2 | **Geometric state baseline / privileged critic** (HiCo-PO component; `RL_survey.md` §7.4) replacing the group-mean baseline | high | medium–high | medium (per-step geometry is available from the env server) | low; enables G = 2 (half the rollout cost) | outcome advantages are constant over ~65 actions; per-step credit for MOVE/TURN |
| 3 | **Cap-aware budget / anti-wandering shaping** (train with the evaluator's cap distribution, or penalise loops) | medium | medium | high | low | forced STOPs rose 114 → 133 after RL |
| 4 | **Efficiency-gated reward (v1)** for SPL | low–medium | medium (SPL) | high | low | SPL gain (+2.8) lags SR (+5.3); TL increased |
| 5 | **RL for recurrent (GDN) memory**: credit through the persistent linear-attention state, state-dropout / window-length curriculum | high (no prior RL work on GDN VLN policies) | uncertain | medium | medium | unique to SimpleMem's architecture; this baseline shows the full-BPTT RL loop is stable |
| 6 | **Pass-rate-filtered instruction pool** (drop always-solved instructions) | low | low–medium | high | very low | 29–36 % of groups were all-success |
| 7 | **On-policy distillation from a privileged geometric teacher** (dense per-step targets) | medium | medium | medium | medium | complementary dense signal; risk of teacher–policy mismatch |

Recommended next experiments (one change at a time, same 1K order, same evaluation):
(a) matched-compute controls — continued SFT/DAgger on the same 1K instructions, and a second GRPO
seed; (b) the full 4K / 500-step GRPO run for the scale curve (exports every 50 steps already supported);
(c) rank 1 (analytic STOP) on top of this exact pipeline; (d) rank 2 replacing the GRPO baseline.

---

## 11. Status against the completion criteria

| Criterion | Status |
|---|---|
| GRPO integration passes its tests | **yes** — `tests/rl` (13 tests), DS accumulation, resume, IPC-helper parity, placement gates (§5) |
| RL training completed | **yes on the 1K subset** (125/125 steps, from scratch, user-directed scope change from 4K) — the 4K/500-step run was stopped at step 22 by user request and not completed |
| Trained model evaluated | **yes** — full R2R val_unseen, 1,839/1,839 episodes, identical protocol |
| Comparison documented with evidence | **yes** — §8, `plans/RL/results/` |
| RxR | not run (user: R2R only); the baseline RxR journal is preserved at 350/3,669 and resumable (`/data/anh/rl_baseline/resume_rxr15_campaign.sh`, sources at worktree `6473321`) |
