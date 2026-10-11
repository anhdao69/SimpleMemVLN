# 00 — Provenance, environment and resource audit (Task 0)

Status labels used in this directory: **proposed**, **implemented**, **CPU-tested**,
**GPU-tested**, **GPU-qualified**, **submitted**, **started**, **completed**,
**evaluated**, **failed**. Nothing here is a navigation result.

Audit date: 2026-10-11 (EDT). Auditor: coding agent in interactive Slurm job 894903.

## Repository and branches

| Branch | Tip | Relation to base `f6b8108` |
|---|---|---|
| `streaming_text_dual` (base) | `f6b8108f6ef8c7b4e3df035688d3210dbd3678a1` | audited starting reference (plan §1.1) |
| `main` | `53ac625611b5778df8a4eaeecbdeda028bfa0309` | 7 report-only commits not in base; source is a strict subset |
| `streaming` | `36a2c9e58ad1f5e75d672f4ed82c4010de4903b8` | 7 commits (reports, resumable eval helpers); source lacks text-dual features |
| `streaming_logits` | `1058d090f58708a9f20d9ad40c7a561ff630a186` | 7 report/eval commits; subset of base source |
| `streaming_logits_dual` | `499f7d81348d0b7eb9a558dcbfae7991d7dba011` | 2 report commits; lacks batched attention/text gates |
| `RL_baseline` | `96eedf1a45fb3396bc94351d40347faafd9599b0` | 31 commits **on top of** base: GRPO code under `src/qwen_vl/rl`, plus RL-specific host-RAM measures (pageable offload, heap trim, expandable segments). Not merged; the generic ideas were re-implemented and re-qualified here (see 02). |

`git diff` of `src/`, `scripts/`, `tests/`, `configs/` against base confirms the other
branches contain no fixes to the shared training/serving modules that base lacks
(their diffs are deletions of text-dual features, reports, or RL-only modules).

Local checkout `/lustre/fs1/home/an221229/code/SimpleMemVLN` (branch
`streaming_text_dual` at `f6b8108`, in sync with origin) had one untracked file,
`plans/episodic_mem.md` (SHA-256 `142a07b6…780b`). It was copied, not moved; the
original checkout was not modified. Isolated worktree:
`/lustre/fs1/home/an221229/code/SimpleMemVLN-c2_episodic`, new branch `c2_episodic`
from `f6b8108`. The plan file was committed there unchanged.

## Environment (actual, this server)

* Node evc103 (cluster `newton`, partition `highgpu`), 4× H100 80GB HBM3 (79.18 GiB
  reported capacity), driver 580.105.08.
* `uv sync --locked` in the worktree: Python 3.12.13, torch 2.10.0+cu129,
  Transformers 5.11.0, FLA/fla-core 0.5.2, FlashAttention 2.8.3 (locked mirror wheel,
  SHA-256 `fa278650…5218c`), DeepSpeed 0.16.4, Accelerate 1.13.0, TileLang 0.1.14,
  Triton 3.6.0. No dependency was upgraded.
* `causal_conv1d` is not installed: HF uses its PyTorch depthwise-conv path for GDN
  (the FLA chunk/recurrent GDN kernels are the approved `fla.ops.gated_delta_rule`
  functions; `load_model` still refuses any non-FLA delta-rule fallback).
* CUDA toolkit for JIT: `/apps/cuda/cuda-12.6.0` (historical reports used 12.8.1; 12.6
  is the closest available). **Failure found:** TileLang's Hopper backward kernel
  compiles with `-std=c++20`; the system host compiler g++ 8.5 ignored it and the
  compile failed (`namespace std has no member conjunction`, 1 baseline test). Fix:
  `CXX=/apps/gcc/gcc-12.2.0/bin/g++` (+ its `lib64` on `LD_LIBRARY_PATH`). The FLA
  Hopper guard was not touched.
* Model: `Qwen/Qwen3.5-4B` snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in
  `/groups/yshang/an221229/cache/huggingface/hub` (matches the pinned revision).
* Hugging Face token: user `anhdao69`, fine-grained `repo.write` on that namespace.
  Target repo `anhdao69/Streaming_episodic` did not exist at audit time; uploads create
  it as a **private** model repo.

## Baseline tests

Full suite on the worktree with the pinned environment (GPU 0): **230 passed, 1 failed,
0 skipped**. The single failure (`test_submission.py::test_training_chain_is_independent_of_uploads`)
shells out to the system `python3` (3.6, no `text=` kwarg); with `.venv/bin` first on
`PATH` it passes. It is an environment PATH issue, not a regression. (The first run
also exposed the TileLang compiler failure above.)

## Data (recomputed, not copied from reports)

Source: `/groups/yshang/an221229/data/JanusVLN_data` — R2R metadata
`R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`, frames `R2R/train/<id>/`; English RxR
15° metadata `RxR_CE_15_deg_Trajectory_data/train/train_guide.json.gz`
(conversion manifest: derived 15° from 30°, `en-IN/en-US`), frames `RxR_15deg/train/<id>/`.
`prepare_r2r.py`/`prepare_joint.py` gained explicit `--source/--frames-root` and
`--rxr-source/--rxr-frames` arguments for this layout (no logic change).

| Quantity | Recomputed | Plan reference |
|---|---:|---:|
| R2R episodes / actions | 10,819 / 631,244 | 10,819 |
| RxR15 English episodes / actions | 19,996 / 2,497,380 | 19,996 |
| Joint episodes / action targets | **30,815 / 3,128,624** | 30,815 / 3,128,624 |
| Rejected / missing English IDs | 0 / 0 | — |
| Max observations | 627 | 627 |

Joint manifest `…/SimpleMemVLN-c2_episodic/artifacts/r2r_rxr15_train.jsonl`, SHA-256
`0e5ee1c3fd1e419727111d19ac208d4a74cdccaa7b41cf3076ab31197cec4515`. It differs from the
historical `5deb425d…2933` because `rgb_path` values are absolute server paths; counts,
UIDs and labels were rebuilt from the same official sources. Exact encoded lengths
(actual processor, 300 tokens/image) reproduce every historical fixture:

| Fixture | UID | Obs | Tokens |
|---|---|---:|---:|
| Longest RxR | `rxrce:…82sE5b5pLXE…:31149:22206` | 627 | 199,735 |
| Longest R2R | `r2rce:…PuKPg4mmafe…:6967:6967` | 183 | 58,290 |
| Just below offload | `rxrce:…Vvot9Ly1tCj…:13476:79656` | 205 | 65,526 |
| Just above offload | `rxrce:…kEZ7cmS4wCh…:10317:39641` | 205 | 65,578 |
| Largest prefix ≥ offload | `rxrce:…E9uDoFAP3SH…:57190:79383` | 239 | 76,679 (prefix 678) |

2,547 of 30,815 episodes (8.3%) are at or above the 65,536-token activation-offload
threshold. Capture-before-action timing remains unverified (inherited qualification;
filenames are not collector evidence).

## Slurm and host resources (discovered, never assumed)

* Interactive allocation **894903**: evc103, 4× H100, 16 CPUs, **128 GiB**, started
  2026-10-11 04:54, ends **22:54 EDT** (18 h limit). Job memory cgroup is **v1**
  (`/sys/fs/cgroup/memory/slurm/uid_1228030620/job_894903`), limit 128 GiB.
  The supervisor/trainer sampler was extended to cgroup v1 (`memory.usage_in_bytes`,
  `total_inactive_file`, `memory.oom_control:oom_kill`, `memory.failcnt`).
* Protected jobs (never touched): 894903 (interactive), 897730, 897732, 897735, 897736
  (running `chrono-*`), 897740, 897741 (pending `chrono-*`); recorded in
  `…/SimpleMemVLN-c2_episodic/protected_jobs.json`.
* `highgpu` nodes evc101–104: 8× H100, 128 CPUs, ~1006 GiB each; at audit every GPU was
  allocated and the queue was long (scheduler start estimates for 4-GPU jobs ≈ next day).
  QOS `normal`: no wall-time limit found.
* Disk: Lustre `/lustre/fs1` 1.2 PB, 395 TB free. `lfs quota` reports the user at
  8.6 TiB against a 1000 GiB soft / 1010 GiB hard quota (flagged `*`), but writes
  succeed (10 GiB direct-I/O write test, 809 MB/s) — the quota is evidently not
  enforced. Recorded as a risk; campaign preflight still reserves ≥400 GiB free.

## Consequences for qualification

128 GiB cannot host the longest-RxR gates on either profile (historical H4 guarded
working set 417.9 GiB; see 05/06 for the new measurements). Those gates therefore run
inside each submitted high-RAM candidate job, on the actual allocation, before it can
claim the run lease. This is recorded as a blocker of the interactive allocation, not
waived.
