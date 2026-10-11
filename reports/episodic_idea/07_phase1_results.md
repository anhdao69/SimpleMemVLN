# 07 — Phase 1: runs, status and evaluation hand-off

**No Phase-1 result exists yet.** Rows move from *submitted* to *started*, *completed*,
*uploaded* and *evaluated* only with recorded evidence (`run_index.json`, campaign ledger).

## Campaign identity

| Field | Value |
|---|---|
| Campaign UUID | `c42b737b-9519-4be5-9886-c8a2c9c2db50` |
| Campaign root | `/groups/yshang/an221229/checkpoints/SimpleMemVLN-c2_episodic` |
| Immutable source | `source/469b107696bde6e0565860af45d2a0e35428427e` (read-only `git archive`; supersedes `a7e5ae5`, see 01 D9/D11) |
| Environment | `/lustre/fs1/home/an221229/code/SimpleMemVLN-c2_episodic/.venv` (uv-locked; must not be modified while jobs run) |
| Manifest | `artifacts/r2r_rxr15_train.jsonl`, 30,815 episodes, SHA-256 `0e5ee1c3…4515` |
| Model | Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Schedule | 2 epochs = 7,704 updates, warmup 232, cosine min 0.1, LR 5e-6, AdamW (0.9, 0.95), wd 0.01, clip 1.0, seed 429; **stop at the epoch-1 boundary (update 3,852)** |
| Hub | private `anhdao69/Streaming_episodic`, folder `<run>/<attempt>/` |

## Runs

| Run | Arm | R | B | Policy | Submission | Status |
|---|---|---:|---:|---|---|---|
| `P1-V0-s429` | V0 | 8 | 0 | recent FIFO | H4 job **898758** (`…-H4-a2`), H2 job **898759** (`…-H2-a2`) — race, same logical run; a1 jobs 898755/898756 superseded before start | submitted (pending) |
| `P1-M1-s429` | M1 | 9 | 0 | recent FIFO | submitted automatically by the verified V0 winner on its profile | not yet submitted |
| `P1-M2-s429` | M2 | 8 | 300 | committed-write top-B (GDN l−1 → attention l) | same | not yet submitted |
| `P1-M2c-s429` | M2c | 8 | 300 | per-layer SHA-256 random priority | same | not yet submitted |

Race protocol (plan §17): each candidate runs preflight and the **full profile suite
(V0 + M2)** on its own allocation; the first to pass claims the lease atomically; the
other exits `DUPLICATE_STANDBY` if it starts later. The winner becomes
`TRAINING_VERIFIED` after ≥50 healthy updates on every rank, a complete and hashed
recovery checkpoint at update 50, and host/GPU limits held; only then is the recorded
pending sibling cancelled (exact job ID + live identity check) and M1/M2/M2c submitted
on the winning profile (each runs its own startup longest-episode gate). No
cross-world-size resume is ever performed.

## What each completed run publishes

`anhdao69/Streaming_episodic/<run>/<attempt>/`:

* `export-e1/` — serving export (`pytorch_model.bin`, `navigation.json` with the full
  episodic contract, processor/tokenizer files, `reload_reference.json`).
* `checkpoint-3852/` — complete training state (DeepSpeed ZeRO-2 shards, scheduler,
  4 or 2 RNG states, trainer/sampler state, resume contract). Epoch 2 resumes exactly
  only on the same profile/world size:
  `--resume_from_checkpoint <dir>/checkpoint-3852` with the identical configs, manifest
  hash, source and profile, **without** `--stop-after-epochs` (the 7,704-update
  schedule is already declared).
* `campaign/` — launch manifest, resolved config, source hashes, qualification summary,
  per-rank profile/exposure/microtime logs, trainer state.

## Evaluation hand-off (another server)

Use branch `c2_episodic` (pushed to GitHub; also `code/c2_episodic_*.bundle` in the Hub repo) at (or after) the source commit above; the evaluator is the
existing `qwen_vl.eval.habitat_r2r` (see `EVALUATION_SETUP.md`), unchanged. The episodic
memory policy is read from the export's `navigation.json`; nothing must be passed on the
command line and mismatches fail closed. The model process reports
`runtime.episodic` (R, B, score, backends) in its handshake.

```bash
# Variables as in EVALUATION_SETUP.md §11–13 (EVAL_PYTHON, MODEL_PYTHON, MODEL_PATH,
# R2R_CONFIG, R2R_DATA, MP3D_SCENES, SIMULATOR_SOURCE, EVAL_GPU). The full R2R split is
# exactly the pre-registered 1,839 IDs (r2r_val_unseen_ids.json, SHA-256 92536e8e…abc3).
CHECKPOINT=/path/to/P1-M2-s429/<attempt>/export-e1
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind r2r --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$R2R_CONFIG" --data-path "$R2R_DATA" --scenes-dir "$MP3D_SCENES" \
  --gpu "$EVAL_GPU" --split val_unseen --out "$RESULT_ROOT/r2r-full/P1-M2-e1"
# RxR15 English dev panel: same command as EVALUATION_SETUP.md §13 (--dataset-kind rxr15
# --simulator-source ... --ground-truth-path ...) plus the fixed panel:
#   --episode-list reports/episodic_idea/rxr15_val_unseen_panel1000.list.json
```

Report per arm at E1 (mid-schedule): SR, SPL, NE, OS, model STOP, forced STOP (500-action
cap), invalid/infra failures, per-episode outcomes (paired across arms), and the
evaluator journals. Label RxR results "1,000-episode English RxR15 val_unseen panel",
never full RxR. Evaluate V0, M1, M2, M2c on the same IDs; Phase 2 opens only through
`08_phase2_decision.md`.

## Operating the campaign after this session

```bash
C=/groups/yshang/an221229/checkpoints/SimpleMemVLN-c2_episodic
SRC=$C/source/469b107696bde6e0565860af45d2a0e35428427e
ENV=/lustre/fs1/home/an221229/code/SimpleMemVLN-c2_episodic/.venv
cd $SRC && PYTHONPATH=$SRC/src:$SRC $ENV/bin/python -m scripts.vln.submit_episodic status --campaign-root $C
tail -f $C/runs/P1-V0-s429/*/slurm-*.log          # lifecycle states (STATE ...) per attempt
```

* Per attempt: `status.json`, `events.jsonl`, `preflight.json`, `qualification-<job>/`,
  `supervisor/` (1 s host-memory samples, GPU samples), `train/` (per-rank profiles,
  exposures with episodic diagnostics, recovery checkpoints), `upload_complete.json`.
* Lease: `$C/leases/<run>/LEASE.json` (+ `events.jsonl`, `history/`).
* **Interrupted attempt** (Slurm pre-termination checkpoint, node failure after a complete
  recovery checkpoint): resubmit the *same* attempt so it resumes on the same profile:
  `sbatch --parsable --job-name=<same> --comment=<same> --partition=highgpu --nodes=1 --ntasks=1
  --gres=gpu:nvidia_h100_80gb_hbm3:<4|2> --cpus-per-task=16 --mem=512G --time=<...>
  --signal=B:USR1@1800 --output=<attempt>/slurm-%j.log $SRC/train/slurm/episodic_candidate.sbatch <attempt_dir>`
  and append the job ID to the ledger. The lifecycle re-gates the new allocation
  (startup longest-episode gate) and resumes with `--resume_from_checkpoint auto`
  (complete markers only). Never resume an H4 attempt on 2 GPUs or vice versa.
* **Failure before any valid checkpoint**: submit a new attempt (`submit_episodic submit
  --run <run> --profiles <winner profile> --qualify startup`).
* **Epoch 2** (only if the Phase-2/evaluation decision asks for it, for all four arms):
  resume each run's `checkpoint-3852` on its own profile without `--stop-after-epochs`.
