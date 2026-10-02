# Pretrained candidate-logit navigation

Branch: `streaming_logits`. Base: `341ec2cfce934a872c4478cb06c4f70354113bc4`
(`streaming`, after synchronizing the joint R2R/RxR production code).

## Architecture and experimental contract

The old `qwen_text` policy greedily projects the full vocabulary and appends
tokens until the assistant terminator, then calls the strict action parser.
Nothing in the vocabulary restricts its output to the four navigation strings;
`MOVE_RIGHT` is therefore possible. Relaxing parsing would hide errors without
removing unnecessary decoding. A random four-way linear head avoids invalid
strings but discards pretrained output geometry and can exploit forward-heavy
class imbalance. Pretrained candidate rows provide a cleaner first baseline,
not a guarantee of improved navigation or STOP recall.

`candidate_logits` reads the final observation/cue hidden state and projects
only four Qwen LM-head rows. `argmax` goes directly through the canonical
`contracts.py` action/Habitat mapping. There is no parsing, response generation
loop, generated EOS, or vocabulary-sized output tensor on this path.

Pinned Qwen3.5-4B revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Its tokenizer verifies `A/B/C/D` as IDs `32/33/34/35`. Bare `Action:A` and
`Action: A` do not preserve the same append boundary. The exact rendered
observation ends in **`Action:\n`**. Startup verifies four distinct, nonspecial,
single-token labels, round-trip decoding, and full context + label + terminator
concatenation equality. No assumption based solely on isolated tokenization.

The native `[248320, 2560]` LM-head weight is tied to input embeddings and has
no bias. The implementation also handles a head bias when present.

| Head mode | Initialization/readout | Trainable parameters / LR |
|---|---|---|
| `lm_rows_trainable` (default) | Direct current LM rows | Shared embedding/head and decoder, normal backbone LR 5e-6 |
| `lm_rows_frozen` | Direct frozen LM rows | Entire shared input/output parameter frozen; decoder still trains |
| `copied_linear` | Four rows and any bias copied into Linear(d,4) | Separate classifier uses `classifier_lr`; backbone uses its own LR |

Freezing only a detached readout would not actually freeze tied rows that can
change through input embeddings. The frozen experiment intentionally freezes
the whole shared parameter. Trainable mode does **not** promise that only four
embedding rows change: normal input gradients and AdamW operate on the shared
parameter. No special classifier LR applies to direct-row mode.

## Feedback and causality

Both feedback controls make exactly one deterministic history append:

- `candidate_token`: chosen A/B/C/D token + fixed assistant terminator/newline.
- `canonical_action_text`: original action string tokens + the same fixed suffix.

The suffix is protocol framing, not autoregressive generation or an extra CE
target. Canonical feedback is a history-length/representation control, not an
identical prompt to old text decoding: the new decision prompt still defines
the candidate mapping.

Training predicts at the cue before gold feedback. Inference predicts before
its own feedback. The cache append happens before returning the action for
execution, exactly once on a successful observation request; retries return
the cached result without appending again. If action execution fails, reset the
episode instead of submitting a new observation with an unexecuted history.
Teacher forcing and rollout share serialization and causal visibility, but
their history values naturally differ after prediction errors.

Window8 retains the instruction prefix plus eight complete observation AND
feedback groups, including the current group. Eviction removes only the oldest
group's full-attention KV; it does not reset GDN recurrent/convolution state.
Logical token and native MRoPE positions continue advancing after eviction.
FullContext keeps all groups. Reset creates fresh KV/GDN/position state.

New serializer: `vln_candidate_logits_v1`. Saved metadata includes candidate
IDs/labels, decision cue, feedback IDs/format, head mode and resolved training
class counts/weights. Strict loader/resume comparisons reject incompatible
contracts. Existing `qwen_text` and old random-classification metadata remain
unchanged; old weights are not silently reinterpreted as candidate checkpoints.

## Objective and diagnostics

Each gold action contributes one four-way float32 CE. Supported weights:
`none`, `sqrt_inverse_frequency`, `effective_number`. Counts come from the
actual selected training manifest, with the same selection ordering as the
dataset. Balanced modes reject missing classes.

For counts n, p=n/sum(n), and raw weights u:

```
sqrt:      u_c = 1/sqrt(n_c)
effective: u_c = (1-beta)/(1-beta**n_c)
w_c = u_c / sum_j(p_j*u_j)
sum_c(p_c*w_c) = 1
```

The global denominator remains the number of valid actions across ranks and
the full accumulation window, not the sum of weights. DDP's averaging is
compensated by world size exactly as in the existing trainer. Default `none`
is the clean baseline; the sqrt overlay is the recommended mild balancing
experiment. Compare both rather than asserting that balancing improves SR.
For the approximate R2R counts in the request, sqrt weights are about
`[0.7102, 1.3554, 1.3991, 4.3448]` in canonical action order; training recomputes
them from its exact manifest rather than hardcoding those estimates.

`action_metrics.jsonl` records globally summed confusion counts, precision and
recall per class, overall accuracy, macro accuracy (mean four-class recall;
absent classes count as zero), predicted/truth distributions, and weighted and
unweighted CE. Recovery rewinds this journal along with existing update logs.
STOP recall here is teacher-forced supervised recall. It is not a closed-loop
Habitat oracle metric; rollout reports model STOP and forced STOP separately.

A uniform predictor has CE `log(4)=1.386294`. Pretrained logits are not uniform.
Four-way CE must not be numerically equated with token-level text-policy CE.

## Verification evidence

Local CPU suite: **64 passed, 10 tokenizer/GPU-dependent skips**. Remote full GPU
suite: **74 passed, zero skips**, 52 dependency warnings. Independent read-only
review found no Critical/Important issues and reproduced the pre-final suite
(63 passed/10 skips). The final added test fixes benchmark provenance for the
legacy `PYTORCH_CUDA_ALLOC_CONF` variable used by the approved recipe.

Real pinned-model gates used one allocated H100, 12 real observations from a
short R2R episode, and a backward pass from its final STOP loss. All four
combinations passed. Vision stayed frozen; gradients reached early observation
embeddings, the first GDN layer, and the first full-attention layer. Tests also
checked complete eviction, FP32 recurrent state, retry/reset, and forbade a full
LM-head call during candidate streaming.

| Memory | Feedback | Initial episode CE | Streaming/offline logit RMS error | Argmax agreement | Peak reserved GiB |
|---|---|---:|---:|---:|---:|
| FullContext | candidate | 0.3705 | 0.1056 | 12/12 | 19.45 |
| Window8 | candidate | 0.3370 | 0.1285 | 12/12 | 19.31 |
| FullContext | canonical text | 1.6655 | 0.0963 | 12/12 | 19.35 |
| Window8 | canonical text | 2.0090 | 0.0942 | 12/12 | 19.33 |

BF16 chunked/offline versus incremental kernels are not bitwise identical;
maximum logit differences were 0.25–0.3125. The gate enforces absolute/RMS
numerical bounds. Review noted one deferred minor: its additional high-margin
criterion is mathematically redundant with the observed error; the reported
12/12 argmax agreement is measured but not asserted. These are short-episode runtime
checks, not full-dataset memory admission or navigation results. The initial
loss difference between feedback formats shows why this control matters.

Two-GPU Window8 candidate-token smoke used the eight shortest R2R episodes,
global batch 8 (2 × 1 × GAS4), LR 5e-6, zero warmup, three cosine-schedule updates.
All three epoch checkpoints and the final model were saved. Exact model reload
passed the existing 1e-5 logit / 1e-4 summed-loss tolerances; reference loss sum
was 4.81544733. No OOM or nonfinite loss occurred.

| Update | Unweighted/weighted CE (`none`) | Update seconds | Max reserved GiB across ranks |
|---|---:|---:|---:|
| 1 | 0.4526 | 6.059 | 59.80 |
| 2 | 0.3652 | 4.059 | 59.80 |
| 3 | 0.6007 | 4.023 | 59.80 |

End-to-end smoke wall time including load and saves was 210.19 s (Trainer
runtime 184.59 s). Update-only times exclude checkpoint I/O and loader prefetch.
First-update accuracy was 90%, but STOP recall was zero and macro accuracy
24.52%: the logs expose majority-class behavior instead of treating low CE as
navigation success. The three losses are not monotonic or a convergence result.
The selected set has 156 forward, 3 left, 3 right and 8 STOP targets per pass;
it is especially unrepresentative of the full data. Do not use its speed to
estimate full-dataset training time or its memory as a longest-episode gate.

Raw validation artifacts are under the isolated remote directory
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/logits-dev-xLLoog`:
`integration-*.json`, `pytest-final-gpu.log`, `reload.log`, `smoke.log`, and
`outputs/candidate-smoke/{action_metrics,profile_rank0,profile_rank1}.jsonl`.
Existing training source snapshots and Slurm jobs were not changed. The current
Janus `.venv` has neither `habitat` nor `habitat_sim`; no closed-loop Habitat
run is claimed. Full 100/1,839-episode commands require the validated simulator
environment used for earlier evaluations.

## Measured H100 serving-speed check

One model process at a time on the same otherwise idle allocated H100 80GB,
Python 3.12.13, Torch 2.10.0+cu129, Transformers 5.11.0, expandable segments,
BF16 and the existing attention/GDN runtime. CUDA-synchronized full decisions,
16 warmups + 64 measured steps, repeated real 640×480 RGB, one reset excluded
from timing; no rendering/IPC/model loading included. Both models used two CPU
threads. This follows the historical timing procedure, but not its Blackwell
hardware, four-thread setting, or exact RGB file. **No cross-report speed claim.**

Input: `JanusVLN/data/trajectory_data/R2R/train/1/step_0000_TURN_RIGHT.png`, SHA256
`a77da7861c69221503832598f13e9e6e4952cf056f96f5912f33a54900c7b579`.
Instruction: “Go around the right side of the center unit and stop by the right
side doorway with the dining table and mirror in it.” Both policies receive
the same frame/instruction and accumulate their own predictions.

| Window8 policy | Median ms/action | Mean | p95 | Peak live GiB | Retained KV | Generated tokens/action |
|---|---:|---:|---:|---:|---:|---:|
| Candidate, three-update smoke | 82.65 | 83.06 | 86.16 | 8.664 | 2,651 | 0 |
| Old qwen_text, epoch 1 `checkpoint-1353` | 122.85 | 123.34 | 126.73 | 8.663 | 2,611 | 3 |

This is about 33% lower median latency in this controlled runtime test, not a
quality-matched trained-policy result. A preceding repeat measured 82.72 versus
128.13 ms, illustrating run-to-run variability. Candidate KV is slightly larger
because its mapping prompt/cue differs, despite shorter action feedback.

Separate synchronized component runs measured candidate preprocessing ~6.23 ms,
vision ~9.51 ms, observation-language append ~32.90 ms, four-row projection
~0.053 ms, and deterministic history append ~32.89 ms. The history time includes
~32.58 ms of language processing and must not be added to it again. The old text
decoder costs roughly 75 ms in the first instrumented run; the original
observation-language/vision work remains. Eliminating token generation does not
eliminate the model forward or the history append, and is not a 3× speedup.

Raw timing files: `speed-verified-{candidate,text}-{headline,components}.json`
in the isolated remote directory above. Both old text checkpoint loading and
its generation/parsing path succeeded unchanged. The final benchmark source
records both modern and legacy allocator environment variable spellings.

## Changed surfaces and compatibility review

- Contracts/serialization: `contracts.py`, `data/candidates.py`,
  `data/episode_serializer.py`, `data/episode_dataset.py`.
- Shared readout/objective: `models/nav_model.py`, `models/action_loss.py`.
- Streaming: `stream/session.py`, opt-in `stream/timing.py`; no changes to
  cache eviction, GDN kernels, positional ledger or Window8 attention engine.
- Training: `train/vln_runtime.py`, `train/train_episode.py`,
  `train/action_reporting.py`, `train/recovery.py`. Existing global-normalization
  and optimizer implementations remain in `trainer.py` and are tested directly.
- Metrics: `eval/action_metrics.py`, `eval/metrics.py`; Habitat itself consumes
  the same navigation result API without a candidate-specific special case.
- Tools: `scripts/vln/{check_candidate_integration,evaluate_actions,benchmark_inference}.py`.
- Configs: baseline candidate overlay plus canonical-feedback, frozen, copied,
  sqrt, effective-number and tiny-smoke overlays.
- Tests: seven `tests/vln/test_candidate_*.py` files. Existing old-mode tests
  also ran; all runtime paths share the same canonical action definitions.

Source paths above are relative to `src/qwen_vl/` unless otherwise qualified.
No checkpoint, dataset, authentication token, or large artifact is committed.
The branch is experimental: full training/convergence, long-episode memory
admission, 100/1,839-episode Habitat results, and the completed quality ablation
remain required before promotion. One reviewer-noted redundant test criterion
is deferred as described above; numerical parity bounds remain active.

## Reproducible commands

Run from this branch's isolated checkout, not an active production snapshot.
Use an allocated GPU shell and the existing validated environment. The training
entry point is `train_qwen`, which dispatches to `train_episode`.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv/bin:$PATH"
export MODEL_PATH=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export R2R_MANIFEST=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_train.jsonl
export JOINT_MANIFEST=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl
export VLN_MODEL_PATH="$MODEL_PATH"
python -m pytest -q
```

### Tiny training smoke (two GPUs, global batch eight)

```bash
torchrun --standalone --nproc_per_node=2 -m qwen_vl.train.train_qwen \
  --vln_config configs/vln_r2r_v0_base.yaml \
  --output_config configs/vln_r2r_b_3epoch_bs8.yaml \
  --policy_config configs/vln_candidate_smoke.yaml \
  --memory_config configs/vln_memory_window8.yaml \
  --manifest "$R2R_MANIFEST" --model_name_or_path "$MODEL_PATH" \
  --episode-limit 8 --selection shortest --debug-repeat-episodes \
  --max-optimizer-updates 3 --gradient_accumulation_steps 4 \
  --output_dir outputs/candidate-smoke --run-name candidate-smoke
python -m scripts.vln.check_reload --checkpoint outputs/candidate-smoke/final \
  --manifest "$R2R_MANIFEST" --model-path "$MODEL_PATH"
```

Remove `--memory_config` for FullContext. The smoke explicitly uses zero warmup
and three updates; the production schedule guard is not applicable to this
debug-only run. Do not extrapolate whole-dataset time from eight shortest episodes.
Use a new output directory for a new run; recovery requires an explicit resume.

### Full training (four GPUs)

```bash
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/vln_r2r_v0_base.yaml \
  --output_config configs/vln_r2r_b_3epoch_bs8.yaml \
  --policy_config configs/vln_r2r_v0_candidate_logits.yaml \
  --memory_config configs/vln_memory_window8.yaml \
  --manifest "$R2R_MANIFEST" --model_name_or_path "$MODEL_PATH" \
  --recovery-save-steps 100 --output_dir outputs/r2r-window8-candidate \
  --run-name r2r-window8-candidate
```

R2R retains 3 epochs, global batch 8, GAS 2, LR 5e-6, 4,059 updates and 122 warmup
steps. For joint R2R+RxR_15deg, substitute
`configs/vln_joint_b_2epoch_bs8.yaml`, `$JOINT_MANIFEST`, and a fresh output
directory: 2 epochs and 7,704 updates, with its existing warmup/admission/offload
recipe. Remove the memory overlay for FullContext. `--policy_config` merges
after the production recipe but does not overwrite its token admission cap.
No new full training job is submitted by this implementation task.

Alternative complete policy overlays: `vln_candidate_canonical.yaml`,
`vln_candidate_sqrt.yaml`, `vln_candidate_effective.yaml`,
`vln_candidate_copied.yaml`, `vln_candidate_frozen.yaml` (all under `configs/`).
Only change one experimental factor at a time. For a recovery of the same run,
add `--resume_from_checkpoint auto` and retain the original manifest/config.

### Supervised held-out per-class evaluation

```bash
python -m scripts.vln.evaluate_actions --checkpoint "$CANDIDATE_CHECKPOINT" \
  --model-path "$MODEL_PATH" --manifest "$HELD_OUT_ACTION_MANIFEST" \
  --episode-limit 100 --out artifacts/candidate-supervised-100.json
```

The manifest must contain held-out, observation-before-action ground truth.
Do not label training-set STOP recall as generalization or closed-loop recall.

### Closed-loop Habitat: fixed 100 and full 1,839 val_unseen episodes

Use the existing Habitat environment/interpreter and camera/task config. Set
`HABITAT_PYTHON` to that interpreter (not the Qwen environment) and
`MODEL_PYTHON` to the validated Qwen interpreter. Fix the same episode-ID lists
for all policies. Derive the official list, once, from the dataset:

```bash
export DATA_ROOT=/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data
export JANUS_ROOT=/mnt/data/vmo-ai-task/anhdh35/JanusVLN
python -c 'import gzip,json,os,pathlib; p=pathlib.Path("artifacts"); p.mkdir(exist_ok=True); d=json.load(gzip.open(os.environ["DATA_ROOT"]+"/datasets/r2r/val_unseen/val_unseen.json.gz","rt")); ids=sorted({str(e["episode_id"]) for e in d["episodes"]},key=int); assert len(ids)==1839; (p/"val_unseen_all.json").write_text(json.dumps(ids)); (p/"val_unseen_100.json").write_text(json.dumps(ids[:100]))'
"$HABITAT_PYTHON" -m qwen_vl.eval.habitat_r2r \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --habitat-config "$JANUS_ROOT/config/vln_r2r.yaml" \
  --data-root "$DATA_ROOT" --simulator-source "$JANUS_ROOT/src" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --episode-list artifacts/val_unseen_100.json --out artifacts/candidate-val-unseen-100
```

For the full evaluation replace the list with `val_unseen_all.json` and use a
fresh output directory. Existing summary output includes SR, SPL, NE, oracle
success, mean steps, model/forced STOP, failures/invalid responses, action
distribution, model latency mean/median/p95, peak GPU memory, and retained KV.
Do not reuse a rollout journal from another checkpoint at the same path.

### Matched inference speed

Use `scripts/vln/benchmark_inference.py` sequentially for each checkpoint on the
same idle GPU, identical runtime, RGB and instruction. For the original report's
exact input, restore `artifacts/smoke_rgb.png` and verify SHA256
`e69cf147c19881da0e08cc69fe0204dc5de0f13fce8579eae5ecd78850558e27`.

```bash
python -m scripts.vln.benchmark_inference --family simple \
  --checkpoint "$CANDIDATE_CHECKPOINT" --base-model "$MODEL_PATH" \
  --image artifacts/smoke_rgb.png --instruction "$BENCHMARK_INSTRUCTION" \
  --warmup 16 --steps 64 --out artifacts/speed-candidate.json
python -m scripts.vln.benchmark_inference --family simple \
  --checkpoint "$WINDOW8_TEXT_CHECKPOINT" --base-model "$MODEL_PATH" \
  --image artifacts/smoke_rgb.png --instruction "$BENCHMARK_INSTRUCTION" \
  --warmup 16 --steps 64 --out artifacts/speed-text.json
```

Set `BENCHMARK_INSTRUCTION` to the instruction in
`R2R_INFERENCE_SPEED_2026-09-30.md`. Repeat with `--steps 500` for long-history
behavior. Repeat separately with `--components` for synchronized preprocessing,
vision, language append, four-row projection, feedback append and old text
autoregressive timing. Component instrumentation changes latency; headline
measurements must be uninstrumented. Feedback timers contain language append
time, so do not sum nested components. Removing generated tokens does not imply
a proportional whole-policy speedup.

## Ablation results: navigation experiments still required

| Output method | Action history | Readout | SR | SPL | STOP recall | ms/action |
|---|---|---|---:|---:|---:|---:|
| qwen_text, FullContext epoch 1 | generated text | full LM generation | 42.90% | 39.30% | not measured | 91.88* |
| qwen_text, Window8 epoch 1/intermediate | generated text | full LM generation | 41.11% | 36.86% | not measured | 84.57* |
| old classification | none | random Linear(d,4) | pending | pending | pending | pending |
| candidate_logits | candidate token | pretrained LM rows | pending | pending | pending | pending |
| candidate_logits control | canonical text | pretrained LM rows | pending | pending | pending | pending |
| copied_linear | candidate token | LM-initialized Linear(d,4) | pending | pending | pending | pending |

`*` Historical Blackwell measurements, not comparable to new H100 checks.
The old Window8 navigation/speed rows must retain the original checkpoint
provenance; do not assume the intermediate navigation snapshot and complete
epoch speed snapshot are interchangeable. **Not yet evaluated in Habitat** for
candidate policies. Full training and evaluated checkpoints are required before
claiming improved SR/SPL or a matched quality/speed tradeoff.
