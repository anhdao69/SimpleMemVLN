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

`action_metrics.jsonl` records globally summed confusion counts, precision and
recall per class, overall accuracy, macro accuracy (mean four-class recall;
absent classes count as zero), predicted/truth distributions, and weighted and
unweighted CE. Recovery rewinds this journal along with existing update logs.
STOP recall here is teacher-forced supervised recall. It is not a closed-loop
Habitat oracle metric; rollout reports model STOP and forced STOP separately.

A uniform predictor has CE `log(4)=1.386294`. Pretrained logits are not uniform.
Four-way CE must not be numerically equated with token-level text-policy CE.

## Verification evidence

Local CPU suite: 63 passed, 10 tokenizer/GPU-dependent skips. Remote GPU suite
before the final profiling additions: 66 passed, zero skips. Final verification
results and smoke details are recorded below when complete.

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
maximum logit differences were 0.25–0.3125. The gate checks numerical bounds and
high-margin decisions, not just matching argmax. These are short-episode runtime
checks, not full-dataset memory admission or navigation results. The initial
loss difference between feedback formats shows why this control matters.

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
  --data-root "$DATA_ROOT" --simulator-source "$JANUS_ROOT" \
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
