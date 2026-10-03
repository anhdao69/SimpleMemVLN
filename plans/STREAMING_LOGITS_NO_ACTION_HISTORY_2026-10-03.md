# Streaming-logits architecture and FullContext no-action-history ablation

## 1. Scope, provenance, and meaning of this experiment

This report describes the implementation on `streaming_logits`, including the
existing candidate-logit architecture and the new explicitly history-free policy.
The task starts at commit `a86f2329c69f2e3df9f8e797961bbf56143e1a89`.
Implementation commit: `cd5ac95` (explicit no-history contract, serializer,
configuration, tests and existing-allocation smoke runner). This report is committed
separately after collecting the final measurements.
The original candidate architecture was introduced from streaming base
`341ec2cfce934a872c4478cb06c4f70354113bc4`; these are different provenance points.

The requested experiment is **candidate-logit decisions + FullContext + no explicit
action history**, not a new random classifier, a memory reset, or an observation-only
single-frame policy. The language decoder still sees earlier observations and the
instruction. Its Gated DeltaNet (GDN) recurrent state is retained. Only the
class-dependent action content previously fed back between observations is removed.

The approved implementation keeps the fixed assistant-turn terminator and newline.
These structural tokens carry no selected-action identity. Removing them as well
would be a different prompt/framing ablation and is not what was approved.

Execution is restricted to the user's existing interactive Slurm allocation **4652**,
on **worker-3**, with four H100 80GB GPUs. No new `sbatch`, `srun`, allocation request,
or cancellation is part of this work. Other running training sources are untouched.
The server uses an isolated source snapshot under:

```
/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/no_history_full_20261003-OqSlcs/source
```

The corresponding validation artifacts are in sibling `smoke/` (initial run) and
`smoke-clean/` (final validated rerun) directories.
The tracked local worktree is
`/Users/hoanganh692004/Desktop/SimpleMemVLN-streaming_logits`.

## 2. Why candidate logits exist

The text policy runs the full vocabulary LM head repeatedly, chooses a token,
appends it through the decoder/cache, and continues until the assistant terminator.
It then decodes the generated tokens and applies the strict navigation parser.
The vocabulary is not the four-action navigation set: a string such as `MOVE_RIGHT`
can therefore be generated but rejected by the parser. Making parsing permissive
would not eliminate generation overhead or define the intended navigation decision.

The original classification mode uses a randomly initialized `Linear(2560, 4)`.
It is valid as an ablation but lacks the pretrained LM readout geometry. High initial
loss and majority-action behavior are plausible; neither is proof of an optimizer
bug. In particular, low overall CE or accuracy dominated by MOVE_FORWARD must not
be mistaken for useful STOP behavior.

Candidate logits instead reuse Qwen's existing vocabulary rows for four verified
labels. The system prompt explicitly defines their semantics:

| Class | Label | Token ID | Canonical navigation action | Habitat ID |
|---:|---|---:|---|---:|
| 0 | A | 32 | MOVE_FORWARD | 1 |
| 1 | B | 33 | TURN_LEFT | 2 |
| 2 | C | 34 | TURN_RIGHT | 3 |
| 3 | D | 35 | STOP | 0 |

`contracts.py` remains the canonical action/Habitat mapping. The labels do not
intrinsically mean navigation actions in the pretrained model; the prompt supplies
that mapping. Reusing pretrained rows is an experimentally clean initialization,
not a guarantee of better SR, SPL, calibration, or rare-class recall.

## 3. Model architecture and readout

The base is `Qwen/Qwen3.5-4B`, pinned to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with verified Transformers 5.11.0.
`models/nav_model.py` is a thin wrapper, not a replacement transformer.
It delegates multimodal positions and language computation to the native backbone.
The native decoder is hybrid: full-attention layers coexist with recurrent GDN
layers: the pinned configuration has 32 decoder layers, with three GDN layers
followed by one full-attention layer, repeated eight times (24 GDN, 8 attention).
Full-attention dimensions are 16 query heads, 4 KV heads and head dimension 256;
GDN uses 16 key heads and 32 value heads with dimension 128 and convolution width 4.
Consequently, **FullContext does not mean converting every GDN layer into
attention**. It specifies the history visibility of the existing attention layers.

Each RGB observation is 640×480. The processor must produce grid `[1,30,40]` and
300 expanded visual tokens. The wrapper embeds textual tokens and replaces image
placeholder embeddings with frozen vision features. Vision runs under `no_grad`,
in image microbatches of four, and remains in evaluation mode even when the language
model is training. The visual module, including its merger, remains frozen.

For a decision-position hidden state `h_t`, the candidate output is:

```
indices = [32, 33, 34, 35]
W_action = lm_head.weight.index_select(0, indices)
b_action = selected LM bias, if one exists
logits_t = F.linear(h_t, W_action, b_action)
class_id = argmax(logits_t)
```

The pinned head has shape `[248320,2560]`, no bias, and is tied to the input
embedding parameter. Projection produces four scores, not a vocabulary-sized
logit tensor. Training selects one hidden state per supervised action and projects
those states to `[number_of_actions,4]`; serving projects a single state to `[4]`.
There is no autoregressive action generation, generated EOS, text parsing, or
16-token response loop on this path.

### Head configurations and tied weights

| Mode | Readout and learning behavior |
|---|---|
| `lm_rows_trainable` (default) | Current selected LM rows, normal backbone LR `5e-6`; no separate classifier parameter |
| `lm_rows_frozen` | Entire LM-head parameter frozen, which also freezes input embeddings when tied |
| `copied_linear` | Four rows and optional bias copied into a dedicated trainable linear layer; `classifier_lr` applies |

Trainable direct rows do not imply that only four vocabulary rows can change.
Input-embedding gradients and AdamW act on the shared parameter. Conversely,
detaching a readout alone would not truly freeze shared rows if embeddings still
updated them. The explicit frozen mode intentionally freezes that entire shared
parameter. The new no-history configuration changes none of these rules.

## 4. Exact tokenizer and framing contract

`data/candidates.py` validates all four candidates at startup: cardinality,
uniqueness, a single token per label, nonspecial status, exact decode round-trip,
and concatenation behavior in the actual observation prompt plus closing suffix.
Checking isolated `encode("A")` is insufficient: whitespace or preceding text
can change token boundaries.

The decision cue is exactly `Action:\n`, not `Action:` or `Action: `.
The serializer uses the existing non-thinking Qwen chat template, validates known
structural token IDs, and records the template hash in navigation metadata.
The full step ends after an assistant-generation prefix and that decision cue.
The decision hidden state is the final token of this observation/cue block.

The no-history test additionally verifies that directly closing this cue with
`<|im_end|>\n` preserves the same separately serialized token boundary. The suffix
contains the assistant terminator token 248046 and the newline separator. It is
fixed protocol framing, not a sampled model response and not a supervised CE target.

## 5. History versus no-history: exact differences

All candidate policies use the same static action legend and decision cue.
The new ablation deliberately leaves that prompt unchanged. Static text naming
the action space is not a record of previous actions. The sentence saying previous
actions *may* be recorded is also preserved to avoid an additional prompt change.

| Output mode / feedback | Decision | Content appended after decision | Explicit prior-action input? |
|---|---|---|---|
| `qwen_text` | Vocabulary autoregression | Generated canonical string plus closing suffix | Yes |
| Old `classification` | Random-initialized linear readout | No action feedback; distinct legacy framing | No |
| `candidate_logits` / `candidate_token` | Four selected LM rows | A/B/C/D plus fixed closing suffix | Yes |
| `candidate_logits` / `canonical_action_text` | Four selected LM rows | Canonical string in one deterministic append plus suffix | Yes |
| `candidate_logits` / `none` | Four selected LM rows | Fixed closing suffix only | **No** |

Conceptually, history-enabled input is:

```
instruction + static action legend
observation_0 + Action:\n + B + fixed_closure
observation_1 + Action:\n + A + fixed_closure
observation_2 + Action:\n ...
```

The no-history sequence is:

```
instruction + the same static action legend
observation_0 + Action:\n + fixed_closure
observation_1 + Action:\n + fixed_closure
observation_2 + Action:\n ...
```

For identical instruction and observations, replacing any nonfinal gold action
with another valid non-STOP action must change `action_class_ids` but **not**
`input_ids`, multimodal token types, pixels, grids, read positions, or step spans.
This is tested against the real tokenizer/processor. History-enabled formats are
positive controls: changing the target changes their serialized history.

At serving time, changing the selected class must likewise leave the appended
input tokens unchanged in no-history mode. The output API still reports the chosen
class/action/Habitat ID, and the environment executes it. This does not imply that
different actions lead to identical future observations: physical ego-motion still
changes subsequent images. No-history removes the *explicit symbolic* action input,
not all indirect evidence about motion.

Candidate-token history costs one action token per step; no-history removes exactly
that one token while retaining the same closure. It does not remove the 300 visual
tokens or most model computation. In particular, the fixed closure still needs a
language-model append during serving. A near-zero readout cost is not near-zero
total decision cost.

## 6. Shared training and streaming causality

`EpisodeSerializer.encode_episode()` builds the instruction prefix, then each
observation block, then the selected feedback format. `read_positions` points to
the last observation/cue token **before** feedback. `action_class_ids` is a separate
target tensor. Candidate mode has no text-response target positions.

Training is an uncached complete-episode forward with causal visibility. A gold
action at step t cannot affect the prediction at t. With action history, it can
affect subsequent decisions through teacher forcing. With no history, it is never
inserted into the input at all. The target still affects loss and parameter gradients;
that is supervision, not input leakage.

`StreamSession.observe()` follows the corresponding online sequence:

1. Validate episode/step identity; return the previous cached result on a retry.
2. Encode the current image and reserve space for the complete observation/closure.
3. Evict expired KV groups only if using Window8.
4. Append the observation once; compute four candidate logits at its final state.
5. Verify finite scores and `[4]` shape; select the canonical class by argmax.
6. Append deterministic feedback—or only the fixed closure in no-history mode.
7. Record the group span, advance the step, and return the navigation result.

No-history requires no special branch in the model, Habitat evaluator, or streaming
engine: the shared `feedback_ids(class_id)` method returns the same closure for
every class. Training and serving use precisely this function.

Inference commits feedback before returning the action to its caller, as before.
If external action execution fails, the caller must reset rather than continue from
an unexecuted action history. Removing explicit feedback does not redefine this API.
An exception invalidates the session; retrying a successful step is idempotent and
does not append tokens or advance GDN twice. Reset creates new KV/GDN/position state.

Offline chunk kernels and incremental recurrent kernels are not bitwise identical
in BF16. The native integration gate measures logit differences with explicit
tolerances; matching causal structure is not a claim of exact floating-point equality.

## 7. FullContext, Window8, GDN and positions

FullContext retains the instruction and all previous observation/closure groups in
the attention cache. It does not evict. The FIFO resident-group deque is only needed
for Window8 and remains empty in FullContext; its emptiness does not mean lost KV.

Window8 retains the fixed prefix plus the most recent eight complete groups,
including the current group. Before step t it evicts groups older than t−7.
Each group contains both its observation and its deterministic feedback/closure.
For no-history the group is shorter but still includes its closing boundary.
Tokens are never left behind merely because they are not visual tokens.

`stream/cache.py` evicts keys/values only from full-attention cache layers. It does
not clear GDN recurrent state or convolution state. Recurrent state is explicitly
stored in FP32 by `FP32LinearAttentionLayer`, even though model weights and much
activation computation use BF16. Therefore Window8 is a bounded-attention-KV policy,
not an eight-step limit on every form of model memory.

Training Window8 uses `simplememvln_step_attention`, retaining prefix plus eight
step spans without constructing a dense production attention mask. Its custom
backward recomputes windows and accumulates gradients into shared Q/K/V buffers.
FullContext instead uses native FlashAttention2. Neither path is changed here.

`PositionLedger` separately tracks total logical token count and native multimodal
MRoPE coordinates. Every observation and closure advances these cursors. Evicting
KV must not reset them. Removing action tokens correctly changes later positions;
reusing positions from a history-enabled episode would be wrong. The serializer's
computed lengths and actual block spans remain the source of truth.

Decoder activation checkpointing is non-reentrant. The production recipe also
offloads checkpoint inputs for sequences of at least 65,536 tokens. Rotary frequency
buffers remain FP32 across module BF16 casts to preserve checkpoint reload behavior.
No-history does not change activation-offload, visual freezing, recurrent kernels,
precision policies, or temporal differentiation.

## 8. Training objective, imbalance and diagnostics

Each supervised action contributes a four-way cross entropy computed in FP32.
Uniform logits give `log(4) = 1.38629436`. Pretrained logits need not be uniform;
initial CE above this number is not automatically a defect. The old qwen-text
objective averages token CE within each action string, then over actions, so its
numeric loss is not directly comparable to candidate four-way CE.

The default is unweighted CE. Optional weights are square-root inverse frequency
or effective-number weights. For counts n, probabilities p=n/sum(n), raw weights u:

```
sqrt inverse:      u_c = 1 / sqrt(n_c)
effective number:  u_c = (1-beta) / (1-beta**n_c)
normalized:        w_c = u_c / sum_j(p_j * u_j)
constraint:        sum_c(p_c * w_c) = 1
```

This is not arithmetic-mean normalization across four classes. Counts are resolved
from the actual selected manifest. Balanced modes reject absent classes. No-history
does not change labels or the count used as the distributed loss denominator.

For N actions across all ranks and accumulation slots, the intended objective is
`sum_i(w_yi * CE_i) / N`. Each rank supplies its local loss sum times world size/N,
compensating for distributed gradient averaging. Trainer/DeepSpeed must not divide
again by GAS. Global episode batch 8 means 4 ranks × 1 episode × GAS 2; it does not
mean eight actions or equal action counts per GPU.

`action_metrics.jsonl` globally aggregates the 4×4 confusion matrix, per-class
precision/recall, overall accuracy, macro accuracy (mean of four recalls; absent
classes count as zero), target/predicted distributions, and weighted/unweighted CE.
STOP recall here is supervised recall on the sampled observations, not Habitat SR.
Exposure logs retain per-episode loss sums and global action denominators. Timings
and GPU memory are recorded separately for every rank.

## 9. Configuration and checkpoint compatibility

New complete policy overlay: `configs/vln_candidate_logits_no_history.yaml`:

```yaml
model:
  output_mode: candidate_logits
  action_head_mode: lm_rows_trainable
observations:
  serializer_version: vln_candidate_logits_no_action_history_v1
  append_action_tokens: false
  feedback_format: "none"
```

It also retains the 384-token step admission cap, unweighted CE, and disabled
assistant-terminator supervision. It does not override the production learning
rate, global batch, epoch count, warmup, or maximum episode length.

`contracts.serializer_version(config)` selects the new version only for candidate
mode with `feedback_format == "none"`. Legacy classification, qwen-text, and both
history-enabled candidate versions keep their existing version strings. Config
validation rejects contradictory serializer/history flags and unknown feedback.

Navigation metadata records the template hash, action map, candidate IDs, cue,
feedback format, all four feedback sequences, head mode, positions/precision
conventions, and full resolved configuration. In no-history metadata all four
feedback sequences are identical. Checkpoint loaders and resume validation compare
these contracts strictly; identical weight shapes do not make the policies
interchangeable. A history-enabled checkpoint must not silently resume as no-history.

This experiment starts from the pinned pretrained model, not a previously trained
history policy. Deliberate cross-policy warm starts would need a separately named,
audited conversion experiment. Raw `load_state_dict` alone bypasses these semantic
checks and is not the supported navigation-checkpoint loading workflow.

## 10. Validation protocol and evidence

Baseline local test suite: **87 passed, 10 skipped**. Four new tests were first run
against unchanged production behavior and failed on the missing no-history contract
and action-bearing feedback. After implementation and expanded exact-tokenizer
tests: **92 passed, 12 skipped** locally. The extra skips are the new real-tokenizer
parameter cases; they subsequently ran in the server allocation, not counted as local passes.
After the temporary-directory runner regression was added, the local suite was
**93 passed, 12 skipped**. The first server run executed all earlier cases:
**104 passed, zero skipped**, with 73 dependency warnings, in 85.80 seconds.

Tests cover explicit contract validation, resume incompatibility, class-independent
feedback, all four action outputs, zero generated tokens, retry idempotency,
FullContext accumulation, Window8 complete-group accounting, and preservation of
the joint production schedule. Real-tokenizer cases compare every model input after
changing targets and verify exact closing-token concatenation.

The existing suite also exercises projection equivalence to full-head slices,
readout gradients/tied weights, optional head modes and bias, CE=log(4), class-weight
normalization, old text decoding, distributed loss normalization, recovery, rotary
precision, activation offload, and Window8 attention/gradient behavior.

### Four-GPU smoke recipe

- Existing allocation 4652; worker-3; four otherwise idle H100 80GB GPUs.
- Eight fixed episodes: four `r2rce`, four `rxrce` from the English RxR_15deg joint data.
- 384 supervised actions/pass; longest selected trajectory: 113 observations.
- Manifest SHA256: `7ae5d831cfaf19634e798c7c16eb4e8c7dd27caf72712e14bb1c5acc7041222f`.
- Global batch 8; four ranks × microbatch 1 × GAS 2; LR `5e-6`.
- Three optimizer updates, seed 429, existing cosine-with-min-LR scheduler.
- Zero warmup **only for this smoke**, to test actual parameter updates immediately.
- Four loader workers/rank, OMP/MKL two threads; expandable CUDA segments.
- DeepSpeed ZeRO2; BF16; frozen vision; non-reentrant decoder checkpointing.
- Profile-only: no large optimizer/model checkpoint is saved by the smoke trainer.

The native integration gate separately performs a backward pass from a final-action
loss and checks early-observation/GDN/full-attention gradients and frozen vision.
It then runs 12 streaming observations, prohibits full-vocabulary head calls,
checks no-history closure, retries, FP32 recurrence, FullContext KV retention,
offline/streaming numerical parity, and fresh reset reproducibility.

### Final measured results: corrected runner

Server suite: **105 passed, zero failures/skips**, 73 dependency warnings, 16.11 s.
The complete runner returned zero. All three optimizer updates had finite loss and
gradient norms; no CUDA OOM, Python traceback, NFS EBUSY cleanup error, or unwritable
kernel-cache warning appeared in the corrected training/integration logs.

| Update | Four-way CE | Max update s across ranks | Max allocated GiB | Max reserved GiB | Overall accuracy | Macro accuracy | STOP recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.077797 | 9.0434 | 30.1217 | 38.2910 | 18.23% | 24.36% | 62.50% |
| 2 | 1.117108 | 6.8368 | 35.7200 | 41.0879 | 51.30% | 33.10% | 0% |
| 3 | 2.622760 | 6.8041 | 36.3133 | 41.2266 | 57.29% | 25.00% | 0% |

Pre-clipping gradient norms were 660.374, 87.399, and 449.528; configured global
gradient clipping is 1.0. Logged learning rates were 5e-6, 3.875e-6, and 1.625e-6.
Trainer runtime was **33.3262 s**; training-entrypoint wall time including model
loading was **46.5713 s**. These times exclude the preceding test suite and following
native integration process. Per-update callbacks exclude loader prefetch, model load,
and saving, so their sum is intentionally less than Trainer runtime.

Five-second GPU telemetry sampled within the update windows averaged 100.0%,
99.875%, and 100.0% utilization (4, 8, and 8 device samples respectively). These are
coarse snapshots, not a continuous utilization integral. The initial uncorrected
run's first update took about 107.95 s with cold runtime compilation; that is not
pooled with the clean warmed-runtime update times or called a model speedup.

Target counts on every tiny dataset pass were `[220,74,82,8]` in canonical order.
Predicted counts reveal the limitations directly:

| Update | MOVE_FORWARD | TURN_LEFT | TURN_RIGHT | STOP |
|---:|---:|---:|---:|---:|
| 1 | 99 | 6 | 2 | 277 |
| 2 | 235 | 96 | 53 | 0 |
| 3 | 384 | 0 | 0 | 0 |

The initial CE exceeds log(4), but the pretrained predictor is strongly nonuniform:
72.14% of first-update predictions are STOP although only 2.08% of targets are STOP.
Wrong confident predictions can give CE above the uniform baseline. Input-independence,
target mapping, projection, and distributed normalization checks passed. This is
evidence of a poorly calibrated tiny-start policy, not evidence of token leakage.
After two parameter updates the third batch is all-forward; STOP recall is zero.
**The smoke passes execution/causality/memory checks, not convergence or policy quality.**
There is no claim of monotonic loss. Longer controlled training and held-out/closed-loop
evaluation are necessary, preferably comparing the unweighted baseline with a separately
specified mild-balancing experiment rather than silently changing this recipe.

The `epoch: 3` line is three passes over the **eight-episode debug dataset** because
`--max-optimizer-updates 3` overrides its nominal epoch limit. It is not three full
R2R/RxR epochs or a modification of the production two-epoch recipe.

### Native FullContext integration measurement

- Passed 12 observation decisions with zero generated action tokens and fixed
  class-independent feedback; no invalid textual action path exists.
- All 12 streaming argmax decisions agreed with replayed offline decisions.
- Maximum logit difference 0.25; RMS difference 0.139463, below the 0.15 gate.
- Fresh reset reproduced the first decision logits exactly; retries did not append.
- Initial CE on the separate short integration episode: 0.956837.
- Final-action backward reached early-observation embeddings (gradient norm 17.2938),
  first GDN `in_proj_b.weight` (6.22472), and first attention `q_proj.weight` (35.2595).
- Vision stayed frozen/eval with no gradients; recurrent state stayed FP32.
- Peak reserved memory in this separate one-model integration process: 19.4258 GiB.

Relevant artifacts under `smoke-clean/`: `tests.log`, `train.log`, `allocation.json`,
`smoke_recipe.yaml`, `gpu_utilization.csv`, `integration.log`, `integration.json`,
and `train/{navigation.json,resolved_config.json,trainer_state.json,run_timing.json,
action_metrics.jsonl,profile_rank0..3.jsonl,exposures_rank0..3.jsonl}`.

### First-run environment issue and corrective rerun

The first run returned exit code zero and completed its three updates and native
integration gate, but emitted multiprocessing-finalizer `OSError: [Errno 16] Device
or resource busy` tracebacks under the inherited NFS directory
`/mnt/data/vmo-ai-task/anhdh35/tmp/pymp-*`. This was not a training-step exception,
OOM, or nonfinite-loss event. The kernel cache also pointed to an unwritable shared
directory. These messages were not suppressed or counted as a clean validation.

The runner was corrected to create its own `/tmp/smv-nohistory-4652-XXXXXX` directory
on the worker and place `PYTORCH_KERNEL_CACHE_PATH` inside it. It does not delete or
modify the inherited shared temporary tree. A regression test executes the real
shell setup with an inherited temp path, first fails on the original behavior, then
verifies that the corrected setup creates a different node-local path and writable
kernel directory. The entire suite/training/integration sequence is rerun into
`smoke-clean/`, preserving original logs under `smoke/`.
The corrected rerun completed without those errors. Temporary storage is scoped
to this invocation and left on the worker for inspection; no broad shared cleanup
or environment/dependency installation was performed.

Two pre-existing runtime warnings remain disclosed. Transformers emits a generic
"fast path unavailable" warning: direct inspection of its loaded bindings found
`causal_conv1d_fn=None` and `causal_conv1d_update=None`, while chunk and recurrent
GDN functions come from `fla.ops.gated_delta_rule.chunk` and
`fla.ops.gated_delta_rule.fused_recurrent`. Thus optional fused causal convolution
uses the native PyTorch fallback; the GDN kernels themselves are FLA, as enforced
by the loader. No extension was installed or runtime changed to hide that warning.
DeepSpeed also warns that its pre-existing Triton autotune cache is on NFS;
the clean process exited normally. These are distinct from the corrected Python
multiprocessing temp-directory and PyTorch kernel-cache problems.

After validation, all four visible GPUs reported 0 MiB usage and 0% utilization.
Allocation 4652 and unrelated jobs 4649/4643 remained running. The original interactive
shell was returned to its prompt; no allocation was submitted, resized, or cancelled.

### Review and coverage boundaries

Independent read-only review found no Critical or Important issue in the policy
change or initial runner. One minor test limitation is deliberately recorded:
the new CPU Window8 no-history controller test uses an empty mock cache, so it
checks group lengths/FIFO/positions but does not itself exercise tensor KV removal.
Existing native eviction tests exercise real cache tensors, while the new native
integration run specifically covers the requested **FullContext** ablation.
No new no-history Window8 native-run result is claimed.

The native integration gate starts from pretrained weights, not the transient
three-update smoke weights. Profile-only training does not publish a checkpoint;
this task therefore does not claim a new real-model no-history checkpoint save/reload
measurement. Existing synthetic tied-readout reload tests and strict metadata/resume
tests ran. A saved trained no-history checkpoint still needs its own production
save/reload gate before use in a navigation experiment.

## 11. How to reproduce and what was not launched

From an isolated checkout in an already allocated four-GPU shell:

```bash
bash scripts/vln/smoke_no_history_interactive.sh \
  /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/NEW_NO_HISTORY_SMOKE \
  "$SLURM_JOB_ID"
```

The output directory must not already exist. The runner asserts the expected job
ID and four visible GPUs, preserves `CUDA_VISIBLE_DEVICES`, runs the exact-tokenizer/
GPU suite, then launches four local workers with `torchrun`. Torchrun is a process
launcher inside the allocation, not a new Slurm resource request. The script does
not call `sbatch` or `srun`. It leaves the interactive shell/allocation intact.

For a future full training experiment, use the production joint schedule and the
new policy overlay, omitting the Window8 memory overlay:

```bash
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/vln_r2r_v0_base.yaml \
  --output_config configs/vln_joint_b_2epoch_bs8.yaml \
  --policy_config configs/vln_candidate_logits_no_history.yaml \
  --model_name_or_path "$VLN_MODEL_PATH" \
  --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl \
  --output_dir outputs/joint-fullcontext-no-history \
  --run-name joint-fullcontext-no-history --recovery-save-steps 100
```

This command is documentation, **not a submitted training run**. It preserves
two epochs, global batch 8, LR 5e-6, 7,704 optimization steps, and 232 warmup steps.
Long-episode/host-RAM admission and recovery validation are still required before
launching that production experiment. The interactive allocation has 480GiB host
RAM, not the 768GiB production allocation; a tiny smoke cannot validate the latter.

## 12. Interpretation, limitations and next experiments

No-history still has visual temporal memory and persistent GDN. It tests whether
explicit previous action identity helps beyond what the instruction and observation
stream provide. It does not isolate every positional effect: removing a token per
step changes subsequent coordinates and input length, which is part of this ablation.
The sampler buckets by encoded length, so a full-dataset comparison should record
episode order and consider a fixed-order control if small rank/order shifts matter.

The clean comparison holds backbone initialization, dataset, split, action mapping,
optimizer schedule, global batch, seed, attention mode, and evaluation episodes
constant, then changes feedback among candidate token, canonical text, and none.
Teacher-forced history has exposure bias; no-history removes that explicit channel
but does not remove distribution shift in closed-loop images.

**Not yet evaluated in Habitat** for this no-history policy. No SR/SPL/NE/oracle
success, held-out STOP recall, full-run convergence, or trained-policy speed claim
is established by this smoke. Three tiny updates do not justify a full-dataset
training-time estimate or a longest-episode no-OOM claim.

For later speed comparison, use the existing `scripts/vln/benchmark_inference.py`:
one policy at a time, identical hardware/runtime/image/instruction, 16 warmups and
64 CUDA-synchronized complete actions, reset/load/simulation/IPC excluded. Repeat
component timing separately. The existing `action_history_append` timer includes
the deterministic closing append even when history is disabled; its historical
name must not be interpreted as evidence of action content. It nests language
append time and must not be summed with that timer. The old Blackwell measurements
in `reports/R2R_INFERENCE_SPEED_2026-09-30.md` are not directly comparable to H100
measurements. No assumption of a 3× speedup from removing three generated tokens.

## 13. Code map

| File | Responsibility / effect of this change |
|---|---|
| `src/qwen_vl/contracts.py` | Canonical action maps, config/resume contracts; new feedback-dependent serializer version |
| `src/qwen_vl/data/candidates.py` | Unchanged candidate prompt and exact-context token validation |
| `src/qwen_vl/data/episode_serializer.py` | Shared observation/feedback serialization; new class-independent closure and metadata |
| `src/qwen_vl/data/episode_dataset.py` | Unchanged length accounting automatically consumes shorter feedback |
| `src/qwen_vl/models/nav_model.py` | Unchanged frozen vision, native hidden states, selected LM rows and loss |
| `src/qwen_vl/models/action_loss.py` | Unchanged balancing and action CE |
| `src/qwen_vl/stream/session.py` | Unchanged decision/append/reset/retry state machine |
| `src/qwen_vl/stream/{cache,positions,window_attention}.py` | Unchanged hybrid cache, eviction, position and training attention logic |
| `src/qwen_vl/train/{trainer,train_episode,vln_runtime}.py` | Unchanged optimizer/distributed loss, metrics, strict checkpoint workflow |
| `configs/vln_candidate_logits_no_history.yaml` | New explicit no-history policy overlay |
| `tests/vln/test_no_action_history.py` | New contract and streaming invariance tests |
| `tests/vln/test_candidate_contract.py` | Exact-tokenizer no-history and action-target perturbation tests |
| `tests/vln/test_no_history_runner.py` | Executes runner setup to reject inherited shared temporary storage |
| `scripts/vln/check_candidate_integration.py` | Native integration gate accepts and checks no-history mode |
| `scripts/vln/smoke_no_history_interactive.sh` | Bounded existing-allocation-only GPU validation runner |

No checkpoint, dataset, model weight, token credential, or large validation artifact
belongs in this code commit. Raw logs remain on the server; this report records their
provenance and measured conclusions separately from architectural expectations.

## 14. Delivery verification

Final local command:

```bash
PYTHONPATH=src:. ACCELERATE_USE_CPU=true OMP_NUM_THREADS=2 \
  /tmp/simplememvln-logits-test-env/bin/python -m pytest -q
git diff --check
bash -n scripts/vln/smoke_no_history_interactive.sh
```

Result: **93 passed, 12 skipped in 5.21 s**; whitespace and shell syntax checks
returned zero. Server final command was the documented runner inside allocation
4652, output directory `smoke-clean`; its full suite was **105 passed** and its
training/integration stages returned zero. Logs were independently checked for
tracebacks/OOM, three profile records on every rank, finite CE, and peak reserved
memory below 78 GiB on this selected sample.

The only model-path changes are in the central contract and serializer. No new
navigation readout, attention kernel, optimizer, or cache implementation was added.
The existing linked worktree and server validation snapshot are preserved. This
task does not merge into `streaming`, launch production training, or publish weights.
