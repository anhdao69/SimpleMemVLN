# R2R streaming checkpoint evaluation

Use `qwen_vl.eval.habitat_r2r` with Habitat 0.2.4 in its own Python environment
and the validated Qwen Python environment supplied through `--model-python`.
The model process loads `navigation.json` through `load_checkpoint`, including
strict serializer metadata and state-dictionary validation. Do not override the
checkpoint's output mode, memory mode, or action-feedback settings.

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet

"$HABITAT_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$BASE_MODEL" \
  --habitat-config "$PWD/configs/eval/vln_r2r.yaml" \
  --data-root "$JANUS_DATA_ROOT" --out "$OUTPUT_DIR"
```

For a smoke test, add `--max-episodes 1` and use a separate output directory.
The full split contains 1,839 episodes. Resume with the identical command and
output directory; the evaluator rejects changed source, checkpoint, data,
protocol, or runtime contracts. Each concurrent run needs its own output directory.

The evaluation uses 640×480 RGB, 0.25 m forward motion, 15-degree turns, seed 42,
a 3 m success threshold, and a 500-action cap. The journal preserves the model's
prediction separately from a forced final STOP. Failures remain in the SR/SPL
denominator. RGB transport is lossless. The same scene-grouped episode order is
used for every policy.

For Window8 candidate logits, retain the instruction prefix and the latest eight
complete observation/feedback groups in full-attention KV. Append the selected
candidate token followed by the fixed closing suffix. GDN state and logical
positions continue across the episode.

For FullContext NoHistory candidate logits, retain visual history and GDN state,
and append only the class-independent `[248046, 198]` closing suffix. Never insert
the predicted candidate token or canonical action text. The current serializer
implements this through `feedback_ids`; the Habitat loop uses the same API for
both policies. Runtime JSON records the serializer and all four feedback sequences,
and per-action journal entries retain the actual feedback tokens.

Concurrent rollouts provide navigation metrics. Their recorded latencies include
GPU contention and must not be compared with the isolated inference-speed table.

## Dual-lane FullContext NoHistory exports

Use the `streaming_logits_dual` model/session implementation and the same Habitat
adapter above. `load_checkpoint` constructs lanes from `navigation.json`, then
strictly restores the complete exported state dictionary, including trained lane
weights. Do not use the training-only `initialize_step_lane_from_policy` API to
evaluate an Adapted export: that would initialize fresh lanes.

Both published models use candidate logits, class-independent `[248046, 198]`
closure, full visual KV history, and `step_end` lanes at layers 16/20/24/28. The
lane reads on every token and writes only on the final existing closure token,
after the current decision. Each session owns four FP32 `[1,4,128,128]` state
banks (1 MiB total), reset for every episode. Native GDN and positional state are
also retained within an episode. Model/runtime JSON records the lane spec and
parameter count; action records include actual state dtypes, bytes and cursors.

Before a new export's Habitat smoke, run this numerical/lifecycle audit in the
model Python environment, with `PYTHONPATH=src` and the same pinned base model:

```bash
"$MODEL_PYTHON" scripts/vln/check_dual_evaluation.py \
  --checkpoint "$CHECKPOINT" --model-path "$BASE_MODEL" \
  --image "$REAL_HABITAT_RGB" --out "$AUDIT_JSON"
```

The audit checks trained nonzero output projections, unchanged state on
nonwriter appends, one closure writer per layer after each decision, FP32 state,
retry idempotency, exact first-decision reproduction after reset, and
whole-sequence/streaming score agreement using the branch's declared tolerances.
Its four repeated frames are a correctness fixture, not a navigation score.
Then run at least two Habitat episodes in a separate smoke output directory to
exercise real scene observations and episode reset, followed by the full split.

The published epoch-1 FromBase and Adapted exports are both at update 3,852 of
7,704-update schedules. FromBase starts from Qwen and uses 232 warmup updates.
Adapted starts from the completed one-pass native NoHistory policy, giving it
one additional parent pass of data exposure; its adaptation schedule was
extended at checkpoint 100, retaining 116 warmup updates. Compare them as
checkpoint results, not a controlled initialization-only or lane-only ablation.
The older implementation report describes the original one-pass adaptation
plan; the published checkpoint metadata/training provenance describes the
actual extended schedule used by these exports.
