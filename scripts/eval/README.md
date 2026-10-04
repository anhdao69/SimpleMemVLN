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
