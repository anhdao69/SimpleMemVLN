# State capture and G-3

This additive module implements TASK_SPEC section 3.5. Synthetic CPU tests pass;
no GPU/model execution was performed while developing it. The actual G-1 and
G-3 results remain unmeasured until the main task runs the gates.

```bash
OMP_NUM_THREADS=1 /storage/anhdh35/SimpleMemVLN/.venv/bin/python -m unittest discover -s scripts/diag -p test_state.py
```

Load the checkpoint through the original runtime, set `model.eval()`, and encode
the same episode once. Keep normal model arguments unchanged (in particular,
do not pass `episodic_diagnostics` to Window8 models). Integrate as follows from
the diagnostic repository root with `PYTHONPATH=src:.`:

```python
import torch
from scripts.diag.state import LAYERS, LayerInputCapture, layout_from_encoded, reconstruct

layers = model.backbone.model.language_model.layers
with torch.inference_mode():
    plain_logits = model(**encoded)['logits'].float().cpu()
    with LayerInputCapture(layers) as capture:
        instrumented_logits = model(**encoded)['logits'].float().cpu()
if not torch.equal(plain_logits, instrumented_logits):
    raise RuntimeError('G-1 failed: captured forward changed logits')

layout = layout_from_encoded(encoded, serializer.image_id)
for layer_index in LAYERS:
    record = reconstruct(layers[layer_index], capture.hidden.pop(layer_index),
                         layout, check_fla=True)
    # Save record before raising, so a failed gate retains its evidence.
    torch.save(record, scratch / f'layer_{layer_index}.pt')
    if record['gate_g3']['status'] != 'PASS':
        raise RuntimeError(f'G-3 failed for layer {layer_index}')
```

Run the prescribed two panel episodes per model. Gate G-1 also needs any other
hooks used by the final diagnostics installed simultaneously; this example
only checks the state-capture hooks. Use `check_fla=False` for remaining panel
measurements after G-3 passes. Output paths and metadata (model, episode UID,
source/weights hashes) are owned by the caller. Do not overwrite prior outputs.

`LayerInputCapture` registers forward prehooks on layers 8,16,22,24,26; each
copies input `[1,T,width]` to independent CPU `[T,width]` storage and returns
None. Hooks are removed on normal and exceptional exit. More than one execution
per layer is rejected, preventing accidental mixing of streaming traces or
checkpoint recomputation. Capture is for ordinary serving-export forwards,
not probe-expanded research forwards that bypass the normal decoder call.

`layout_from_encoded` uses the original first action-target positions and
`step_plan` to derive prefix, observation and response segments through the
native layout builder. This also works for Window8, whose serializer omits
`episodic_layout`; if an existing M5 layout disagrees, the helper fails.

`reconstruct` calls the unchanged `CommittedWriteScorer._prepare` and `_scan`
once per canonical prefix/observation/response segment. The scorer internally
uses blocks of at most 64 tokens, carrying the convolution tail and state
between segments. It snapshots after each observation and before its response.
Shapes for the real backbone are:

| Field | CPU dtype | Shape |
|---|---|---|
| states | FP32 | `[frames,32,128,128]` (key by value) |
| visual_keys | FP16 default | `[frames,300,16,128]` |
| visual_values | FP16 default | `[frames,300,32,128]` |
| visual_beta | FP16 default | `[frames,300,32]` |
| visual_token_ids | int64 | `[frames,300]` |
| anchor_token_ids | Python integers | one per frame |

Use `visual_dtype=torch.float32` if lossless prepared visual tensors are needed.
Key heads remain unrepeated in storage; the native scorer repeats each key
head consecutively twice with `repeat_interleave` for the 32 value heads.

For G-3, each observation anchor makes a separate call to the decoder's actual
`fla.*.chunk_gated_delta_rule` with `output_final_state=True` over the entire
prepared prefix. Keys, values, log decay and beta are identical FP32 inputs to
the scorer and reference. In-kernel normalization is disabled because keys
are already normalized. Queries equal repeated keys, which does not affect the
recurrent state. This is independent native-kernel recurrence verification;
it does not independently check fused-convolution preparation parity. The
reference restarts from zero for each full prefix, so this check has quadratic
work in episode length and should be limited to the specified gate episodes.
There is no torch fallback or altered threshold. Every anchor must have relative
Frobenius error <=.01. Undefined zero-reference comparisons and nonfinite values
raise rather than producing a passing result.

The module retains tensors and per-anchor gate values. It makes no navigation
claims and performs no optimization, parameter updates or changes to policy
state. CPU tests cover the recurrence with an independent sequential FP64
reference, nonzero initial state, head repetition, conv carry, anchor timing,
visual source retention, metadata derivation and capture transparency. Full
GPU-dependent repository tests were left to the main task's scheduled run.
