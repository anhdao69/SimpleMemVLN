# G3+A1 epoch-1 evaluation

This branch publishes the source packaged with `anhdao69/g3a1_training`.
The production implementation is commit
`877815e97e46a8bcc37cafee919b6348c18d0f35`. The packaged follow-up commit
`260e6d8` adds the original implementation report and a replay diagnostic;
it does not change the model or training implementation.

The run resumed checkpoint 2100 and completed joint R2R + English RxR15deg
epoch 1 at update 3852 on worker-2. Its optimizer, scheduler, RNG and original
trajectory sampling identities were retained. The original report describes
the earlier Newton launch; its running status is historical.

## Published serving weights

- Repository: https://huggingface.co/anhdao69/Streaming_VGGT
- Folder: `G3A1/epoch-1`
- Verified publication revision: `86a537b6eb4d72baedbbf6b9ec003b3ab20f0a24`
- Provenance: `G3A1/provenance/`, including exact training source and export audit.
- Base assets: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

In the existing pinned Linux/CUDA model environment (see `pyproject.toml` and
`uv.lock`), download the serving files:

```python
from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    "anhdao69/Streaming_VGGT",
    revision="86a537b6eb4d72baedbbf6b9ec003b3ab20f0a24",
    allow_patterns=["G3A1/epoch-1/*"],
)
checkpoint = snapshot + "/G3A1/epoch-1"
print(checkpoint)
```

Use this branch's strict loader with `PYTHONPATH` pointing to its `src`:

```python
from qwen_vl.train.vln_runtime import load_checkpoint

model, serializer = load_checkpoint(checkpoint, "/path/to/pinned/Qwen3.5-4B")
model.eval()
```

This is a Window8 text policy with canonical action history. The four trained
rank-8 GeoAddress Q/K adapter tensors at decoder layer 24 (45,056 parameters)
are required during inference. The serving export retains them and removes
only the training G3 state readout and research configuration. Do not remove
the adapters, initialize new ones, use a plain AutoModel loader, or override
the saved serializer/memory/action-history contract. VGGT-Omega and its
training assets are not needed for inference.

For Habitat R2R evaluation, follow [the evaluation instructions](scripts/eval/README.md)
with `CHECKPOINT` set to the printed `G3A1/epoch-1` directory, `MODEL_PYTHON`
set to the pinned model environment, and `BASE_MODEL` set to the pinned base
assets. Start with `--max-episodes 1` and a fresh output directory. The
candidate-logit and dual-lane-specific audits in that document do not apply
to this model.

The [original implementation report](reports/vggt_five/G3A1/implementation_report.md)
records a hardware-dependent streaming/offline replay tolerance failure as
well as successful export parity checks. Evaluate serving parity on the
evaluation hardware; successful training and export do not establish
navigation quality. No new Habitat evaluation was performed for this
GitHub publication.

## Publication verification

All 271 code/config/test files in the exact training source archive published
with the weights were compared byte-for-byte against this checkout and
matched. `src/`, `configs/`, `train/`, `pyproject.toml`, and `uv.lock` have no
changes relative to production commit `877815e`. This publication adds only
this evaluation guide to the packaged follow-up commit.
