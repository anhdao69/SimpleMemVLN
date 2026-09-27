# SimpleMemVLN

Minimal Qwen3.5-4B supervised fine-tuning for R2R navigation with up to eight
uniformly sampled history frames and the current observation. The visual
backbone stays frozen while the vision merger and language model are trained.

## Requirements

- Linux x86-64 with one or more NVIDIA GPUs
- An NVIDIA driver compatible with the locked CUDA 12.9 PyTorch wheels
- A CUDA toolkit discoverable through `CUDA_HOME` or `nvcc`
- [uv](https://docs.astral.sh/uv/) 0.12 or newer
- Uniform-8 annotations and their referenced RGB images

Create the locked Python 3.12 environment:

```bash
uv sync --locked
```

## Data format

The annotation file is a JSON array. Each record must contain a non-empty
ordered `images` list and exactly two `conversations`: one human prompt with
one `<image>` placeholder per image, followed by one assistant action. Image
paths may be relative to `DATA_ROOT`. Valid actions are `MOVE_FORWARD`,
`TURN_LEFT`, `TURN_RIGHT`, and `STOP`.

Each example may contain at most eight history frames plus its current frame.
The launcher does not resample frames; the annotation file must already use
the intended Uniform-8 selection.

## Train

Export the two required data locations, then run the launcher:

```bash
export ANNOTATION_PATH=/path/to/r2r_uniform8.json
export DATA_ROOT=/path/to/image/dataset
bash train.sh
```

By default, the launcher uses every visible GPU, downloads or loads
`Qwen/Qwen3.5-4B`, and writes to `output/uniform8`. Common overrides include:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export OUTPUT_DIR=/path/to/checkpoints/simplememvln
export CACHE_DIR=/path/to/huggingface/cache
export NPROC_PER_NODE=4
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8
export GRADIENT_CHECKPOINTING=True
bash train.sh
```

`ENV_DIR` selects an environment other than `.venv`. Other training values in
`train.sh` can also be overridden through variables of the same name.

The source and launcher receive local syntax, configuration, lockfile, and
preflight validation. A complete GPU training run was not performed on the
local development machine because it does not have the required dataset and
NVIDIA environment.
