# SimpleMemVLN

[Detailed R2R and RxR evaluation setup for a new server](EVALUATION_SETUP.md).

[R2R `val_unseen` evaluation and inference-speed comparison](reports/R2R_ALL_VERSIONS_COMPARISON_2026-10-03.md).

## Candidate-logit experiment (`streaming_logits`)

The new `candidate_logits` mode selects four verified single-token pretrained
Qwen LM-head rows, without autoregressive action generation. Default
`lm_rows_trainable` uses the normal backbone LR; explicit feedback supports
candidate tokens or deterministic canonical action text. Both FullContext and
Window8 preserve action history and GDN state. Existing text checkpoints keep
their original contract.

See the [implementation report and exact run commands](reports/archive/CANDIDATE_LOGITS_IMPLEMENTATION_2026-10-02.md)
for production-recipe overlays, class balancing, smoke/reload evidence,
profiling, and pending Habitat ablations. The joint FullContext candidate
checkpoints have completed R2R `val_unseen` evaluations at epochs 1 and 2;
see the comparison report above for SR, SPL and inference speed.

## Full-episode streaming R2R

The episode path implements the supplied [v2 plan](plans/SimpleMemVLN_Implementation_Plan.md):
pretrained Qwen3.5-4B, frozen vision/merger, full-episode text gradients,
an action classifier or sparse action-text loss, and native persistent GDN
with FullContext or prefix + eight-step attention. See
[implementation evidence](reports/archive/H100_R2R_SMOKE_2026-09-28.md) and the
[execution ledger](IMPLEMENTATION_PROGRESS.md) for measured gates and limitations.

The H100 environment uses Python 3.12, Torch 2.10/cu129, Transformers 5.11,
FA2 2.8.3 and FLA 0.5.2. **TileLang 0.1.14 is required for the verified H100
backward path:** FLA rejects Triton 3.6's gated backward on Hopper. Do not
disable that guard. Set `CUDA_HOME` to the available CUDA toolkit used for
TileLang compilation (12.8.1 in the measured environment). FlashAttention
downloads from the supplied Hugging Face mirror with its locked SHA256.

```bash
export CUDA_HOME=/path/to/cuda-12.8.1
uv sync --locked
export MODEL_PATH=/path/to/pinned/Qwen3.5-4B/snapshot
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

.venv/bin/python -m scripts.vln.prepare_r2r \
  --data-root /path/to/JanusVLN/data --out artifacts/r2r_train.jsonl
.venv/bin/python -m scripts.vln.audit_data \
  --manifest artifacts/r2r_train.jsonl --model-path "$MODEL_PATH"

./train.sh --vln_config configs/vln_r2r_v0_base.yaml \
  --output_config configs/vln_r2r_v0_classification.yaml \
  --manifest artifacts/r2r_train.jsonl --model_name_or_path "$MODEL_PATH" \
  --episode-limit 64 --selection shortest --debug-repeat-episodes \
  --max-optimizer-updates 4 --output_dir outputs/r2r_smoke
```

Default global batch is **64 complete episodes**: four ranks × one episode ×
16 accumulation steps. Loss is an action mean across that entire update,
not an episode mean. Repeated/debug exposures and a partial final update
are counted explicitly. No episode is truncated or padded. The short-episode
smoke is not representative of full-corpus training time.

Add `--memory_config configs/vln_memory_window8.yaml` for Window8; use
`configs/vln_r2r_v0_qwen_text.yaml` for text output. Ordinary Uniform8
source/CLI remains below; its historical results are not revalidated by this migration.

```bash
VLN_MODEL_PATH="$MODEL_PATH" .venv/bin/python -m pytest tests/vln -q
.venv/bin/python -m scripts.vln.check_compat --model-path "$MODEL_PATH"
.venv/bin/python -m scripts.vln.check_parity --manifest artifacts/r2r_train.jsonl \
  --model-path "$MODEL_PATH" --memory-mode window8 --atol .5 --max-rms .15
.venv/bin/python -m scripts.vln.summarize_run outputs/r2r_smoke
.venv/bin/python -m scripts.vln.check_reload --checkpoint outputs/r2r_smoke/final \
  --manifest artifacts/r2r_train.jsonl --model-path "$MODEL_PATH"
```

Numerical budgets are recorded with each result. Initial stricter BF16
parity failures are retained; bounded numerical differences can change
near-tie actions. FP32 reference tests use a separate tight tolerance.
Do not equate BF16 parity or a decreasing smoke loss with navigation success.

Resume requires the same model/serializer/visibility contract and the same
manifest hash, ordered episode selection, seed, world size and accumulation.
Pass `--resume_from_checkpoint outputs/.../checkpoint-N`; the `final`
directory is a serving export, not an optimizer checkpoint.

Habitat uses a separate simulator interpreter and the validated model process:

```bash
PYTHONPATH=src /path/to/habitat/python -m qwen_vl.eval.habitat_r2r \
  --habitat-config /path/to/JanusVLN/config/vln_r2r.yaml \
  --simulator-source /path/to/JanusVLN/src --data-root /path/to/JanusVLN/data \
  --model-python "$PWD/.venv/bin/python" --model-path "$MODEL_PATH" \
  --checkpoint outputs/r2r_smoke/final --split val_unseen \
  --episode-list artifacts/eval_episode_ids.json --out artifacts/rollouts
```

Rollout failures remain in the SR/SPL denominator. The current server's
available Janus environment lacks Habitat; simulator execution and raw
capture-before-action replay remain separate, unverified gates.

## Legacy Uniform8 workflow (not revalidated)

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
