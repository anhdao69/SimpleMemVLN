# R2R and RxR evaluation: complete setup on a new Linux server

Verified against the existing server on **2026-10-10**. This guide covers
SimpleMemVLN inference in Habitat, from an empty server to smoke tests, complete
validation runs, result checks, and environment migration. Commands use Bash.
Run sections in order, in the same shell unless a section says otherwise.

The benchmark here is **continuous-environment VLN-CE**: R2R-VLNCE and the
project's **English-guide RxR converted to 15-degree turns**. This is not the
discrete Matterport3DSimulator evaluation, the original multilingual RxR task,
or an official held-out test submission.

## Contents

1. [What must be installed](#1-what-must-be-installed)
2. [Choose the right branch and checkpoint family](#2-choose-the-right-branch-and-checkpoint-family)
3. [Prepare Linux, GPU, and storage](#3-prepare-linux-gpu-and-storage)
4. [Clone and define portable paths](#4-clone-and-define-portable-paths)
5. [Create the model environment](#5-create-the-model-environment)
6. [Install Habitat-Sim and Habitat-Lab](#6-install-habitat-sim-and-habitat-lab)
7. [Install the RxR registrations](#7-install-the-rxr-registrations)
8. [Prepare Matterport3D scenes and R2R episodes](#8-prepare-matterport3d-scenes-and-r2r-episodes)
9. [Prepare the exact RxR15 data and ground truth](#9-prepare-the-exact-rxr15-data-and-ground-truth)
10. [Download and verify models](#10-download-and-verify-models)
11. [Check data, configs, and rendering before inference](#11-check-data-configs-and-rendering-before-inference)
12. [Run R2R smoke and full evaluation](#12-run-r2r-smoke-and-full-evaluation)
13. [Run RxR smoke and full evaluation](#13-run-rxr-smoke-and-full-evaluation)
14. [Other checkpoints, multiple GPUs, and launchers](#14-other-checkpoints-multiple-gpus-and-launchers)
15. [Monitor, stop, resume, and validate results](#15-monitor-stop-resume-and-validate-results)
16. [Legacy evaluator branches](#16-legacy-evaluator-branches)
17. [Troubleshooting](#17-troubleshooting)
18. [Archive a reproducible setup](#18-archive-a-reproducible-setup)
19. [What was verified and what still needs a new-server check](#19-what-was-verified-and-what-still-needs-a-new-server-check)

## 1. What must be installed

### 1.1 Two interpreters are required

The Habitat process owns scenes, observations, actions, and metrics. It starts a
separate model process through `--model-python`; they communicate through local
stdin/stdout. No HTTP model server or port configuration is required.

| Component | Simulator process | Model process |
|---|---|---|
| Python | 3.10; observed 3.10.20 | 3.12; observed 3.12.13 |
| Environment | Dedicated Conda environment | Repository `.venv`, created by `uv` |
| Habitat-Sim | **0.2.4**, tag `v0.2.4` | Not installed |
| Habitat-Lab | **0.2.4**, tag `v0.2.4` | Not installed |
| NumPy | **1.26.4** | **2.4.4**, from `uv.lock` |
| PyTorch | Unneeded by the modern Habitat adapter | **2.10.0**, CUDA **12.9** wheels |
| Transformers | Unneeded | **5.11.0** |
| FlashAttention | Unneeded | **2.8.3**, repository's exact cp312/Torch 2.10 wheel |
| FLA / fla-core | Unneeded | **0.5.2 / 0.5.2** |
| TileLang | Unneeded | **0.1.14** |
| Hydra / OmegaConf | Observed **1.3.4 / 2.3.1** | Branch lock controls dependencies |
| Gym | **0.23.0** | Not needed by the model |
| fastdtw | **0.3.4**, required for RxR nDTW | Not needed |

Do not install the repository's model dependencies into the simulator
environment. In particular, its NumPy 2.x pin conflicts with the old Habitat/Gym
stack. An activated Conda environment does not replace `--model-python`.

The exact Habitat source commits are:

```text
Habitat-Sim: f179b584bcd713c5a2a998132211e2cae881d6d1
Habitat-Lab: 1639e1ae732ba1e84199a1a04b79c7243c3f8586
```

The observed simulator supports **headless EGL rendering**, with Bullet and
CUDA-specific simulator features disabled. `habitat_sim.cuda_enabled == False`
does **not** mean RGB rendering runs without an NVIDIA GPU: EGL/OpenGL rendering
is separate from CUDA buffer support. This evaluator transports CPU RGB arrays
to the model process and does not require Habitat-Sim CUDA interop.

The original VLN-CE upstream README describes Habitat 0.1.7. Use it for dataset
information, but use the **0.2.4 pins in this guide** for SimpleMemVLN.

### 1.2 Evaluation protocol

| Setting | Value used here |
|---|---|
| R2R dataset class | `R2RVLN-v1` |
| RxR dataset class | `RxRVLNCE-v1`, supplied by bundled registrations |
| Default split | `val_unseen` |
| R2R full denominator | **1,839** episodes for the audited file |
| RxR full denominator | **3,669** English-guide episodes for the audited file |
| RxR roles / languages | `guide`; `en-US`, `en-IN` |
| RGB width / height / HFOV | **640 / 480 / 79°** |
| Forward displacement | **0.25 m** |
| Left / right turn | **15°** |
| Episode decision cap | **500**, with separately recorded forced final STOP |
| Success radius | **3.0 m**, Habitat's success implementation |
| Evaluation seed | **42** |
| Depth sensor | Configured at 640×480, HFOV 79°, range 0–10 m; model uses RGB |
| Model state | Reset between episodes; retained within an episode |
| Failure accounting | Failed episodes contribute zero SR/SPL |

The evaluator validates the camera, movement, success radius, and cap. Load the
checkpoint's saved memory, serializer, feedback, and output policy. Do not turn a
FullContext checkpoint into Window8, insert action history into a NoHistory
checkpoint, or initialize new lane weights for a trained DualLane export.

## 2. Choose the right branch and checkpoint family

**Having this guide on a branch does not add an evaluator to that branch.**
The documentation is shared across branches; model implementations and evaluator
interfaces differ. Before installing, decide what checkpoint you will evaluate.

| Branch / branch group, at the inspected tips | Evaluation guidance |
|---|---|
| `main`, `report_all_results_main` | Legacy Uniform8 training/report branch; no `qwen_vl.eval.habitat_r2r`. Switch to a compatible evaluation branch. |
| `streaming`, `report_all_results_streaming`, `report_refresh_streaming_20261005` | Native text/classifier model family. Committed evaluator is the older R2R interface; see section 16. This server's `streaming` worktree additionally has uncommitted modern evaluation tools. |
| `streaming_logits`, `report_all_results_logits`, `report_refresh_streaming_logits_20261005` | Candidate-logit family; older committed R2R adapter. Use `eval_candidate_variants_20261004` for modern R2R/RxR commands. |
| `streaming_logits_dual`, `report_refresh_streaming_logits_dual_20261005` | Candidate DualLane family; older adapter. Use `eval_dual_variants_20261005` for modern R2R/RxR commands. |
| `eval_candidate_variants_20261004` | Modern evaluator, candidate logits / NoHistory / Window8 evaluation source. |
| `eval_dual_variants_20261005` | Modern evaluator and candidate DualLane implementation. |
| `streaming_text_dual`, `eval_text_dual_20261007` | Modern evaluator and text DualLane implementation; the complete workflow below uses `streaming_text_dual`. |

Known modern source snapshots, before the documentation commits:

```text
eval_candidate_variants_20261004: 9786b15a9889712caee8bf9ab16a523a3b5b546d
eval_dual_variants_20261005:      c30489bf55678d425c64ea2c9739cf4f9b2e9508
streaming_text_dual / eval_text_dual_20261007:
                                f59133d4f0277454517c926c6c0a94d5e408da91
```

Use the updated branch tip to get this guide and its setup assets. Record the
actual `git rev-parse HEAD` in your experiment. A documentation-only commit does
not change the model source, but a different implementation commit can.

If a branch is unavailable on the remote, use the all-branches Git bundle
provided with the setup handoff, or transfer that branch from the existing server.
Do not substitute an unrelated branch just because its name looks similar.

## 3. Prepare Linux, GPU, and storage

The observed host runs **Ubuntu 22.04.5 LTS, x86-64**. The measured evaluation GPU
is an **RTX PRO 6000 Blackwell Max-Q, approximately 96 GiB**, driver **580.178.04**.
H100 model checks also exist. These are reference machines, not a promise that
every supported-looking GPU has been evaluated.

Plan for one GPU worker initially. The recorded three-worker 500-step FullContext
probe peaked near **14.62 GiB reserved per model worker** on that host, excluding
Habitat and other processes. This is a measured fixture, **not a universal VRAM
minimum**; leave headroom and validate your actual checkpoint's longest episodes.
Use a 24 GiB or larger GPU as a practical starting point, with uncertainty until
your smoke and long-horizon checks pass. Three workers need much more aggregate
memory than one. Longer/full-history paths may dominate runtime and memory.

Check the machine first:

```bash
uname -m
cat /etc/os-release
nvidia-smi
df -hT
df -ih
free -h
command -v nvcc || true
nvcc --version 2>/dev/null || true
```

Use a writable filesystem with room for MP3D, model weights, caches, source
builds, and outputs. Check actual archive/checkpoint sizes; allow hundreds of GB
if transferring full scenes and multiple model families. Model caches and local
copies can duplicate large weights. On the original server `/storage` was nearly
full, while `/data14t` had space; choose your new server's path independently.

Install OS build/runtime dependencies, using an administrator if you lack sudo:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs curl wget unzip rsync ca-certificates build-essential \
  cmake ninja-build pkg-config libjpeg-dev libglm-dev \
  libgl1-mesa-dev libegl1-mesa-dev libglvnd-dev \
  mesa-utils xorg-dev freeglut3-dev
```

The NVIDIA driver must expose GPU compute **and graphics/EGL**. In containers,
use NVIDIA Container Toolkit and expose the GPU with graphics capability, for
example `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. Run `nvidia-smi`
inside the container too. Setting `DISPLAY` is not required for headless EGL.

PyTorch's wheel bundles CUDA runtime libraries; it does not install the NVIDIA
driver or a full development toolkit. The H100 TileLang path was verified with
**CUDA toolkit 12.8.1**; the model's PyTorch wheel reports **CUDA 12.9**. Those are
different components. If using an installed toolkit, set its real location:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version
```

Replace that example path with the actual toolkit. Do not use `nvidia-smi`'s
displayed CUDA compatibility number as proof that `nvcc` is installed.

## 4. Clone and define portable paths

Choose `EVAL_ROOT` once. The example uses your home directory; change it to a
large writable local filesystem before running the commands.

```bash
export EVAL_ROOT="$HOME/smv-eval"
mkdir -p "$EVAL_ROOT"
git clone --branch streaming_text_dual \
  https://github.com/anhdao69/SimpleMemVLN.git "$EVAL_ROOT/SimpleMemVLN"
export SMV_ROOT="$EVAL_ROOT/SimpleMemVLN"
cd "$SMV_ROOT"
git status --short
git rev-parse HEAD

export HABITAT_SRC_ROOT="$EVAL_ROOT/habitat-source"
export DATA_ROOT="$EVAL_ROOT/data"
export CHECKPOINT_ROOT="$EVAL_ROOT/checkpoints"
export RESULT_ROOT="$EVAL_ROOT/results"
export HF_HOME="$EVAL_ROOT/huggingface-cache"
export UV_CACHE_DIR="$EVAL_ROOT/uv-cache"
export TMPDIR="$EVAL_ROOT/tmp"
mkdir -p "$HABITAT_SRC_ROOT" "$DATA_ROOT" "$CHECKPOINT_ROOT" \
  "$RESULT_ROOT" "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
```

The desired layout is:

```text
smv-eval/
  SimpleMemVLN/                    chosen implementation branch
    .venv/                        Python 3.12 model environment
    EVALUATION_SETUP.md
    docs/evaluation_setup/        portable configs + RxR registrations
  habitat-source/
    habitat-sim/                  exact 0.2.4 source, if built locally
    habitat-lab/                  exact 0.2.4 source
  conda/                          optional private Miniconda installation
  data/
    scene_datasets/mp3d/<scan>/<scan>.glb
    scene_datasets/mp3d/<scan>/<scan>.navmesh
    datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
    datasets/rxr_15deg/val_unseen/val_unseen_guide.json.gz
    datasets/rxr_15deg/val_unseen/val_unseen_guide_gt.json.gz
  checkpoints/
    qwen-base/
    joint-fullcontext-b/epoch-1/
  results/
    r2r-smoke/epoch-1/
    r2r-full/epoch-1/
    rxr-smoke/epoch-1/
    rxr-full/epoch-1/
```

Source code and setup assets are in Git. Dataset files, checkpoints, caches,
compiled environments, and evaluation journals are not downloaded by cloning Git.

## 5. Create the model environment

Install `uv` if missing. Its installer is documented by
[Astral](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh -o "$TMPDIR/install-uv.sh"
sh "$TMPDIR/install-uv.sh"
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The project requires **uv >= 0.12.0**; the observed server has 0.12.21. Build the
model environment from the selected branch's lock, without upgrading packages:

```bash
cd "$SMV_ROOT"
unset VIRTUAL_ENV
uv python install 3.12.13
uv sync --locked --python 3.12.13
export MODEL_PYTHON="$SMV_ROOT/.venv/bin/python"
"$MODEL_PYTHON" --version
"$MODEL_PYTHON" - <<'PY'
import importlib.metadata as m
import torch
for name in ('torch', 'torchvision', 'transformers', 'flash-attn',
             'flash-linear-attention', 'fla-core', 'tilelang', 'numpy'):
    print(name, m.version(name))
print('wheel CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Model interpreter cannot access NVIDIA CUDA'
print('GPU:', torch.cuda.get_device_name(0))
print('compute capability:', torch.cuda.get_device_capability(0))
PY
```

The FlashAttention download is an exact wheel URL in `pyproject.toml` and its
SHA256 is locked in `uv.lock`. A generic `pip install flash-attn` can select a
different wheel/ABI or trigger a source build. Keep the locked source. Linux
x86-64, CPython 3.12, Torch 2.10/cu129 must match that wheel.

On Hopper, TileLang 0.1.14 matters for the verified FLA backward path; do not
disable FLA's Triton/Hopper checks. Evaluation still needs the branch's validated
inference backend. Do not infer correctness just because imports succeed.

## 6. Install Habitat-Sim and Habitat-Lab

### 6.1 Create a separate Conda environment

If Conda is already installed, use it. Otherwise install it under `EVAL_ROOT`:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o "$TMPDIR/miniconda.sh"
bash "$TMPDIR/miniconda.sh" -b -p "$EVAL_ROOT/conda"
source "$EVAL_ROOT/conda/etc/profile.d/conda.sh"
conda create -n habitat-vln310 python=3.10.20 pip -y
conda activate habitat-vln310
export EVAL_PYTHON="$(command -v python)"
"$EVAL_PYTHON" --version
```

If your Conda installation requests channel terms acceptance, follow its displayed
instructions or use your organization's approved Conda channels. Record the
channels/builds you use. Do not silently change Python major/minor versions.

### 6.2 Habitat-Sim installation: choose one route

**Route A: prebuilt headless Conda package.** This is the upstream-supported
convenient route. It has the same requested version but is not the exact binary
build used on the observed server; run the rendering checks in section 11.

```bash
conda install -n habitat-vln310 -c conda-forge -c aihabitat \
  habitat-sim=0.2.4 headless -y
```

If the solver cannot find a compatible Python 3.10 package, use Route B instead
of upgrading Habitat or downgrading to Habitat 0.1.7. Do not combine source and
Conda simulator installations in one environment.

**Route B: source build, closest to the observed installation.** Use a fresh
Python 3.10 simulator environment without a preinstalled Habitat-Sim package:

```bash
cd "$HABITAT_SRC_ROOT"
git clone https://github.com/facebookresearch/habitat-sim.git
git -C habitat-sim checkout f179b584bcd713c5a2a998132211e2cae881d6d1
git -C habitat-sim submodule update --init --recursive
git -C habitat-sim describe --tags --always

# Old setup.py uses distutils. These are compatibility build-tool choices,
# not a claim that they equal the original server's build-time tool versions.
"$EVAL_PYTHON" -m pip install 'setuptools<70' wheel 'cmake>=3.14,<4' ninja
"$EVAL_PYTHON" -m pip install \
  -c "$SMV_ROOT/docs/evaluation_setup/habitat-constraints.txt" \
  -r "$HABITAT_SRC_ROOT/habitat-sim/requirements.txt"
cd "$HABITAT_SRC_ROOT/habitat-sim"
"$EVAL_PYTHON" setup.py install --headless --no-bullet
cd "$SMV_ROOT"
```

Do not add `--with-cuda` just to render on a GPU. The observed build has CUDA
interop and Bullet disabled. The existing server's cached CMake directory also
contains historical interpreter paths, so rebuilding from source is safer than
copying its `build/` directory. Compiler/CMake compatibility on a new distribution
must be checked locally; Ubuntu 22.04 is the reference baseline.

The upstream [0.2.4 installation notes](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.2.4/README.md)
and [source build instructions](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.2.4/BUILD_FROM_SOURCE.md)
describe headless installation. Use the pinned commit above rather than `stable`
or `main`, which may move.

### 6.3 Install Habitat-Lab at the matching commit

```bash
cd "$HABITAT_SRC_ROOT"
git clone https://github.com/facebookresearch/habitat-lab.git
git -C habitat-lab checkout 1639e1ae732ba1e84199a1a04b79c7243c3f8586
git -C habitat-lab describe --tags --always
"$EVAL_PYTHON" -m pip install \
  -c "$SMV_ROOT/docs/evaluation_setup/habitat-constraints.txt" \
  -e "$HABITAT_SRC_ROOT/habitat-lab/habitat-lab"
"$EVAL_PYTHON" -m pip install \
  -c "$SMV_ROOT/docs/evaluation_setup/habitat-constraints.txt" fastdtw
cd "$SMV_ROOT"
```

The constraints file captures the observed important simulator dependencies.
It is **not a full transitive lock** or a portable copy of the original mixed
ROS/training environment. Inspect dependency conflicts instead of suppressing
them:

```bash
"$EVAL_PYTHON" -m pip check
"$EVAL_PYTHON" - <<'PY'
import sys, importlib.metadata as m
import habitat, habitat_sim
print(sys.executable, sys.version)
for name in ('habitat-lab', 'habitat-sim', 'numpy', 'gym',
             'hydra-core', 'omegaconf', 'fastdtw'):
    print(name, m.version(name))
assert m.version('habitat-lab') == '0.2.4'
assert m.version('habitat-sim') == '0.2.4'
assert m.version('numpy') == '1.26.4'
print('Habitat-Lab:', habitat.__file__)
print('Habitat-Sim:', habitat_sim.__file__)
print('Simulator CUDA interop:', habitat_sim.cuda_enabled)
print('Bullet:', habitat_sim.built_with_bullet)
PY
```

The **modern** evaluator imports `habitat.config.default.get_config`, so
Habitat-Baselines, ROS, SpatialStack training dependencies, Torch 2.8, and its
cp310 FlashAttention wheel are not required for that process. The older adapter
uses Habitat-Baselines; its separate instructions are in section 16.

## 7. Install the RxR registrations

Habitat-Lab alone does not supply this project's `RxRVLNCE-v1` dataset loader
and nDTW registrations. The guide bundles the exact files used locally:

```text
docs/evaluation_setup/habitat_extensions/__init__.py
docs/evaluation_setup/habitat_extensions/task.py
docs/evaluation_setup/habitat_extensions/measures.py
```

They are unchanged copies of `SpatialStack` commit
`b1f435642e5397dbd1523d4c58efe504402b881e`, with its license and file hashes in
the adjacent `SPATIALSTACK_LICENSE` and `provenance.json`. This avoids requiring
an entire SpatialStack clone on the new server.

Set the directory **containing** `habitat_extensions`, not the package itself:

```bash
export SIMULATOR_SOURCE="$SMV_ROOT/docs/evaluation_setup"
export PYTHONPATH="$SMV_ROOT/src:$SIMULATOR_SOURCE"
"$EVAL_PYTHON" - <<'PY'
from habitat_extensions import task, measures
from habitat.core.registry import registry
assert registry.get_dataset('RxRVLNCE-v1') is not None
assert registry.get_measure('NDTW') is not None
print('RxR dataset and nDTW registrations imported successfully')
PY
```

`task.py` supplies role/language filtering and Hydra's `rxrvlnce_v1` dataset
config. `measures.py` supplies `OracleSuccess` and `NDTW`. Import them before
composing the RxR YAML. The modern evaluator does this when given
`--dataset-kind rxr15 --simulator-source ...`.

## 8. Prepare Matterport3D scenes and R2R episodes

### 8.1 Matterport3D scene files

Obtain the Habitat-format Matterport3D scene reconstructions through the
[Matterport3D project](https://niessner.github.io/Matterport/) and its access
instructions, or transfer your existing authorized copy. RGB training frames
are insufficient: Habitat needs 3D assets and navigation meshes.

Expected paths:

```text
$DATA_ROOT/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
$DATA_ROOT/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh
...
```

There are 90 scans in the full MP3D scene collection. Keep the full collection if
you will run both datasets and multiple splits. A partial R2R-only scene mirror
can pass an R2R smoke and still fail on RxR. Transfer `.house` and semantic assets
too when copying whole scan directories, even though the policy reads RGB.

`--scenes-dir` points to **`scene_datasets`**, the parent of `mp3d`, because the
episode scene paths contain `mp3d/<scan>/<scan>.glb`. Pointing it directly to
`scene_datasets/mp3d` commonly produces `mp3d/mp3d` in resolved paths.

Example from the **new server**, replacing the SSH host/user with your values:

```bash
mkdir -p "$DATA_ROOT/scene_datasets"
rsync -aL --info=progress2 \
  YOUR_USER@OLD_SERVER:/storage/anhdh35/JanusVLN/data/scene_datasets_full/ \
  "$DATA_ROOT/scene_datasets/"
```

`-L` transfers symlink targets rather than leaving links to paths that only exist
on the old machine. Trailing slashes copy directory contents into the destination.

### 8.2 R2R-VLNCE annotations

Use `R2R_VLNCE_v1-3_preprocessed`, with:

```text
datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
```

The [official VLN-CE dataset instructions](https://github.com/jacobkrantz/VLN-CE#episodes-room-to-room-r2r)
link the preprocessed archive. Download it using a small separate utility
environment, keeping download tools out of the two evaluation environments:

```bash
uv venv "$EVAL_ROOT/download-tools" --python 3.12.13
uv pip install --python "$EVAL_ROOT/download-tools/bin/python" gdown
"$EVAL_ROOT/download-tools/bin/gdown" \
  'https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr' \
  -O "$TMPDIR/R2R_VLNCE_v1-3_preprocessed.zip"
unzip -l "$TMPDIR/R2R_VLNCE_v1-3_preprocessed.zip" | head -30
mkdir -p "$DATA_ROOT/datasets"
unzip "$TMPDIR/R2R_VLNCE_v1-3_preprocessed.zip" -d "$DATA_ROOT/datasets"
```

Inspect the archive's root folder before extracting. The final location must
match the path below; move a redundant nesting level if the archive layout
differs. Google Drive quota/login restrictions may require browser download or
transfer from the original server instead:

```bash
rsync -aL --info=progress2 \
  YOUR_USER@OLD_SERVER:/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/ \
  "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/"
```

Define the path template, preserving literal braces:

```bash
export R2R_DATA="$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
export MP3D_SCENES="$DATA_ROOT/scene_datasets"
```

Do not supply raw discrete R2R JSON to the continuous Habitat dataset class.

## 9. Prepare the exact RxR15 data and ground truth

The evaluated files on this server are:

```text
/storage/anhdh35/JanusVLN/data/datasets/rxr_15deg/val_unseen/
  val_unseen_guide.json.gz
  val_unseen_guide_gt.json.gz
  conversion_manifest.json
```

Transfer the exact conversion rather than renaming an upstream 30-degree file:

```bash
mkdir -p "$DATA_ROOT/datasets/rxr_15deg"
rsync -aL --info=progress2 \
  YOUR_USER@OLD_SERVER:/storage/anhdh35/JanusVLN/data/datasets/rxr_15deg/ \
  "$DATA_ROOT/datasets/rxr_15deg/"
export RXR_DATA="$DATA_ROOT/datasets/rxr_15deg/{split}/{split}_{role}.json.gz"
export RXR_GT="$DATA_ROOT/datasets/rxr_15deg/{split}/{split}_{role}_gt.json.gz"
```

The local conversion manifest records English `en-US`/`en-IN` guide trajectories
and expands each original 30° turn into two 15° turns. For `val_unseen` it records
3,669 episodes, 323,274 original actions, and 430,692 converted actions. Ground
truth locations were copied because the source stores start/post-forward-motion
positions. Its manifest explicitly calls for a 15° replay check before training;
do not claim that this conversion process was independently recreated here.

The public upstream RxR archive is a source dataset, **not a verified public
download of this exact derived conversion**. If the old server is unavailable,
obtain the derived files plus conversion manifest from the dataset owner, or
reproduce the conversion and validate it as a new dataset revision. There is no
verified standalone RxR15 conversion downloader in these branch tips.

The GT gzip is a mapping keyed by the string episode ID; each entry contains
`locations`. Matching IDs and nonempty locations are required for nDTW. Keep
guide episodes paired with guide GT. Follower trajectories or Hindi/Telugu
episodes do not belong to this evaluated denominator.

Audited compressed-file hashes, from the existing evaluation contracts:

```text
R2R val_unseen:
1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3
RxR15 val_unseen guide episodes:
0bf7358953a1b59502d7876611bf5da817d19ce7a8d9922b740a53a93812b94e
RxR15 val_unseen guide GT:
ac381254bfc019a55c62d993a770ffb7a0b77c02fa0c18118a99d02e6bc6b959
```

Check them after transfer:

```bash
sha256sum \
  "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz" \
  "$DATA_ROOT/datasets/rxr_15deg/val_unseen/val_unseen_guide.json.gz" \
  "$DATA_ROOT/datasets/rxr_15deg/val_unseen/val_unseen_guide_gt.json.gz"
```

Recompressing identical JSON can change the gzip hash. Exact reproduction uses
the original gzip bytes; a different hash requires investigation and explicit
dataset provenance, not an assumption of equivalence. Hashes/counts above are
for `val_unseen`; do not apply them to `val_seen`.

## 10. Download and verify models

### 10.1 Base model and a concrete joint text checkpoint

The workflow below uses joint R2R+RxR15 FullContext text **epoch 1** and the exact
Qwen base revision. This family can be loaded by the modern text implementation
used in this guide. Base weights and navigation weights serve different roles;
download both.

```bash
export MODEL_PATH="$CHECKPOINT_ROOT/qwen-base"
export POLICY_REPO=anhdao69/SimpleMemVLN-R2R-RxR15deg-FullContext-B
export POLICY_REVISION=caee0029ffb3ac2bfafdd4fbd1e0984d5c5bf10c
export POLICY_DIR="$CHECKPOINT_ROOT/joint-fullcontext-b"
export CHECKPOINT="$POLICY_DIR/epoch-1"
"$MODEL_PYTHON" - <<'PY'
import os, json, hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download('Qwen/Qwen3.5-4B',
    revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    local_dir=os.environ['MODEL_PATH'], max_workers=4,
    ignore_patterns=['*.md', '*.gitattributes'])
root = Path(snapshot_download(os.environ['POLICY_REPO'],
    revision=os.environ['POLICY_REVISION'],
    local_dir=os.environ['POLICY_DIR'], max_workers=4)).resolve()
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
assert isinstance(manifest, dict) and manifest, 'Empty checksum manifest'
checkpoint = Path(os.environ['CHECKPOINT']).resolve()
prefix = str(checkpoint.relative_to(root)) + '/'
for name in ('navigation.json', 'config.json', 'tokenizer.json',
             'processor_config.json'):
    assert prefix + name in manifest, f'Missing published checksum: {name}'
assert any(prefix + name in manifest for name in
           ('pytorch_model.bin', 'model.safetensors')), 'No checksummed weights'
for name, expected in manifest.items():
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path.is_file(), name
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, f'Checksum mismatch: {name}'
(root / 'verified_revision.json').write_text(json.dumps({
    'repository': os.environ['POLICY_REPO'],
    'revision': os.environ['POLICY_REVISION'],
    'sha256_verified': True}, indent=2) + '\n')
print('Verified:', checkpoint)
PY
```

If access is gated/private, authenticate the Hugging Face client using its
supported login mechanism. Keep credentials out of logs and Git. A repository
name without an immutable revision is insufficient for reproducing a published
checkpoint. The download can be large; wait for checksum verification to finish.

### 10.2 Other known immutable checkpoint revisions

Names below are under `anhdao69/`. Select the implementation branch from section
2, change `POLICY_REPO`, `POLICY_REVISION`, `POLICY_DIR`, and `CHECKPOINT`, then use
the same download/verification pattern. Inspect the repository's epoch folders;
a particular revision may not contain every epoch.

| Repository suffix | Epoch | Recorded revision | Family |
|---|---|---|---|
| `SimpleMemVLN-R2R-FullContext-B` | 1–3 | `73d866c31e398596a67cd8be6f7dbfd25ab9241d` | Native text, R2R-only |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-B` | 1 | `caee0029ffb3ac2bfafdd4fbd1e0984d5c5bf10c` | Native text, joint |
| `SimpleMemVLN-R2R-RxR15deg-Window8-B` | 1 | `3c17fdd7519742d4510fb1e2f6c3151a2bb9ae67` | Native text, Window8 |
| `SimpleMemVLN-R2R-RxR15deg-Window8-B` | 2 | `0a111c781477440d45afef600f3b79c340416d9f` | Native text, Window8 |
| `SimpleMemVLN-R2R-RxR15deg-Window8-CandidateLogits` | 1 | `04d3b2264a532a8ad7f778a3e23391fa12f6c98f` | Candidate |
| `SimpleMemVLN-R2R-RxR15deg-Window8-CandidateLogits` | 2 | `4e1b922fb7d243a01170d963d4b675a0483bab72` | Candidate |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-CandidateLogits-NoHistory` | 1 | `40de30d7febd215849f6bf56679eb1a11a47a073` | Candidate, NoHistory |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-DualLane-NoHistory-FromBase` | 1 | `394847e9072106ee08abc052812c98b972aa6bf7` | Candidate DualLane |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-DualLane-NoHistory-FromBase` | 2 | `1f36ade02b4036d33af7793599d43ea16ce45675` | Candidate DualLane |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-DualLane-NoHistory-Adapted` | 1 | `6036f64ead4164731d02351ea0ccd9b5d3b67dba` | Candidate DualLane |
| `SimpleMemVLN-R2R-RxR15deg-FullContext-DualLane-NoHistory-Adapted` | 2 | `e7e4d3dc7357d34e78646827fd9556240131899c` | Candidate DualLane |
| `SimpleMemVLN-R2R-RxR15deg-Window8-Text-DualLane-FromBase` | 1 | `6e98a6b3803068ae6122eb3cd7cf6414a8559440` | Text DualLane |

These revisions come from local verified downloads and the consolidated report,
not from a live claim that every Hub repository remains publicly accessible.
For an unlisted epoch/family, obtain its recorded revision/checksums. Never invent
a revision or replace it with `main` to claim the same experiment.

### 10.3 Inspect the saved policy contract

```bash
test -s "$MODEL_PATH/model.safetensors.index.json"
test -s "$CHECKPOINT/navigation.json"
"$MODEL_PYTHON" - "$CHECKPOINT/navigation.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d, indent=2))
PY
```

Keep `navigation.json`, config, processor/tokenizer files, and the complete
weight export together. A training optimizer checkpoint is not automatically a
serving export. Strict load failures are useful evidence of an incompatible
implementation or incomplete download; do not bypass missing/unexpected keys.

## 11. Check data, configs, and rendering before inference

### 11.1 Define runtime variables and confirm the modern CLI

```bash
cd "$SMV_ROOT"
unset VIRTUAL_ENV
export PYTHONPATH="$SMV_ROOT/src:$SIMULATOR_SOURCE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
export EVAL_GPU=0
export SPLIT=val_unseen
export R2R_CONFIG="$SMV_ROOT/docs/evaluation_setup/vln_r2r.yaml"
export RXR_CONFIG="$SMV_ROOT/docs/evaluation_setup/vln_rxr15.yaml"

"$EVAL_PYTHON" -m qwen_vl.eval.habitat_r2r --help
```

The modern help must include `--data-path`, `--scenes-dir`, `--dataset-kind`,
`--ground-truth-path`, `--max-episodes`, and `--gpu`. If help requires
`--data-root`/`--episode-list` and lacks those flags, you have the old adapter;
switch to the appropriate modern branch or follow section 16.

The portable YAML files bundled with this guide are copies of the actual local
evaluation configs. They are intentionally in `docs/evaluation_setup` so older
branches can carry the guide without replacing their production config trees.

### 11.2 Load both datasets, resolve every scene, and check GT coverage

Run this using the **simulator** interpreter. It composes the actual configs and
loads the same dataset classes the evaluator uses. It does not load model weights.

```bash
"$EVAL_PYTHON" - <<'PY'
import os, gzip, json
from pathlib import Path
import habitat
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from habitat_extensions import task, measures

split = os.environ['SPLIT']
for kind, config_var, path_var, expected in (
    ('r2r', 'R2R_CONFIG', 'R2R_DATA', 1839),
    ('rxr15', 'RXR_CONFIG', 'RXR_DATA', 3669),
):
    config = get_config(os.environ[config_var])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = os.environ[path_var]
        config.habitat.dataset.scenes_dir = os.environ['MP3D_SCENES']
    dataset = make_dataset(config.habitat.dataset.type,
                           config=config.habitat.dataset)
    ids = [str(e.episode_id) for e in dataset.episodes]
    assert len(ids) == len(set(ids)), f'{kind}: duplicate IDs'
    assert ids, f'{kind}: empty split'
    if split == 'val_unseen':
        assert len(ids) == expected, (kind, len(ids), expected)
    scenes = sorted({e.scene_id for e in dataset.episodes})
    missing = [s for s in scenes if not Path(s).is_file()]
    assert not missing, f'{kind}: missing scene files: {missing[:10]}'
    missing_nav = [s for s in scenes if not Path(s).with_suffix('.navmesh').is_file()]
    assert not missing_nav, f'{kind}: missing navmeshes: {missing_nav[:10]}'
    if kind == 'rxr15':
        gt_path = os.environ['RXR_GT'].format(split=split, role='guide')
        with gzip.open(gt_path, 'rt') as f:
            gt = json.load(f)
        for uid in ids:
            assert uid in gt and gt[uid]['locations'], f'Missing GT: {uid}'
    print(kind, 'episodes=', len(ids), 'scenes=', len(scenes), 'OK')
PY
```

For strict reproduction, compare the hashes in section 9 as well as counts.
Matching counts alone does not prove identical episode contents/order.

### 11.3 Render actual RGB and exercise the metrics

The model child inherits the parent's environment; **`--gpu` alone selects only
Habitat's GPU, not the model's GPU**. The commands below pair
`CUDA_VISIBLE_DEVICES="$EVAL_GPU"` with `--gpu "$EVAL_GPU"`, as the existing
launchers do. CUDA sees that single selected device as device 0; Habitat's EGL
configuration uses the physical device ID. Verify this mapping with rendering
and the model's runtime record on your host/container before using multiple GPUs.

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" - <<'PY'
import os
from pathlib import Path
import habitat
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from habitat_extensions import task, measures
from PIL import Image

for kind, config_var, path_var in (
    ('r2r', 'R2R_CONFIG', 'R2R_DATA'),
    ('rxr15', 'RXR_CONFIG', 'RXR_DATA'),
):
    config = get_config(os.environ[config_var])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = os.environ['SPLIT']
        config.habitat.dataset.data_path = os.environ[path_var]
        config.habitat.dataset.scenes_dir = os.environ['MP3D_SCENES']
        config.habitat.seed = 42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(os.environ['EVAL_GPU'])
        if kind == 'rxr15':
            config.habitat.task.measurements.ndtw.gt_path = os.environ['RXR_GT']
    dataset = make_dataset(config.habitat.dataset.type,
                           config=config.habitat.dataset)
    dataset.episodes = dataset.episodes[:1]
    with habitat.Env(config=config, dataset=dataset) as env:
        obs = env.reset()
        assert obs['rgb'].shape == (480, 640, 3), obs['rgb'].shape
        image = Path(os.environ['RESULT_ROOT']) / f'{kind}-first-rgb.png'
        Image.fromarray(obs['rgb']).save(image)
        env.step(0)  # STOP, for simulator/metric wiring only
        metrics = env.get_metrics()
        assert {'success', 'spl', 'distance_to_goal'} <= metrics.keys()
        if kind == 'rxr15':
            assert 'ndtw' in metrics
        print(kind, env.current_episode.episode_id, metrics, 'RGB:', image)
PY
```

A black image, EGL error, missing navmesh, or nDTW exception must be investigated
before loading an expensive model. This STOP check validates wiring; it does not
measure navigation quality. In Slurm, use the assigned GPU/device mapping and
verify it in both processes; the standalone physical-ID example assumes the full
GPU device set is exposed to the job.

### 11.4 Model backend check

On branches that have the compatibility script:

```bash
cd "$SMV_ROOT"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$SMV_ROOT/src" \
  "$MODEL_PYTHON" -m scripts.vln.check_compat \
  --model-path "$MODEL_PATH" --native-continuation --check-flash-routing \
  --out "$RESULT_ROOT/model-compat.json"
```

This loads the base model and exercises the implementation's declared numerical
tolerances. It does not validate the checkpoint's navigation score. The real
checkpoint smoke in the next sections also checks strict export loading, actual
observations, actions, and episode state reset. For DualLane, run the branch's
`scripts/vln/check_dual_evaluation.py --help` and follow its checkpoint-specific
audit before the smoke if that script is present; use the saved real RGB frame.

## 12. Run R2R smoke and full evaluation

### 12.1 Two real episodes first

```bash
cd "$SMV_ROOT"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind r2r \
  --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$R2R_CONFIG" \
  --data-path "$R2R_DATA" --scenes-dir "$MP3D_SCENES" \
  --gpu "$EVAL_GPU" --split "$SPLIT" \
  --max-episodes 2 --out "$RESULT_ROOT/r2r-smoke/epoch-1" \
  > "$RESULT_ROOT/r2r-smoke-epoch-1.log" 2>&1
cat "$RESULT_ROOT/r2r-smoke/epoch-1/summary.json"
```

Require `complete: true`, two completed episodes, and investigate every failure.
Two successful process executions are not evidence of a good SR. The value of
this check is exercising real simulator frames and reset between episodes.

### 12.2 Full `val_unseen`, 1,839 episodes

Use a **new output directory**; removing `--max-episodes 2` changes the selection
and cannot resume a smoke directory.

```bash
cd "$SMV_ROOT"
mkdir -p "$RESULT_ROOT/r2r-full"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" setsid nohup \
  "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind r2r \
  --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$R2R_CONFIG" \
  --data-path "$R2R_DATA" --scenes-dir "$MP3D_SCENES" \
  --gpu "$EVAL_GPU" --split "$SPLIT" \
  --out "$RESULT_ROOT/r2r-full/epoch-1" \
  > "$RESULT_ROOT/r2r-full/epoch-1.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RESULT_ROOT/r2r-full/epoch-1.pid"
```

If running under Slurm or another scheduler, use its job script and assigned
resources instead of daemonizing outside the allocation. Set all variables in
the batch script; a new login shell does not retain your interactive exports.

## 13. Run RxR smoke and full evaluation

The same Python module handles RxR. These three inputs are additional:
`--dataset-kind rxr15`, `--simulator-source`, and `--ground-truth-path`, plus the
RxR config/episode template. R2R and RxR share scenes and model environment.

### 13.1 Two RxR episodes

```bash
cd "$SMV_ROOT"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind rxr15 --simulator-source "$SIMULATOR_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$RXR_CONFIG" \
  --data-path "$RXR_DATA" --ground-truth-path "$RXR_GT" \
  --scenes-dir "$MP3D_SCENES" --gpu "$EVAL_GPU" --split "$SPLIT" \
  --max-episodes 2 --out "$RESULT_ROOT/rxr-smoke/epoch-1" \
  > "$RESULT_ROOT/rxr-smoke-epoch-1.log" 2>&1
cat "$RESULT_ROOT/rxr-smoke/epoch-1/summary.json"
```

Inspect nDTW coverage as well as SR/SPL and failures. A missing GT file must not
be worked around by removing the nDTW measurement when reproducing this protocol.

### 13.2 Full RxR English-guide `val_unseen`, 3,669 episodes

```bash
cd "$SMV_ROOT"
mkdir -p "$RESULT_ROOT/rxr-full"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" setsid nohup \
  "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind rxr15 --simulator-source "$SIMULATOR_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$RXR_CONFIG" \
  --data-path "$RXR_DATA" --ground-truth-path "$RXR_GT" \
  --scenes-dir "$MP3D_SCENES" --gpu "$EVAL_GPU" --split "$SPLIT" \
  --out "$RESULT_ROOT/rxr-full/epoch-1" \
  > "$RESULT_ROOT/rxr-full/epoch-1.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RESULT_ROOT/rxr-full/epoch-1.pid"
```

Start with sequential R2R then RxR runs on a new server. Concurrent jobs share
GPU memory and make decision latency include contention. To evaluate `val_seen`,
set `SPLIT=val_seen`, ensure both split/GT files and scenes exist, and choose new
output paths. Use that split's actual expected count, not 1,839/3,669.

## 14. Other checkpoints, multiple GPUs, and launchers

For epoch 2 or another family, update `CHECKPOINT` to its verified export and
change **all smoke/full output paths and log/PID filenames**. Re-run the model
audit and two-episode smoke. Keep base revision, sensor protocol, dataset hashes,
episode selection, and failure accounting fixed when comparing policies.

For one modern direct worker on physical GPU 1:

```bash
export EVAL_GPU=1
```

Keep the `CUDA_VISIBLE_DEVICES="$EVAL_GPU"` prefix on the launch command, and
pass the same physical ID through `--gpu`. If you only change `--gpu`, the model
can stay on CUDA device 0 while Habitat moves to GPU 1. When a scheduler/container
already restricts the visible devices, validate its CUDA/EGL mapping first rather
than applying the standalone host example unchanged.

The original `streaming` worktree contains these local convenience launchers:

```text
scripts/eval/run_r2r_epochs.sh       three R2R-only FullContext-B epochs
scripts/eval/run_rxr15_epoch1.sh     one joint epoch on RxR
scripts/eval/run_window8_epoch1.sh   one Window8 epoch on R2R
scripts/eval/run_rxr15_after_r2r.sh  waits for an existing R2R job, then RxR
```

They were untracked at inspection and are **not guaranteed to exist in a fresh
checkout**. The direct commands above do not require them. If using a branch or
handoff containing these launchers, override their original-machine defaults:

```bash
export EVAL_PYTHON MODEL_PYTHON MODEL_PATH MP3D_SCENES R2R_DATA RXR_DATA RXR_GT
export SIMULATOR_SOURCE
export EVAL_GPUS='0 1 2'  # exactly three IDs for run_r2r_epochs.sh
export CHECKPOINT_ROOT="$EVAL_ROOT/checkpoints/fullcontext-b"
OUTPUT_DIR="$RESULT_ROOT/r2r-three-epoch-smoke" \
  bash scripts/eval/run_r2r_epochs.sh --max-episodes 2
```

That launcher expects the R2R-only epochs 1/2/3 under `CHECKPOINT_ROOT`, not the
joint one-epoch example. It sets masks internally; do not add another outer mask.
Its existing default `'0 0 0'` shares one GPU across all three models and may OOM
on a smaller new machine. Use single direct workers if resources are limited.

If present, `scripts.vln.download_eval_checkpoints` downloads/checksums the pinned
R2R-only FullContext-B epochs and the base model. It is also untracked in the
inspected original worktree. Use the explicit download snippet in section 10 when
it is unavailable. The original Window8 downloader resolves a live Hub revision
and then records it; retain `verified_revision.json` before comparing reruns.

## 15. Monitor, stop, resume, and validate results

### 15.1 Progress and logs

```bash
tail -n 60 "$RESULT_ROOT/r2r-full/epoch-1.log"
cat "$RESULT_ROOT/r2r-full/epoch-1/summary.json"
wc -l "$RESULT_ROOT/r2r-full/epoch-1/episodes.jsonl"
nvidia-smi
ps -eo pid,ppid,pgid,stat,etime,%cpu,%mem,cmd | \
  rg 'qwen_vl.eval.habitat_r2r|qwen_vl.stream.server'
```

Before the first episode finishes, `summary.json` may not exist. Check the log
for checkpoint loading and first-scene initialization. Kernel compilation and
large-weight loading can make the first decision slower than later decisions.

The modern output directory contains:

| File | Meaning |
|---|---|
| `evaluation_contract.json` | Checkpoint/data/config/source/selection identity |
| `runtime.json` | Model runtime and saved policy details |
| `episodes.jsonl` | One journal record per completed or failed episode |
| `summary.json` | Current metrics, denominator, completion, and latency |

The dataset is deterministically selected in scene/episode order by the modern
adapter. For a comparison subset, pass the same `--episode-list` JSON to every
worker and keep it in the experiment archive. A list contains **official string
IDs**, not integer array offsets.

### 15.2 Stop one background run

The `setsid` commands create a process group including the simulator and model.
First inspect the recorded PID and group, then terminate that run's group:

```bash
run_pid=$(cat "$RESULT_ROOT/r2r-full/epoch-1.pid")
ps -p "$run_pid" -o pid,pgid,stat,etime,cmd
kill -TERM -- "-$run_pid"
```

Confirm the PID still belongs to your run, since stale PID files can outlive a
process and PIDs may be reused. Avoid killing all Python or all GPU processes.
RxR uses its own PID file. Scheduler jobs should be canceled through the scheduler.

### 15.3 Resume a partial run

Re-run the **identical full command** with the same output directory and input
paths. The modern evaluator skips completed journal IDs and rejects changes to
weights, navigation metadata, source, data, config, runtime, or selection.

Do not use the smoke directory for a full run. Do not change the branch, model
environment, or data files halfway through a journal. A moved installation can
also change recorded absolute paths/interpreter identity and trigger rejection;
start a new run unless you preserve the original contract. Preserve a rejected
directory for diagnosis rather than deleting/editing its contract to force resume.

### 15.4 Verify completion and denominator before reporting scores

For R2R, run:

```bash
"$MODEL_PYTHON" - "$RESULT_ROOT/r2r-full/epoch-1" 1839 <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
summary = json.loads((root / 'summary.json').read_text())
contract = json.loads((root / 'evaluation_contract.json').read_text())
records = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
ids = [str(row['episode_id']) for row in records]
assert summary['complete'] is True, 'Partial run'
assert summary['scheduled_episodes'] == summary['episodes'] == expected
assert len(ids) == len(set(ids)) == expected, 'Missing/duplicate journal IDs'
assert set(ids) == set(map(str, contract['episode_ids'])), 'Wrong evaluated set'
failures = sum(bool(r['failed']) for r in records)
assert failures == summary['failures']
assert not any(r.get('fatal') for r in records), 'Fatal infrastructure failure'
sr = sum(0 if r['failed'] else r['metrics'].get('success', 0) for r in records) / expected
spl = sum(0 if r['failed'] else r['metrics'].get('spl', 0) for r in records) / expected
assert abs(sr - summary['sr']) < 1e-12
assert abs(spl - summary['spl']) < 1e-12
print('Complete:', expected, 'failures:', failures)
print(f'SR: {100 * sr:.2f}%  SPL: {100 * spl:.2f}%')
print('Failure reasons:', summary.get('failure_reasons'))
print('nDTW:', summary.get('ndtw'), 'coverage:', summary.get('ndtw_coverage'))
PY
```

For RxR, run the same validator with
`"$RESULT_ROOT/rxr-full/epoch-1" 3669`. Additionally inspect `ndtw_coverage` and
missing GT/metric errors. The R2R core config does not request nDTW, so `ndtw:
null` is expected there. The RxR config requests it explicitly.

`sr` and `spl` are fractions; multiply by 100 for percentages. Failed episodes
remain in those denominators. Secondary metrics such as navigation error and
nDTW report the average over available finite values with coverage counts; do
not call that failure-adjusted coverage a complete-denominator mean unless it
actually covers all episodes. Retain failure reasons alongside scores.

`model_seconds` covers policy preprocessing/encoding/decision and commitment.
`transport_and_model_seconds` includes pipe/RGB transfer. `environment_seconds`
includes Habitat stepping and rendering. Concurrent latency is not comparable
with an isolated inference-speed probe. The model retains FullContext KV for
FullContext exports, so long-episode decisions can cost more than early ones.

## 16. Legacy evaluator branches

Use this section only when intentionally running the committed older adapter on
`streaming`, `streaming_logits`, `streaming_logits_dual`, or their older report
branches. Prefer the compatible modern branch for fresh benchmark work.

The old CLI has **required** `--data-root`, `--habitat-config`, and
`--episode-list`; it lacks `--dataset-kind rxr15`, `--max-episodes`, and explicit
data/GT flags. It imports `habitat_baselines.config.default`, so add
Habitat-Baselines **0.2.4** in the simulator environment. Its dependencies may
include Torch and training packages; keep them separate from the model `.venv`:

```bash
"$EVAL_PYTHON" -m pip install \
  -c "$SMV_ROOT/docs/evaluation_setup/habitat-constraints.txt" \
  -e "$HABITAT_SRC_ROOT/habitat-lab/habitat-baselines"
"$EVAL_PYTHON" -m pip check
```

The adapter hardcodes these paths under `--data-root`:

```text
scene_datasets/
datasets/r2r/{split}/{split}.json.gz
```

Create that layout without duplicating all scene files:

```bash
export LEGACY_ROOT="$EVAL_ROOT/legacy-data"
mkdir -p "$LEGACY_ROOT/datasets/r2r"
ln -s "$MP3D_SCENES" "$LEGACY_ROOT/scene_datasets"
for split in val_seen val_unseen; do
  mkdir -p "$LEGACY_ROOT/datasets/r2r/$split"
  ln -s "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/$split/$split.json.gz" \
    "$LEGACY_ROOT/datasets/r2r/$split/$split.json.gz"
done
"$EVAL_PYTHON" - <<'PY'
import gzip, json, os
from pathlib import Path
path = os.environ['R2R_DATA'].format(split=os.environ['SPLIT'])
with gzip.open(path, 'rt') as f:
    episodes = json.load(f)['episodes']
ids = [str(e['episode_id']) for e in episodes]
assert len(ids) == len(set(ids))
root = Path(os.environ['RESULT_ROOT'])
(root / 'r2r-legacy-full-ids.json').write_text(json.dumps(ids))
(root / 'r2r-legacy-smoke-ids.json').write_text(json.dumps(ids[:2]))
PY
CUDA_VISIBLE_DEVICES=0 "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --checkpoint "$CHECKPOINT" \
  --habitat-config "$SMV_ROOT/docs/evaluation_setup/vln_r2r.yaml" \
  --data-root "$LEGACY_ROOT" --simulator-source "$SIMULATOR_SOURCE" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --split "$SPLIT" --episode-list "$RESULT_ROOT/r2r-legacy-smoke-ids.json" \
  --out "$RESULT_ROOT/r2r-legacy-smoke"
```

Skip `ln -s` for links already present; verify existing links rather than blindly
replacing them. For the full legacy run, use the full ID list and a new directory.
This legacy example uses physical GPU 0, matching the bundled YAML's GPU field;
the old adapter has no `--gpu` option. For another physical GPU, use a copied
config with its GPU field explicitly changed and a matching CUDA mask.
The legacy adapter has weaker resume/progress contracts, PNG RGB transport, and
different selection handling; it does not produce the modern `complete` fields.
Check its journal against the exact supplied ID list. Do not use the modern
validator unchanged on legacy output.

**RxR15 is not supported by that old CLI.** Select the modern implementation for
your model family; changing only YAML or renaming files does not add RxR loading,
language filtering, or GT metric setup. `main` has no adapter at all. This guide's
distribution deliberately does not replace branch implementations.

## 17. Troubleshooting

| Symptom | Check / corrective action |
|---|---|
| `No module named qwen_vl` | Run from the repo and export `PYTHONPATH="$SMV_ROOT/src:$SIMULATOR_SOURCE"`; use the intended interpreter. |
| `No module named habitat` | You launched with the model Python or did not install Habitat-Lab in `EVAL_PYTHON`. |
| `No module named habitat_baselines` | You have the legacy adapter. Switch to the modern branch or install matching Baselines in the simulator environment. |
| `No module named habitat_extensions` | `SIMULATOR_SOURCE` must be the directory containing that package. |
| Hydra cannot find `rxrvlnce_v1` / `ndtw` | Import bundled `task`/`measures` before config composition; pass RxR's simulator-source flag. |
| Gym / NumPy ABI or `np.float` errors | Keep simulator NumPy 1.26.4; inspect `pip check` and dependency versions. Do not install model requirements there. |
| `unrecognized arguments: --dataset-kind` | Old branch adapter; use section 2/16 rather than modifying flags at random. |
| Cannot find `.../mp3d/mp3d/...` | `--scenes-dir` should be parent of `mp3d`. |
| A later episode has no scene | Transfer all scenes referenced by both datasets; rerun the all-scene check. |
| `.navmesh` missing or infinite goal distance | Transfer original navmeshes with GLBs; verify the scan/split paths. A render alone does not prove navigation geometry works. |
| EGL device/context error | Check host/container NVIDIA graphics/EGL driver libraries and GPU exposure. Unset an unintended global CUDA mask. Temporarily unset quiet log variables to inspect renderer errors. |
| `habitat_sim.cuda_enabled` is false | Expected for the observed renderer; test real RGB instead of assuming CUDA interop is required. |
| Source build fails in CMake policy checks | Use CMake <4 and a supported Ubuntu compiler; preserve build logs. |
| `distutils` missing during source build | Use Python 3.10 and compatible setuptools in the simulator build environment. |
| GT `KeyError` or missing `locations` | Pair the exact guide/English episode file and guide GT; run the coverage check. |
| 30° dataset with 15° simulator | Use the verified RxR15 conversion; a filename change is insufficient. |
| FlashAttention undefined symbol / import error | Use locked cp312/Torch2.10/cu129 wheel in `.venv`; check for an injected `PYTHONPATH` or copied incompatible environment. |
| Model CUDA unavailable | Check driver/GPU allocation in the model interpreter; `nvidia-smi` alone does not validate Python imports. |
| Strict state-dict/serializer mismatch | Wrong model family/branch, base revision, or incomplete export. Inspect `navigation.json`; do not relax strict loading. |
| Model timeout or fatal CUDA error | Inspect the worker log, GPU memory, selected checkpoint and native backend. Preserve journal/contract; do not report partial scores as full results. |
| GPU OOM with parallel epochs | Stop that run's group; run one worker at a time, free unrelated memory, and retain expandable allocations. Do not crop FullContext to claim the same policy. |
| Resume contract changed | Use original source/environment/selection/input paths, or start a new result directory. |
| No summary yet | First episode may be loading/compiling; inspect log and process activity. |
| Hub download produces HTML / Git redirects to proxy authentication | Resolve network/proxy/account access or transfer verified snapshots. HTML login pages are not checkpoint files. |
| `uv sync --locked` wants to change the lock | Confirm correct branch/uv/Python; do not regenerate the lock for an evaluation reproduction. |

## 18. Archive a reproducible setup

Save a record of both environments and source pins after checks pass:

```bash
mkdir -p "$RESULT_ROOT/setup-record"
git -C "$SMV_ROOT" rev-parse HEAD > "$RESULT_ROOT/setup-record/simplememvln-commit.txt"
git -C "$SMV_ROOT" status --short > "$RESULT_ROOT/setup-record/simplememvln-status.txt"
git -C "$HABITAT_SRC_ROOT/habitat-lab" rev-parse HEAD \
  > "$RESULT_ROOT/setup-record/habitat-lab-commit.txt"
"$EVAL_PYTHON" -m pip freeze > "$RESULT_ROOT/setup-record/habitat-pip-freeze.txt"
conda list -n habitat-vln310 --explicit > "$RESULT_ROOT/setup-record/habitat-conda-explicit.txt"
uv pip freeze --python "$MODEL_PYTHON" > "$RESULT_ROOT/setup-record/model-pip-freeze.txt"
cp "$SMV_ROOT/uv.lock" "$RESULT_ROOT/setup-record/uv.lock"
cp "$SMV_ROOT/docs/evaluation_setup/provenance.json" "$RESULT_ROOT/setup-record/"
nvidia-smi > "$RESULT_ROOT/setup-record/nvidia-smi.txt"
"$MODEL_PYTHON" --version > "$RESULT_ROOT/setup-record/model-python.txt"
"$EVAL_PYTHON" --version > "$RESULT_ROOT/setup-record/habitat-python.txt"
sha256sum "$R2R_CONFIG" "$RXR_CONFIG" \
  > "$RESULT_ROOT/setup-record/habitat-config-sha256.txt"
```

For a source-built simulator, also record its commit and build flags. Record
datasets/GT hashes, the chosen Hub revision/checksum manifest, and complete result
contracts/journals. An explicit Conda export will not capture editable Git source
or a manually built simulator; archive those separately.

Do not copy `.venv` to another server: it can contain absolute interpreter
symlinks and compiled wheel/toolchain assumptions. Recreate with `uv.lock`. Conda
packing can help migrate a compatible simulator environment, but editable
Habitat-Lab paths must still be restored, and EGL/driver compatibility must be
tested again. A raw full `pip freeze` from the existing simulator environment also
includes ROS/local paths and unrelated training packages; it is a provenance
record, not a clean new-server requirements file.

To reproduce offline, transfer the selected Git branch/bundle, locked model
wheels and Python distribution, Habitat source including submodules or matched
Conda packages, exact dataset/scene files, verified base/policy snapshots, and
the experiment record. A Git bundle alone contains none of those large assets.

## 19. What was verified and what still needs a new-server check

This guide's version pins come from live imports/package metadata in the existing
Python environments, the local source commits, configs, checkpoint manifests,
dataset conversion manifest, and previous evaluation contracts. Bundled RxR
registrations retain the original bytes and license. All-branch publication
adds documentation/setup assets and preserves each branch's implementation.

The accompanying checks validate that the bundled configs/registrations compose,
both audited datasets load with the expected counts, GT covers RxR IDs, scenes
and navmeshes resolve, and the existing server can render real observations.
They do not substitute for installing on your new machine and running its model
smoke. No fresh full benchmark or full clean OS install is implied by writing
this tutorial, and no public download of the derived RxR15 files is assumed.

Primary upstream references:

- [Habitat-Sim 0.2.4 installation](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.2.4/README.md)
- [Habitat-Sim 0.2.4 source build](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.2.4/BUILD_FROM_SOURCE.md)
- [Habitat-Lab 0.2.4 source](https://github.com/facebookresearch/habitat-lab/tree/v0.2.4)
- [VLN-CE dataset release instructions](https://github.com/jacobkrantz/VLN-CE)
- [Matterport3D access](https://niessner.github.io/Matterport/)
- [SpatialStack extension source](https://github.com/phamquandung/SpatialStack/tree/b1f435642e5397dbd1523d4c58efe504402b881e/src/habitat_extensions)

The simulator versions and derived RxR recipe in this guide are determined by
this project's observed evaluation environment, not by upstream's latest defaults.
