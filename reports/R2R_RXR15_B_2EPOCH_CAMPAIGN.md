# Joint R2R + RxR_15deg FullContext+B

Approved recipe: one model from pretrained Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, two epochs, LR `5e-6`,
global batch eight (four H100 ranks, one episode/rank, GAS two), seed 429.
Frozen vision/merger; full-episode gradients; native causal FlashAttention;
serializer `vln_append_only_chat_v1`. No cropping, episode dropping or TBPTT.

The audited manifest contains 10,819 R2R and 19,996 English RxR_15deg episodes,
3,128,624 action labels, and a maximum of 627 frames / 199,735 encoded tokens.
RxR annotations come from `RxR_CE_15_deg_Trajectory_data/train/train_guide.json.gz`,
not the original 30-degree metadata. All English 15-degree episodes are present.
Filename/action/chronology/final-STOP checks passed. Capture-before-action Habitat
replay is not independently verified; this model is **not yet evaluated in Habitat**.

Schedule: 3,852 updates/epoch, 7,704 total, 232 warmup updates, one two-epoch
cosine-with-minimum-LR schedule. Save and retain both epoch checkpoints including
optimizer state. Epoch one is a mid-schedule snapshot, not a separately trained run.
The runtime schedule guard aborts on any mismatch. Reports cover the first 50
updates and each epoch. No automatic publication is requested for this campaign.

## Memory and correctness gates

The original configuration OOMed on the 627-frame episode during checkpoint
recomputation (75.69 GiB allocated plus a requested 1.91 GiB).
The accepted configuration offloads non-reentrant checkpoint inputs to pinned CPU
memory for episodes of at least 65,536 tokens. It does not change attention,
tokenization, or gradient horizon. CPU/GPU tests verify exact gradient agreement;
the GPU test also verifies reduced allocation. The full H100 suite passed 35 tests.

The four-rank short/median/p95/longest smoke completed two optimizer updates with
optimizer state allocated, using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Peak reserved memory was **76.787109375 GiB**, below the 78 GiB gate. No OOM;
logged aggregate smoke loss was about 0.275. This is not a navigation-quality result.
The failed non-offloaded profile is retained separately for provenance.

Resources: four H100s, 24 CPUs, 256 GiB host memory, four loader workers/rank,
OMP/MKL threads two, 48-hour Slurm limit. Node-local temporary storage avoids NFS
multiprocessing cleanup failures. Free shared disk was about 7.8 TiB; the runner
requires at least 250 GiB. Source/config and manifest SHA256 must still match the
successful smoke before training starts.

Remote campaign: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_b_bs8_e2_20260929`.
The source snapshot and all smoke logs are retained there. Interactive allocation
4469 is preserved; only the previously authorized Window8 training step was canceled.
