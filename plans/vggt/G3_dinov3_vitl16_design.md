# G3 with an online DINOv3 ViT-L/16 teacher

## Scope and baseline

Replace the frozen VGGT-Omega teacher in original G3 with DINOv3 ViT-L/16
LVD1689M. Start from c2_vggt_omega_g3_fixed01_seed430 at a989b81; isolate
implementation on c2_dinov3_g3. This is original G3, without M5 or A1.
The user selected ViT-L/16 and requested revision and implementation.

## Teacher and target contract

Use the installed, pinned Transformers 5.11.0 DINOv3ViTModel with an authorized
local snapshot of facebook/dinov3-vitl16-pretrain-lvd1689m, revision
ea8dc2863c51be0a264bab82070e3e8836b02d51.
No runtime download, remote code, dependency upgrade, PCA, feature cache, teacher
projection, or teacher optimization. Load strictly; reject incomplete weights.
The HF implementation is deliberately selected because the repository publishes
native safetensors for it and this environment already pins its dependency.

Decode original RGB 640x480, preserve aspect and all pixels, scale to [0,1],
normalize with ImageNet mean (0.485,0.456,0.406) and std (0.229,0.224,0.225).
No resize/crop/augmentation. Independently batch selected source frames as
[B,3,480,640]. Teacher is resident FP32, eval, frozen, inference_mode, SDPA;
autocast is disabled. This is the reference numerical contract; BF16 is not
silently substituted to improve speed.

Use final normalized hidden states after all 24 transformer blocks. Remove
exactly one CLS and four register tokens. The remaining [B,1200,1024] tokens
are row-major patches on a 30x40 grid. Average each non-overlapping 2x2 cell
block to obtain 15x20; apply non-affine per-cell LayerNorm(1024, eps=1e-6).
Apply the unchanged G3 region reducer: top-left/top-right (7x10),
bottom-left/bottom-right (8x10), and all 300 cells. Output [B,5,1024].
The global mean is area weighted, not the unweighted mean of four regions.
No extra positional features. These are dense visual feature targets, not
an assertion that DINO reproduces Omega's explicit geometry supervision.
Return detached ordinary tensors (not inference tensors) for loss backward.

## Student and objective

Preserve native GDN state supervision at zero-based decoder layer 24. Keep
source/read lag 8..32, quotas 3/3/2, at most eight pairs, trajectory-path sampling,
causal observation anchors, detached query context, identity probe rows, full
BPTT, and global valid-pair normalization. Teacher images/features cannot enter
the student navigation input or probe-query computation.

Retain the state head's query MLP and 4096->1024->D decoder; only D changes from
2048 to 1024. No inference architecture changes. Teacher selection determines
and validates D; old Omega configurations keep their original defaults and
serialized contracts. Serving export removes research_aux and research config.

Recipe: original Qwen3.5-4B base, joint R2R + RxR15deg, seed/head/sample seed430,
one epoch, 3852 updates, warmup116, cosine to 0.1 of peak LR, backbone5e-6,
head1e-4, Adam betas0.9/0.95, WD0.01, clip1, global batch8, 4H100, GAS2,
constant L_nav + 0.1*L_aux from update zero. Preserve Window8 canonical text
history, ZeRO2 CPU optimizer, activation offload and loader settings.

## Identity, recovery and operational safety

Record official model/revision, actual weight/config SHA256, installed teacher
implementation SHA256/version, preprocessing and adapter hashes, token offset,
feature dimension, pooling, normalization, precision and attention backend.
Verify these before loading a teacher. Existing config recovery comparison
must reject teacher, dimension, or identity changes. Separate run directories;
never resume an Omega checkpoint with this changed head. Preserve CPU/CUDA RNG
when constructing the frozen teacher. The model is trainer-owned, never inside
student DDP, optimizer or serving state. Empty pair batches return [0,5,1024]
and use the existing zero-loss participation path.

Only worker-2 may be used for GPU qualification. Do not touch existing jobs.
Training launcher must require a prepared, identity-bound runtime config and
use an isolated output directory with the existing recovery machinery.

## Validation and limits

CPU: exact preprocessing and patch order; CLS/register exclusion; five region
means; empty/malformed/nonfinite tensors; frozen teacher and student gradients;
construction RNG preservation; old Omega config regression; resume identity
rejection; dataset source-frame alignment; namespace-only serving export.
Use a deterministic synthetic model only as an explicitly labeled adapter
fixture, never as training weights or a claimed pretrained smoke result.

Before a production launch: real authorized weights, four-H100 batch1/2/4/8
measurements, forward/backward and several updates, longest R2R/RxR episodes,
checkpoint reload/recovery and serving export. Existing G3 GPU qualification
is not evidence that this new teacher is qualified. Report unavailable checks
and measured performance separately. Current account access returned HTTP403;
implementation can proceed but pretrained GPU qualification requires access.

## Primary references

- https://github.com/facebookresearch/dinov3
- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- Installed transformers/models/dinov3_vit implementation (version5.11.0).
