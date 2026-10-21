# DINOv3 G3 Implementation Plan

> Execute inline using superpowers:executing-plans. User requested implementation.

**Goal:** Replace G3's online teacher with frozen ViT-L/16 while preserving its student contract.
**Architecture:** Dedicated preprocessing/teacher adapter; explicit teacher dispatch;
validated 1024D head; identity-bound runtime preparation and existing recovery/export.
**Tech Stack:** Existing PyTorch2.10, Transformers5.11, Pillow, YAML, pytest.
**Spec:** G3_dinov3_vitl16_design.md.

## Constraints and review focus

No PCA/cache, no dependency upgrade, no changes to G3 sampling/native recurrence.
Only worker-2; no modification of existing jobs. Reference inference FP32 SDPA.
Reject mismatched model geometry, partial weights, stale identities, nonfinite
features, and attempted Omega recovery. Empty episodes and duplicate source
indices must preserve global normalization and sample order.

## Tasks

- [x] Add failing CPU tests for pixel normalization, patch extraction/pooling,
  selected-teacher dimensions, old Omega defaults, and malformed config rejection.
- [x] Implement data/dinov3_online.py and research/dinov3_teacher.py. Validate
  ViT-L/16 geometry and load locally, freeze, isolate RNG, strip special tokens,
  pool and reuse region_targets. Validate identity before model loading.
- [x] Wire teacher dispatch in episode_dataset/train_episode and pass validated
  target_dim to NativeStateHead in nav_model. Preserve existing Omega code.
- [x] Add preparation CLI and template/launcher with one-epoch fixed0.1 recipe;
  seal actual teacher identity only after authorized weights are present.
- [x] Test adapter gradients, empty support, batching, source mapping, recovery
  mismatch, and export. Run relevant existing CPU regressions.
- [ ] Qualify real pretrained teacher and training on worker-2 if access permits;
  otherwise document this exact outstanding gate without claiming readiness.
- [x] Record implementation, commands, results, and remaining limitations in reports.

## Execution record

Implemented inline under the user's instruction to revise and implement.
Ruling: the existing Transformers5.11 DINOv3 class loads the official converted
HF safetensors; pin its installed implementation and official model/config hashes,
rather than adding a second vendor checkout and conversion path.
Ruling: no extra approvals for implementation; the user explicitly authorized it.
Ruling: no production run or pretrained claim while model access is pending;
user confirmed they will request access and asked to finish implementation.
Independent review required config pinning and launcher preflight; both addressed
with failing-then-passing regressions. See reports/dinov3_g3/implementation.md for
verification results, environment corrections and the pending GPU gate.
