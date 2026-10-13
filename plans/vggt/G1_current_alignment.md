# G1 — Online VGGT-Omega current-frame supervision

**Active specification, revised 2026-10-13 at the user's request.** This supersedes the offline/PCA G1 sections of the original five-run plan. Other arms remain future proposals. No teacher pre-extraction or PCA is used in G1 training.

## Architecture

Base c2_h2_speed@2430fd2, native Window8 canonical-text policy, no lane or episodic bank, frozen student visual tower/merger, trainable native decoder with full-episode BPTT. At <=16 fixed stratified current frames per trajectory, align complete zero-based decoder23 output visual rows. Original image-token order, MRoPE, action targets and sequence lengths are unchanged.

Frozen official VGGT-Omega-1B-512 sourcead02778c and weights SHA256c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934 run ONLINE per rank. Every teacher call is [B,1,3,448,592], single view; only aggregator23 cached.17 special tokens,28x37 patches,2048channels. Heads disabled. Teacher preprocessing follows official balanced512 exactly. CPU loader stages sampled RGB tensors; trainer computes detached targets before student forward. Teacher is never registered in the student optimizer/DDP/checkpoint. Resume binds actual source/weight/preprocessing/dtype hashes.

Teacher features: adaptive average pool grid to15x20; FP32 no-affine LayerNorm2048 eps1e-6. No PCA, no feature cache and no trainable teacher-side compressor. Position term is .5*E0 at width2048, where E0 follows the original G1 normalized-center sin/cos rule, now512frequencies/axis and globalRMS1. Teacher LN makes target RMS approximately1 without fitting a dataset statistic. Position formula is versioned in resume provenance.

## Student dimension conversion — exact SpatialForcing architecture

Follow SpatialForcing-VLN/src/qwen_vl/model/spatial_forcing.py:SpatialForcingProjector:

LayerNorm(2560,elementwise_affine=True,eps=1e-5)
-> Linear(2560,4096,bias=True) -> GELU -> Linear(4096,2048,bias=True).

18,885,632 trainable parameters. Normal(.02) matrices, zero biases, unit LN weight; CPU-local seed429 without resetting CUDA RNG. The teacher supplies [F,300,2048], student projection matches it directly. This updates the old2560->2048->256 no-affine projector as explicitly requested.

FP32 cosine eps1e-6, average300positions per frame, sum frames. Per-update global rank/GAS denominator. Lambda ramps0->.2 over771updates; headLR1e-4, backboneLR5e-6. Navigation remains global per-action mean with EOS. Log both losses separately.

## Memory and runtime

Active allocation4925,worker-2,4H100; global8 via4ranks*GAS2. Try resident frozen teacher and ZeRO2 GPU optimizer first. Compare teacher microbatches4/8/16, CPU loader/threads and vision/attention batch settings while preserving frame/action exposures. If memory requires it, qualify sequential teacher CPU offload (online recompute each episode) or optimizer CPU offload; report the overhead. FP32 teacher chosen after BF16-vsFP32 diagnostic mean cosine.97589; FP32 batch-size relativeL2error1.3e-6, independent batch-image replacement exactlyzero. Student remains BF16. Clone teacher targets outside inference_mode. No graph through teacher.

Re-audited longest R2R183frames/58290tokens; RxR627frames/199735tokens. Profile complete longest episodes, near65536threshold, mixed lengths and at least3optimizerupdates. Record allrankGPU/hostpeaks and newOOMcounters. Never truncate or switch toTBPTT to pass.

## Recovery and export

Recovery includes student+projector, optimizer/scheduler/RNG/sampler and exact teacher contract. Strict serving export removes only research_aux and researchconfig; retains all native weights and uses original Window8 path. No Omega process/projector/targets at inference. Verify strict reload and response replay.

## Scope and status

Implement G1 and diagnostic speed/resource/recovery runs only. No other full scientific arm, no baseline retraining. A full production launch requires matched baseline/schedule provenance and completed qualification; diagnostic steps are not production progress. Report measured R2R/RxR timing and uncertainty separately. Existing M2job4879 and all existing weights remain untouched.
