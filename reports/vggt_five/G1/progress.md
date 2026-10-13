# G1 Omega execution ledger — plans/vggt/G1_omega_execution.md

Base2430fd2; branch c2_vggt_omega_g1; job4924 worker-2,2H100,768GiB. No existing job mutated.

- Read all G1/shared/master/resources specs and source audit, H2 implementation report, SpatialForcing VGGT adapter and Omega official code. Source audit had no executed VGGT evidence.
- User overrides teacher with Omega. Existing512 checkpoint chosen; pinned server sourcead02778c. Full teacher provenance will be recorded by extraction.
- Test-first: new targets/head imports failed before implementation; new functional-forward import failed before implementation. First implementation:17passed,1failed due float32 reassociation in test using exact equality. Replaced that mathematical comparison with1e-7 tolerance, not a scientific accuracy tolerance.
- Native-off and checkpoint/no-future-gradient tiny model tests passed in first run.
- Full corpus audit and selected diagnostic cache underway. No production launch or quality evaluation yet.

## User revision: online teacher, SpatialForcing MLP, four GPUs

Pre-extraction/PCA diagnostic completed608frames but user superseded this design; these targets are NOT used for training. Removed cache/extractor from active implementation. Teacher path now online FP32, full2048dimension targets, SpatialForcing2560->4096->2048 affineLayerNorm projector. User replaced allocation4924 with4925(worker-2,4H100,60CPU,768GiB). Global8=>GAS2. No production run started.

Independent review caught CUDA RNG reset by torch.manual_seed inside CPU-only fork. Added real CUDA RNG test, observed FAIL; replaced with CPU default_generator.manual_seed. Projection-reuse/cachePE issues obsolete after removingcache/PCA. Online review requires FP32 image staging, inference-tensor clone, teacherhash binding and actual grid/replay/resource checks.

Initial regression328passed12skipped4failed: four existing helper tests require real.git checkout; remedy transfer actual branchGitobjects; no patch to weaken tests. Enable VLN_MODEL_PATH to run12processor tests rather than skip.
