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

## Four-GPU acceptance update

Mixed3update smoke passed (17.78s/update,56.64GiBreserved). Full627frameRxR3update passed residentteacher/GPUoptimizer (90.30s,76.85GiBreserved,device81073MiB): too little practical margin. TeacherCPUbetween3updates passed (86.78s,76.97GiBreserved); CPUoptimizer3updates passed (80.31s,active~62GiB,reserved76.31GiB). CPUoptimizer is the safer tuning base. Initial selection mistakenly treated all reserved allocator cache as active demand and rejected both; selection now requires active allocation<=72GiB plus the unchanged78GiBreserved gate. No failed/OOM experiment was relabeled.

Real4rankNCCL/DDP/GAS reduction passed unequal support, all-empty and equal support, maxgradienterror2.98e-8. Broad regression350passed1failed; failure was optionalCPUAdam extension CUDA12.8-vsPyTorch12.9 builder mismatch. Added isolated hash-verifiedNVIDIA12.9nvcc/cudart/CCCL components, reusing matchingTorch-bundledcuBLAS12.9.1.4/cuRAND10.3.10.19 throughread-onlysymlinks. No driver/system/Pythonpackagechange. CPUAdam targeted2tests passed, including5AdamWtransitions/moments vsTorch and serializedstate continuation.

Sampling identity explicitly revised to collectedRGBpathsequence; no physicalalias guarantee across separatelynamedepisodefolders. Numeric sample algorithm unchanged. Additional gradient tests, actualprocessor/merger roworder, localteacheradapterhashes and correctedteachertiming passed in broadregression. Independentreview requiresprojectorbiasmovement (notmatrixdecayalone) and source/exportlogitparity; acceptance scripts nowenforceboth.

User now explicitly requests one full joint R2R+RxR epoch on4H100. Ruling: use a real one-epoch cosine schedule (3852updates,3%warmup116), retain G1auxramp771. Separate overlay/locked guarded launcher prepared; immutable snapshot and production launch after acceptance. Fulljoint12updatespassed; warmupdates3..12=27.5356s =>29.46h/epoch short-run projection, more conservative than sum of repeated representative estimates3.75+13.36h. Mixed-extremepassed; CPUAdam4rankrecovery underway. Final focused reviewer found no blocker in one-epoch config/launcher; one repeated final exposure and 20%auxramp explicitly documented.

Final acceptance:23/23fourGPUtrainingdiagnosticsPASS; mixedextremeactive61.79GiB/reserved68.42GiB/RAM263.98GiB. CPUAdamfreshresumePASS(headexact,globalrelL2=9.45e-9,finallossexact). Exportcheckerattrbugfixed(getattrmissingnative lane); source/exportlogitsexact,bothdatasetsreplay/retry/reset/evictionPASS. Cleanfinalsuite351passed115warnings32.95s,noskips. All src hashesmatchjointbenchmark. OneepochconfigvalidationPASS. Proceedingto committedimmutableproductionlaunch.
