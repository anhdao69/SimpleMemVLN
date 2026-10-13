# G1: online VGGT-Omega supervision on four H100 GPUs

**Status: implementation, speed/resource qualification and final acceptance passed.** All 351 regression tests passed (115 warnings, 32.95 seconds), and all 23 four-GPU training diagnostics passed. The user-authorized one-epoch production run is now active in Slurm step **4925.16**, worker-2, from committed source **86d68b2eff9db7c5cffdd86f0d98ca98f7e1c93a**. Diagnostic updates are not production progress.

## Scope and source isolation

The implementation starts from `c2_h2_speed` commit `2430fd235b0752e6119d1af809cb14182dc0d404`, in branch `c2_vggt_omega_g1`. The first implementation commit is `8861f72`. Local work is isolated in `/Users/hoanganh692004/Desktop/SimpleMemVLN-c2_vggt_omega_g1`; the server checkout is `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-c2_vggt_omega_g1`. Existing training snapshots and M2 job 4879 were not modified.

The user replaced the original two-GPU interactive allocation with job **4925**, on **worker-2**, with **four H100 80GB GPUs, 60 CPUs and 768 GiB host RAM**. All distributed G1 tests use four ranks, one complete episode per rank per microstep, and two accumulation slots: **eight episodes per optimizer update**.

The original five-run plan used offline VGGT features and PCA. The user explicitly changed G1 to online VGGT-Omega and asked to follow SpatialForcing's dimension conversion. The active G1 implementation uses neither pre-extracted targets nor PCA. The other proposed scientific arms are outside this implementation.

## Exact model and loss

The student is the existing native Window8 Qwen3.5-4B text policy, with canonical action-text history and frozen vision tower/merger. It has no dual lane or episodic bank in G1. Decoder training uses full episodes and full BPTT. No frame, action, response token or long episode is removed to make a test fit.

At selected frames, the auxiliary path reads the original 300 visual-token positions from the full residual output of zero-based decoder layer 23. It adds no tokens and does not change MRoPE, logical positions, causal/window masks, action labels or the native vocabulary head.

The projector follows `SpatialForcing-VLN/src/qwen_vl/model/spatial_forcing.py:SpatialForcingProjector`:

```
LayerNorm(2560, affine=True, eps=1e-5)
Linear(2560, 4096, bias=True)
GELU
Linear(4096, 2048, bias=True)
```

There are **18,885,632 trainable projector parameters**. Matrix initialization is Normal(0, 0.02), biases are zero, and LayerNorm weight is one. Initialization uses a CPU-local generator seed of 429 and preserves both the CPU and CUDA random streams. The teacher has no trainable compressor; the student learns the conversion into the teacher's full 2048-dimensional feature space.

For every selected frame, frozen Omega receives an independent single-view input. Shape is `[B, 1, 3, 448, 592]`: the second dimension is always one. Batching multiple sampled images along the view dimension would change supervision by mixing views, and is rejected. Only aggregator layer 23 is retained; camera, depth and alignment heads are disabled.

The teacher returns 17 special tokens followed by a 28×37 patch grid with 2048 channels. The implementation uses the returned special-token offset, checks the actual shape, pools the spatial grid to 15×20 and applies FP32 no-affine LayerNorm over 2048 channels with epsilon 1e-6. It adds `0.5 * E0`, where E0 is the original G1 center-coordinate sin/cos position rule expanded to 2048 channels and normalized to global RMS one. It does not use SpatialForcing's separate positional-target recipe; only the requested projector architecture is adopted from that project.

The loss is FP32 cosine distance, averaged over the 300 spatial positions of each frame and summed over selected frames. Across ranks and both accumulation slots:

```
L = sum(navigation action losses) / global action count
  + lambda(update) * sum(auxiliary frame losses) / global selected-frame count
lambda(update) = 0.2 * min(1, completed_updates / 771)
```

The local loss includes the world-size factor that cancels distributed gradient averaging. The existing Trainer/DeepSpeed path does not divide by accumulation a second time. Auxiliary and navigation counts remain separate. The auxiliary coefficient is zero on the first update and begins increasing afterward. Projector LR is 1e-4; native backbone LR is 5e-6; diagnostics retain the original two-epoch schedule, while the later user-requested production overlay supplies the explicit one-epoch schedule described below.

## Sampling and declared plan adaptation

Sampling selects `min(16, T)` unique frames by fixed SHA256 draws within equal temporal strata. It is deterministic across epochs. Every navigation action still trains.

The audited manifest has no verified map of physical trajectory aliases across instruction-specific episode folders. The implemented key therefore identifies a **collected image sequence**: dataset, scene and ordered RGB paths normalized below `trajectory_data/`. Changing instruction text or UID while keeping those paths does not change samples. Physically identical trajectories copied into different folders are not merged. This is explicitly documented in the revised plan, rather than claiming the original cross-instruction physical-alias guarantee. Such a guarantee would require a separate verified metadata/pose or image-identity registry; that registry would not require teacher feature extraction.

## Online teacher lifecycle and precision

CPU loader workers preprocess only sampled images. The Trainer consumes these FP32 images before DeepSpeed's ordinary input transfer can cast floating inputs to BF16. The teacher runs frozen, in eval mode and under inference mode. Targets are cloned outside inference mode so student backward can safely save detached target tensors. Only small pooled targets survive into the active student microbatch.

The teacher belongs to the Trainer, not the student module. It is absent from DDP, optimizer groups, student state dictionaries and serving exports. It is reconstructed from pinned assets after recovery. Resident mode keeps one teacher on each GPU; `cpu_between` moves it back to CPU before each student forward. CPU optimizer offload is a separate option and does not change online target computation.

FP32 teacher inference was chosen after explicit diagnostics. BF16 versus FP32 target features had mean cosine 0.975892 and relative L2 difference 0.21958. FP32 batch-two versus batch-one inference had relative L2 difference 1.306e-6, maximum absolute difference 0.000227928 and mean cosine approximately one. Replacing another independent batch image did not affect the retained image's BF16 output. These diagnostics do not claim BF16 Omega is generally invalid; they explain this implementation's conservative precision choice. The student remains BF16.

Teacher-only FP32 batch sizes 4/8/16/32 measured 17.49/17.94/18.14/17.88 frames/s on one GPU. These are not distributed student-training rates.

## Immutable inputs and evidence location

| Item | Value |
|---|---|
| Qwen revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Official Omega source commit | `ad02778c052e636a623ef425377f342ee41c4fa6` |
| Omega source-tree SHA256 | `b0e8d9d7ca0ee3ac8f819621a3d1c2ab5523bc57cbbc93e995aadfc5260a97cb` |
| Omega weight SHA256 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |
| Training manifest SHA256 | `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933` |
| Evidence root on server | `/mnt/data/vmo-ai-task/anhdh35/g1-omega-20261013` |

The teacher identity also records hashes of the local adapter, preprocessing implementation and actual positional-grid values, plus dtype, layer and pooling versions. Resume rejects a changed teacher contract. Every benchmark records its source manifest, source-tree hash, Git revision/status, runtime config, fixture hash, command and resource logs.

The full manifest contains 30,815 episodes and 3,128,624 actions: 10,819 R2R episodes / 631,244 actions, and 19,996 RxR episodes / 2,497,380 actions. Fresh token auditing found longest R2R 183 frames / 58,290 tokens and longest RxR 627 frames / 199,735 tokens. The logical context cap remains 262,144.

The native Window8 baseline's existing checkpoint directories were read-only verified at `SimpleMemVLN/outputs/window8_text_2gpu_reportfix_20261002-xguUNX/train`: checkpoint-3852, checkpoint-7704 and final, with trainer state 7704/7704 and epoch 2. The similarly named episodic V0 run is not substituted for this baseline. No new navigation-quality comparison has been performed.

## Measured resources and speed

All rows below are three-update, complete-episode, four-GPU diagnostics, with eight episode exposures per update. Longest fixtures repeat one real episode to fill all slots and intentionally stress all ranks simultaneously. Mixed fixtures contain eight distinct complete episodes including both sides of the 65,536-token activation-offload threshold. Steady times are differences between the latest rank completion timestamps, excluding first-update compilation/load; they include intervening loader and teacher work.

| Configuration / fixture | Steady s/update | Maximum GPU reserved GiB | Result |
|---|---:|---:|---|
| Resident teacher, GPU optimizer / mixed | 17.78 | 56.64 | Passed |
| Resident teacher, GPU optimizer / longest RxR | 90.30 | 76.85 | Passed; almost no practical headroom |
| Teacher CPU between episodes, GPU optimizer / mixed | 20.21 | 51.22 | Passed |
| Teacher CPU between episodes, GPU optimizer / longest RxR | 86.78 | 76.97 | Passed; later reservation remains high |
| Resident teacher, CPU optimizer / longest RxR | 80.31 | 76.31 | Passed; active allocation about 62 GiB |

The first resident-teacher longest test peaked at 225.18 GiB job RAM and 81,073 MiB device usage. All three updates completed with no CUDA OOM or new cgroup OOM event. Passing that stress test is not a universal no-OOM guarantee, and its margin is insufficient for the recommendation.

Reserved GPU memory includes reusable allocator cache. Runtime selection additionally examines peak active allocations, CUDA-context/device usage, arena release and repeated-update behavior. The safety supervisor limits job working set to 691.2 GiB, below the 768 GiB allocation, and terminates only its own process group if necessary. It never cancels the interactive allocation or another job.

The selected profile is **DeepSpeed CPU AdamW, resident FP32 teacher batch 16, attention batch 32, vision microbatch 8, four OMP/MKL threads and eight loader workers per rank**. It measured 18.09 s/update on the mixed fixture, versus 20.69 for the initial Torch CPU optimizer (12.6% lower time). Default-batch CPUAdam was 18.17 s/update; that 0.5% difference is too small to claim statistical superiority. This is the fastest observed qualified profile, not an exhaustive global optimum.

| Candidate (CPU optimizer unless specified) | Mixed s/update | Active GPU GiB | Job RAM GiB |
|---|---:|---:|---:|
| optimizer-cpu-mixed | 20.69 | 39.72 | 181.3 |
| tune-cpu-adam | 18.17 | 39.72 | 164.2 |
| tune-cpu-adam-fast | 18.09 | 39.72 | 167.3 |
| tune-optimizer-cpu-fast | 20.25 | 39.72 | 180.7 |
| tune-teacher16-attention16 | 20.42 | 39.72 | 180.6 |
| tune-teacher4 | 19.89 | 39.72 | 192.4 |
| tune-threads4-workers2 | 20.69 | 39.72 | 186.8 |
| tune-threads4-workers8 | 20.89 | 39.72 | 178.9 |
| tune-vision8 | 20.06 | 39.72 | 189.9 |

The selected longest-RxR test passed at **73.45 s/update, 61.79 GiB peak active GPU allocation, 76.40 GiB reserved, and 327.92 GiB peak job RAM**. Active memory leaves roughly 17 GiB below physical GPU capacity before context/other allocations; reserved cache is reclaimable. The same profile's longest-R2R test passed at 22.79 s/update. The additional eight-distinct-episode mixed-extreme fixture, including both the 627-frame RxR and 183-frame R2R tails, passed three updates at 50.70 s/update with 61.79 GiB active / 68.42 GiB reserved GPU memory and 263.98 GiB peak job RAM. The comparable native-off mixed fixture took 16.19 s/update, so online G1 cost about **11.7%** extra on that particular fixture.

A two-CPU extension-build/unit-test process overlapped part of the vision8 candidate's wall-clock lifetime; do not interpret its small timing advantage in isolation. The final selected-profile representative and real-manifest timings did not overlap other GPU tests. Existing work on worker-1 was untouched.

### Dataset projections

For the dataset estimates below, discard updates 1 and 2 and average subsequent **rank-completion endpoint intervals**, including data loading and teacher work. Representative fixtures contain 16 length-quantile episodes, repeated for 64 exposures / eight updates. The joint run reads the actual 30,815-episode manifest for 96 exposures / 12 updates. It is not a completed epoch.

| Scope | Warm measured s/update | Projected updates/epoch | Projected compute hours/epoch |
|---|---:|---:|---:|
| R2R representative, warm repeated images | 9.97 | 1,353 | 3.75 |
| RxR representative, warm repeated images | 19.24 | 2,500 | 13.36 |
| Joint actual-manifest short run | 27.54 | 3,852 | 29.46 |

The R2R and RxR figures are warm-fixture projections, **not independent cold-data full-epoch measurements**. Their sum (17.10 hours) substantially underestimates the actual-manifest projection (29.46 hours); do not use that sum to promise a joint completion time. The joint sample averages 34,533 tokens/episode, compared with approximately 32,411 for the full mixture, and has different image-cache, length/rank and shape histories. This experiment does not isolate the cause of the whole difference. Prefer **about 30 hours per joint epoch, about 59 hours for two**, then allow planning margin of 20–30% plus checkpoint/startup costs. That margin is a planning allowance, not a confidence interval. Re-estimate from several hundred updates of a real production run.

All timing candidates preserve full episodes, loss normalization, teacher precision, sampled frames and effective batch eight. No quality-equivalence claim follows merely from a speed test. Small timing differences can reflect shared storage and CPU scheduling. The diagnostic JSON retains raw per-rank profiles, intervals and source/config provenance in [evidence/benchmark_summary.json](evidence/benchmark_summary.json).


## Verification and recovery status

Initial targeted online tests: 21 passed. A separate real Qwen processor/vision-merger test verified the 15×20 row-major correspondence of all 300 visual positions. Tests cover deterministic sampling, actual teacher preprocessing, special-token slicing, position scaling, projector shape/count, detached targets, empty support, native-off forward/input-gradient parity, full-residual tap, checkpoint recomputation, no future-token gradient, optimizer grouping and namespace-only export.

Independent review found and fixed a CUDA RNG reset in projector initialization. It also prompted code-hash binding for teacher preprocessing and correction of a timing field that had counted teacher inference inside backward overhead. Additional acceptance scripts exercise actual four-rank unequal/empty-support reduction, full native parameter-gradient parity at zero auxiliary weight, auxiliary-only gradient reach, source-versus-export logits, streaming response replay and full-state recovery.

The first broad regression run had 328 passes, 12 skips and four failures in existing Git helper tests because the copied remote source lacked `.git` metadata. A real Git bundle was installed; no test was weakened to hide those failures. After restoring Git metadata and enabling the real processor tests, the next broad run passed 350 tests and failed only the optional CPUAdam build because of the CUDA toolkit mismatch. Installing the isolated matching compiler resolved that failure; both CPUAdam tests passed. The clean final regression run passed **351 tests with no failures or skips**, with 115 warnings in 32.95 seconds (`full-suite-final.log`).

Recovery checkpoints retain the student, projector, optimizer/scheduler, per-rank RNG and sampler/data contracts. Serving export removes exactly `research_aux.*` and the research config, then reloads strictly into native Window8. The actual four-rank CPUAdam fresh-process continuation from checkpoint 2 to update 3 passed the predeclared numerical gate. The projector is bitwise identical to uninterrupted training, and the final logged losses are exactly equal (0.41682443022727966). Whole-state relative L2 difference is 9.4496e-9 with maximum absolute difference 1.1921e-7; 31,439 tensor elements differ, so the entire model is **not** claimed bitwise identical. Projector movement from step 2 to 3 affects 126,650 elements and all three bias groups (2,559 / 4,096 / 2,048 nonzero changes), proving more than matrix weight decay. All 22 checkpoint inventory files and four-rank state were validated. Optimizer tensor files are not compared bitwise; the test validates complete inventories and continuation behavior. Both-dataset export/replay checks passed: source-versus-export logits are bitwise identical; retry and reset are exact; all 17 serving steps preserve Window8 eviction. R2R replay covers 50 response tokens, maximum logit difference 0.75 / RMS 0.061505; RxR covers 50 tokens, maximum 0.50 / RMS 0.048351. These satisfy the existing predeclared native BF16 tolerances (atol0.75, rtol0.03, RMS0.15), not a newly relaxed threshold. Controlled response tokens let the streaming/offline paths follow the same history; an additional greedy reset check is exact. This is a serving-prefix diagnostic, not navigation-quality evaluation or training truncation. The first export checker stopped on an incorrect assertion that accessed `model.step_lane_spec` directly. Native models omit that attribute. The checker now uses the same `getattr(..., None)` convention as the model; no training/serving implementation or numerical tolerance changed. The failed log is retained as `export-replay.log`; corrected passing evidence is `export-replay-verified.log` and [evidence/g1_export_check.json](evidence/g1_export_check.json). Recovery details are in [evidence/recovery-comparison.json](evidence/recovery-comparison.json). All final `src/` files were verified to match the source hashes used by the real-manifest timing run.

## Historical work excluded from the current method

Before the user changed the plan, an offline diagnostic extracted 608 frames and fitted a train-only PCA. No policy training used those targets. The active offline extractor/cache modules were removed. Those artifacts and their processing time are historical evidence, not prerequisites or speed claims for online G1.

## Files and operational recipe

**User update:** after the timing tests, the user explicitly requested one complete joint R2R+RxR epoch on four H100s. The production launcher is `train/launch_g1_omega_joint_e1.sh`, with `configs/vggt/g1_joint_one_epoch.yaml`: 3,852 updates, 116 navigation LR warmup updates (3%) and cosine decay to 0.1 of peak LR. The G1 auxiliary ramp remains 771 updates, now 20% of the epoch. One final repeated episode fills the 30,816 batch slots for 30,815 manifest records. This is an explicit schedule override; no matched two-epoch scientific-quality comparison is claimed. Production starts from the original Qwen base, not smoke-test weights.

The launcher saves recovery state every 100 updates, first saves at update 10, retains two rolling recovery snapshots plus epoch checkpoints, and resumes only complete checkpoints. A per-run file lock rejects duplicate launches. Each restart gets its own supervisor log directory and the same 691.2 GiB host guard. The production path and initialization evidence are recorded below.

The implementation boundaries are:

| Path | Responsibility |
|---|---|
| `src/qwen_vl/research/vggt_sampling.py` | Collection identity, deterministic strata and actual visual-token positions |
| `src/qwen_vl/data/vggt_online.py` | CPU image preprocessing and fixed positional target |
| `src/qwen_vl/research/vggt_omega.py` | Pinned online teacher, FP32 targets and residency modes |
| `src/qwen_vl/models/vggt_aux_heads.py` | SpatialForcing projector and cosine numerator |
| `src/qwen_vl/models/research_forward.py` | Full-residual layer tap integrated with nonreentrant checkpointing |
| `src/qwen_vl/models/nav_model.py` | Optional training head and auxiliary forward dispatch |
| `src/qwen_vl/data/episode_dataset.py` | Selected CPU images, positions and frame counts |
| `src/qwen_vl/train/trainer.py` | Global normalization, head LR groups and optional vectorized CPU AdamW |
| `src/qwen_vl/train/train_episode.py` | Teacher ownership, FP32 input staging, logs and checkpoint contracts |
| `configs/vggt/g1_omega_h4.yaml` | Measured four-H100 runtime with pinned teacher identity |
| `scripts/vln/profile_vggt_g1.py` | Guarded diagnostic run and immutable evidence |
| `scripts/vln/benchmark_vggt_g1.py` | Sequential memory/speed comparisons |
| `scripts/vln/accept_vggt_g1.py` | Mixed extreme, recovery, export and regression acceptance |

The optional DeepSpeed CPU AdamW path uses the same parameter groups, independent head LR, bias/no-decay rules, betas `(0.9, 0.95)`, epsilon `1e-8`, bias correction and decoupled weight decay. Its FP32 states are offloaded by ZeRO-2. Tests compare multiple numerical transitions with Torch AdamW and serialize/restore the optimizer. Changing optimizer implementation in the middle of an existing run is rejected by the saved config contract.

This cluster's shared PyTorch is built with CUDA 12.9; the previous local toolkit was 12.8. The optional CPUAdam extension's builder rejected that mismatch. `prepare_g1_toolchain.py` installs hash-verified official NVIDIA 12.9.1 nvcc/cudart/CCCL components in a separate user directory and reuses matching Torch-bundled cuBLAS/cuRAND through read-only symlinks. No driver, system toolkit or shared Python package was changed. A manifest records every component. Because CPUAdam compiles with `-march=native`, the environment wrapper uses a worker-local extension cache. See the [NVIDIA installation guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html) for the component-installation mechanism.

For the authorized one-epoch production run, use the committed profile with `science_common.yaml`, `vln_memory_window8.yaml`, and `deepspeed_zero2_cpu_optimizer.json`; set `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and `G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1` through `train/g1_omega_env.sh`. Use four torchrun ranks. Eight workers per rank means 32 loader workers total within the 60-CPU allocation. The profile's `window_attention_batch_steps: 32` batches independent attention work; it does **not** change the eight-frame attention window.

Equivalent direct training command inside a four-H100 allocation (the actual launcher additionally supplies the lock, resource supervisor and unique restart logs):

```bash
R=/mnt/data/vmo-ai-task/anhdh35
env G1_CUDA_HOME="$R/cuda-12.9.1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash "$R/SimpleMemVLN-c2_vggt_omega_g1/train/g1_omega_env.sh" \
  torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/episodic/science_common.yaml \
  --output_config configs/vln_empty.yaml \
  --memory_config configs/vln_memory_window8.yaml \
  --policy_config configs/vggt/g1_joint_one_epoch.yaml \
  --runtime_profile configs/vggt/g1_omega_h4.yaml \
  --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
  --manifest "$R/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" \
  --model_name_or_path "$R/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
  --output_dir "$R/SimpleMemVLN-c2_vggt_omega_g1/outputs/g1_omega_h4_joint" \
  --run-name g1_omega_h4_joint --recovery-save-steps 100 \
  --resume_from_checkpoint auto
```

This applies the requested one-epoch/3,852-update schedule to the full joint manifest, with eight episodes per update and base initialization. It deliberately omits the diagnostic repetition/update-limit flags. Re-running the same command resumes only a completed checkpoint; an incomplete checkpoint cannot silently be treated as a fresh start. Keep the same world size, manifest, model/data/teacher contract and optimizer implementation. The online teacher is rebuilt from the pinned assets after resume. A saved serving export intentionally lacks the training projector and optimizer, so it is not a substitute for a recovery checkpoint.

## Production launch on 2026-10-13

- Launch recorded at **05:20:56 UTC / 01:20:56 America/New_York**; Slurm allocation **4925**, step **4925.16**, worker-2, four H100s, 60 CPUs, 768 GiB RAM.
- Run root: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-c2_vggt_omega_g1-runs/g1_omega_joint_e1_20261013`.
- Training outputs: `<run-root>/train`. Restart logs: `<run-root>/supervisor-<UTC>-<PID>/command.log`; top-level launch log: `<run-root>/slurm-launch.log`.
- Source snapshot: `<run-root>/source`, archived from **86d68b2eff9db7c5cffdd86f0d98ca98f7e1c93a**, with read-only permissions, exact file hashes in `<run-root>/source_manifest.json`, and commit marker in `<run-root>/source.commit`. Later changes to the branch checkout cannot alter this training source.
- All four ranks printed **`SCHEDULE_GATE_PASS (3852, 116)`**. The trainer verified four ranks, GAS2, batch8, 30,815 unique records, 3,128,624 actions, and a 700–199,735 token range. No diagnostic repetition, episode limit or max-update override is present.
- Every rank installed a 30.48 GiB host activation arena sized for the full corpus maximum. Teacher and optimizer settings are the qualified profile, with the guarded RAM limit at691.2GiB.
- Initial production observation: **all four ranks completed at least update 7**, with no CUDA/cgroup OOM events; peak sampled working set 329.31 GiB. Losses observed at startup were finite. See [evidence/initial_progress.json](evidence/initial_progress.json).
- The run is ongoing; no completed G1 epoch or navigation-quality result is claimed. Expect about30hours plus startup/checkpoint costs, with a practical30–40hour planning range until enough production updates give a firmer estimate.

The exact launch record is [evidence/production_launch.json](evidence/production_launch.json). To resume after replacing a canceled allocation, run this same snapshot launcher and run directory inside the new four-GPU allocation:

```bash
R=/mnt/data/vmo-ai-task/anhdh35
RUN="$R/SimpleMemVLN-c2_vggt_omega_g1-runs/g1_omega_joint_e1_20261013"
# Execute from the new allocation (or prefix with srun --jobid=<new-id> ...).
bash "$RUN/source/train/launch_g1_omega_joint_e1.sh" "$RUN"
```

The launcher automatically chooses the latest **complete** recovery checkpoint. It cannot recover the unsaved updates between that checkpoint and a cancellation. The first planned recovery point is update10; subsequent saves are every100updates and at epoch end. The source and full-state checkpoints should be retained until the run is finished and any desired continuation/export is secured.
