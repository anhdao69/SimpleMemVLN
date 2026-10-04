# R2R FullContext-B evaluation setup, 2026-09-29

Implementation starts from streaming branch `62da70c`. Evaluation follows
`plans/SimpleMemVLN_Implementation_Plan.md`, particularly sections 5, 7–10 and 13.
Habitat setup is ported from the local StageVLN-v2 evaluator/configuration;
SimpleMemVLN owns the model and all recurrent/attention state.

## Checkpoints and protocol

- Repository: `anhdao69/SimpleMemVLN-R2R-FullContext-B`.
- Immutable revision: `73d866c31e398596a67cd8be6f7dbfd25ab9241d`.
- All published SHA256 entries verified for epochs 1, 2 and 3.
- Base: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Strict wrapper state loading and exact serializer metadata checks succeeded.
- R2R-VLNCE `val_unseen`: 1,839 episodes. Three independent concurrent workers.
- RGB 640×480, HFOV 79°, forward 0.25 m, turn 15°, cap 500, success radius 3 m.
- Instructions are stripped exactly as in training preparation.
- Greedy normal-vocabulary text decoding, 16-token response budget including EOS.
- FullContext KV and FP32 GDN state persist for an episode. No history re-encoding,
  KV eviction, action-trie restriction, invalid-output STOP fallback or quantization.
- Inference failures remain in the denominator with zero SR/SPL; infrastructure
  failure stops the affected worker. Partial summaries explicitly mark incompletion.

## Executed checks

- 31 tests passed with the real pinned processor. This includes strict action
  history commitment, CPU reference continuation, CUDA attention/FLA numerical
  checks, raw RGB identity, instruction normalization, resume integrity and launcher tests.
- An inherited Hopper-specific backend assertion now applies only to compute
  capability 9; its numerical forward/backward checks still run on Blackwell.
- Native BF16/FA2/FLA continuation passed on the actual GPU. Isolated GDN maximum
  error was 0.0000610, with a nontrivial reset control. Native Flash routing was exercised.
- Epoch 1 generated-history parity passed on 12 repeated real Habitat frames:
  max logit error 0.25, RMS 0.02194, token argmax agreement 100%, no high-margin flips.
  This replay is a numerical check, not a navigation result.
- All three epochs completed the same two real Habitat episodes without inference
  failures. Epoch 1 succeeded on one; epochs 2/3 succeeded on neither. This small,
  deterministic smoke set is not a benchmark estimate.
- A completed smoke evaluation resumed without duplicate records or model loading.
- An initial 500-step resource replay passed, including reset/identical first
  response. Peak live allocation was 13.69 GiB, KV length 159,075 tokens. The
  default allocator reserved 94.12 GiB, motivating expandable CUDA allocations
  and a subsequent three-worker horizon test.
- Independent code review identified and verified fixes for recoverable policy
  failures stopping the split, truncated final journal records and incomplete
  preprocessing/runtime resume identity.

## Efficiency and runtime

The GPU reports compute capability 12.0 with approximately 96 GiB memory.
Model runtime: Python 3.12.13, Torch 2.10.0/cu129, Transformers 5.11.0,
FA2 2.8.3, FLA 0.5.2. Habitat uses its existing Python 3.10 environment.
FLA GDN bindings are enforced on every layer; convolution uses the supported
Torch path because causal-conv1d is absent. The aggregate Transformers warning
about a missing fast-path package does not mean GDN silently fell back to Torch.

On a captured real RGB image, median encode/decode time across 30 repetitions
was 52.88 ms for PNG versus 2.22 ms for raw RGB/base64, with identical pixels.
These figures exclude pipe/JSON transfer; the evaluator records that overhead
in transport-plus-model latency. Initial concurrent smoke model latency was
approximately 0.24 seconds per decision. FullContext cost still grows with history.

## Artifacts and running

- Launcher: `scripts/eval/run_r2r_epochs.sh`.
- Download/verification: `scripts/vln/download_eval_checkpoints.py`.
- Corrected smoke: `artifacts/evaluation/smoke-final/epoch-{1,2,3}`.
- Native gate: `artifacts/blackwell_compat.json`.
- B replay: `artifacts/epoch1_generated_history_trimmed.json`.
- Resource checks: `artifacts/epoch*_horizon500_parallel.json`.
- Test output: `artifacts/tests.log`.
- Full benchmark outputs: `artifacts/evaluation/fullcontext-b/epoch-{1,2,3}`.

Each benchmark output directory contains its identity contract, runtime,
append-only episode journal and progress summary. The summary's `complete`
flag must be true and `episodes == scheduled_episodes == 1839` before reporting
full-split scores. SR/SPL are stored as fractions. Oracle success is computed from
visited distance-to-goal; nDTW is undefined with the inherited core Habitat config.
The training collector's raw capture timing was not independently reconstructed
by this evaluation work; the rollout loop itself uses observation-before-action.

## Parallel horizon result and full launch

All three checkpoints passed 500 streaming steps concurrently with expandable
allocations. Each peaked at 13.69 GiB live allocation and 14.62 GiB reserved, with
159,074 retained KV tokens. Every checkpoint reproduced its first response after
reset. The reduction from 94.12 GiB reserved under the default allocator was
measured; the underlying FullContext retention policy was not changed.

The full three-epoch val_unseen run was launched on 2026-09-29 after these gates.
Launcher log: `artifacts/evaluation-launch.log`; launcher PID:
`artifacts/evaluation-launch.pid`; per-worker PIDs:
`artifacts/evaluation/fullcontext-b/worker_pids.txt`.
Full results are pending until all three summaries report `complete: true`.
