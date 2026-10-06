# Streaming R2R evaluation implementation plan

Goal: evaluate all three FullContext-B epochs on R2R val_unseen concurrently, preserving the saved policy and benchmark protocol.

Specification: plans/SimpleMemVLN_Implementation_Plan.md, especially sections 5, 7–10, 12–13.

Architecture: retain StreamSession and its native FA2/FLA continuation. Use one Habitat process and one model subprocess per checkpoint; simulator Python 3.10 remains separate from pinned model Python 3.12. Transport current RGB losslessly. Keep checkpoint identities and separate resumable output directories.

Constraints: 640x480 RGB, HFOV 79, forward 0.25 m, turn 15 degrees, 500 decisions, 3 m success radius. Greedy normal-vocabulary decoding, 16 tokens including EOS, committed response history, no KV eviction. Inference failures contribute zero SR/SPL. No changes to training or checkpoint semantics.

- [x] Add regression tests for selection/resume, transport, failure accounting and checkpoint integrity.
- [x] Make evaluator portable to local R2R paths, use core Habitat configuration and deterministic scene grouping, explicit episode assignment, safe resume, durable results and raw response logging.
- [x] Replace PNG compression on the critical path with lossless raw RGB bytes over the existing JSON protocol; retain PNG compatibility.
- [x] Add pinned checkpoint downloader with published SHA256 verification and three-worker parallel launcher with independent logs, progress and exit status.
- [x] Validate existing streaming tests and model/serializer contract, then GPU continuation and simulator smoke for each epoch.
- [ ] Run all 1,839 val_unseen episodes per epoch concurrently and report actual results and performance.

Review focus: duplicate/unexpected journal IDs; failures before simulator reset; dead model subprocess causing thousands of spurious failures; generated/executed action mismatch at cap; same output directory reused with different checkpoint/data/code.

Execution notes: use expandable CUDA allocations after the default allocator reserved 94.1 GiB in the initial 500-step profile; verify three concurrent 500-step profiles before launching the full split. Match training preparation by trimming instruction whitespace. Independent code review findings on recoverable failures, truncated journals and preprocessing/runtime resume identity were fixed and regression-tested.

All three parallel 500-step resource gates passed (13.69 GiB live, 14.62 GiB reserved per model). Full 3×1,839 val_unseen evaluation launched; final metrics pending.
