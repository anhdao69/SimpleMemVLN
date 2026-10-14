# A1 qualification and continuation implementation plan

> Execute inline with executing-plans; retain one final independent review.

Goal: qualify the existing direct-native-Q/K online-Omega A1 and run it after A0 on worker-2, independently from base.
Spec: `A1_geoaddress_sv.md`, `A0_address_control.md`, and the user's latest request to resume A1 supersede the previous deferral.
Architecture: retain the implemented scientific objective and A0-matched one-epoch schedule. Add arm-specific qualification and an A1-only queue. Validate on the four GPUs after A0 releases them; production is fail-closed on every required gate.

## Global constraints
- Existing A0 immutable source, process, weights and allocation remain untouched.
- Worker-2 only; allocation4925; no overlap with A0 or another active training step.
- Qwen base revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, seed429, joint30815 episodes, global8,3852updates,warmup116. No PCA/MLP; native Q/K KL; frozen online single-view Omega.
- Keep full BPTT, CPUAdam/offload and pinned arena from A0. Preserve full checkpoint recovery and strict serving adapter export.

## Review focus
- An A0-only or incomplete seal cannot authorize A1.
- Queue restart must not replay completed A0 or duplicate A1.
- A successful final checkpoint cannot bypass a failed/running supervisor or unfinished A0 export.
- Qualification failures cannot fall through to training; source and arm evidence must match.
- Longest and empty-support A1 examples must train with teacher detached and earlier source keys differentiable.

### Task 1: Arm-specific qualification and launch gate
Files: `scripts/vln/qualify_geoaddress.py`, `scripts/vln/queue_geoaddress.py`, `train/launch_a_omega_joint_e1.sh`, `tests/vln/test_geoaddress_queue.py`.
Interfaces: `check_qualification(source, record, arm=None)`; `write_qualification_seal(source,path,records,expected,arms)`; selected qualification arms.
- [x] Add failing tests rejecting A0-only seals for A1, missing required gates, failed records, duplicate arm requests, and accepting a complete A1 seal.
- [x] Run `PYTHONPATH=src python -m pytest tests/vln/test_geoaddress_queue.py -q`; expected new regressions fail on missing validation.
- [x] Implement explicit qualified arms and required gate names; launch supplies its requested arm. Qualification can run A1 alone without repeating A0.
- [x] Run targeted suite; expected pass. Commit code and tests.

### Task 2: Durable A1-only continuation
Files: `scripts/vln/queue_geoaddress.py`, `tests/vln/test_geoaddress_queue.py`.
Interfaces: selected `--arms A1` and `--predecessor <existing A0 run>`; queue uses Task1 seals, existing predecessor validator and allocation guard.
- [x] Add failing tests for A1-only sequence selection, no A0 replay, fresh qualification selection, failed predecessor, and duplicate launch prevention where applicable.
- [x] Implement arm selection and clear phase state; keep existing whole-sequence default compatible.
- [x] Run targeted suite and whole available CPU suite; distinguish environment skips/failures from regressions. Commit.

### Task 3: Qualification, review and deployment
Files: plans and `reports/vggt_five/A0_A1/` continuation report/evidence; immutable server snapshot and persistent queue.
- [x] Run existing scientific CPU tests; inspect loss/indexing/normalization, online targets and recovery/export paths without rewriting passed implementation.
- [x] Obtain fresh independent review of continuation changes and remaining A1 runtime risks; fix important findings with regressions.
- [x] Deploy a committed source snapshot. Start persistent worker-2 watcher for A0 -> A1 qualification -> A1 production, with source/arm checks, no premature GPU use and no fallback node.
- [ ] H100 gates: full regression, teacher QC, mixed+empty support, longest RxR/R2R, threshold/mixed extremes, fresh-process resume, trained export, distributed denominator, representative timing, final regression. Record PASS only after actual results.
- [x] Report exact queue state and which GPU gates are completed versus pending; update the historical deferral notice. Preserve failed evidence.

## Pre-flight
Task1 seal schema is consumed by Task2 queue and launcher. Explicit arm validation must be applied at both boundaries. GPU gates require A0 completion; CPU work and source deployment do not.
Ruling: do not pause A0 or use other workers. Queue the H100 qualification immediately after A0 completion, followed by A1 only on PASS. This preserves current training at the cost of qualification time between runs.

Deployment: runtime70455f0, campaign `/mnt/data/vmo-ai-task/anhdh35/a1-omega-worker2-20261014-continuation`, detached watcher3361209, allocation4925 on worker-2. CPU scientific/queue39passed; server queue21passed; full local369passed/35skipped/4environmentfailures. Fresh review restart-race fix verified. H100 gates remain pending A0; do not mark Task3 hardware qualification complete until the source-bound A1 seal exists.

## Repair continuation after initial hardware failure

- Original worker-2 qualification:408server tests and teacherQC passed; mixed initialization failed on teacher identity, zero updates/OOMs.
- Remove A1's dependency on unused positional-grid numeric hashes while preserving every used teacher component and strict validation. No PCA, cached targets, or MLP; frozen online Omega and direct native Q/K unchanged.
- Verify regression RED->GREEN, deploy new immutable source, rerun the mandatory H100 gates in a new campaign path. Do not overwrite failed evidence or replay A0.
- Start A1 only after a complete arm/source-bound PASS seal. Observe real optimizer progress and full recovery checkpoint before reporting running successfully.
