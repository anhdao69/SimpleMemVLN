# Episodic-memory campaign — report index

For the later October 11 two-H100 campaign on the `/mnt/data` cluster, see the [H2 speed, qualification and launch report](../c2_h2_speed_20261011/implementation_and_runs.md). The source IDs, job IDs and `/groups` paths below describe the earlier campaign.

Plan: [`plans/episodic_mem.md`](../../plans/episodic_mem.md). Branch `c2_episodic`.
Production source: `e28bd01f10612851cf9ef303fbddbd43c2cae58a` (immutable copy in the
campaign root). Campaign root: `/groups/yshang/an221229/checkpoints/SimpleMemVLN-c2_episodic`.
Hub (public): `anhdao69/Streaming_episodic`.

| File | Content | Status |
|---|---|---|
| [00_audit.md](00_audit.md) | branches, environment, data, Slurm, disk | complete |
| [01_design_decisions.md](01_design_decisions.md) | D1–D14 (epoch-1 on 2-epoch schedule, helper placement, FP32 grad scatter, RAM engineering, chunking, batch steps, superseded submissions, fork-safe arena, loader workers, review fixes) | complete |
| [02_implementation.md](02_implementation.md) | modules and data/serving paths | implemented |
| [03_score_validation.md](03_score_validation.md) | selector + committed-write correctness; train/serve skew (98.2–98.3% set overlap) | CPU- and GPU-tested |
| [04_attention_and_replay.md](04_attention_and_replay.md) | attention oracle, offline==serving, real replays; one retained 96-step R2R budget failure for B300 arms | GPU-tested |
| [05_resource_4h100.md](05_resource_4h100.md) | H4 measurements; final-snapshot interactive gates all PASS; longest-RxR/mixed/speed gates in-job | partially GPU-qualified |
| [06_resource_2h100.md](06_resource_2h100.md) | H2 path smoke; full qualification in-job | in-job pending |
| [07_phase1_results.md](07_phase1_results.md) | runs, jobs, Hub layout, evaluation hand-off, operating guide, lifecycle rehearsal | submitted |
| [08_phase2_decision.md](08_phase2_decision.md) | pre-registered Phase-2 gate | closed |
| [run_index.json](run_index.json) | machine-readable run/attempt/job index | submitted |
| `r2r_val_unseen_ids*.json`, `rxr15_val_unseen_panel1000*.json` | pre-registered evaluation ID lists | frozen |

Tests: `tests/vln` — 303 passed on H100 (0 skipped) at `e28bd01`.
No navigation (SR/SPL) result exists yet; none is implied by any file here.
