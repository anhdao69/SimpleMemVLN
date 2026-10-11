# H2 speed tuning and three independent epoch-one runs

User scope: maximize practical V0/M1/M2 training speed on two H100s, compare alternatives, then submit exactly three Slurm jobs. Start each arm independently from Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the complete joint R2R + English RxR15 manifest. Global batch stays eight (two ranks, one episode/microbatch, accumulation four). Keep full episodes, full BPTT, labels, action history, resolution, objective normalization and the existing AdamW implementation.

Epoch one means the D1 protocol: 3,852 updates, stopped at the epoch boundary of the unchanged 7,704-update schedule (232 warmup). Retain complete distributed checkpoints for same-profile resumption. Do not initialize any production run from benchmark weights.

## Experiments and selection

Compare CPU threads 2/4/8, loader workers 2/4, heap trimming on/off, attention batches 8/16/32 and vision microbatches 4/8/16. First screen with four updates of eight complete 79-observation episodes, four from each dataset. Combine the best thread count with the two leading loading/batching configurations. Confirm the chosen candidate using at least 20 ordinary full-manifest updates; compare the same episode exposures and include between-update overhead. A short-fixture speedup alone is not acceptance.

Qualify the candidate against the original plan's H2 V0/M2 resource matrix (three updates each of short mixed, longest R2R, just below offload, just above offload, longest RxR, mixed tail), stressed save/reload, and representative speed. Exercise M1 as well. Keep the 65,536-token pinned-arena threshold unless an explicitly recorded additional experiment qualifies a change. Use CPU optimizer offload; GPU optimizer and reduced-precision optimizers are not silent substitutes.

The existing hard GPU ceiling is 78 GiB reserved, with 76 GiB preferred for headroom. Host tests use a 460.8-GiB working-set guard when considering a 512-GiB production request. Any OOM, kill, guard termination, nonfinite value, missing update or incorrect exposure fails. Preserve failed candidates and choose a qualified alternative; do not loosen budgets after a failure.

## Numerical and operational requirements

Run the mathematical/kernel suite and real R2R/RxR train/serve replay with the selected batching settings applied equally to the feature-disabled comparator. Preserve atol=0.75, rtol=0.03 and RMS<=0.15. Report selection mismatches and argmax disagreements. Enforce the already declared independent-audit helper-state threshold of 1%, as well as finiteness; this is an engineering qualification budget, not a claim of exact equality.

Close the false-success helper and recoverable-OOM guard defects with regression tests. A campaign must carry immutable, passing numerical evidence matching its source commit and runtime-profile hash; its launcher rejects absent or changed evidence. Each submitted job also repeats its own longest-episode startup gate on the actual allocated node. Preserve immutable source snapshots, separate output directories and single-writer leases.

Submit only H2 V0, H2 M1 and H2 M2 on this cluster's `main` partition. Do not race H4, automatically submit further ablations, cancel existing allocations or upload weights automatically. Tune CPU requests within measured resources; retain sufficient measured host/checkpoint margin. Keep the existing recovery-save and epoch-export machinery. Report source, selected settings, measurements, known plan deviations, job IDs and scheduler states.
