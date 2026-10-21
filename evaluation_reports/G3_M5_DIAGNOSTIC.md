# G3/M5 failure diagnostic — completed 2026-10-21

[Read the complete diagnostic report](https://github.com/anhdao69/SimpleMemVLN/blob/diag/g3m5-failure-20261019/reports/diagnostics/g3m5_failure_20261019/REPORT.md).

The report, six figures, results, scientific review, reproduction commands, and `scripts/diag` code are published on [diag/g3m5-failure-20261019](https://github.com/anhdao69/SimpleMemVLN/tree/diag/g3m5-failure-20261019). The runtime source matches `c2_vggt_omega_g3_m5_layers`; this index preserves each branch's runtime code.

The study completed panels D-a through D-f. Its scientific review records `qualified: false`: rendering parity failed (`failed_rendering_parity_stop`). Teacher-forced measurements do not establish the cause or magnitude of closed-loop navigation degradation. The report's decisions are R1 refuted, R2 refuted, R3 inconclusive, R4 supported, and R5 refuted.

Source: `ws_anhdh35`, `/data14t/anhdh35/simplememvln_diag_g3m5_20261019`, commit `747677faae442c06e90019bbe013b88ab5ddf53f`. The original report, results, and scientific review retain their source checksums. Large raw export tables remain on the server; the publication manifest records their original paths, sizes, and SHA-256 checksums.

Validation on the source checkout: `PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /storage/anhdh35/SimpleMemVLN/.venv/bin/python -m pytest scripts/diag -q` — 140 passed, 4 skipped, 6 subtests passed.
