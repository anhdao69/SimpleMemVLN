# Local publication of the server diagnostic

This directory contains the complete final report, results, six final figures, scientific review, panel and environment metadata, and reproduction records from `ws_anhdh35:/data14t/anhdh35/simplememvln_diag_g3m5_20261019` at commit `747677faae442c06e90019bbe013b88ab5ddf53f`. Supporting code and tests are in `scripts/diag` on this branch. Its `src` tree matches the diagnostic server (`8aaf241096ce54fef87e81a71da19a7ea89a4dff`).

Read [REPORT.md](REPORT.md) and [SCIENTIFIC_REVIEW.json](SCIENTIFIC_REVIEW.json). The study is **not qualified**: rendering parity failed (`failed_rendering_parity_stop`). The teacher-forced results do not establish the cause or size of closed-loop navigation degradation. This publication preserves that qualification and the original report, results, and review bytes.

`PUBLICATION_MANIFEST.json` lists each published source path, size, and SHA-256 checksum. Its `server_evidence_files` section indexes 183 raw export files retained on the source server. The raw `exports`, `d_b_exports`, and `completed_exports` directories are not included in this Git publication. Their source files remain unchanged, including the 202 MB CSV that exceeds GitHub's per-file limit. Full offline re-analysis requires fetching those exports from the source worktree or the asset locations recorded in the original report. Existing source manifests describe the complete server bundle; they must not be interpreted as a claim that all raw exports are included here.

The original `STATUS.md` is an immutable run history and includes the server's earlier failed push attempt. This publication was prepared locally after retrieving the final snapshot; it does not rewrite that history.

Verification before publication:

- SHA-256 and byte-size comparison for every published source file.
- Original `COMPLETE.json` hashes for the report, results, and scientific review.
- Six final figures present; JSON parsing, Python syntax checks, and credential-pattern scan.
- Source environment: 140 tests passed, 4 skipped, 6 subtests passed (`python -m pytest scripts/diag -q`, `PYTHONPATH=src`, CPU-only).

The branch is based on the current `c2_vggt_omega_g3_m5_layers` runtime, with a single snapshot publication rather than importing the diagnostic branch's oversized historical Git blobs. Each existing branch links to this canonical report through `evaluation_reports/G3_M5_DIAGNOSTIC.md`.
