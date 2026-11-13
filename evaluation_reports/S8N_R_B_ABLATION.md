# S8N trained R/B ablation: R2R and RxR

Updated: **2026-11-14**. Final observed success rates for eight trained R/B configurations, including the original S8N baseline. Every configuration has complete R2R and RxR results; R12B0's resumed R2R evaluation has finished.

| Model | SR R2R (%) | SR RxR (%) |
|---|---:|---:|
| R2B1200 | 49.21 | 56.23 |
| R4B0 | 48.72 | 55.00 |
| R4B600 | 50.84 | 58.27 |
| R4B1200 — S8N baseline | 52.42 | 56.58 |
| R4B2400 | 51.33 | 57.37 |
| R8B0 | 48.99 | 55.30 |
| R8B1200 | 50.95 | 56.01 |
| R12B0 | 51.22 | 56.75 |

R2R uses **1,839 val_unseen episodes**. RxR uses **3,669 English-guide val_unseen episodes** (en-US and en-IN) with **15° turns**. All episodes remain in the denominators; failed episodes receive zero success credit. SR is shown as a percentage rounded to two decimal places.

R counts recent observation groups including the current observation. B is the episodic visual-token budget. R4B1200 is the original S8N baseline. The earlier [inference-only budget sweep](S8N_R_B_SWEEP.md) has its own results and evidence.

[SR table CSV](evidence/S8N_R_B_ABLATION_SR_20261113.csv) · [Full-precision counts and source evidence](evidence/S8N_R_B_ABLATION_SR_20261113.json). The evidence index records success counts, full episode counts, journal digests, source-result paths, and the completed R12B0 recovery certificate.

## RxR update provenance

The updated RxR values combine **3,570 original episodes and 99 targeted reruns** for each model. The same fixed 99 episode IDs were replaced regardless of outcome. Their rerun body geometry is height **0.88 m**, radius **0.18 m**; camera height stays **1.25 m** and turns stay **15°**. The exact original and rerun configuration hashes and episode IDs are recorded in the JSON evidence. These are combined results with a targeted configuration change, not a fresh uniform-configuration evaluation of all 3,669 episodes. R2R values are unchanged.

The baseline retains failed episode **8680**, and R4B2400 retains failed episode **10866**, each with zero success credit and the full denominator. The [server campaign audit](evidence/RXR_RERUN_RESULTS_FINAL_AUDIT_20261114.json) records preservation of the other 3,570 original rows per model and serial replay controls. Its published Markdown hashes refer to the [unaltered server reports](archive/WS_RXR_20261114/server_original/ALL_RESULTS.md), before this documentation merge.

[Fresh sync verification](evidence/WS_RXR_20261114/SYNC_VERIFICATION.json) checks all nine combined journals (33,021 records): full-file hashes, unique episode IDs, counts and failure-aware success rates. [Portable per-episode metrics and certificates](evidence/WS_RXR_20261114/README.md) accompany the reports. This sync did not rerun inference or independently repeat simulator/media validation.

Previous results remain in the [prior R/B report](archive/WS_RXR_20261114/prior_main/S8N_R_B_ABLATION.md) and its archived JSON/CSV evidence. The [earlier sync verification](evidence/S8N_R_B_ABLATION_SYNC_VERIFICATION_20261113.json) describes the original, pre-rerun journals.
