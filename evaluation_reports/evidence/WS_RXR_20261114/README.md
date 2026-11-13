# RxR report sync: 2026-11-14 server time

Source: `ws_anhdh35:/storage/anhdh35/SimpleMemVLN`. Report timestamps use +07:00 (2026-11-13 in America/New_York).

Each model directory contains the unchanged source certificate and compact per-episode records exported from its verified combined journal. The compact records retain all fields except `trajectory`, `actions`, and `instruction`; full original and combined journals remain on the server at the paths in `SYNC_VERIFICATION.json`. No source results were deleted.

Fresh verification reads all full journals, checks their SHA-256 against source certificates, checks unique episode IDs and recomputes failure-aware success counts and rates. This is a data-transfer and metric verification; it does not independently repeat inference, simulator/media checks or the campaign's serial replay controls. The separate server campaign audit records those earlier checks.

| Model | Episodes | Successes | Execution failures | SR (%) |
|---|---:|---:|---:|---:|
| R2B1200 | 3669 | 2063 | 0 | 56.23 |
| R4B0 | 3669 | 2018 | 0 | 55.00 |
| R4B600 | 3669 | 2138 | 0 | 58.27 |
| R4B1200 — S8N baseline | 3669 | 2076 | 1 | 56.58 |
| R4B2400 | 3669 | 2105 | 1 | 57.37 |
| R8B0 | 3669 | 2029 | 0 | 55.30 |
| R8B1200 | 3669 | 2055 | 0 | 56.01 |
| R12B0 | 3669 | 2082 | 0 | 56.75 |
| S8N_DAGGER_SF_REMAINING | 3669 | 2464 | 0 | 67.16 |

Reproduce the offline transfer and report checks with `python3 evaluation_reports/evidence/WS_RXR_20261114/verify_sync.py` from the repository root.
