# L2 learned-ranker R2R evaluation

## Latest measurements — 2026-10-25T10:00:28.687170+07:00

| Model | Median ms | p95 ms | Mean ms | Actions/s | R2R successes / completed | SR | SPL | Navigation status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| L2 | 109.716 | 115.079 | 110.094 | 9.083 | 678 / 1384 | 48.99% | 45.42% | Interrupted; partial / 1,839 |

Measured 2026-10-25T10:00:28.687170+07:00 on the same RTX PRO6000 Blackwell Max-Q, driver580.178.04 and 300W limit. This is a **separate post-reboot session** using the unchanged original benchmark worker: batch1, the same 80 preloaded RGB frames/instruction, three trials of16 warm-up +64 timed actions (192 samples/model), CUDA-synchronized observe wall time; simulator, transport and disk reads excluded. No detected foreign GPU process. Native BF16 backbone and embedded learned rankers retained (feature preparation FP32, ranker FP64, priorities FP32). Warm-up/initialization excluded; this is not a cold-start benchmark or a proof of cross-session statistical superiority.

Navigation results were independently recomputed from saved journals with contract-order and source/config identity checks. The workstation rebooted at **2026-10-25 08:35:33 +07:00**; both evaluation processes are absent. These SRs are **partial, interrupted results**, not completed evaluations. L2 and L3 cover different-sized ordered prefixes, so their SRs are not a matched full-panel comparison. No navigation evaluation was restarted by this latency measurement.

Latency p99: 119.178 ms; peak allocated GPU memory: 8.698 GiB. Three-trial token sequences identical: True. Raw result: `/data14t/anhdh35/latency_l_20261025/results/l2/result.json`.

The setup and launch sections below are historical evidence, not current process status.


Prepared 2026-10-25T02:35:14.064319+07:00. Status: **INTERRUPTED after server reboot; 1384/1,839 completed, 678 successes, SR 48.99%, SPL 45.42%, zero recorded execution errors.** Evaluation is not running.

## Identity and implementation review

- Latest branch `m5_gdn_l_sweep`, commit `ba8a1ca9f3dda5356b31e7502b31d95ea054cad5`; isolated checkout. Training source `022e526b279b6d7af0f0c5d861429a0d59a7f7e5`.
- Hub `anhdao69/M5_L`, folder `L2`, pinned revision `2af7da5d11fa77bc80a0e12f6066948cfe95c63f`.
- Published weights SHA256 `fe9bf43822f5d26a3fde39a91d8ee732eaf8775a98d81957fa8727d87beb3d5a`.
- Embedded ranker SHA256 `711196dd5db764457e7fdcfa973ea7deac190b57ba3315f46e40603d1cd5d6e9`.
- Read the published completed-epoch L2 README and campaign implementation report; reviewed the learned ranker, strict contract/parser, feature extraction, offline-plan and streaming feature paths, and tests.

L2 has eight frozen `288→64→1` SiLU MLPs (148,488 parameters). For each attention layer it concatenates F0_v1 features from the preceding three GDN layers: log beta, log negative decay and log value norm, preserving native head channels and fit-only frozen normalizers. Unlike L3, it does not use residual, prediction-norm or cosine-error features and its extractor does not scan recurrent state for scoring. Native policy GDN recurrence remains intact.

Memory: R=4 including current; B=1200 visual tokens; token selection; KV guard4784. Priorities are computed once at birth, fixed thereafter. Native K/V positions and text action history remain intact; feature buffers are released after scoring. The ranker is embedded in navigation metadata; missing/mismatched artifacts fail closed. Backbone BF16, feature preparation FP32, frozen MLP computation FP64 and final priorities FP32. No precision override or model change.

Publication reports completed epoch1, 3,852 updates, seed429, matched joint R2R/English RxR15deg training and action-weighted loss0.12583. Its offline holdout capture@96 of0.755 and Spearman0.780 are ranking diagnostics, **not navigation SR**. Published qualification used M5_E1 with the ranker; the publisher explicitly does not claim GPU replay of the completed L2 policy.

## Local evaluation protocol

- Verify all remote file identities, published checksums, ranker hash/artifact, exact training/export configs and source compatibility; check all 724 model tensors are finite.
- **151 focused CPU tests passed** on this checkout.
- On the actual trained checkpoint: strict reload; real-feature NumPy FP64 ranker reference at each attention layer; three saved 64/128/previous-M5-128 RGB/action replays using unchanged tolerances; bitwise greedy reset; idempotent retry; all-layer cache bounds and feature release; 520-step bounded stream.
- Two Habitat smoke episodes and strict result validation precede the full **1,839 R2R val_unseen episodes**. Same ordered episode panel, dataset, scenes, action mapping, seed42 and max500 steps as L3/S campaign.
- L3 continues concurrently on the same GPU. Runtime latency observed here is shared-load latency, not a controlled speed comparison.
- Model environment `/storage/anhdh35/SimpleMemVLN/.venv/bin/python`; simulator `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python`.
- Existing S0 smoke-control evidence is preserved by hash and reused; no new S0 navigation run is claimed.

## Evidence

Root: `/data14t/anhdh35/m5_l2_epoch1/evaluation`.
Download and preflight evidence: `hf_inventory.json`, `download.log`, `../checkpoint/download_manifest.json`, `published_checksums_verified.json`, `models.json`, `preflight.json`, `tensor_inventory.json`.
Qualification: `tests.log`, `audit.log`, `audit/L2/`.
Monitoring: `campaign_status.json`, `campaign.log`, `smoke_validation.json`, `full.log`.
Full results: `results/full/L2_s429/episodes.jsonl`, `summary.json`, `runtime.json`, `evaluation_contract.json`.
The controller runs `verify_result.py full` after completion to check coverage, order, hashes, action metadata, execution errors and recomputed metrics. Partial validation is never final validation.

## Completed local qualification

All 26 downloaded file identities and 25 published checksums verified; embedded ranker and exact training configuration match, all 724 backbone tensors are finite, and latest `src/` and `configs/` match the recorded training source. No source edits or serving overrides.

| Replay | Max logit error | RMS error | Top-token agreement | Mean bank overlap | Gate |
|---|---:|---:|---:|---:|---|
| 64 steps | 0.6875 | 0.03662 | 99.476% | 98.894% | PASS |
| 128 steps | 0.6875 | 0.03492 | 99.478% | 98.577% | PASS |
| Previous M5 128 steps | 0.5625 | 0.03405 | 100.000% | 98.531% | PASS |

The unchanged numerical gate uses torch.allclose(atol=0.75, rtol=0.03) and RMS≤0.15. All forced replay actions decode correctly, but that is a forced-history protocol check, not free-greedy equivalence. Bank selections and some argmax tokens still differ; qualification is tolerance-based, not bit-identical parity.

All eight embedded ranker heads exactly matched the independent NumPy FP64 MLP after FP32 priority rounding on the tested 300 real-feature rows per head. Bitwise greedy reset passed. The 520-step bounded audit retained 1200 episodic rows at every attention layer, peaked at2546 resident KV rows (guard4784), and used8.698 GiB peak allocated memory. All 24 extractors were verified to use the stateless F0 feature path; native model recurrence remains unchanged.

## Verified full-run launch

Both smoke episodes completed with zero execution errors (1 success /2; setup evidence only). Campaign PID569368, evaluator PID569537. The full-run journal, action mapping, runtime memory metadata, checkpoint/data/config hashes and episode order passed the initial partial check. `full_validation.json` remains explicitly `complete:false` until all1839 episodes finish; the controller validates final results afterward.

L3 was verified alive with 603 completed episodes and 0 execution errors at this snapshot.

Launch verification (`/data14t/anhdh35/m5_l2_epoch1/evaluation/launch_verification.json`).
