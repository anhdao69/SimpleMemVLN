# L1 epoch-1 R2R evaluation

Updated 2026-10-27T00:08:14.387607+00:00. Controller phase: **complete**. Completion status is determined by the final validation below.

## Checkpoint and implementation

- Branch `m5_gdn_l_sweep`, isolated source `/data14t/anhdh35/m5_l1_epoch1/source`, commit `b98539aff67a77736b6c520befbf95b89e63769c`.
- Training source `022e526b279b6d7af0f0c5d861429a0d59a7f7e5`. Latest src/configs are exactly equal to training source; no model changes or serving overrides.
- Hub `anhdao69/M5_L/L1`, pinned revision `1fcce9539f9763007afbff9f4d75a933abb0c6e3`. All 26 remote file identities and 25 published checksums verified.
- Weights SHA256 `331b01b5c15f190f0b12572f5177fd98c0e353d92443238a2ec50406b8af0a87`.
- Embedded ranker SHA256 `8c8bf98dddc4d2f5d76604023c20ba7378039578832202e44dfb8d797f61b263`; artifact/config/export identity verified. All 724 backbone tensors finite.
- Reviewed the campaign implementation report and published completed-epoch README. Frozen ranker: eight heads, 96 → 64 → 1 SiLU MLP; 50,184 parameters total. F0_v1 features from the immediately preceding GDN layer: log beta, log negative decay, log value norm; frozen fit normalizers. Offline ranking metrics are not navigation SR.

BF16 backbone, native FP32 feature preparation, FP64 frozen-ranker arithmetic, FP32 priorities. Scores computed once per 300-row observation and retained at birth values. R4 including current observation, B1200 visual-token bank, KV guard4784, original K/V positions and text action history. Eight stateless feature extractors; model GDN recurrence stays intact. Feature buffers released after scoring.

## Qualification

Independent NumPy oracle checks real features for all eight ranker heads. Strict reload, dtype/cache invariants, three fixed-history numerical replays, bitwise greedy reset and 520-step bounded-stream checks precede Habitat smoke and full evaluation.

| Replay | Max logit error | RMS error | Gate |
|---|---:|---:|---|
| 64 | 0.500000 | 0.039852 | PASS |
| 128 | 0.625000 | 0.039660 | PASS |
| previous_m5_128 | 0.750000 | 0.035823 | PASS |

Numerical gate unchanged: torch.allclose(atol=0.75, rtol=0.03), RMS≤0.15. Passing is tolerance-based, not bit-identical parity. Forced-history decoded-action agreement does not establish free-greedy equivalence. Detailed bank overlap and token disagreements are saved with each replay.

CPU evidence reused from L2: 151 tests passed. Exact src/configs/tests tree equality was verified; this is not a fresh test run. See cpu_test_reference.json. Existing S0 control evidence is hash-verified and reused.

## Navigation protocol and current result

R2R val_unseen, fixed ordered 1,839 episodes, seed42, max500 actions, same dataset/scenes/action mapping as the S/L campaign. Two Habitat smoke episodes must validate before full launch. Full-result verifier checks runtime metadata, checkpoint/data/config hashes, ordered unique episode IDs, action mapping, errors and independently recomputed metrics. Other evaluations ran concurrently; shared-load latency is not a controlled benchmark.

Validated journal: 1839/1839 episodes; SR 50.68%; SPL 47.17%; errors 0; complete=True. A complete=true validation confirms all 1,839 episodes; incomplete results remain partial.

helper.json: passed=True
greedy_reset.json: passed=True
bounded.json: passed=True, peak resident KV=2546, guard=4784
passed.json: passed=True

Smoke validation: `{   "passed": true,   "complete": true,   "episodes": 2,   "scheduled_episodes": 2,   "failures": 0,   "strict_action_metadata": true,   "journal_is_contract_ordered_prefix": true,   "metrics_recomputed": true,   "sr": 0.0,   "spl": 0.0,   "contract_sha256": "4631a52d00688a3173e731d9d1618e98ba1c11fd9a2243ec03643480fa308478" }`


## Evidence and operation

Root: `/data14t/anhdh35/m5_l1_epoch1/evaluation`. Download manifest: `../checkpoint/download_manifest.json`. Setup: `models.json`, `preflight.json`, `published_checksums_verified.json`, `tensor_inventory.json`. Qualification: `audit/L1/`, `audit.log`. Controller: `campaign.py`, `campaign_status.json`, `campaign.log`. Navigation: `results/full/L1_s429/episodes.jsonl`, `summary.json`, `runtime.json`, `evaluation_contract.json`. Partial/final validation: `full_validation.json`.

Launch command (already launched; do not duplicate): `/storage/anhdh35/SimpleMemVLN/.venv/bin/python -u /data14t/anhdh35/m5_l1_epoch1/evaluation/campaign.py`.
Final journal verification passed with complete=true. The budget sweep was later resumed separately and is complete; L2/L3 remain interrupted.
