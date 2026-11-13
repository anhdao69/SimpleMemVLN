## Table R5 rows (SR / SPL; full: SR / SPL / nDTW)

| Model | A SR / SPL | B SR / SPL | C SR / SPL | 2k SR / SPL | full SR / SPL / nDTW | full OS |
|---|---|---|---|---|---|---|
| `sfr` (Stage 2) | 66.50 / 54.87 | 65.20 / 53.19 | - | 65.85 / 54.03 | - | - |
| `rewind_sfr_s125` (CoRe, R2R-only prompts (B)) | 74.00 / 59.09 | 72.20 / 58.50 | - | 73.10 / 58.80 | - | - |
| `rewind_path_sfr_mix1k_s125` (path-aware rewind, mix) | 72.20 / 58.44 | 70.80 / 57.16 | - | 71.50 / 57.80 | - | - |
| `grpo_sfr_mix8k_aplus_s125` (GRPO run 1, mix) | 72.10 / 58.92 | 72.20 / 60.14 | 72.80 / 59.76 | 72.15 / 59.53 | 72.44 / 59.64 / 70.28 | 80.02 |
| `rewind_sfr_mix1k_s125` (CoRe run 1, mix) | 73.30 / 59.63 | 73.10 / 58.45 | 74.12 / 60.18 | 73.20 / 59.04 | 73.62 / 59.56 / 70.68 | 80.68 |
| `rewind_sfr_mix8k_aplus_s125` (CoRe run 2 (gpu2), 125) | 73.80 / 59.26 | 71.90 / 58.20 | 73.40 / 60.41 | 72.85 / 58.73 | 73.10 / 59.49 / 70.47 | 80.05 |

## Full-set details (NE, TL, forced stops, failures, episodes)

- `grpo_sfr_mix8k_aplus_s125`: SR 72.44 SPL 59.64 nDTW 70.28 OS 80.02 NE 3.719 TL 17.48 forced 314 failures 1 invalid 1 episodes 3669 (local)
- `rewind_sfr_mix1k_s125`: SR 73.62 SPL 59.56 nDTW 70.68 OS 80.68 NE 3.551 TL 18.07 forced 349 failures 0 invalid 0 episodes 3669 (local)
- `rewind_sfr_mix8k_aplus_s125`: SR 73.10 SPL 59.49 nDTW 70.47 OS 80.05 NE 3.552 TL 17.84 forced 343 failures 0 invalid 0 episodes 3669 (gpu2)

## CoRe runs 1 + 2 (seed 20261010), full RxR: mean +- sd (sample sd)

- sr: 73.62, 73.10 -> mean 73.358 sd 0.366
- spl: 59.56, 59.49 -> mean 59.527 sd 0.047
- ndtw: 70.68, 70.47 -> mean 70.577 sd 0.148
- oracle_success: 80.68, 80.05 -> mean 80.362 sd 0.443
- navigation_error: 3.55, 3.55 -> mean 3.552 sd 0.001

## Paired tests (local; base -> new)

| base | new | split | SR base | SR new | Delta SR | Delta SPL | only base | only new | McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 72.20 | 72.10 | -0.10 | +0.48 | 123 | 122 | 1 | [-3.2, 3.0] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 73.30 | 72.10 | -1.20 | -0.71 | 127 | 115 | 0.48 | [-4.3, 1.9] |
| `rewind_sfr_mix1k_s125` | `rewind_path_sfr_mix1k_s125` | rxr | 73.30 | 72.20 | -1.10 | -1.19 | 107 | 96 | 0.483 | [-3.9, 1.7] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 74.00 | 72.10 | -1.90 | -0.17 | 127 | 108 | 0.24 | [-5.0, 1.2] |
| `rewind_sfr_s125` | `rewind_path_sfr_mix1k_s125` | rxr | 74.00 | 72.20 | -1.80 | -0.65 | 116 | 98 | 0.245 | [-4.7, 1.1] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_s125` | rxr | 74.00 | 73.30 | -0.70 | +0.54 | 111 | 104 | 0.682 | [-3.5, 2.2] |
| `sfr` | `grpo_sfr_mix8k_aplus_s125` | rxr | 66.50 | 72.10 | +5.60 | +4.05 | 117 | 173 | 0.0012 | [2.3, 9.0] |
| `sfr` | `rewind_path_sfr_mix1k_s125` | rxr | 66.50 | 72.20 | +5.70 | +3.57 | 93 | 150 | 0.000309 | [2.7, 8.8] |
| `sfr` | `rewind_sfr_mix1k_s125` | rxr | 66.50 | 73.30 | +6.80 | +4.76 | 92 | 160 | 2.19e-05 | [3.7, 9.9] |
| `sfr` | `rewind_sfr_s125` | rxr | 66.50 | 74.00 | +7.50 | +4.22 | 84 | 159 | 1.72e-06 | [4.5, 10.5] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 71.50 | 72.15 | +0.65 | +1.73 | 227 | 240 | 0.579 | [-1.5, 2.8] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 73.20 | 72.15 | -1.05 | +0.49 | 250 | 229 | 0.361 | [-3.25, 1.15] |
| `rewind_sfr_mix1k_s125` | `rewind_path_sfr_mix1k_s125` | rxr2k | 73.20 | 71.50 | -1.70 | -1.24 | 216 | 182 | 0.098 | [-3.7, 0.25] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 73.10 | 72.15 | -0.95 | +0.73 | 230 | 211 | 0.391 | [-2.95, 1.1] |
| `rewind_sfr_s125` | `rewind_path_sfr_mix1k_s125` | rxr2k | 73.10 | 71.50 | -1.60 | -1.00 | 234 | 202 | 0.138 | [-3.65, 0.4] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_s125` | rxr2k | 73.10 | 73.20 | +0.10 | +0.25 | 212 | 214 | 0.961 | [-1.95, 2.15] |
| `sfr` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 65.85 | 72.15 | +6.30 | +5.50 | 210 | 336 | 7.77e-08 | [4.05, 8.5] |
| `sfr` | `rewind_path_sfr_mix1k_s125` | rxr2k | 65.85 | 71.50 | +5.65 | +3.77 | 183 | 296 | 2.74e-07 | [3.5, 7.7] |
| `sfr` | `rewind_sfr_mix1k_s125` | rxr2k | 65.85 | 73.20 | +7.35 | +5.02 | 165 | 312 | 1.61e-11 | [5.25, 9.45] |
| `sfr` | `rewind_sfr_s125` | rxr2k | 65.85 | 73.10 | +7.25 | +4.77 | 178 | 323 | 9.29e-11 | [5.1, 9.4] |
| `grpo_sfr_mix8k_aplus_s125` | `rewind_sfr_mix1k_s125` | rxrfull | 72.44 | 73.62 | +1.17 | -0.08 | 423 | 466 | 0.159 | [-0.44, 2.78] |
