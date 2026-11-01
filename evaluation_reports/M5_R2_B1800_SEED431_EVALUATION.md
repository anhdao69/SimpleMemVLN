# Final M5 E1 R2/B1800 seed431 — 2026-10-21

Full R2R VLN-CE val_unseen evaluation completed on 2026-10-21. Final validation
recomputed metrics from the complete journal and verified unique episode IDs,
contract order, checkpoint hashes and action metadata.

| Model | Successes / episodes | SR (%) | SPL (%) | NE (m) | Oracle success (%) | State |
|---|---:|---:|---:|---:|---:|---|
| M5 E1 R2/B1800 seed431 | 944 / 1,839 | 51.332246 | 47.525482 | 5.726252 | 58.945079 | FINAL |

Zero execution failures and zero invalid responses. Mean episode length:
104.2447 actions; mean trajectory length: 9.6024 m; forced STOPs: 175.
Peak KV occupancy: 2,610 rows against the 4,616 structural guard;
peak allocated GPU memory: 8.6931 GiB.

This is the separately trained **seed431, R2/B1800, one-epoch-schedule** model,
not the historical P2-M5 seed429 R4/B1200 export. It retains two recent groups
including current and 1,800 episodic visual tokens per attention layer,
committed_mean_l2_v1 scoring, text action history and the saved serving contract.
The inactive selector seed is 429. Training completed 3,852 updates with 116 warmup
updates. No teacher, auxiliary heads, PCA or step lanes are enabled.

Checkpoint: `anhdao69/Streaming_VGGT/M5_E1_R2_B1800_seed431/epoch-1`.
Pinned HF revision: `e1d614ab3eb5f2737dbe27de69906b05bea372e3`.
Weights SHA256: `5ed04b21391ce8a92c070cdd53ab4e3b0be56b30952302f7c999daadd766c76c`.
Source branch: `c2_vggt_omega_g3_m5_layers`, frozen commit
`fde1cbef213e27320920400a826addea8d5e114a`; src/config trees match training revision
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`.

Qualification: all 18 published artifact hashes verified, 724 finite backbone
tensors, strict GPU load, 59 targeted tests, native scorer check, 64/128-step
replays, reset/retry checks and 520-step bounded stream passed. The 128-step replay
had one raw next-token disagreement out of 383 but passed the unchanged
atol0.75 + rtol0.03 and RMS<=0.15 criteria; passing does not imply bitwise or greedy
trajectory equivalence. Full evaluation used Habitat-Lab/Sim0.2.4, seed42,
500-action cap, 3m success radius, 640x480 RGB, 79-degree HFOV, 0.25m forward
steps and 15-degree turns. Final journal validation passed.

Run root: `/data14t/anhdh35/m5_e1_r2_b1800_seed431_epoch1/evaluation/`.
Results: `results/full/M5_R2_B1800_s431/{episodes.jsonl,summary.json}`.
Validation: `full_validation.json`; qualification: `audit/M5/passed.json`.
This additional result does not replace historical checkpoint identities or
establish a causal memory-budget/seed advantage.

[Machine-readable evidence](evidence/M5_R2_B1800_SEED431_FINAL_20261021.json).
