| Policy | SR | SPL | NE | OSR | TL | forced stops | failures | infra retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m5_e1 | 51.22 | 47.96 | 5.78 | 57.97 | 9.50 | 142 | 0 | 0 |
| grpo | 55.68 | 49.66 | 5.32 | 65.69 | 11.04 | 174 | 0 | 0 |

| Run | updates | main rollouts (+discarded) | trained seqs | simulator steps | policy forward steps | wall h |
|---|---:|---:|---:|---:|---:|---:|
| grpo | 125 | 4,000 (+1,124) | 3,680 | 281 k | 281 k | 14.66 |

| new vs base | dSR [95% CI] | dSPL [95% CI] | dNE | dOSR | gained / lost | McNemar p | Holm p |
|---|---|---|---:|---:|---|---:|---:|
| grpo vs m5_e1 | +4.46 [+2.2, +6.7] | +1.71 [-0.3, +3.7] | -0.46 | +7.72 | 265 / 183 | 0.000125 | — |

pending: grpo_fb, memfork, a3_actfork, a2_memfork_grpo, a1_frame_greedy_fb, grpo_g8
