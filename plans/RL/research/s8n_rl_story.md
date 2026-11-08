# S8N + RL: the paper story and its evidence (living document, started 2026-10-30)

Status tags: **[done]** established by a finished, verified run; **[pending]** run queued or running;
**[negative]** tested and not supported. Numbers come only from finished journals (HiPerGator rows are marked
HPG; rows from this workstation are local, never paired with HPG rows).

## One-sentence claim (target)

A streaming VLN agent with a constant-size, surprise-gated memory (S8N) is not only cheap to run but cheap to
*counterfactually re-run*: its whole belief state forks in O(1), so RL can ask "what if I had remembered
differently?" (memory twins) and "what if I had been corrected here?" (rescue twins) from the agent's own state,
which is where outcome-only GRPO stops improving it.

## 1. S8N: constant-memory streaming with surprise-gated frames [done]

- Qwen3.5-4B hybrid (24 Gated DeltaNet + 8 full-attention layers). Each attention layer keeps the instruction,
  the 4 most recent observation groups and its own top-4 past frames ranked by the FP32 normalised GDN state
  drift ||S_after - S_before||_F / ||S_before||_F (birth-fixed). Constant KV (2.6k rows) and compute per step.
- R2R-CE val_unseen SR 52.42 (orig. server) / 51.50 (HPG); best of 43 memory variants in `ALL_RESULTS.md`.
- RxR-CE val_unseen: **[pending]** (no S8N RxR journal exists yet; eval launcher `ops/local/eval_rxr.sh`).

## 2. RL amplifies the memory design [done, descriptive]

Same ActiveVLN GRPO recipe (R2R 1k, 125 steps, G = 4), each delta against its own SFT journal:

| memory | SFT SR | + GRPO | + GRPO+F-B |
|---|---:|---:|---:|
| Window8 (recent 8 + GDN state) | 51.55 | 56.88 / 57.37 (2 seeds) | 58.40 / 58.08 |
| M5 (token bank, committed-write score) | 51.22 | 55.68 | 60.03 |
| **S8N** | 51.50 (HPG) | **60.41** (HPG) | 60.63 |

At equal SFT accuracy, plain GRPO gains +8.9 on S8N vs +4.5 (M5) and +5.3/+5.8 (Window8); tree credit (F-B)
closes the gap for M5 but adds nothing on S8N (+0.2, p = 0.89). Caveat: different machines; deltas are within
machine.

## 3. What does not help (negative results that shape the method) [negative]

- Action-level fork credit (F-B) on S8N: = GRPO at 2.2x simulator steps.
- Event-boundary forks: S8N's drift does not predict sibling divergence (pre-registered H1 fails).
- Surprise-placed interventions: drift predicts deviation onsets only through the preceding turns (z 7.5 -> 1.8
  with turn controls; `s8n_local_campaign.md`, 2026-10-30).
- Per-decision memory credit (MemFork on M5): retention collapsed onto the prior.

## 4. Memory twins: learning what to remember [negative at step 100; val_unseen pending]

Per-layer retention rule (consensus a_l, birth b_l; 0 = S8N) learned during GRPO by antithetic ES on
common-random-number twins (a twin replays its main's action noise with the opposite memory perturbation).
- The rule matters for the SFT policy and is policy-dependent: start-frame rule +5.7 on base S8N (val_seen-300,
  p = 0.04, Holm 0.12), -2.0 on the RL policy [done, HPG; val_unseen confirmation pending].
- Run 1 (local), steps 0-100, 808 CRN twin pairs: twins diverge in 50-100 % of pairs and flip success in ~15 %
  (both directions), but the rule's return gradient is zero within noise: sum z^2 = 5.74, sign-flip
  randomization p = 0.99; 95 % upper bound 0.62 reward (~5.8 pp SR) per one-sigma move along the best of 16
  directions, ~1.5 pp per coordinate. w random-walks around S8N's own rule (Adam-replayed nulls for path
  efficiency and for a restoring force: n.s.) [negative, local]. Reading: on the RL policy, S8N's drift
  ranking is already a stationary point of return in this family; the retention rule is not where RL gains.
- Val_unseen of the step-125 rule and val_seen-300 per export [pending]; step 25: 55.33 vs base 53.00
  (val_seen-300, p = 0.44).

## 5. Rescue / rewind twins: learning from corrections at one's own mistakes [done on the SFR line, §8]

When the agent's pose drifts >= 1 m beyond its closest approach (privileged, causal), fork its full S8N state,
let a geodesic expert correct it for ~14 decisions *inside the agent's own memory*, let the policy finish, and
credit (i) the policy's post-correction decisions by the contrast with the main's own continuation and (ii) the
expert's decisions only if the correction beat the policy's continuation (a counterfactual Q-filter).
- Probe on base S8N: 6/16 hard failures rescued (37.5 %) [done, local probe]; the same corrections replayed into a
  fresh (empty) memory rescue 5/16, so "own memory makes recovery easier" is not claimed.
- Run 2 (local, fork at the trigger t): **stopped at step 22 [negative]**. Imitating the expert's turn-arounds
  from states where the agent is already lost (79 % turns) made the policy spin: mains' turn share 0.36 -> 0.53,
  truncation 0.03 -> 0.23, paired against run 1 on the same prompts. Lesson for the paper: a privileged
  correction is only learnable where the agent could have known it; after the error the right turn depends on
  the unseen goal.
- Run 4 = **rewind** (from base S8N: not run; rewind was trained from SFR instead, §8): same trigger, but the branch forks from the agent's closest approach k* (S8N
  state snapshotted at every new minimum, one constant-size clone), so the expert re-decides the decisions
  where the trajectory left its best point, in the agent's own memory, and may STOP there (the "passed the
  goal" failure, which GRPO increases 110 -> 146). Guarded against the spinning failure (`drift_guard.py`).
- Rewind probe (base S8N, 32 hard instructions, identical mains): **12/16 triggered failures rescued vs 6/16 for
  fork-at-t** (paired 7 vs 1, p = 0.07), with 8.3 instead of 13.9 forced decisions [done, local probe].
- Systems [done, measured]: every S8N snapshot is 135.7 MiB at any t (49.5 MiB FP32 GDN state + <= 2,245 KV rows),
  cloned in 1.4 ms (vs ~280 ms per decision); a full-history cache would be 914 MiB at t = 100 and 3.5 GiB at
  t = 400. Rewinding to *every* new closest approach is only affordable because the state is constant-size.

## 6. Imitation comparator: stage-2 DAgger on S8N [done, local]

Two S8N models fine-tuned (by the user, `anhdao69/stage2_S8N` @ 59c4d00d) from the same S8N weights on 14,195
DAgger episodes (R2R 3,043 + RxR 11,152, 1,775 updates, LR 5e-6). Collection let S8N drive and a geodesic expert
take over after deviations; a STOP near the goal was overridden by the expert until the final waypoint (0.25 m);
only recovered episodes were kept. S8N_DAGGER trains on the actions as executed, S8N_DAGGER_ORACLE on oracle
relabels. Evaluated locally with the base journal's frozen evaluator and protocol (all 1,839 val_unseen episodes,
checkpoint files checked against the published SHA256SUMS, per-episode verifier: 0 problems).

| local, val_unseen | SR | SPL | OSR | NE | TL | steps | forced stops | reached <= 3 m, stopped outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base S8N | 52.04 | 48.35 | 58.08 | 5.59 | 9.41 | 90.4 | 124 | 111 |
| S8N_DAGGER | 52.04 | 42.38 | **70.26** | 5.93 | 15.96 | 148.6 | 262 | **335** |
| S8N_DAGGER_ORACLE | 58.84 | 44.97 | **72.65** | **4.75** | 15.72 | 126.5 | **68** | 254 |
| run 1 (GRPO, twins) s125 | **60.03** | **53.61** | 68.35 | **4.87** | 10.61 | 108.7 | 161 | 153 |

- S8N_DAGGER vs base: SR identical (317 vs 317 discordant, McNemar p = 1) but not the same episodes; it gets
  within 1 m of the goal in 1,047 episodes vs 651 (RL: 915). vs run 1: -7.99 SR (238 vs 385, p = 4.2e-9).
- Reading: imitation of in-memory corrections teaches S8N to *reach* the goal better than outcome RL does (OSR
  +12.2 vs +10.3), and loses all of it at the *stop* decision (335 "reached then left"), because the as-executed
  labels near the goal say "keep walking". The value of a correction depends on where it is applied and how it is
  credited: that is the rewind design (correct at the closest approach, credit by the counterfactual outcome
  instead of copying the labels).
- S8N_DAGGER_ORACLE (expert relabels on the same visited states): +6.80 SR over base (433 vs 308, p = 5.0e-6;
  CI +3.9..+9.7) and +6.80 over S8N_DAGGER (p = 3.1e-6): the stop failure of S8N_DAGGER was the labels. It ties
  run 1 (-1.20, 312 vs 334, p = 0.41) with the opposite error profile: it reaches more (OSR 72.65 vs 68.35, within
  1 m 1,031 vs 915) and stops worse (254 vs 153 "reached then left"; SPL 44.97 vs 53.61; 15.7 m vs 10.6 m paths).
  Imitation and outcome RL fix different decisions; RL from the ORACLE init is the natural IL -> RL pipeline row.
- Next (queued 2026-10-31 19:05): run 5 = GRPO G = 5 from ORACLE (IL -> RL baseline), run 6 = rewind from ORACLE.
  Claim to test: rewind's corrections at the closest approach fix the stop decision that imitation leaves broken,
  beyond what outcome-only GRPO recovers from the same IL policy [superseded].
- Status 2026-11-08: run 5 stopped at step 98 when the power failed on 2026-11-01 08:57 (kept, not evaluated). Run 6
  was not run. The IL -> RL rows moved to SFR (stage-2 DAgger with the STOP-label fix, §8).

## 7. Controls and protocol

- Local base S8N val_unseen: SR 52.04 [done]; run 1 (memory twins, effectively GRPO) +7.99 paired (p = 2.6e-11)
  [done]. Compute control GRPO G = 5 (local, run 3): 58.29 (+6.25 over base, p = 2.2e-7); run 1 vs G5 +1.74
  (p = 0.13, n.s.): memory twins = GRPO at matched rollouts [done].
- Paired McNemar over 1,839 val_unseen episodes, one evaluation per final checkpoint, selection on val_seen-300.
- Systems: MPS (1.6x rollout throughput), longest-first update order (-15 % update time), ZeRO-2 optimizer
  offload to host RAM with a fixed DeepSpeed 0.16.4 accumulation path (both GPUs train; 3 helpers/rank for host
  RAM). Run 1 changed systems mid-run (GPU AdamW for steps 0-49, offload after), rounding-level by check.

## 8. Rewind RL from SFR (stage-2 DAgger with the STOP-label fix) [1k done; 8k pending]

SFR = `anhdao69/stage2_S8N` S8N_DAGGER_SF_REMAINING (local R2R val_unseen SR 62.59). Every run: 125 steps (1,000
prompts), G = 4, seed 20261010, ActiveVLN's recipe (constant LR 1e-6). Local journals, paired McNemar over the 1,839
R2R val_unseen episodes. Full table: `s8n_local_campaign.md` §4.1.

| run | what | R2R SR / SPL | RxR 2k SR / SPL |
|---|---|---:|---:|
| SFR | starting point | 62.59 / 55.59 | 63.55 / 52.15 |
| A | ActiveVLN GRPO, R2R 1k | 64.33 / 57.32 | - |
| E | GRPO G 5 (compute-matched with B) | 66.50 / 58.71 | - |
| C | self-rewind (policy retries from k*, no expert) | 67.26 / 59.58 | - |
| G | expert fork at t (no rewind), drift-guard stop at step 20 | 65.85 / 56.65 | - |
| B | **rewind** (expert at the closest approach k*), R2R 1k | 68.73 / 60.87 | 70.70 / 56.91 |
| mix1k | **rewind**, 500 R2R + 500 RxR | **71.94 / 63.65** | 70.75 / 57.04 |
| path_mix1k | path-aware rewind, 500 R2R + 500 RxR | 69.66 / 62.29 | 69.00 / 55.92 |
| ActiveVLN 1k | GRPO on mix1k's 1,000 prompts (8k pool rows 0-999) | 66.83 / 60.01 | 69.75 / 57.61 |
| ActiveVLN 2k | the same run at 2,000 prompts | 67.92 / 61.27 | (later) |
| ActiveVLN 4k | the same run at 4,000 prompts | 69.33 / 61.44 | (later) |

- **Rewind beats outcome-only GRPO** [done]: B vs A +4.40 (p 7e-5) on the same prompts. At matched compute, B vs E
  +2.23 (p 0.037).
- **The rewind point is what makes the expert usable** [done]: forking at the trigger t spins (G: -2.88 vs B,
  p 0.016, stopped at step 20; the same failure as run 2 on base S8N).
- **The privileged expert matters at matched compute** [done]: self-rewind C = E (+0.76, p 0.50).
- **Mixed R2R + RxR prompts help** [done]: mix1k vs B +3.21 on R2R (p 0.0026), RxR equal. Goal-distance rewind
  beats path-aware rewind (path_mix1k vs mix1k -2.28, p 0.036).
- **Same prompts, only the estimator differs** [done]: ActiveVLN GRPO on mix1k's prompts in the same order is
  -5.11 vs mix1k (p 2e-6). At twice the prompts (2k) it is still -4.02 (p 4e-4). At 4k it is -2.61 (p 0.023), which
  ties B (+0.60, p 0.63).
- RxR transfer: R2R-only rewind B lifts RxR val_unseen A from 64.40 to 71.80 (p 2e-6). Full RxR (3,669) exists so
  far only for ActiveVLN 1k (69.96 / 57.67).
- **Scaling to 8k** [pending]: rewind (gpu2) and ActiveVLN (local) on the same A+ 8k pool, compared at 1k / 2k / 4k /
  6k / 8k.
  - Rewind 1k on the first 1,512 R2R episodes: 69.91 vs mix1k 71.36 on the same episodes (p 0.22). This is the same
    algorithm, prompts and order on another machine, so the gap is run-to-run spread.
  - Checkpoints kept, and on Hugging Face `anhdao69/stage3_S8N` (ActiveVLN 1k, 2k so far).
