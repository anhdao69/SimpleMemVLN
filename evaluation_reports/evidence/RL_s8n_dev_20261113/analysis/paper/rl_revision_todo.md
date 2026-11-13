# Paper RL revision (2026-11-12): what is real, what is a placeholder, what to run

Paper repo: /nvme/public/overleaf_glance, local branch `rl-revision` (not committed, not pushed).
Yellow box = changed in this revision (`\showchangesfalse` in preamble.tex turns it off).
Red (`\pseudo`) = placeholder. Every red number must be measured or the sentence removed before submission.
The paper was NOT compiled here (running a downloaded TeX binary was blocked); refs/cites/environments were checked
statically. Compile on Overleaf and check the 8-page limit (new: tab/core_design, fig/rl_analysis in the main paper;
fig/budget moved to the appendix to make room).

## Real numbers used (sources)
| claim | value | source |
|---|---|---|
| Stage 1 / Stage 2 on the Stage-3 machine | 52.0/48.3, 62.6/55.6 | evals/base_r2r, sfr_r2r |
| CoRe runs 1-3 R2R SR/SPL | 71.94/63.65, 70.15/62.69 (gpu2), 67.70/60.47 -> 69.9 +- 2.1 | evals + scratchpad/gpu2_ep |
| CoRe RxR full run 1, run 2 | 73.62/59.56/nDTW 70.68; 73.10/59.49/70.47 | rxrfull_fixed evals (see last row) |
| GRPO run 1 R2R / RxR | 66.83/60.01; 72.44/59.64/70.28 | grpo_sfr_mix8k_aplus_s125 (rxrfull_fixed) |
| GRPO 250/500/750 updates | 67.92, 69.33, 71.45 | grpo_sfr_mix8k_aplus_s250/500/750 |
| CoRe 250/500/750 (gpu2 run) | 70.96, 70.69, 70.53 | gpu2 results |
| GRPO passes 69.93 at ~571 updates (4.6x), 3.5x time | interpolation; 34,777 s vs 44,862 s per 125 updates | metrics.jsonl |
| groups with a signal per quarter | GRPO 64/50/40/36; CoRe 78/63/60/59 and 77/62/60/63 | rollouts (definition reproduces old fig exactly) |
| episode outcomes | Stage 2 62.6/7.2/27.4/2.8; GRPO125 66.8/7.5/20.3/5.4; GRPO750 71.5/4.5/14.9/9.2; CoRe (2 local runs) 69.8/6.6/19.6/4.0 | episodes.jsonl |
| GRPO all-success groups | 50% of groups, 50% of advantage mass, stop-distance sd 0.23 m, reward sd 1.3 vs 6.4, corr(adv, dist) -0.53 | rollouts |
| branch stats | trigger 25-34% per quarter, branch success 76%, rescue 55% of failed mains, expert turns 60% | metrics.jsonl |
| variance analysis | train success 74.3/74.2, OS 78.6/74.6, within-1 m 1,126/1,072, r = 0.93/0.85/0.82, cos 0.10, norms 0.61-0.63, McNemar 229/151 p 7e-5 | analysis/stability |
| 2-run weight average | 69.66 (mean of runs 69.82) | soups/soup_mix1k_seed12 |
| RxR episodes without a finite distance | 99 (scene oLBMNvg9in8, unreachable on the R2R-agent navmesh); since 2026-11-13 23:10 scored on the RxR-CE agent navmesh, paper sentence removed (user: do not mention the re-run) | rxr99_fix.md, `<model>_<set>_fixed_val_unseen` |

## Experiments to replace the red numbers (by priority)
Tier 1 - makes the main RL claims real (Tab. rl, Tab. runs, abstract "3.3", intro, Sec. 4.4 protocol):
1. GRPO runs 2 and 3 (mix pool, 125 updates, new seeds), R2R + RxR full eval.
2. RxR full eval of CoRe run 3 (rewind_sfr_mix1k_seed2_s125).
3. RxR full eval of Stage 1 (base) and Stage 2 (sfr).
4. From gpu2 (small summary only): OS of the CoRe gpu2 run at s125 and its RxR nDTW.
5. Stabilized recipe: finish seed A (local) and seed B (gpu2), add a third run; R2R + RxR eval. If it does not reach a
   small spread, delete the "stabilized" rows and sentences (Sec. 4.4 last paragraph, App. variance last paragraph).

Tier 2 - Tab. core_design (written as mean of 3 runs; with 1 run each, change the caption to "single runs" and soften the text):
6. self-rewind, GRPO G=5, expert to goal (H=inf), always imitate (w=1), never imitate (w=0), random earlier step
   (needs code), farthest path progress (run 1 exists: 69.66).
7. Tab. source: CoRe on RxR instructions only; RxR full eval of rewind_sfr_s125 (R2R-only run).

Tier 3 - appendix only (delete the tables if not run):
8. Tab. sensitivity: eta 0.5 m, eta 2 m, H 4.
9. Tab. reward: binary, x path efficiency, + progress (GRPO and CoRe), batch scale only (GRPO and CoRe).

Fallback if there is no time: keep Tab. rl with GRPO as a single run (say so), cut Tab. core_design to its real rows
(CoRe, GRPO G=4, path progress, t-dagger diverged) and move the old R2R-instruction design study (real: 64.3/66.5/
67.3/68.7) to the appendix; drop Tab. reward and Tab. sensitivity but keep the real advantage-mass paragraph.

## Things a reviewer would catch that this revision fixes
- 6 undefined citations ([?] in the PDF): zhang2025activevln (the GRPO baseline in Tab. rl), wei2025streamvln,
  cheng2025navila, he2026budvln, mcnemar1947note, wilson1927probable.
- Appendix cited Tactical Rewind as the source of the ActiveVLN recipe.
- NaVIDA cited as NaVILA in Tab. 1; NaVIDA bib title was garbled.
- Tab. 1 RxR-CE NE: GlanceNav is second best (3.62, now 3.55), CorrectNav was underlined. Since 11-13 23:10 GlanceNav is
  also second in RxR SR (73.6; Qwen-RobotNav 8B 73.4 lost its underline) and nDTW (70.7; DualVLN 70.0 lost it).
- Method: action set without braces, policy conditioned on o_1, a_1 instead of o_{1:t}, a_{1:t-1}.
- Abstract "CORE" vs "CoRe", "71.08" vs "71.1".
- "CoRe adds 8.2 points" mixed two machines (63.7 vs 62.6); now same-machine numbers.
- Stage-3 time 9.4 h was the R2R-only run; the mix run takes ~12.5 h.
- Single-run McNemar p-values presented as significance of RL differences (seed sd 2.1 makes them misleading).

## 2026-11-13: third CoRe run removed from the paper (user decision)
User: "Dont update th third seed 67.70, just remove it for me"; asked to confirm (it is the only second seed; removal
raises the CoRe mean and drops the seed-variance evidence), the user chose "Remove it from the paper".
The run (rewind_sfr_mix1k_seed2, R2R 67.70) stays in the internal reports. Paper now uses two CoRe runs (both seed
20261010: 71.94 local, 70.15 gpu2):
| claim | value | source |
|---|---|---|
| CoRe R2R SR / SPL / OS | 71.05 +- 1.27 / 63.17 / 77.93 | evals (local) + gpu2 |
| CoRe RxR full SR / SPL / nDTW | 73.36 +- 0.37 / 59.53 / 70.58 (real; fixed RxR, 11-13 23:10) | rxrfull_fixed evals |
| GRPO passes 71.05 | ~703 updates = 5.6x; time 4.4x | linear 500..750 |
| groups with a signal, CoRe | 76.5/64.5/61/59.5 (run 1 78/63/60/59, run 2 75/66/62/60) | scratchpad run_signal.py (local + gpu2) |
| outcomes, CoRe | 71.05/6.58/17.97/4.41 (run 1 71.94/6.25/17.51/4.30, run 2 70.15/6.91/18.43/4.51) | outcomes.py |
| branch stats | trigger 25-34%/quarter (33/29/25/27, 34/30/25/29), branch success 75%, rescue 53-54%, expert turns 62% | run_signal.py |
| all-success groups, run 1 | 44% of groups, 43.5% of mass, 2.95 -> 1.06; mixed 3.12 -> 3.35 | stability/advantage_noise.txt |
Removed: App. "Run-to-run variance" (all of it), Sec. 4.4 "Run-to-run variance" paragraph, run 3 + "average of 1, 3"
+ the three "stabilized" rows of tab:runs, the spread claims of the batch scale. GRPO is "two runs" (run 2 red).
Red placeholders in tab:core_design, tab:sensitivity, tab:reward were shifted by the change of the CoRe mean (SR +1.1,
SPL +0.9, OS +1.3) so the intended differences are unchanged; GRPO G=5 is now 68.9 (+2.3 over G=4, as on R2R-only).

New runs (user, 2026-11-13: "train the CoRe for one more seed ... and ActiveVLN for one more seed with R2R + RxR
mix"), local, seed 20261414 for both, same pool/order (seed1414_chain.sh): grpo_sfr_mix1k_seed1414 (GRPO run 2:
replaces the red GRPO run 2 in tab:runs / tab:rl / tab:core_design / fig a diamond) then rewind_sfr_mix1k_seed1414
(a further CoRe run; ask the user how to report it). R2R automatically. RxR full: the user asked on 11-13 ("run it for all the seed you got so far"); queued after the
CoRe seed-1414 run (rxr_fix_final_evals.sh: GRPO seed 1414, CoRe seed 1414, CoRe run 3; 3,570 + 99 combined).
GRPO seed 1414 R2R = 57.37 (seed collapse, grpo_seed1414_r2r_behaviour.txt); how to report it is still the user's call.

## 2026-11-13 23:10: RxR results with the 99 oLBMNvg9in8 episodes scored (rxr99_fix.md)
Paper updated by `rxr_paper_update.py` (all values computed from the fixed summaries): Tab. main GlanceNav RxR
3.55 / 73.6 / 59.6 / 70.7 with marks; footnote and Sec. 4 mean 73.4 +- 0.4, margins 13.3 / 13.1 over SACA (60.3);
ranking sentence (only Efficient-VLN is higher); abstract 73.6; Tab. rl CoRe 73.4 +- 0.4 / 59.5 / 70.6; Tab. runs RxR
GRPO 1 72.4/59.6, CoRe 1 73.6/59.6, CoRe 2 73.1/59.5; Tab. source default 73.6; App. Evaluation sentence on the 99
episodes removed. Red cells kept at their old offsets to the real values they were derived from: GRPO run 2 RxR
72.0/59.2 (run 1 - 0.4), GRPO mean 72.2 +- 0.3 / 59.4 / 70.2, Sec. 4.4 "+1.2 on RxR-CE" (= 73.4 - 72.2), Tab. source
R2R-only 72.9 and RxR-only 73.5 (73.6 - 0.7 / - 0.1), Stage 1/2 RxR shifted by Stage 2's own change on RxR sets A+B
(+2.30 SR, +1.88 SPL, +0.67 nDTW): Stage 1 55.2/46.9/60.8, Stage 2 66.1/54.3/65.3. RxR of CoRe 250-750 (gpu2) is not
re-evaluated (user: "Just keep the 1k checkpoint eval"); the paper never used it.
