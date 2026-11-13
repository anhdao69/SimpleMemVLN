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
| CoRe RxR full run 1, run 2 | 71.08/57.50/nDTW 69.51; 70.59/57.44 | rxrfull evals |
| GRPO run 1 R2R / RxR | 66.83/60.01; 69.96/57.67/69.39 | grpo_sfr_mix8k_aplus_s125 |
| GRPO 250/500/750 updates | 67.92, 69.33, 71.45 | grpo_sfr_mix8k_aplus_s250/500/750 |
| CoRe 250/500/750 (gpu2 run) | 70.96, 70.69, 70.53 | gpu2 results |
| GRPO passes 69.93 at ~571 updates (4.6x), 3.5x time | interpolation; 34,777 s vs 44,862 s per 125 updates | metrics.jsonl |
| groups with a signal per quarter | GRPO 64/50/40/36; CoRe 78/63/60/59 and 77/62/60/63 | rollouts (definition reproduces old fig exactly) |
| episode outcomes | Stage 2 62.6/7.2/27.4/2.8; GRPO125 66.8/7.5/20.3/5.4; GRPO750 71.5/4.5/14.9/9.2; CoRe (2 local runs) 69.8/6.6/19.6/4.0 | episodes.jsonl |
| GRPO all-success groups | 50% of groups, 50% of advantage mass, stop-distance sd 0.23 m, reward sd 1.3 vs 6.4, corr(adv, dist) -0.53 | rollouts |
| branch stats | trigger 25-34% per quarter, branch success 76%, rescue 55% of failed mains, expert turns 60% | metrics.jsonl |
| variance analysis | train success 74.3/74.2, OS 78.6/74.6, within-1 m 1,126/1,072, r = 0.93/0.85/0.82, cos 0.10, norms 0.61-0.63, McNemar 229/151 p 7e-5 | analysis/stability |
| 2-run weight average | 69.66 (mean of runs 69.82) | soups/soup_mix1k_seed12 |
| RxR episodes without a finite distance | 99 (paper said 98) | rxrfull episodes.jsonl |

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
- Tab. 1 RxR-CE NE: GlanceNav (3.62) is second best, CorrectNav was underlined.
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
| CoRe RxR full SR / SPL / nDTW | 70.84 +- 0.35 / 57.47 / 69.54 (now real) | rxrfull evals |
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
(a further CoRe run; ask the user how to report it). R2R automatically; RxR full needs the user's go-ahead.
