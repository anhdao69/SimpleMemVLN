# Hard-2k: decision at the 1k-hard milestone (step 125)

Written 2026-11-10 ~12:45, before the step-125 evaluation exists (training at step 99).
User: "After finish eval the local and get the result. Decide the next step for me. If the result good improvement.
Continue train for 1k hard. if not then think deeply why then propose next step".

Comparison: `rewind_mix1k_hard2k_s125` vs its start checkpoint `rewind_sfr_mix1k_s125` (SR 71.94, SPL 63.65),
R2R val_unseen, the same 1,839 episodes, paired (compare_evals.py; McNemar). Noise: the paired SR difference has a
standard error of about 1.1 pp (~430 discordant episodes); two runs of the same recipe differ by up to ~2 pp.

GOOD improvement -> continue to 2k hard prompts (step 250; hard2k_chain.sh resumes on its own), if ALL hold:
- SR delta >= +1.0 pp (SR >= 72.94), and
- SPL delta >= -0.5 pp, and
- verify_eval: 0 problems, 0 policy failures, and no drift-guard stop.

NOT GOOD (anything else) -> stop the run at once (stop_run.sh with the reason; exports/step_0125 and the latest
resumable state kept; the chain then ends), diagnose with
`scripts/rl/s8n_analysis/diagnose_pair.py` (vs mix1k and vs ActiveVLN 6k: failure taxonomy, paired flips by path
length, instruction length and scene, turn share, forced stops), check training-side signals (training success
trend, rescue use, drift-guard turn shares), and propose the next step to the user.

## Outcome (2026-11-10 16:25)

`rewind_mix1k_hard2k_s125`: R2R val_unseen SR 68.84 / SPL 62.72 vs mix1k 71.94 / 63.65: SR -3.10 (McNemar p 0.0036,
CI [-5.11, -1.09]), SPL -0.93. NOT GOOD -> run stopped at 16:26 (STOPPED with the reason; exports/step_0125 and
resumable/step_0125 kept); diagnosis in diag_mix1k_vs_hard1k.json.

Diagnosis:
- Broad drop: every path-length and instruction-length tertile, 10 of 11 scenes (157 gained, 214 lost).
- Reaches the goal less: oracle success 78.6 -> 73.7. Wrong-route failures (stopped > 5 m away, never within 3 m)
  260 -> 347; passed-the-goal 122 -> 89, ran-out-of-steps 72 -> 56. Lost episodes end as wrong-route in 123/214.
- Explores less: val turn share 0.455 -> 0.403, steps 97 -> 83, path length 11.5 -> 10.6 m. Same on the training
  prompts (main turn share 0.517 -> 0.433, steps 110 -> 91); the drift guard checks only for MORE turning.
- No more learning signal than mix1k: trained trajectories per step ~40 in both runs; all-success groups 0.32-0.36
  (hard) vs 0.47 (mix1k on the A+ pool; the 66-70 % figure was ActiveVLN's). Training success plateaued at ~0.69
  after step 25.
- Likely mechanism: failures on the hard prompts were long, turn-heavy "lost" episodes; GRPO pushed down turning
  and long searches, so on unseen scenes the policy commits to a route earlier and more often picks the wrong one.
