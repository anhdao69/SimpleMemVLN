# S8N event-boundary analyses (s8n_rl_report.md §5)

Scripts as run on HiPerGator; outputs in `plans/RL/research/results/s8n/h1h2/`. REPO = this checkout,
RUN = the GRPO+F-B run directory (`svln_runs/rl/hico_grpo_fork_bs_s8n_r2r1k`), HI = first RL step not read
(41 for the early read `*_steps0_40`, 125 for `*_all`).

| script | output | usage |
|---|---|---|
| `../analyze_event_forks.py` | `h1_fb_*.json` (pre-registered H1) | `python scripts/rl/analyze_event_forks.py RUN [--steps 0:HI] --out FILE` |
| `h1_position_controls.py` | `h1_position_controls_*.json` | `python h1_position_controls.py REPO RUN HI` |
| `h1_success_flip.py` | `h1_success_flip_*.json` (exploratory target) | `python h1_success_flip.py REPO RUN HI` |
| `h2_geometry.py`, `h2_geometry_ff.py` | `h2_geometry*_*.json` (H2 proxy) | `python h2_geometry_ff.py REPO RUN HI` |
| `drift_by_t.py` | `drift_by_t_fb_step0.txt` | `python drift_by_t.py RUN` |
