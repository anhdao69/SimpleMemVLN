# G3+M5 diagnostics (D-a through D-f)

Execution resumed after the user confirmed the JanusVLN R2R and RxR 15-degree datasets and instructed us to use them. The original three-training-episode rendering gate remains failed and is disclosed as a limitation, as recorded in `SCOPE_UPDATE.md`. D-g is removed. Model parity and instrumentation tolerances remain unchanged.

Implemented preparation and gate tools:

1. Verify checkpoint, teacher, original journal/source and scene identities using the saved report manifests. The original OnlineG3Teacher constructor passed its identity gate.
2. `python -m scripts.diag.panels --data-root /storage/anhdh35/JanusVLN/data/datasets/r2r/val_unseen --output-dir NEW_EMPTY_OUTPUT_DIRECTORY` selects deterministic panels. Existing frozen report panels are preserved; the command refuses overwrite.
3. Training references were recovered with the extraction command recorded in `TRAINING_REFERENCE_RECOVERY.json` from the verified JanusVLN archives.
4. `PYTHONPATH=src /storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python -m scripts.diag.render_panel` ran the rendering gate. It returned failure. Reproduction must use a new `--output` and `--report` destination to preserve the original failure evidence.
5. G-2 tooling and exact five-model commands are in [REPLAY_GATE.md](REPLAY_GATE.md). Execution evidence is saved in the external assets `g2_replay` directory; consult each result.json for completion.

Remaining execution order (all launchers refuse overwriting existing outputs):

6. `prepare_expert.py` renders the frozen 240 episodes after confirming the archive contains only training frames. See [PREPARE_EXPERT.md](PREPARE_EXPERT.md).
7. `cache_teacher.py` computes verified online FP32 Omega targets and stores FP16 frame caches. No teacher enters a policy forward.
8. `gate_queue.py` runs two fixed panel episodes (1007 and 1013) on all five models with all state/scorer and attention hooks installed. It checks bitwise G-1, native-state G-3 and probability-times-V attention parity. See [INSTRUMENT_GATE.md](INSTRUMENT_GATE.md).
9. `measurement_queue.py` verifies every completed gate and runs `measure_panel.py` sequentially on all five complete panels. Outputs include native recall, fixed Gaussian state features, current-view features, attention measurements, selection statistics and dummy-head probe drift. It stops on any gate or integrity failure.
10. `fit_probes.py` fits scene-held-out ridge probes from completed frozen captures. Current-view controls always use layer 24. Saved fits are reused for ablation inference; navigation weights are never fitted.
11. `run_bank_ablation.py` measures global and matched targeted/control interventions after its copied full-forward logits/plan checks. Targeted/control Metric 1 uses an unmodified prefix baseline at the same anchor and sequence length; full-vs-prefix numeric sensitivity is retained as descriptive evidence, not a parity gate. The approved KL is decision-token vocabulary KL, in nats.
12. `selector_eval.py` runs the two-episode smoke before four disjoint workers for each model's 600 random-selector episodes and 60 original-selector repeats. Outputs are explicitly diagnostic and unqualified. Its original absolute scene path is preserved through a read-only bind mount.
13. Aggregate paired episode-level results with 2,000 within-scene bootstrap resamples; produce figures and the final report. No hypothesis decision is justified until its required measurements exist.

Use `python -m scripts.diag.MODULE` from this worktree with `PYTHONPATH=src:.`. Direct-script invocation can make `scripts/diag/statistics.py` shadow Python's standard library; isolated replay workers explicitly use Python safe-path mode.

The active server's exact commands and logs are under `/data14t/anhdh35/diag_g3m5_20261019_assets`. The first post-measurement output at `post_measurements/` is preserved as a failed attempt; the corrected active queue is `post_measurements_retry1/queue_status.json`. Inspect it with its child PIDs before deciding a process stopped. Do not launch duplicates just because a polling interval expires.

CPU checks (policy environment):

```bash
PYTHONPATH=src:. /storage/anhdh35/SimpleMemVLN/.venv/bin/python - <<'PYTEST'
from pathlib import Path
import unittest
names = ['scripts.diag.' + p.stem for p in sorted(Path('scripts/diag').glob('test_*.py'))]
result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

Helper tests do not establish numerical model parity. The final report must distinguish completed measurements, fixture-only evidence, queued work and unavailable metrics. The full requested scope remains in `IMPLEMENTATION_PLAN.md`, `TASK_SPEC.md`, `SCOPE_UPDATE.md` and `KL_SCOPE_UPDATE.md`.

### Final report stage

Once post_measurement_queue completes, assemble the hash-pinned prerequisite JSON described in `report.py`. Retain the rendering failure, its user-authorized continuation, and the decision-token KL scope amendment. Normalize recorded queue commands into a separately hashed reproduction command file with their actual working directories; preserve original command records. Run the report generator with the simulator Python (which includes matplotlib), writing a new review directory:

```bash
PYTHONPATH=src:. /storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python -m scripts.diag.report --post-root /data14t/anhdh35/diag_g3m5_20261019_assets/post_measurements_retry1 --panel reports/diagnostics/g3m5_failure_20261019/panel_ids.json --prerequisites REPORT_PREREQUISITES_JSON --output NEW_REPORT_REVIEW_DIRECTORY
```

Review all primary contrasts, missing-match coverage, denominators and figures against raw evidence before installing the final report. The generator requires actual completed captures, probes, ablations and selector runs; tests with synthetic inputs are not scientific results.

The final report generator also exports D-a Metric 1 as a long-form pair table. It checks every selected episode's `COMPLETE.json`, all required recall-layer CSV SHA-256 values and exact frozen-panel coverage; the table carries model, episode/instruction/scene, layer, anchor, source, lag, region, epsilon and forgetting. The five models use layers 22/24/26, with layers 8/16 additionally for TRIPLE. This approximately gigabyte-scale CSV is stored as verified split-gzip parts under `reports/diagnostics/g3m5_failure_20261019/completed_exports/`; `native_recall_pairs.manifest.json` records source and part hashes, and `native_recall_pairs.capture_manifest.json` pins each capture and its recall CSV hashes. Reconstruct it with `gzip -cd native_recall_pairs.csv.part-*.gz > native_recall_pairs.csv` from the completed_exports directory. The report's `results.json#/D_a_metric1_pair_export` records coverage and the reconstructed CSV identity.

Completed teacher-forced figures can be reviewed while D-e is still live by
calling `build_figures(result, output, selection_csv, include_selector=False)`.
This explicit mode produces only D-a, D-b, D-c, D-d and D-f; it does not invent
selector results or change completion status. The default final-report path
still requires D-e before writing figures. The exact current invocation and
input hashes are saved in `COMPLETED_FIGURE_COMMAND.json` and
`completed_figure_review_20261021/figure_results.json` in the report directory.

Completed preparation/replay recipes are pinned in the report directory under
`reproduction_preparation/`; completed model gates, captures and analyses are
pinned under `reproduction_completed/`. These inventories distinguish original
argv from reconstructed launch recipes and retain completion evidence. They
preserve historical output paths: use new roots for reproduction. D-e and final
report commands remain pending until those stages finish.
