# Frozen expert-panel preparation

The completed run used render mode after the archive inventory found no
val_unseen frames. All 240 episodes and 13,847 RGB frames were verified. The
exact completed command, launcher settings and evidence are preserved under
`reports/diagnostics/g3m5_failure_20261019/reproduction_preparation/`.
The original training-rendering parity failure remains disclosed; the user
authorized use of the specified data after that failure.

`prepare_expert.py` accepts the frozen 240 IDs in `panel_ids.json`. It never chooses replacement episodes or reruns the training rendering gate. The original failed gate is embedded in external `PREPARATION.json`, together with the scope-update identity and an explicit unqualified-data limitation.

Run after archive inventory confirms whether all frozen val_unseen frames exist. Archive mode requires an already extracted directory containing `<episode_id>/step_XXXX_<ACTION>.png`:

```bash
/storage/anhdh35/miniconda3/envs/janusvln_host/bin/python -m scripts.diag.prepare_expert \
  --mode archive \
  --source /storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --ground-truth /storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --panel reports/diagnostics/g3m5_failure_20261019/panel_ids.json \
  --frames-root /absolute/path/to/extracted/val_unseen \
  --output /data14t/anhdh35/diag_g3m5_20261019_assets/expert_panel_archive
```

Only after confirming no usable archive val_unseen panel, use `--mode render` with `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python` and fresh external `--frames-root` and `--output` paths. This invokes Habitat 0.2.4 with unchanged evaluation camera/actions, seed 42, and observations before actions including the STOP frame. It does not tune camera parameters or claim the failed training-pixel gate passed. The renderer refuses existing per-episode output directories rather than overwriting or silently resuming partial trajectories.

Both modes verify the official source gzip hash, fixed panel metadata, GT action counts, every action filename including terminal STOP, and every PNG's RGB 640×480 layout. They produce a deterministic filtered metadata gzip retaining original episode fields and vocabulary, then call unchanged `scripts/vln/prepare_r2r.py --split val_unseen --source <subset> --frames-root <frames>`. The resulting `dataset_version` is honestly the filtered gzip SHA-256; original gzip and panel hashes are retained separately. Every frame hash and the unchanged converter's audit are saved externally. `validate_episode` runs inside the unchanged converter. Policy `EpisodeSerializer.encode_episode` validation is a later policy-environment step, not performed by this simulator-only helper.

CPU verification:

```bash
/storage/anhdh35/miniconda3/envs/janusvln_host/bin/python -m unittest \
  scripts.diag.test_prepare_expert scripts.diag.test_render_panel -v
```

The original helper validation recorded thirteen passing tests, including a
subprocess integration test of the unchanged converter. The subsequently
completed GPU render is documented above; those helper tests alone do not
establish rendering parity.
