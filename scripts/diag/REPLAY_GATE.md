# G-2 replay gate

All five model G-2 gates completed after the user authorized continuation with
the specified dataset despite the failed rendering gate. The failed rendering
result remains a limitation; it was not relabeled passed. See `SCOPE_UPDATE.md`
in the report directory for that authorization.

Use the verified recipes and saved worker invocations in
`reports/diagnostics/g3m5_failure_20261019/reproduction_preparation/` for the
completed runs. The successful G1 result is in `G1_retry_safe_path`. The shell
examples below are templates, not an exact record of the executed commands;
in particular, the completed runs supplied the local base-model path explicitly.
Always choose new output roots and preserve the existing replay evidence.

CPU verification:

```bash
/storage/anhdh35/SimpleMemVLN/.venv/bin/python -m unittest discover -s scripts/diag -p test_replay_gate.py
```

Six tests cover the exact combined absolute/relative tolerance and RMS boundary,
argmax flips, elementwise failures, nonfinite/shape rejection, bank overlap and
cardinality rejection, and immutable fixture identity/label preservation.

The orchestrator launches two sequential independent processes with distinct
`PYTHONPATH` and working directories, imports each source's `load_checkpoint`,
uses its stored serving configuration and BF16 loader, disables gradients, and
runs offline eval inference. It records all response-decision vocabulary logits,
per-layer/per-step selected IDs, encoded tensor hashes, RGB mappings/hashes,
fixture identity, checkpoint hash, commands and logs. It compares diagnostic
logits against frozen-source logits with atol=.75, rtol=.03, RMS<=.15. Bank overlap
is descriptive: no extra bank-overlap pass threshold is invented. Empty bank
steps are counted separately and excluded from the nonempty mean.

Each output path must not already exist. The original fixture JSON is never
rewritten: only `rgb_path` is relocated in memory by exact filename. Episode UID,
scene ID, step IDs, and action labels remain unchanged. The checkpoint directory
is used for both processor/model construction and strict checkpoint loading;
`--model-path` can explicitly supply the verified local base directory if needed.
Neither loader nor scientific source is patched.

Set these shell variables and helper for the commands below:

```bash
replay_python=/storage/anhdh35/SimpleMemVLN/.venv/bin/python
replay_code=/data14t/anhdh35/simplememvln_diag_g3m5_20261019
replay_assets=/data14t/anhdh35/diag_g3m5_20261019_assets
replay_output=/data14t/anhdh35/diag_g3m5_20261019_assets/g2_replay
run_remote_replay() {
  "$replay_python" "$replay_code/scripts/diag/replay_gate.py" run \
    --model "$1" --checkpoint "$2" --weights-sha256 "$3" \
    --fixture "$replay_assets/remote_evaluation/$4/fixture.json" \
    --rgb-dir "$replay_assets/remote_evaluation/$4/fixture" \
    --frozen-source "$replay_assets/remote_evaluation/$4/source" \
    --diagnostic-source "$replay_code" --output "$replay_output/$1"
}
run_remote_replay M5_E1 "$replay_assets/checkpoints/M5_E1/epoch-1" \
  92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6 m5_e1_epoch1_20261019
run_remote_replay TRIPLE "$replay_assets/checkpoints/G3_M5_L8_L16_L24/epoch-1" \
  c206e32e4bbe884ae6ac5dbcd1a46ac7589d9218b7bc8f27a70e29a821b295ce g3_m5_l8_l16_l24_epoch1_20261018
run_remote_replay G3_M5 /data14t/anhdh35/simplememvln_g3_m5_combined_eval/checkpoints/G3_M5/epoch-1 \
  4e2d2095325fa779dcdece278f52213013d6889e72705ff860ab2eba1b13194f g3_m5_epoch1_20261017
"$replay_python" "$replay_code/scripts/diag/replay_gate.py" run \
  --model G1 --checkpoint /data14t/anhdh35/simplememvln_g1_eval/checkpoints/G1/epoch-1 \
  --weights-sha256 a620ab17b1c0cbec6f7c5f95d67177a4b078ae53e83ea08c6696c5c4797a63ae \
  --fixture /data14t/anhdh35/simplememvln_g1_eval/replay_128.json \
  --rgb-dir /data14t/anhdh35/simplememvln_g1_eval/replay_frames \
  --frozen-source /data14t/anhdh35/simplememvln_g1_eval/source \
  --diagnostic-source "$replay_code" --output "$replay_output/G1"
```

G3 uses the existing local serving directory verified by the main task:

```bash
run_remote_replay G3 /data14t/anhdh35/simplememvln_g3_m5_eval/checkpoints/G3/G3/epoch-1 \
  a71be6b5291b6acddbeb389318befaded5818c5b1438d5d44cf351c64248333b g3_epoch1_20261018
```

The five completed G-2 results report zero logit RMS error, zero maximum absolute
error, and zero argmax flips against each checkpoint's frozen source. The three
M5-family models also have exact bank selections. These are results for the fixed
replay fixture, not proof of equivalence on every possible input. Numeric evidence
and hashes are retained in `REPLAY_G2_PROGRESS.json` and the report prerequisites.
CPU unit tests alone do not establish GPU parity. The helper records imported
loader paths and enforces the supplied checkpoint weight hash; artifact
verification separately establishes the frozen-source identity.
