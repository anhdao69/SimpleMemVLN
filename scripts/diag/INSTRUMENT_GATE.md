# G-1/G-3/D-c instrumentation runner

`instrument_gate.py` loads one checkpoint through its stored native runtime,
checks plain versus simultaneously captured state/attention logits byte for byte,
then reconstructs layers 8,16,22,24,26 and compares every observation anchor against
native FLA. M5-family input hooks include the union 2,6,8,10,14,16,18,22,24,26,30
for D-d scorer-state measurements; Window8 uses 8,16,22,24,26. The registered
project attention wrapper reconstructs pre-output-gate attention at layers
3,7,11,15,19,23,27,31 on the anchor plus32 fixed random visual queries per step. It
retains raw logits, full reconstructed state/source tensors, per-anchor errors,
image mapping hashes, input hashes, checkpoint hash and incremental status.
No optimization or policy-parameter updates are performed.

The initial GPU runs use each model's original replay fixture and are explicitly
partial. Passing one fixture does not satisfy two panel episodes per model.
The current runner includes all state/scorer input hooks and the attention
wrapper together. Any additional hooks introduced later require a new G-1 check.
All errors and numeric failures remain in `result.json`, and dependent checks
stop on G-1/G-3 failure. A new output directory is required for every invocation.

CPU test:

```bash
OMP_NUM_THREADS=1 /storage/anhdh35/SimpleMemVLN/.venv/bin/python -m unittest discover -s scripts/diag -p test_instrument_gate.py
```

Fixture invocation (substitute model, checkpoint/hash and frozen fixture root):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
/storage/anhdh35/SimpleMemVLN/.venv/bin/python scripts/diag/instrument_gate.py \
 --model M5_E1 \
 --checkpoint /data14t/anhdh35/diag_g3m5_20261019_assets/checkpoints/M5_E1/epoch-1 \
 --weights-sha256 92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6 \
 --model-path /storage/anhdh35/SimpleMemVLN/artifacts/checkpoints/qwen-base \
 --fixture /data14t/anhdh35/diag_g3m5_20261019_assets/remote_evaluation/m5_e1_epoch1_20261019/fixture.json \
 --rgb-dir /data14t/anhdh35/diag_g3m5_20261019_assets/remote_evaluation/m5_e1_epoch1_20261019/fixture \
 --input-kind fixture \
 --output /data14t/anhdh35/diag_g3m5_20261019_assets/instrument_gates/M5_E1_fixture
```

For panel checks, replace `--fixture` and `--rgb-dir` with `--manifest` pointing
to the prepared expert JSONL, `--input-kind panel`, and two explicit
`--episode-uid` arguments. IDs and order must be declared before measurement.
JSON arrays are also accepted. RGB paths in a prepared manifest need no
relocation. Never label replay fixtures as panel episodes.

Run sequentially after all prior GPU replay work completes. The full-prefix
G-3 check is quadratic in trajectory length; peak memory/runtime are measured
by the actual invocation, not established by CPU tests. See `STATE.md` for
reference scope: identical prepared FP32 tensors are supplied to native FLA;
independent fused-convolution preparation parity is not measured by this gate.

## Recorded fixture execution (2026-10-20)

Both independent processes completed with exit code 0 and status `PARTIAL`.
M5_E1 and TRIPLE each have one original 128-step fixture, zero panel episodes,
95,106,560 bitwise-identical BF16 decision logits, and 640 passing G-3 anchor
checks (128 anchors at each of five layers). Worst relative state error was
0.002258927260933112 for M5_E1 and 0.002486466917908736 for TRIPLE; the unchanged
limit is 0.01. These are numerical checks on replay fixtures, not complete
two-panel-episode qualification. Raw records live under
`/data14t/anhdh35/diag_g3m5_20261019_assets/instrument_gates/`, in the
`M5_E1_fixture` and `TRIPLE_fixture` directories.

## Persistent panel queue

`gate_queue.py` uses `gate_models.json` and runs all five models sequentially on
lexicographically selected frozen panel IDs1007 and1013. Its startup writes
`selection.json` with the original panel SHA256 before any model measurements.
It waits for PREPARATION status complete, teacher supervisor complete/exit0 and
240-episode COMPLETE.json, then verifies manifest/index hashes. Every model
re-verifies its complete checkpoint weight SHA256. No model is launched after
a failed or incomplete gate. Exact child commands and source hashes are saved.

Active queue output:
`/data14t/anhdh35/diag_g3m5_20261019_assets/instrument_gates/panel_allhooks/`.
The root measurement queue must wait for `queue_status.json` status `complete`,
not a single model result. Supervisor launch command/PID are recorded in the
parent directory's `panel_allhooks_process.json`.

Per-model results include `selected_episode_ids`, `selected_episode_uids`,
`input_capture_layers`, `attention_capture_layers`, `weights_sha256`,
`panel_episodes_completed`, and rows with `g1_pass`, `g3_pass`, `dc_pass`, and
`attention_pass` (alias of dc_pass). PASS requires two panel episodes with all
three gates passing. Fixture runs remain PARTIAL.

Attention storage is bounded: raw post-RoPE Q/K/V, visible IDs and native/output
tensors are retained for steps0 andlast in every layer; every step has tensor
hashes, shapes/dtypes, complete per-query/head mass/count metrics and numerical
PV gate evidence. These raw-step choices are fixed before observing results.
The per-step PV relative Frobenius tolerance remains .01.
