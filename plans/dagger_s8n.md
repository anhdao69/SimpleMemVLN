# On-policy DAgger for S8N (R2R-VLNCE train) — design and runbook

> Historical R2R-only experiment retained for comparison. The supported joint
> R2R/RxR15 collection on `dagger_s8n` uses the [new runbook](../docs/S8N_DAGGER.md)
> and `s8n_dagger` entry points. See the
> [comparison and validation report](../evaluation_reports/S8N_DAGGER_COLLECTION.md)
> for the immediate-takeover, memory-feedback, length and merge decisions.

Code: `src/qwen_vl/dagger/`, `scripts/dagger/`, tests `tests/dagger/` (44 CPU tests, no simulator or
GPU needed). Branch base: `m5_s8n_followup` (S8N training/serving source).

## 1. Purpose

**Experiment:** CONT-DAgger vs S8N_E2.
- Both are one weights-only epoch from the S8N epoch-1 export, with fresh AdamW, the same
  warmup/cosine schedule and the same 3,852 updates.
- Only the manifest differs. S8N_E2 used the oracle joint manifest (49.59 SR). CONT-DAgger uses
  oracle rows mixed with on-policy DAgger rows collected by S8N itself.
- The paired difference is the effect of recovery data, separated from "one more epoch".

## 2. Reference implementations read

| Code | Expert | Who acts | Labels | Kept episodes |
|---|---|---|---|---|
| StreamVLN `streamvln/streamvln_dagger.py` (`dagger_p=0`) | `ShortestPathFollower` along `reference_path` waypoints: radius 1.8 m, final point 0.25 m | Policy; expert on takeover until the next waypoint | **Executed actions** (policy or expert) | Final distance < 0.5 m and (PL < 0.93 after a takeover, or PL < 0.85) |
| JanusVLN `src/dagger.py` | Same (copy of StreamVLN, one action per step) | Same | Same | Same |
| VLN-CE `vlnce_baselines/dagger_trainer.py` | `ShortestPathFollower` to the goal | Expert with prob. β = p^it, otherwise policy (sampled) | Oracle action at every visited state; history = executed | All |

**What we follow:** StreamVLN/JanusVLN.
- Their expert follows the instruction's reference path, the same path the oracle data follows.
- Their labels equal the executed actions. This keeps our serializer contract unchanged: the
  appended action history *is* the supervised target.
- VLN-CE's split of "label ≠ executed" would need a new loss path in `EpisodeSerializer` and the
  trainer.

## 3. Per-decision algorithm (`rules.py`, `episode.py`, `expert.py`)

At each decision t:

1. **Advance waypoints.** Advance past every intermediate reference waypoint the expert follower
   reports as reached. Reaching a waypoint ends an expert takeover. The follower is queried once
   per target per decision: habitat-sim's GreedyFollower records each answer, and duplicate
   entries would disable its thrashing detection.
2. **Final point.** If the expert action is STOP (final point within 0.25 m), record the frame with
   STOP, step STOP and end the episode.
3. **Expert takeover.** The model process is called with `forced_class = expert`. The frame and
   the expert action stream into the policy's memory. Label: `expert`.
4. **Policy control.** The model decodes greedily, as in evaluation.
   - **First token starts STOP:** STOP is intercepted before any token is committed. The expert
     action is committed and executed instead. Label: `stop_override`. The STOP counts as an error
     (as in StreamVLN). If the agent is ≥ 3 m from the goal, a takeover starts immediately.
   - **Otherwise:** the policy's action is executed. Label: `model`. If it differs from the expert
     action, the error count increments.
   - **Takeover trigger:** errors / (oracle actions per reference segment) > 0.8, or errors > 12.
     The takeover starts at the **next** decision, and the error count resets.
5. **Save criterion.** The episode is saved if all of the following hold:
   - final distance < 0.5 m;
   - steps ≤ 183 (the longest proven training episode);
   - PL = start geodesic / max(start geodesic, travelled) is < 0.93 after a takeover, or < 0.85
     otherwise;
   - the episode contains at least one deviation (see deviation d below).

**Deliberate deviations from StreamVLN** (all tested in `tests/dagger/`):

a. **Deferred error takeover.** StreamVLN replaces the action at the decision that crosses the
   error threshold. A streaming text policy has already committed those response tokens, and only
   STOP is identifiable from the first token. So the takeover starts one decision later, and the
   crossing action is executed and labelled `model`. A deferred takeover counts (and relaxes the
   save threshold to PL < 0.93) only after it has forced at least one expert decision. If the
   waypoint is reached first, the takeover is cancelled (`cancelled_takeovers`).
b. **Advance before comparing.** Waypoints are advanced before the policy action is compared with
   the expert's. StreamVLN compares first, which scores a spurious error every time the agent
   enters a waypoint radius, and lets the expert finish that decision. Here control returns to the
   policy at that decision.
c. **Follower failures drop the episode.** A follower failure raises (`stop_on_error=False`) and the
   episode is journaled as an `expert_error` failure. StreamVLN reads it as STOP, which skips
   waypoints silently. Watch the rate in the smoke run.
d. **Clean episodes are dropped.** An episode where the policy agreed with the expert at every
   decision is not saved (`clean`, `--save-clean` restores StreamVLN). A winding reference path can
   put such an oracle-like episode below PL 0.85.

## 4. Model side (`session.py`, `server.py`)

**`DaggerSession.step`:**
- It is `StreamSession.observe` with forced and intercepted outcomes added (`policy_step.py`).
- Forced tokens are appended exactly as greedy decoding appends them: one `response_token` block
  per action token, then `<|im_end|>` + separator as `response_close`.
- The streaming state (GDN/conv, full-attention KV, S8N bank, scorers) after a forced action
  therefore equals the state of a policy that generated it.
- Interception requires STOP's first token to be unique among the four actions. `ResponseLayout`
  enforces this at load time.

**`server.py`:** the evaluator's model-process protocol, with operations `reset` and `step`.

**`collect_r2r.py`:** the Habitat-side driver.
- Settings are the evaluator's camera/action/cap/success settings, on the train split.
- Each shard has its own journal, contract and runtime records, and resumes from its journal.
- Frames are saved as lossless PNG of the exact RGB the policy received, named
  `step_{t:04d}_{ACTION}.png`, using the `observation_before_action` alignment.
- **Manifest rows:**
  - They follow the `scripts/vln/prepare_r2r.py` schema.
  - Scene, instruction ID and instruction come from the same R2R json the oracle manifest was
    built from.
  - The dataset field is `r2rce_dagger`; every step carries `dagger` provenance.

## 5. Runbook

0. **CPU tests:** `PYTHONPATH=src python -m pytest tests/dagger -q`
1. **GPU gate (model interpreter).** Must print `"passed": true`:
   `python scripts/dagger/check_session.py --checkpoint /data14t/anhdh35/m5_s8n_epoch1/checkpoint/S8N --model-path QWEN --manifest ORACLE.jsonl --out check.json`
   It checks:
   - decode parity with `observe` (actions and bitwise state), both unarmed and armed with
     `intercept_with` as during collection;
   - forced parity (bitwise state);
   - interception commits exactly the forced action and the episode continues identically.
2. **Smoke (4 episodes).**
   - Run `MAX_EPISODES=4 GPUS="0" bash scripts/dagger/run_r2r_shards.sh` with a fresh `OUT`.
   - Inspect `journal.shard000.jsonl`: reasons, takeovers, steps, PL.
   - Inspect a few saved frame folders.
3. **Full collection.**
   - Run `GPUS="0 1 2 3 ..." bash scripts/dagger/run_r2r_shards.sh`.
   - Each shard is resumable: rerun the same command after an interruption.
4. **Merge and mix.**
   - DAgger-only manifest: `python -m qwen_vl.dagger.merge --run-dir OUT --out OUT/dagger.jsonl`.
   - Training mixture: `python -m qwen_vl.dagger.merge --run-dir OUT --out OUT/mix_s8n_e2.jsonl --oracle-manifest ORACLE.jsonl --replace-dataset r2rce --dagger-fraction 0.5 --seed 429`.
   - The mixture keeps the oracle manifest's size (30,815 rows, so 3,852 updates) and **every RxR
     row**. DAgger rows (all R2R) replace up to half of the 10,819 R2R oracle rows, so the
     dataset balance of the S8N_E2 control is unchanged.
   - Merge refuses shards whose contracts disagree (checkpoint, rules, source) or that are
     incomplete (`--allow-partial` overrides). The audit JSON reports counts per dataset and
     whether the R2R source hash matches the oracle rows.
5. **Train CONT-DAgger** exactly as S8N_E2, with `--manifest OUT/mix_s8n_e2.jsonl`:
   - `--init-weights-checkpoint /data14t/anhdh35/m5_s8n_epoch1/checkpoint/S8N`
   - `--init-weights-sha256 c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b`
   - `--sampler-epoch-offset 1`
   - the S8N memory/runtime/policy/deepspeed files of the follow-up campaign.

   **Campaign caveats** (`scripts/memory_sweep/s8n_followup.py`):
   - The controller binds one manifest SHA for its parent and arms. Add a DAgger arm with its own
     manifest and identity, or a separate continuation campaign.
   - The continuation gate compares data order with S8N_E2's second-epoch order. That gate must use
     the mixture's own order.
6. **Evaluate** on the standard R2R val_unseen panel and compare paired (McNemar) with S8N_E2 and
   with S8N.

## 6. Known limits and open choices

- **R2R only.** RxR needs the `RxRVLNCE-v1` dataset path of the evaluator (simulator source +
  `habitat_extensions`). The expert and rules are dataset-agnostic.
- **Greedy decoding only** (the deployed policy's own state distribution). Sampling would increase
  coverage but is not implemented.
- **Policy deviations are supervised.** As in StreamVLN/JanusVLN, a policy action that differs from
  the expert but stays within tolerance remains a training target. Provenance is stored per step
  (`dagger.source`, `dagger.expert`). Masking `model` steps whose action ≠ expert would need an
  optional per-step loss mask in `EpisodeSerializer.encode_episode` (`response_target_positions`)
  and `per_action_token_loss`.
- **Turn-only takeovers are dropped.** PL ignores in-place turns, so an episode corrected only from
  turning stays near-oracle and is not saved (StreamVLN behaviour; `tests/dagger/test_collect_driver.py`).
- **Missed waypoints pull the agent back.** The expert never skips a waypoint the agent did not come
  within 1.8 m of; it walks back to it (StreamVLN behaviour).
- **Volume.** Frames are PNG (~0.3–0.6 MB). Expect roughly 100 frames per saved episode.
- **Throughput.** Collection costs about 6× an R2R val_unseen evaluation in decisions
  (10,819 train episodes).
