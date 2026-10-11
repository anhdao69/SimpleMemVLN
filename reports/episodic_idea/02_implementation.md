# 02 — Implementation (Tasks 1–9)

Branch `c2_episodic` (worktree `SimpleMemVLN-c2_episodic`), parent `f6b8108`. Every file
below is **implemented**; test status is in 03/04 and the resource status in 05/06.
Lanes remain loadable for legacy checkpoints but are disabled (the episodic contract
rejects `model.step_lane.enabled`).

## New modules

| File | Responsibility |
|---|---|
| `src/qwen_vl/research/episodic_contract.py` | `EpisodicSpec` (frozen), memory mode `episodic_window_v1`, validation (B0⇔score none, no cap, no `visual_group` in v1, derived integer KV guard `max_prefix + R·max_group + B`, text-only, no lanes), scoring pairs from `layer_types` (GDN l−1 → attention l: (2,3)…(30,31)), versioned backend IDs. |
| `src/qwen_vl/data/episodic_metadata.py` | `EpisodicLayout` (prefix, group spans, observation ends, 300 contiguous visual IDs/group) derived and validated from actual serialized IDs; canonical score segments. Metadata only. |
| `src/qwen_vl/stream/episodic.py` | SHA-256 random priorities, exact running top-B (priority desc, ID asc), immutable read-only `LayerSelectionPlan` with digest, bounded `plan_summary` diagnostics (occupancy, sources, early-frame fraction, turnover, cutoff margins). |
| `src/qwen_vl/models/committed_write.py` | Native GDN preparation from a layer's input (no grad), exact FP32 WY block scan, `CommittedWriteScorer` (offline whole-sequence; serving per-append with identical segmentation). |
| `src/qwen_vl/models/episodic_forward.py` | Project language forward mirroring HF `Qwen3_5TextModel.forward` (source fingerprint pinned: `e56d7184…16db`), scorer side pass before each scoring GDN layer, explicit per-layer plan kwargs, serving score routing. |
| `src/qwen_vl/stream/episodic_cache.py` | `EpisodicSessionState`: per-layer resident original-ID map, running top-B, recent groups with fixed source scores, atomic promotion/repacking `[prefix | E sorted | recent]` by `index_select` of resident rows, invariants (prefix, uniqueness, disjointness, equal counts, row/ID agreement). No K/V backups. |
| `src/qwen_vl/train/host_memory.py` | Read-only cgroup v1/v2 job memory, working-set heuristic, OOM/limit counters, guard formula. |
| `src/qwen_vl/train/campaign_lease.py` | Atomic `O_EXCL` lease, heartbeats, scheduler-evidenced stale archival (rename, never delete), exact-ID sibling cancellation with live identity check. |
| `scripts/vln/check_episodic.py` | Real-model gates A–E (feature-disabled parity, helper vs FLA state, forced-history replay with ID/score comparison, retry/reset/bounded KV, ≥500-step dry stream). |
| `scripts/vln/profile_episodic.py` | Fixture selection from exact encoded lengths; supervised Trainer stress gates with per-rank/update verification; stressed checkpoint save → fresh reload → continue. |
| `scripts/vln/run_episodic_gate.py` | Owned-command supervisor (1 s host samples, GPU samples, OOM delta, process-group-only guard). |
| `scripts/vln/submit_episodic.py` | Campaign manifest, read-only source snapshot (`git archive`), attempt dirs, `sbatch --parsable`, append-only ledger/allowlist, status. |
| `scripts/vln/episodic_job.py` | In-job lifecycle (preflight → standby check → qualification → lease → supervised training + agent → upload). |
| `scripts/vln/upload_episodic.py` | Hardlink staging into `<run>/<attempt>/{export-e1,checkpoint-N,campaign}` and resumable private Hub upload with remote size verification. |
| `train/slurm/episodic_candidate.sbatch` | Thin environment wrapper; `exec`s the Python lifecycle so Slurm's `USR1` reaches it. |
| `configs/episodic/*.yaml` | `science_common` + arms `v0/m1/m2/m2c` (Phase 1) and `m5/m5c` (Phase 2, gated) + runtime profiles `profile_h4/h2`. |

## Modified modules (all backward compatible for legacy modes)

* `contracts.py` — accepts `episodic_window_v1`; rejects stray episodic keys in legacy
  modes; validates staging knobs (`dataloader_*`, `stage_pixels_bf16`,
  `activation_offload_backend`).
* `episode_serializer.py` — adds `episodic_layout` (metadata) and the spec to
  `navigation.json` metadata; tokens/images/targets unchanged.
* `episode_dataset.py` — optional BF16 pixel staging.
* `nav_model.py` — routes episodic models through the project forward; strict errors if
  episodic metadata reaches a non-episodic model; offload backend from config.
* `window_attention.py` — new `_EpisodicWindowAttention` (index-gathered prefix + plan
  rows + R recent groups; bounded packed FA2 varlen batches; recompute-per-batch VJP
  with FP32 deterministic scatter); legacy Window8 paths unchanged.
* `session.py` — episodic promotion before each observation; score segment routing;
  group open/close; diagnostics only when `action_diagnostics`.
* `activation_offload.py` — `pinned_arena_v1` / `pageable_v1` backends.
* `train_episode.py` — `--runtime_profile`, loader/pinning knobs, arena installation
  sized from the dataset maximum, per-update host/cgroup/arena/CUDA-OOM records,
  `malloc_trim`, `--stop-after-epochs`, `--first-recovery-save-step`, Slurm checkpoint
  request callback, bounded per-episode episodic diagnostics in `exposures_rank*.jsonl`.
* `recovery.py` — optional early recovery save (update 50) besides the 250 cadence.
* `vln_runtime.py` — runtime overlay merged last.
* `prepare_r2r.py`, `prepare_joint.py` — explicit source/frame paths for this server.

## Training path (offline, one complete episode per rank/microbatch)

1. Dataset → serializer: unchanged tokens; `episodic_layout`; BF16 pixels.
2. Frozen vision encoder + merger (no grad), embeddings, logical+MRoPE positions.
3. Project loop over the 32 native decoder layers. Before GDN layer 2, 6, …, 30 (M2
   only): no-grad scores for all tokens → visual `[G,300]` → CPU → running top-B plan for
   attention 3, 7, …, 31. B0 arms build no plan; M2c plans come from hashes only.
4. Each attention layer gets `(step_plan, R, plan)`; queries of group t see prefix +
   E_t (sorted by original ID) + groups t−R+1..t (causal within t). Positions are the
   original ones (K already rotary-transformed).
5. Loss/normalization unchanged (per-action mean CE incl. EOS, summed; global action
   count; no GAS double division). No new parameters.

## Serving path

`reset` creates `EpisodicSessionState`; `observe(t)`: promote group t−R (admit visual
rows by fixed priority, rebuild every attention cache from resident rows), open group,
append observation (scores for its 300 visual rows at each scoring layer), greedy text
tokens (buffered for the helper), EOS+separator (closes the helper's response segment),
close group. Retries return the saved result; any exception invalidates the session;
reset recreates everything.
