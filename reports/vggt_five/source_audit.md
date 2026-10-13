# Live-source audit for the five VGGT runs — 2026-10-12

## Scope

Read-only GitHub connector inspection. A direct `git ls-remote` in the working container failed (`Could not resolve host: github.com`); no local repository clone is claimed. No H100/Slurm/Habitat execution or private checkpoint/journal inspection was performed. Remote reports are source-authored evidence, not measurements independently reproduced in this session.

The user says V0/R0 are already done. We reuse those artifacts. Public labels named V0 do not establish which local baseline the user means, and public reports may lag local completion.

## Remote branches observed

| Branch | SHA | Use here |
|---|---|---|
| `main` | `53ac625611b5778df8a4eaeecbdeda028bfa0309` | historical explicit-history code, not the chosen base |
| `streaming` | `36a2c9e58ad1f5e75d672f4ed82c4010de4903b8` | native streaming reference |
| `streaming_logits` | `1058d090f58708a9f20d9ad40c7a561ff630a186` | candidate-policy reference; do not switch this campaign to candidate mode |
| `streaming_logits_dual` | `499f7d81348d0b7eb9a558dcbfae7991d7dba011` | older parallel lane lineage |
| `streaming_text_dual` | `f6b8108f6ef8c7b4e3df035688d3210dbd3678a1` | stable native text/lane implementation and reports |
| `c2_episodic` | `a06c7896dd4ca8df79d20a49ea537f8b2019631b` | episodic implementation/resource infrastructure, not a method to combine now |
| `c2_h2_speed` | `2430fd235b0752e6119d1af809cb14182dc0d404` | **selected starting branch**, legacy Window8 path only |
| `RL_baseline` | `96eedf1a45fb3396bc94351d40347faafd9599b0` | out of scope |

No remote `c2_vggt`/`c2_vggt_five` was returned by this branch listing. Recheck before execution; local/unpushed files may exist. Never overwrite them.

## Why c2_h2_speed, with episodic disabled

The selected branch retains the legacy native/lane forward and adds RAM-management fixes documented and partly checked in source. Its `parse_episodic_spec()` returns None for `memory.mode=window8`; the wrapper then takes the legacy native language-model path. Therefore the extra episodic implementation need not participate in this experiment. It is not enough to set episodic B=0 while retaining `episodic_window_v1`.

Important live-source facts:

1. `models/nav_model.py`: vision/merger frozen; features microbatched in no_grad; whole-episode hidden return; sparse response-token loss. There is no geometry auxiliary-output contract yet. Optional lane parser and episodic parser are separate.
2. `models/parallel_step_lane.py`: 4 value heads by default, FP32 state, step-end write mask, original output projection zero initialized, kernel read before head norm/output available only internally. It does not yet expose read-only auxiliary queries.
3. `models/activation_offload.py`: newer branch has `pinned_arena_v1` plus original pinned and pageable alternatives. Arena uses mmap, MADV_DONTFORK and explicit lifetime accounting. Its capacity estimator assumes one `[T,hidden]` checkpoint input per decoder layer; new saved inputs need their own budget review.
4. `train/trainer.py`: global action denominator, native/lane learning-rate grouping, optional same-hyperparameter group chunking. New auxiliary denominators and adapter/head groups must be implemented.
5. `research/episodic_contract.py`: legacy mode rejects episodic keys, and episodic mode currently rejects lane combination. This campaign must not bypass these guards.
6. `streaming_text_dual` report documents four installed layers [16,20,24,28], not a confirmed local single-lane R0.

## Latest resource report: evidence and caveats

`reports/c2_h2_speed_20261011/implementation_and_runs.md` reports H2 tests with ZeRO-2 CPU optimizer offload, bounded pinned activation arena, 8 threads/rank, 4 loader workers/rank, attention batch8 and vision microbatch4. It reports 12 ordinary H2 resource gates passing, maximum reserved GPU memory 72.9414 GiB, and V0 save/resume host peaks approximately 274/310 GiB. It also explicitly records incomplete M2 reload and omitted additional tests after the user requested quicker launches.

Those measurements contain NO VGGT extraction, auxiliary projectors, GDN probe insertion or GeoAddress adapters. New branches require new tests. The supervisor/arena improvements are a reason to use this code base, not a claim the five new runs already fit.

## Baseline status must not be inferred from branch names

The original reports identify native Window8 text E2 (52.20 SR/48.19 SPL) and four-lane text E1 (46.55 SR/41.73 SPL). They are independent checkpoints with historical runtime differences. The new episodic reports use another V0 label for a two-epoch schedule stopped at E1. None of these descriptions resolves the user's exact local V0/R0 automatically. Bind the local artifacts to the baseline registry, then compare like progress; no baseline retraining.

## Pinned source locations read

- https://github.com/anhdao69/SimpleMemVLN/tree/2430fd235b0752e6119d1af809cb14182dc0d404
- `src/qwen_vl/models/nav_model.py` blob `679c091feca062b7a8f1e4c3417ee4cfe8316e35`
- `src/qwen_vl/models/activation_offload.py` blob `40535560d9389696691db4f7b30c296eb46123b2`
- `src/qwen_vl/train/trainer.py` blob `30f39cf4b01965473b035b4dff25a2000a9234b4`
- `src/qwen_vl/research/episodic_contract.py` blob `cf3f005a88dcfe657c7351f1a75441995f76ca4e`
- `reports/c2_h2_speed_20261011/implementation_and_runs.md` blob `33427292478d0d27943aa373e5a570d1b252d86a`
- At `streaming_text_dual@f6b8108...`, `models/parallel_step_lane.py` blob `89caa0dac7f364b4bcc320d63dc7648e8f4e9ca4`
- Transformers v5.11.0 `src/transformers/models/qwen3_5/modeling_qwen3_5.py` blob `64705d768243202f7d8f4817d5b6e23542ebe516`; decoder returns a Tensor and runs mixer/residual/MLP/residual.

Re-resolve the corresponding blob in the execution checkout before patching; hashes above record what was read, not a promise future main branches remain unchanged.

## Public primary references

- Gated Delta Networks: https://arxiv.org/abs/2412.06464 . Motivation for decay + delta update, not evidence for the proposed auxiliary objective.
- VGGT: https://arxiv.org/abs/2503.11651 . Teacher architecture, not a guarantee that independently encoded descriptor similarity is physical correspondence.
- VGGT code ref inspected: https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce . Resolve actual weight hash at extraction.
- HF decoder reference: https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py . Preserve native semantics; no global patching of unrelated models.

This artifact is a design audit, not a full audit of every file or all branches' execution.
