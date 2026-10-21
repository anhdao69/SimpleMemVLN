# Original G3, constant geometric loss, seed430

User request: replace the unstarted G1-future queue stage with original G3 using exactly `L_nav + 0.1 * L_geo` and a different seed, then run it automatically after G3+A1 finishes. The user authorizes pausing G3+A1 for the queue change.

- Start at latest original G3 `034f2eb` (training code identical to qualified2259383), separate branch/worktree. No M5, A1, future teacher or PCA.
- Keep G3 native GDN24 delayed reads, original online frozen Omega regional targets, full BPTT, Window8 text/action history, joint R2R/RxR15deg, baseQwen3.5-4B initialization, one epoch3852updates/warmup116/cosine floor0.1, globalbatch8/fourH100.
- Add G3 ramp_updates=0 as constant weight, including update0. Default771 ramp and G1 contracts remain unchanged. Existing global action/pair normalization remains unchanged.
- Set training/data-order seed, teacher-pair sample seed, and newly configurable readout head seed to430. Preserve429 head default for old configs; no teacher identity/architecture changes.
- CPU tests first: constant weight at0/1/770/771/resumed/final steps, original ramp, invalid config, actual Trainer loss/gradient normalization, deterministic isolated head RNG, exact recipe diff, queue restart/failure/completion gates.
- Prepare immutable source and a server-side qualification driver before pausing. Request native checkpoint-and-stop, wait for the old queue lock to release, archive the handled flag, and run H100 qualification. Failed qualification resumes G3+A1 alone; it cannot launch unqualified G3 or the replaced G1-future.
- GPU full suite and two-update mixed-extreme training include the longest627-frame RxR episode, fixed0.1 exposure weights, finite gradients/losses, complete four-rank checkpoint2, and actual readout bias updates.
- On success, launch a new persistent queue with the exact old G3+A1 source/state first and fresh G3 second. Require complete epoch3852/checkpoint/export before transition. Preserve current allocation4925 on worker-2 and its interactive shell.
- Verify G3+A1 resumes from the newly saved checkpoint; document the replacement queue and push the branch.

Completed verification:381GPUtests pass with no skips; two full mixed-extreme updates pass including627-frame RxR, all16exposures have constant0.1, checkpoint2 validates22files, and readout biases update. G3+A1 safely saved2649 and resumed through2654 with preserved one-epoch schedule and40nonrepeated exposures. Active sequence in4925.139 isG3+A1→G3fixed0.1seed430; G1-future launcher disabled and retained as superseded evidence. Full G3 remains queued, not yet trained/evaluated.
