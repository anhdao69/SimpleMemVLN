# 08 — Phase-2 evidence gate (frozen before any Phase-1 result)

Status: **closed** (gate not evaluated). 2026-10-12: the user explicitly chose to train M5/M5c
now, before E1 evaluation; jobs 899883–899886 carry that override as `phase2_reason` (see 09).
This does not open the gate. Outcome A–D must still be recorded from E1 results, and no
claim may cite it as satisfied.

Original text: No Phase-2 run may be submitted until Phase-1 E1 evaluations of all
four arms (V0, M1, M2, M2c) exist on the same episode IDs, and one outcome below is
recorded here with the evaluator journals. The thresholds are planning choices copied
from plan §9.5, not significance claims, and are not redefined after seeing results.

Panels: full R2R val_unseen (1,839 IDs, `r2r_val_unseen_ids.json`) and the fixed
1,000-episode English RxR15 val_unseen panel (`rxr15_val_unseen_panel1000.json`).
Statistics: paired per-episode outcomes; scene-cluster bootstrap intervals; predeclared
subgroups only (reference route length terciles, dataset).

Note on checkpoint choice: the user requested epoch-1 training only. The plan's primary
comparison is at E2; with E1 only, the gate is evaluated at **E1 (mid-schedule)** for every
arm (common, predeclared rule). Any E2 continuation must be applied to all four arms.

* **A — selector-positive.** M2 ≥ +1.0 SR pt over V0 on R2R, ≥ +0.5 pt above both M1 and
  M2c, and SPL not > 0.5 pt below V0; **or** ≥ +2.0 SR pts on the RxR panel, ≥ +1.0 above
  both controls there, with R2R SR and SPL each no worse than V0 by > 0.5. Plus no
  protocol/failure-accounting/numerical violation and bounded-cache/resource evidence.
  → Phase 2 default (M5, M5c, V0-s430, winner-s430; M3 when relevant).
* **B — generic episodic signal.** M2 and/or M2c improve over V0 under the same quality
  guard without a robust selector advantage. → M5/M5c may run; no claim that committed
  write is the mechanism.
* **C — inconclusive.** Small differences, broad intervals, or gains confined to a tiny
  subgroup. → one diagnostic or seed follow-up, not all variants.
* **D — negative/invalid.** No useful signal, or correctness/resource failure. → stop
  expansion and diagnose.

Pre-registered caveat to evaluate with any outcome: M2's committed scores differ between
training (full-sequence BF16) and serving (incremental BF16) enough that per-step selected
sets often differ (03). A selector advantage must therefore survive this train/serve
skew; the M2c control has none.

Bounded extra experiments (≤2, plan §10.4) need a written hypothesis, one changed factor,
matched control, GPU-hour estimate and tests before submission. Candidates noted so far:
(i) rank-bucket/quantized committed score (stabilization of the skew), (ii) pinned
first-frame control if early-frame concentration dominates the bank.
