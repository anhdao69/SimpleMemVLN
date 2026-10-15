# Independent R1 review

Fresh-context read-only reviewer `r1_review` examined `4f772d3..89b194c7` and then fixes through `8ab23d6`. It ran21 targeted CPU tests on the first review and11 on the follow-up. It did not inspect or rerun remote GPU qualification.

Initial verdict: core causal/native/lane implementation sound; production conditional on fixes and pending runtime gates.

Findings resolved:

1. P1: longest episode profiles omitted full checkpoint saving. Both RxR and R2R longest profiles now save checkpoints. A subsequent five-line change in5635357 additionally checks the final saved inventory and four-rank/update count; the actual longest tests passed this check.
2. P2: the launch checked baseline weights/metadata but not the external manifest. Qualification and immediate launch now compare the actual manifest path and SHA-256 against the registered RxR15 corpus, with a mutation/path regression test.

The reviewer also requested narrow baseline claims. An eighteenth gate now audits trained R0 same-history raw scores, streaming response replay, reset/retry and Window8 eviction read-only. This verifies consistency in the current pinned runtime, not equivalence to an independently executed historical environment.

Follow-up verdict at8ab23d6: code approved, no remaining critical or important findings; production still conditional on runtime qualification.

Strengths noted: native and lane kernels/residual execute once; probes have zero write/decay and bypass output projection; query/target detach and causal writer gradients; original other-lane wrapper paths; bound checkpoint closures; G3-identical online targets; strict export retaining all four lanes; worker-2 restriction and complete-checkpoint resume.

Declined to judge: remote GPU qualification, navigation gains, final speed choice, and historical-runtime score equivalence. These are not implied by code review. The user-authorized R0/R1 LR schedule mismatch is documented as an intentional comparison limitation.


## Audit-only continuation review (2026-10-15)

Fresh reviewer approved the canonical-reference decomposition after actual H100 diagnostics. Source/export offline and forced streaming are exact; the legacy direct R2R BF16 discrepancy remains explicit. Review requested source-preserving separate attempt roots, full command equivalence except output relocation, fixture hash checks, and recovery dependency binding for carried training evidence. All are implemented. Eleven targeted CPU tests passed independently. No remaining critical/important findings; final changed audits and server full suites still required before launch. Runtime audit commit: dca86b64742f779d6f4b5ebdb50e21dd1a5e7d3b.
