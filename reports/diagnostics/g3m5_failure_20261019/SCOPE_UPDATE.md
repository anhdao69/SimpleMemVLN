# User-authorized scope update

The user instructed: “Skip it, remove the D-g”.

The active study consists of D-a through D-f and the original R1–R5 decision table. D-g gradient measurements, its 32-episode subset, gradient tests, figures, and recovery-checkpoint requirements are removed. D-e no longer depends on D-g completion. No objective-interference conclusion will be reported.

TASK_SPEC.md remains the unchanged original specification for provenance; this later user instruction overrides its D-g requirements. All remaining parity gates, diagnostic definitions, tolerances, no-training constraints, and deliverables remain in force. Trained-head readouts in D-a/D-b remain optional under the original specification; their absence does not block the study. D-f may use dummy heads only after its specified invariance gate passes.

## Dataset confirmation after rendering failure

After receiving the failed rendering-gate report, the user confirmed the R2R training data as XinyuYan/JanusVLN_Trajectory_Data and the RxR data as the supplied 15-degree version, and instructed: “Just use them for me”. This is interpreted as authorization to proceed with these specified datasets despite the recorded rendering discrepancy. The original failed result remains intact and must be disclosed as a limitation; it is not relabeled passed and its tolerance is not changed. The model parity gates and R1–R5 definitions remain unchanged. Prefer existing archive val_unseen images when available; otherwise retain the prescribed evaluation renderer and explicitly report the unmatched training-image condition. No substitution of training episodes for the frozen val_unseen panel is authorized.
