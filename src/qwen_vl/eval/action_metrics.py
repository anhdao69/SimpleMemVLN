"""Supervised action metrics; rows=truth, columns=prediction, canonical order."""

from qwen_vl.contracts import ACTIONS


def classification_metrics(matrix):
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Expected a 4x4 confusion matrix")
    truth = [sum(row) for row in matrix]
    predicted = [sum(matrix[r][c] for r in range(4)) for c in range(4)]
    total = sum(truth)
    per_class = {
        name: dict(
            precision=matrix[c][c] / predicted[c] if predicted[c] else 0.0,
            recall=matrix[c][c] / truth[c] if truth[c] else 0.0,
            support=truth[c],
            predicted=predicted[c],
        )
        for c, name in enumerate(ACTIONS)
    }
    return dict(
        confusion=matrix,
        per_class=per_class,
        overall_accuracy=sum(matrix[c][c] for c in range(4)) / total if total else 0.0,
        macro_accuracy=sum(v["recall"] for v in per_class.values()) / 4,
        macro_accuracy_definition="mean recall over four classes; absent classes count as zero",
        ground_truth_action_counts=dict(zip(ACTIONS, truth)),
        predicted_action_counts=dict(zip(ACTIONS, predicted)),
        ground_truth_action_distribution={
            a: n / total if total else 0.0 for a, n in zip(ACTIONS, truth)
        },
        predicted_action_distribution={
            a: n / total if total else 0.0 for a, n in zip(ACTIONS, predicted)
        },
    )
