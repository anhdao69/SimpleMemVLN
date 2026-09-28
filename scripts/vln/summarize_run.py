"""Summarize consumed actions, repetitions, global loss and measured timing."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    args = p.parse_args()
    root = Path(args.run)
    updates = defaultdict(list)
    profiles = defaultdict(list)
    for path in root.glob("exposures_rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            updates[row["update"] + 1].append(row)
    for path in root.glob("profile_rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            profiles[row["update"]].append(row)
    rows = []
    for update, profile in sorted(profiles.items()):
        examples = updates[update]
        counts = Counter(e["episode_uid"] for e in examples)
        actions = sum(e["actions"] for e in examples)
        seconds = max(p["seconds"] for p in profile)
        matrix = np.asarray([e["confusion"] for e in examples if "confusion" in e]).sum(
            0
        )
        row = dict(
            update=update,
            episode_exposures=len(examples),
            unique_episodes=len(counts),
            repeated_exposures=sum(v - 1 for v in counts.values()),
            actions=actions,
            global_action_mean_loss=sum(e["loss_sum"] for e in examples) / actions,
            seconds=seconds,
            actions_per_second=actions / seconds,
            tokens_per_second=sum(e["tokens"] for e in examples) / seconds,
            peak_allocated_gib=max(p["peak_allocated_gib"] for p in profile),
            peak_reserved_gib=max(p["peak_reserved_gib"] for p in profile),
        )
        if matrix.shape == (4, 4):
            row.update(
                confusion=matrix.tolist(),
                accuracy=float(matrix.trace() / matrix.sum()),
                per_class_recall=np.divide(
                    matrix.diagonal(),
                    matrix.sum(1),
                    out=np.zeros(4),
                    where=matrix.sum(1) != 0,
                ).tolist(),
                stop_precision=float(matrix[3, 3] / max(1, matrix[:, 3].sum())),
                stop_recall=float(matrix[3, 3] / max(1, matrix[3].sum())),
            )
        rows.append(row)
    all_examples = [e for group in updates.values() for e in group]
    summary = dict(
        completed_updates=len(rows),
        measured_update_seconds=sum(r["seconds"] for r in rows),
        total_episode_exposures=len(all_examples),
        total_unique_episodes=len({e["episode_uid"] for e in all_examples}),
        total_action_exposures=sum(e["actions"] for e in all_examples),
        updates=rows,
    )
    if (root / "run_timing.json").exists():
        summary["timing"] = json.loads((root / "run_timing.json").read_text())
    microtimes = defaultdict(list)
    for path in root.glob("microtimes_rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            microtimes[row["episode_uid"]].append(row)
    summary["microtiming_by_episode"] = [
        {
            "episode_uid": uid,
            "exposures": len(records),
            **{
                f"{field}_p{int(q * 100)}": float(
                    np.quantile([r[field] for r in records], q)
                )
                for field in ("forward_seconds", "backward_and_step_overhead_seconds")
                for q in (0.5, 0.95)
            },
        }
        for uid, records in microtimes.items()
    ]
    (root / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
