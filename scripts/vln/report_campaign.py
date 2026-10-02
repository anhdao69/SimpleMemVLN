"""Action-weighted loss and slowest-rank timing at update 50 and each epoch."""
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics


def build_report(run, start, end, *, world_size):
    run = Path(run)
    if not isinstance(world_size, int) or world_size < 1 or start < 1 or end < start:
        raise ValueError("Invalid report world size or update range")
    expected_updates = set(range(start, end + 1))
    profiles, examples = defaultdict(list), []
    for prefix in ("profile", "exposures"):
        expected = {f"{prefix}_rank{rank}.jsonl" for rank in range(world_size)}
        if {p.name for p in run.glob(f"{prefix}_rank*.jsonl")} != expected:
            raise ValueError(f"Incomplete distributed {prefix} ranks; expected {world_size}")
    for rank in range(world_size):
        path = run / f"profile_rank{rank}.jsonl"
        seen = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if start <= row["update"] <= end:
                if row["update"] in seen:
                    raise ValueError(f"Duplicate update measurement in rank {rank}")
                seen.add(row["update"])
                profiles[row["update"]].append(row)
        if seen != expected_updates:
            raise ValueError(f"Incomplete distributed update measurements in rank {rank}")
        path = run / f"exposures_rank{rank}.jsonl"
        seen_exposures = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if start <= row["update"] + 1 <= end:
                seen_exposures.add(row["update"] + 1)
                examples.append(row)
        if seen_exposures != expected_updates:
            raise ValueError(f"Incomplete distributed exposures in rank {rank}")
    flat = [r for rows in profiles.values() for r in rows]
    begin, finish = min(r["unix_start"] for r in flat), max(r["unix_end"] for r in flat)
    util = []
    if (run / "gpu_utilization.csv").exists():
        with (run / "gpu_utilization.csv").open() as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                try:
                    timestamp = datetime.strptime(
                        row["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
                    ).timestamp()
                    value = float(row["utilization.gpu [%]"].split()[0])
                    if begin <= timestamp <= finish:
                        util.append(value)
                except (KeyError, ValueError):
                    continue
    times = [max(r["seconds"] for r in rows) for _, rows in sorted(profiles.items())]
    losses = {
        step: sum(r["loss_sum"] for r in examples if r["update"] + 1 == step)
        / sum(r["actions"] for r in examples if r["update"] + 1 == step)
        for step in sorted(profiles)
    }
    return dict(
        world_size=world_size,
        start_update=start,
        end_update=end,
        completed_updates=len(profiles),
        loss_action_weighted=sum(r["loss_sum"] for r in examples)
        / sum(r["actions"] for r in examples),
        first_update_loss=losses[start],
        last_update_loss=losses[end],
        mean_update_compute_seconds=statistics.mean(times),
        median_update_compute_seconds=statistics.median(times),
        mean_wall_seconds_per_update=(finish - begin) / len(profiles),
        peak_allocated_gib=max(r["peak_allocated_gib"] for r in flat),
        peak_reserved_gib=max(r["peak_reserved_gib"] for r in flat),
        gpu_utilization_mean_pct=statistics.mean(util) if util else None,
        gpu_utilization_samples=len(util),
        updates=[
            {
                "step": step,
                "loss": loss,
                "seconds": max(r["seconds"] for r in profiles[step]),
            }
            for step, loss in losses.items()
        ],
    )


def write_report(run, start, end, name, *, world_size):
    report = build_report(run, start, end, world_size=world_size)
    directory = Path(run) / "reports"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "CAMPAIGN_REPORT",
        name,
        json.dumps({k: v for k, v in report.items() if k != "updates"}),
        flush=True,
    )
