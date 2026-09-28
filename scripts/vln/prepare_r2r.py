"""Build complete R2R episodes from official metadata and ordered raw frames.

Frame naming alone does not prove capture timing. The report explicitly keeps
collector/replay verification separate from structural alignment checks.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from qwen_vl.contracts import validate_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    args = p.parse_args()
    root = Path(args.data_root)
    source = root / f"datasets/r2r/{args.split}/{args.split}.json.gz"
    with gzip.open(source, "rt") as f:
        records = json.load(f)["episodes"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    episodes, failures, counts = [], [], Counter()
    for record in records:
        eid = str(record["episode_id"])
        scene = record["scene_id"]
        folder = root / f"trajectory_data/R2R/{args.split}/{eid}"
        steps = []
        for image in sorted(folder.glob("step_*.png")):
            match = re.fullmatch(
                r"step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.png", image.name
            )
            if not match:
                raise ValueError(f"Invalid frame name {image}")
            steps.append(
                dict(
                    step_id=int(match[1]),
                    rgb_path=str(image),
                    action_name=match[2],
                    is_valid=True,
                )
            )
        instruction = record["instruction"]
        iid = str(instruction.get("instruction_id", eid))
        episode = dict(
            schema_version=1,
            episode_uid=f"r2rce:{args.split}:{scene}:{eid}:{iid}",
            dataset="r2rce",
            dataset_version=hashlib.sha256(source.read_bytes()).hexdigest()
            if not episodes
            else episodes[0]["dataset_version"],
            split=args.split,
            scene_id=scene,
            episode_id=eid,
            instruction_id=iid,
            instruction=instruction["instruction_text"].strip(),
            observation_action_alignment="observation_before_action",
            alignment_evidence="raw filename labels; collector/replay gate reported separately",
            steps=steps,
        )
        try:
            validate_episode(episode)
        except (ValueError, FileNotFoundError) as exc:
            failures.append({"episode_uid": episode["episode_uid"], "reason": str(exc)})
            continue
        episodes.append(episode)
        counts.update(s["action_name"] for s in steps)
    report = dict(
        official_episodes=len(records),
        valid_episodes=len(episodes),
        action_counts=dict(counts),
        actions=sum(counts.values()),
        rejected=failures,
        capture_before_action_replay_verified=False,
        lengths=sorted(len(e["steps"]) for e in episodes),
    )
    out.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise ValueError(
            f"{len(failures)} invalid episodes; inspect audit, no manifest published"
        )
    with out.open("w") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("lengths", "rejected")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
