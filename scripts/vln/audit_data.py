"""Token admission and transition diagnostics for the actual complete corpus."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from transformers import AutoProcessor
from qwen_vl.contracts import ACTIONS
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.data.episode_dataset import EpisodeDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-modes", nargs="+", default=["classification", "qwen_text"])
    p.add_argument("--out", default="artifacts/data_audit.json")
    args = p.parse_args()
    results = {}
    for mode in args.output_modes:
        cfg = resolve_config(
            "configs/vln_r2r_v0_base.yaml", f"configs/vln_r2r_v0_{mode}.yaml"
        )
        serializer = EpisodeSerializer(
            AutoProcessor.from_pretrained(args.model_path), cfg
        )
        dataset = EpisodeDataset(args.manifest, serializer)
        lengths = np.array(dataset.encoded_lengths)
        transitions = np.zeros((4, 4), dtype=int)
        action_counts = Counter()
        runs = {a: [] for a in ACTIONS}
        for ep in dataset.episodes:
            ids = [ACTIONS.index(s["action_name"]) for s in ep["steps"]]
            action_counts.update(ep["steps"][i]["action_name"] for i in range(len(ids)))
            for a, b in zip(ids, ids[1:]):
                transitions[a, b] += 1
            previous, count = ids[0], 0
            for action in ids:
                if action != previous:
                    runs[ACTIONS[previous]].append(count)
                    previous, count = action, 0
                count += 1
            runs[ACTIONS[previous]].append(count)
        results[mode] = dict(
            episodes=len(dataset),
            actions=sum(action_counts.values()),
            action_counts=dict(action_counts),
            token_quantiles={
                str(q): float(np.quantile(lengths, q)) for q in [0, 0.5, 0.95, 1]
            },
            exceeds_12800=int((lengths > 12800).sum()),
            exceeds_65536=int((lengths > 65536).sum()),
            previous_action_copy_accuracy=float(
                transitions.trace() / transitions.sum()
            ),
            transition_matrix=transitions.tolist(),
            run_length_mean={a: float(np.mean(v)) for a, v in runs.items()},
            token_length_method="Native processor step measured at required 640x480; exact per-instruction and per-action tokenization; actual lengths rechecked on every loaded episode",
            selected_complete_profiles={
                label: dataset.episodes[
                    int(np.argsort(lengths)[int(q * (len(lengths) - 1))])
                ]["episode_uid"]
                for label, q in [
                    ("short", 0),
                    ("median", 0.5),
                    ("p95", 0.95),
                    ("longest", 1),
                ]
            },
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
