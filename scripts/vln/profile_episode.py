"""Select complete length representatives, then use the production Trainer.

Use --max-optimizer-updates >=2 to include allocated optimizer states.
Profiles are explicitly repeated to fill complete global batches; these
exposures are logged, never described as unique corpus coverage.
"""
import argparse
import json
from pathlib import Path
import sys
from qwen_vl.train.train_episode import train_episode


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--length-buckets", nargs="+", default=["short", "median", "p95", "longest"]
    )
    p.add_argument("--out", default="artifacts/profile")
    p.add_argument("--vln_config", required=True)
    p.add_argument("--output_config", required=True)
    p.add_argument("--memory_config")
    p.add_argument("--model_name_or_path", required=True)
    args, rest = p.parse_known_args()
    from transformers import AutoProcessor
    from qwen_vl.train.vln_runtime import resolve_config
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.data.episode_dataset import EpisodeDataset

    cfg = resolve_config(args.vln_config, args.output_config, args.memory_config)
    serializer = EpisodeSerializer(
        AutoProcessor.from_pretrained(args.model_name_or_path), cfg
    )
    dataset = EpisodeDataset(args.manifest, serializer)
    order = sorted(range(len(dataset)), key=lambda i: dataset.encoded_lengths[i])
    episodes = [dataset.episodes[i] for i in order]
    q = {"short": 0, "median": 0.5, "p95": 0.95, "longest": 1}
    selected = [episodes[int(q[b] * (len(episodes) - 1))] for b in args.length_buckets]
    path = Path(args.out) / "profile_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch.distributed as dist
    import os

    # All ranks select identical records; only rank zero publishes the manifest.
    if int(os.environ.get("RANK", 0)) == 0:
        path.write_text(
            "".join(json.dumps(selected[i % len(selected)]) + "\n" for i in range(64))
        )
        (path.parent / "representatives.json").write_text(
            json.dumps(
                [
                    dict(
                        bucket=b,
                        episode_uid=ep["episode_uid"],
                        actions=len(ep["steps"]),
                        tokens=dataset.encoded_lengths[
                            order[int(q[b] * (len(episodes) - 1))]
                        ],
                    )
                    for b, ep in zip(args.length_buckets, selected)
                ],
                indent=2,
            )
            + "\n"
        )
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        import torch

        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        dist.barrier()
    recipes = [
        "--vln_config",
        args.vln_config,
        "--output_config",
        args.output_config,
        "--model_name_or_path",
        args.model_name_or_path,
    ]
    if args.memory_config:
        recipes += ["--memory_config", args.memory_config]
    sys.argv = [
        sys.argv[0],
        *rest,
        *recipes,
        "--manifest",
        str(path),
        "--output_dir",
        args.out,
        "--debug-repeat-episodes",
        "--max-optimizer-updates",
        "2",
        "--profile-only",
    ]
    train_episode()


if __name__ == "__main__":
    main()
