"""Teacher-forced action metrics, distinct from closed-loop Habitat navigation."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_vl.eval.action_metrics import classification_metrics


def summarize_outputs(outputs):
    confusion = torch.zeros(4, 4, dtype=torch.long)
    loss_sum = 0.0
    for result in outputs:
        targets = result["targets"].detach().cpu().long()
        predictions = result["predictions"].detach().cpu().long()
        confusion += torch.bincount(4 * targets + predictions, minlength=16).reshape(
            4, 4
        )
        loss_sum += float(result["loss_sum"])
    count = int(confusion.sum())
    if not count:
        raise ValueError("No supervised actions selected")
    return dict(
        classification_metrics(confusion.tolist()),
        actions=count,
        unweighted_ce=loss_sum / count,
    )


def main():
    from qwen_vl.data.episode_dataset import EpisodeDataset
    from qwen_vl.train.vln_runtime import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    if serializer.mode not in ("classification", "candidate_logits"):
        raise ValueError("Four-way supervised metrics require a four-way readout")
    model.eval()
    dataset = EpisodeDataset(args.manifest, serializer, args.episode_limit)
    device = next(model.parameters()).device

    def outputs():
        with torch.inference_mode():
            for sample in dataset:
                data = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in sample.items()
                }
                result = model(**data)
                targets = data["action_class_ids"]
                yield dict(
                    predictions=result["predictions"],
                    targets=targets,
                    loss_sum=F.cross_entropy(
                        result["logits"].float(), targets, reduction="sum"
                    ),
                )

    report = summarize_outputs(outputs())
    report.update(
        test_kind="teacher_forced_not_closed_loop",
        episodes=len(dataset),
        episode_uids=[ep["episode_uid"] for ep in dataset.episodes],
        checkpoint=str(Path(args.checkpoint).resolve()),
        navigation_contract=serializer.metadata(),
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
