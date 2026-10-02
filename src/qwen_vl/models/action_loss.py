"""Action-normalized CE; balancing never changes the global action denominator."""

import json
import math
from pathlib import Path
import torch
from qwen_vl.contracts import ACTIONS


def class_weights(counts, mode="none", beta=0.9999):
    n = torch.as_tensor(counts, dtype=torch.float64)
    if (
        n.shape != (len(ACTIONS),)
        or not torch.isfinite(n).all()
        or (n < 0).any()
        or n.sum() <= 0
    ):
        raise ValueError(
            "Four finite nonnegative class counts with positive total required"
        )
    if mode == "none":
        return torch.ones_like(n)
    if (n == 0).any():
        raise ValueError(
            "Balanced training requires every action class in the selected manifest"
        )
    if mode == "sqrt_inverse_frequency":
        raw = n.rsqrt()
    elif mode == "effective_number":
        if not 0 <= beta < 1:
            raise ValueError("effective_number_beta must be in [0,1)")
        raw = (
            torch.ones_like(n)
            if beta == 0
            else (1 - beta) / (-torch.expm1(n * math.log(beta)))
        )
    else:
        raise ValueError("Unknown class weighting")
    # E_{c~training distribution}[w_c] == 1, NOT arithmetic mean(w)==1.
    return raw / ((n / n.sum()) * raw).sum()


def action_losses(logits, targets, weights):
    ce = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    weights = torch.as_tensor(weights, device=logits.device, dtype=torch.float32)
    return ce, ce * weights[targets]


def resolve_class_balance(cfg, manifest, limit=None, selection="first"):
    """Resolve counts before model/serializer construction and checkpoint checks."""
    if cfg["model"]["output_mode"] != "candidate_logits":
        return
    episodes = [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ]
    if selection == "shortest":
        episodes.sort(key=lambda e: (len(e["steps"]), e["episode_uid"]))
    if limit:
        episodes = episodes[:limit]
    counts = [0] * len(ACTIONS)
    for ep in episodes:
        for step in ep["steps"]:
            counts[ACTIONS.index(step["action_name"])] += 1
    t = cfg["training"]
    weights = class_weights(
        counts, t.get("class_weighting", "none"), t.get("effective_number_beta", 0.9999)
    )
    t["action_class_counts"] = counts
    t["action_class_weights"] = weights.tolist()
    t["class_weight_normalization"] = "training_probability_mean_one_v1"
