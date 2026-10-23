"""Frozen-ranker input drift between checkpoints on identical teacher-forced histories (plan 11.7).

Each checkpoint runs the same expert episodes under the model-independent
``C_RANDOM`` memory (selector seed 7001), so banks and therefore histories are
identical across checkpoints and any difference in the GDN features is weight
drift alone.  Passive pre-hooks on all 24 GDN layers run the shared
``GdnFeatureExtractor`` on visual rows; every frozen L-ranker artifact then
scores those rows.  Reported per (arm, attention layer), against the first
checkpoint (the one the rankers were fitted on):

* Spearman of all visual-row scores within an episode;
* the static top-B bank overlap at every decision (B = 1200 over frames older
  than the R = 4 recent window), which is exactly what a static learned
  priority would retain;
* mean/std shift of the standardized ranker inputs (per input feature).

Diagnostic only: no labels, no navigation claim, never alters execution.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

BUDGET, RECENT = 1200, 4


def configure(model, serializer, seed=7001):
    from qwen_vl.research.gdn_sweep_contract import arm_memory_config
    from scripts.vln.check_episodic import switch
    cfg = deepcopy(model.navigation_config)
    cfg['memory'] = arm_memory_config('C_RANDOM', selector_seed=seed)['memory']
    return switch(model, serializer, cfg)


def load(role, args):
    from qwen_vl.train.vln_runtime import load_checkpoint, load_model, resolve_config
    root = Path(__file__).resolve().parents[2]
    if role == 'base':
        cfg = resolve_config(root / 'configs/episodic/science_common.yaml', root / 'configs/vln_empty.yaml',
                             root / 'configs/episodic/arm_m5.yaml', root / 'configs/vggt/g3_joint_one_epoch.yaml',
                             args.runtime_profile)
        return load_model(cfg, args.model_path)
    return load_checkpoint(role, args.model_path)


@torch.no_grad()
def episode_features(model, serializer, episode, device):
    """GDN layer -> FP32 [G*300, Hv, 6] F1_v2 encodings of every visual row (F0 = first 3 channels)."""
    from qwen_vl.models.gdn_features import RAW_FEATURE_SCHEMA, GdnFeatureExtractor, encode_features
    text = model.backbone.model.language_model
    types = list(text.config.layer_types)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in serializer.encode_episode(episode).items()}
    layout = data['episodic_layout']
    rows = layout.visual_matrix().reshape(-1).to(device)
    segments = layout.score_segments()
    out, hooks = {}, []
    for index, kind in enumerate(types):
        if kind != 'linear_attention':
            continue

        def hook(module, args, kwargs, index=index):
            hidden = (args[0] if args else kwargs['hidden_states']).detach()[0]
            fields = GdnFeatureExtractor(module, hidden.device).sequence(hidden, segments)
            raw = {name: value[rows] for name, value in fields.items()}
            out[index] = encode_features(dict(schema=RAW_FEATURE_SCHEMA, fields=raw), 'F1_v2')['values'].cpu()
        hooks.append(text.layers[index].register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.inference_mode():
            model(**data)
    finally:
        for h in hooks:
            h.remove()
    return out, layout.groups


def bank_overlaps(a, b, groups, budget=BUDGET, recent=RECENT):
    """Mean/min |top-B(a) ∩ top-B(b)| / B over decisions whose eligible pool exceeds B."""
    values = []
    for u in range(groups):
        eligible = max(0, u - recent + 1) * 300
        if eligible <= budget:
            continue
        top_a = np.argpartition(-a[:eligible], budget - 1)[:budget]
        top_b = np.argpartition(-b[:eligible], budget - 1)[:budget]
        values.append(np.intersect1d(top_a, top_b).size / budget)
    return values


def spearman(a, b):
    from scripts.memory_sweep.train_ranker import spearman as rank_correlation
    return rank_correlation(a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help="'base' or export directories; the first is the reference (fit checkpoint)")
    p.add_argument('--artifacts', nargs='+', required=True, help='L-ranker artifact JSON files')
    p.add_argument('--selection', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--runtime-profile', required=True)
    p.add_argument('--partition', default='ranker_dev')
    p.add_argument('--episodes', type=int, default=24)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    from qwen_vl.research.gdn_ranker import FrozenRanker, canonical_sha256, flatten_window
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = json.loads(Path(args.selection).read_text())
    chosen = [e for e in selection['episodes'] if e['partition'] == args.partition and e['steps'] > 12]
    chosen = sorted(chosen, key=lambda e: e['cell_rank'])[:args.episodes]
    wanted = {e['episode_uid'] for e in chosen}
    episodes = {}
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['episode_uid'] in wanted:
                    episodes[e['episode_uid']] = e
    rankers = {}
    for path in args.artifacts:
        artifact = json.loads(Path(path).read_text())
        rankers[artifact['ranker_id']] = FrozenRanker.load(json.dumps(artifact, sort_keys=True),
                                                           canonical_sha256(artifact))
    scores = {}            # role -> uid -> arm -> layer -> np.ndarray
    moments = {}           # role -> arm -> layer -> (sum, sumsq, n) of standardized inputs
    for role in args.checkpoints:
        model, serializer = load(role, args)
        model.eval()
        device = next(model.parameters()).device
        serializer = configure(model, serializer)
        scores[role], moments[role] = {}, {}
        for row in chosen:
            features, groups = episode_features(model, serializer, episodes[row['episode_uid']], device)
            per_arm = {}
            for arm, ranker in rankers.items():
                per_arm[arm] = {}
                for layer, record in ranker.layers.items():
                    window = record['feature_layers']
                    channels = 3 if ranker.schema == 'F0_v1' else 6
                    x = flatten_window({g: features[g][..., :channels] for g in window}, window)
                    per_arm[arm][layer] = ranker.score(layer, x).numpy()
                    z = (x.double() - record['mean']) / record['std']
                    acc = moments[role].setdefault(arm, {}).setdefault(layer, [0.0, 0.0, 0])
                    acc[0] = acc[0] + z.sum(0)
                    acc[1] = acc[1] + z.square().sum(0)
                    acc[2] += z.shape[0]
            scores[role][row['episode_uid']] = (groups, per_arm)
            print('DRIFT_EPISODE', role, row['episode_uid'], groups, flush=True)
        del model
        torch.cuda.empty_cache()
    reference = args.checkpoints[0]
    report = dict(version='gdn_ranker_drift_v1', reference=reference, memory='C_RANDOM selector 7001',
                  partition=args.partition, episodes=[e['episode_uid'] for e in chosen],
                  budget=BUDGET, recent=RECENT, comparisons={})
    for role in args.checkpoints:
        per = {}
        for arm, ranker in rankers.items():
            per[arm] = {}
            for layer in ranker.layers:
                rho, banks = [], []
                for uid, (groups, per_arm) in scores[role].items():
                    a = scores[reference][uid][1][arm][layer]
                    b = per_arm[arm][layer]
                    rho.append(spearman(a, b))
                    banks += bank_overlaps(a, b, groups)
                s, sq, n = moments[role][arm][layer]
                mean = s / n
                std = (sq / n - mean.square()).clamp_min(0).sqrt()
                per[arm][str(layer)] = dict(
                    spearman_mean=float(np.mean(rho)), spearman_min=float(np.min(rho)),
                    bank_overlap_mean=float(np.mean(banks)) if banks else None,
                    bank_overlap_min=float(np.min(banks)) if banks else None, decisions=len(banks),
                    input_mean_abs=float(mean.abs().mean()), input_std_mean=float(std.mean()))
        report['comparisons'][role] = per
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    for role, per in report['comparisons'].items():
        for arm, layers in per.items():
            print('DRIFT', role, arm, 'spearman', round(np.mean([v['spearman_mean'] for v in layers.values()]), 4),
                  'bank_overlap', round(np.mean([v['bank_overlap_mean'] for v in layers.values()
                                                 if v['bank_overlap_mean'] is not None]), 4), flush=True)


if __name__ == '__main__':
    main()
