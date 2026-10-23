"""Holdout ranking report for the fitted rankers (plan 12.3); diagnostic only, never feeds training.

For every scorer on the ranker_holdout scenes: per-(episode, attention layer)
capture@96 and Spearman against the raw near16 utility, macro-averaged over
episodes within a domain, then domains, then layers (as in fitting).  Paired
differences against S0 get a scene-cluster bootstrap (episodes resampled by
scene, 5,000 draws, percentile 95% interval).

Scorers: the frozen L0-L3 artifacts, the fitted controls, the fixed S scores
and random from ``train_ranker``, plus two specificity diagnostics that are not
GDN rankers: a static raster-position prior (mean fit log-utility per visual
position) and newest-source recency.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from qwen_vl.research.gdn_ranker import FrozenRanker, canonical_sha256
from qwen_vl.research.gdn_ranker_dataset import RankerData
from scripts.memory_sweep import train_ranker as tr

BOOTSTRAP, SEED = 5000, 429


def per_pool(data, index, raw, scores, budget=tr.BUDGET_ROWS):
    """episode -> (capture, spearman) for pools larger than the budget."""
    episode = data.rows['episode'][index]
    order = np.argsort(episode, kind='stable')
    episode, raw, scores = episode[order], raw[order], scores[order]
    out = {}
    for part in np.split(np.arange(episode.size), np.flatnonzero(np.diff(episode)) + 1):
        if part.size <= budget:
            continue
        y, s = raw[part], scores[part]
        best = np.sort(y)[::-1][:budget].sum()
        top = np.argsort(-s, kind='stable')[:budget]
        out[int(episode[part[0]])] = (float(y[top].sum() / best) if best > 0 else 1.0, tr.spearman(s, y))
    return out


def macro(pools, domain_of, episodes=None, metric=0):
    """Mean over layers of the equal-domain mean of episode means (optionally on resampled episodes)."""
    layer_values = []
    for per_episode in pools.values():
        domains = {}
        for e in (episodes if episodes is not None else per_episode):
            if e in per_episode:
                domains.setdefault(domain_of[e], []).append(per_episode[e][metric])
        layer_values.append(np.mean([np.mean(v) for v in domains.values()]))
    return float(np.mean(layer_values))


def control_result(path, layer):
    record = torch.load(path, weights_only=False)[str(layer)]
    return record


def scores_for(name, data, layer, index, rankers, controls, prior):
    k = data.layer_column(layer)
    if name in rankers:
        x = tr.learned_features(data, name, layer, index)
        keep = (x.shape[0] // 300) * 300
        head = rankers[name].score(layer, x[:keep]).double().numpy()
        tail = np.empty(0)
        if keep < x.shape[0]:                     # pad the remainder to one frame, score, drop the pad
            pad = torch.cat([x[keep:], x[:300 - (x.shape[0] - keep)]])
            tail = rankers[name].score(layer, pad).double().numpy()[:x.shape[0] - keep]
        return np.concatenate([head, tail])
    if name in controls:
        record = controls[name][str(layer)]
        x = tr.learned_features(data, name, layer, index)
        head = tr.head_for(name, x.shape[1])
        head.load_state_dict(record['state'])
        head.eval()
        with torch.no_grad():
            s = torch.cat([head(c) for c in ((x - record['mean']) / record['std']).split(65536)]).double().numpy()
        if name == 'L0_S0res':
            s = s + tr.s0_baseline(data, index, k, record['kappa_s0']).double().numpy()
        return s
    if name == 'position_prior':
        return prior[layer][data.rows['position'][index]]
    if name == 'newest_source':
        return data.rows['source'][index].astype(np.float64)
    return tr.fixed_scores(data, name, layer, index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--rankers', required=True, help='Directory with fit_report.json, L*_artifact.json, *_control.pt')
    p.add_argument('--selection', required=True)
    p.add_argument('--partition', default='ranker_holdout')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    data = RankerData(args.dataset)
    rankers_dir = Path(args.rankers)
    ledger = json.loads((rankers_dir / 'fit_report.json').read_text())
    rankers = {}
    for arm in ('L0', 'L1', 'L2', 'L3'):
        artifact = json.loads((rankers_dir / f'{arm}_artifact.json').read_text())
        digest = canonical_sha256(artifact)
        assert digest == ledger['arms'][arm]['artifact_sha256'], arm
        rankers[arm] = FrozenRanker.load(json.dumps(artifact, sort_keys=True), digest)
    controls = {name: torch.load(rankers_dir / f'{name}_control.pt', weights_only=False) for name in tr.CONTROLS}
    selection = json.loads(Path(args.selection).read_text())
    uids = data.manifest['episode_uids']
    scene_of_uid = {e['episode_uid']: e['scene_id'] for e in selection['episodes']}
    scene = {i: scene_of_uid[u] for i, u in enumerate(uids)}
    domain_of = {i: int(d) for i, d in zip(data.rows['episode'], data.rows['domain'])}
    prior = {}
    for layer in data.attention_layers:
        fit = np.flatnonzero(data.mask('ranker_fit', layer))
        log_y = np.log(data.rows['near'][fit, data.layer_column(layer)])
        position = data.rows['position'][fit]
        table = np.full(300, log_y.mean())
        sums = np.bincount(position, weights=log_y, minlength=300)
        counts = np.bincount(position, minlength=300)
        table[counts > 0] = sums[counts > 0] / counts[counts > 0]
        prior[layer] = table
    names = ['L0', 'L1', 'L2', 'L3', *tr.CONTROLS, *tr.FIXED_SCORES, 'position_prior', 'newest_source']
    pools = {name: {} for name in names}
    for layer in data.attention_layers:
        index = np.flatnonzero(data.mask(args.partition, layer))
        raw = data.rows['near'][index, data.layer_column(layer)]
        for name in names:
            pools[name][layer] = per_pool(data, index, raw, scores_for(name, data, layer, index, rankers, controls, prior))
        print('LAYER', layer, {n: round(np.mean([v[0] for v in pools[n][layer].values()]), 3) for n in names}, flush=True)
    episodes = sorted(pools['S0'][data.attention_layers[0]])
    scenes = sorted({scene[e] for e in episodes})
    by_scene = {s: [e for e in episodes if scene[e] == s] for s in scenes}
    rng = np.random.default_rng(SEED)
    draws = [[e for s in rng.choice(len(scenes), len(scenes)) for e in by_scene[scenes[s]]] for _ in range(BOOTSTRAP)]
    report = dict(version='gdn_ranker_holdout_report_v1', partition=args.partition, budget_rows=tr.BUDGET_ROWS,
                  episodes=len(episodes), scenes=len(scenes), bootstrap=BOOTSTRAP, seed=SEED, scorers={})
    for name in names:
        record = dict(capture=macro(pools[name], domain_of), spearman=macro(pools[name], domain_of, metric=1),
                      layers={str(l): float(np.mean([v[0] for v in pools[name][l].values()])) for l in pools[name]})
        for d, label in ((0, 'r2rce'), (1, 'rxrce')):
            eps = [e for e in episodes if domain_of[e] == d]
            record[f'capture_{label}'] = float(np.mean([np.mean([pools[name][l][e][0] for e in eps if e in pools[name][l]])
                                                        for l in pools[name]]))
        if name != 'S0':
            difference = record['capture'] - macro(pools['S0'], domain_of)
            boot = [macro(pools[name], domain_of, draw) - macro(pools['S0'], domain_of, draw) for draw in draws]
            record['capture_minus_S0'] = dict(estimate=difference, ci95=[float(np.quantile(boot, .025)),
                                                                         float(np.quantile(boot, .975))])
        report['scorers'][name] = record
        print('SCORER', name, round(record['capture'], 4), round(record['spearman'], 4),
              record.get('capture_minus_S0'), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
