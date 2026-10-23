"""Fit and evaluate the Phase-B learned rankers on the near16 target (plan Sections 11-12).

Every model sees the same rows, folds, target and tuning budget:

* target ``log1p(Y_near16 / kappa[layer])``, ``kappa = max(median positive
  fit target, 1e-12)`` per attention layer (raw aggregate first, then log);
* Huber(delta=1) with equal total weight per domain and per episode
  (rows equal within an episode); one predictor per attention layer;
* AdamW, LR in {3e-4, 1e-3}, weight decay 1e-2, batch 4096, <= 20 epochs,
  gradient clip 1.0, seed 429; early stopping (patience 3) on the
  predeclared dev metric ``capture@96`` (tie-break: dev loss); the LR is
  chosen per arm by the mean best dev capture over layers;
* fit-partition normalizers (std floor 1e-4) frozen before any dev row is used.

``capture@96``: within each (episode, attention layer) pool of labeled rows,
the raw near16 utility of the 96 highest-scored rows divided by that of the
96 highest-utility rows.  96 = (B / 300) frames x 24 sampled rows, the
sample-relative analogue of the deployed top-B; pools of <= 96 rows are
uninformative and skipped.  Macro-averaged over episodes within a domain,
then over domains, then layers.  Spearman per pool is reported, never gated.

Arms: L0-L3 (primary), controls L0_raw, L0_S0res, H_linear, H_mlp_matched,
and the fixed S0/S1/S2/S3/S4/S5/S9/random scores computed from the same rows.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from qwen_vl.models.gdn_features import encode_features
from qwen_vl.research.gdn_ranker import (
    ARMS, CHANNELS, RankerHead, STD_FLOOR, build_artifact, canonical_sha256, huber, input_width,
    kappa_for, parameter_count, transform_target)
from qwen_vl.research.gdn_ranker_dataset import RankerData

SEED, LRS, WEIGHT_DECAY, BATCH, EPOCHS, CLIP, PATIENCE = 429, (3e-4, 1e-3), 1e-2, 4096, 20, 1.0, 3
BUDGET_ROWS = 96
CONTROLS = ('L0_raw', 'L0_S0res', 'H_linear', 'H_mlp_matched')
FIXED_SCORES = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S9', 'random')
PROJECTION_SEED, PROJECTION_WIDTH = 0, 576


# ---------------------------------------------------------------- features
def learned_features(data, arm, layer, index):
    """[n, width] float32 inputs in GDN-layer, head, channel order."""
    if arm in ARMS:
        spec = ARMS[arm]
        window = list(range(layer - spec['window'], layer))
        fields = data.fields_torch(layer, index, window)
        encoded = encode_features(dict(schema='gdn_raw_v2', fields=fields), spec['schema'])['values']
        return encoded.reshape(index.size, -1)                 # [n, W, Hv, C] -> layer, head, channel
    if arm in ('L0_raw', 'L0_S0res'):
        fields = data.fields_torch(layer, index, [layer - 1])
        if arm == 'L0_raw':
            x = torch.stack([fields['beta'], fields['log_decay_g'], fields['value_norm']], -1)
            return x.reshape(index.size, -1)
        encoded = encode_features(dict(schema='gdn_raw_v2', fields=fields), 'F0_v1')['values']
        return encoded.reshape(index.size, -1)
    if arm in ('H_linear', 'H_mlp_matched'):
        bits = np.asarray(data.hidden(layer)[np.sort(index)])[np.argsort(np.argsort(index))]
        hidden = torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32))   # exact BF16 -> FP32
        if arm == 'H_linear':
            return hidden
        weight = projection(hidden.shape[1])
        return torch.cat([chunk @ weight for chunk in hidden.split(262144)])
    raise ValueError(arm)


def projection(width):
    g = torch.Generator().manual_seed(PROJECTION_SEED)
    return torch.randn(width, PROJECTION_WIDTH, generator=g) / math.sqrt(width)


def head_for(arm, width):
    if arm in ARMS:
        return RankerHead(width, ARMS[arm]['hidden'])
    if arm in ('L0_raw', 'L0_S0res', 'H_linear'):
        return RankerHead(width, None)
    return RankerHead(width, 64)                                # H_mlp_matched: L3's head


def fixed_scores(data, name, layer, index):
    k = data.layer_column(layer)
    if name == 'S0':
        return data.rows['s0'][index, k].astype(np.float64)
    if name == 'random':
        key = data.rows['episode'][index].astype(np.int64) * 1_000_003 + data.rows['token_id'][index] * 31 + layer
        return np.asarray([int(hashlib.sha256(f'rank-random|{x}'.encode()).hexdigest()[:12], 16) for x in key],
                          dtype=np.float64)
    f = data.fields_torch(layer, index, [layer - 1])
    f = {n: v[:, 0].double() for n, v in f.items()}            # [n, Hv]
    per_head = dict(S1=f['error_norm'], S2=f['beta'] * f['value_norm'], S3=f['value_norm'],
                    S4=f['cosine_error'], S5=f['cosine_error'], S9=(1 - f['beta']) * f['error_norm'])[name]
    return (per_head.min(-1).values if name == 'S4' else per_head.mean(-1)).numpy()


# ---------------------------------------------------------------- metrics
def average_ranks(x):
    """Ranks 0..n-1 with ties sharing their mean rank (scipy's 'average')."""
    order = np.argsort(x, kind='stable')
    sorted_x = x[order]
    starts = np.flatnonzero(np.r_[True, sorted_x[1:] != sorted_x[:-1]])
    ends = np.r_[starts[1:], x.size]
    ranks = np.empty(x.size, np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(np.asarray(a, np.float64)), average_ranks(np.asarray(b, np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / denominator) if denominator > 0 else float('nan')


def pool_metrics(data, index, raw, scores, budget=BUDGET_ROWS):
    """Macro capture@budget and Spearman over (episode) pools of one layer."""
    episode = data.rows['episode'][index]
    domain = data.rows['domain'][index]
    order = np.argsort(episode, kind='stable')
    episode, domain, raw, scores = episode[order], domain[order], raw[order], scores[order]
    bounds = np.flatnonzero(np.diff(episode)) + 1
    per_domain = {0: [], 1: []}
    rho = {0: [], 1: []}
    for part in np.split(np.arange(episode.size), bounds):
        if part.size <= budget:
            continue
        y, s = raw[part], scores[part]
        top = np.argsort(-s, kind='stable')[:budget]
        best = np.sort(y)[::-1][:budget].sum()
        d = int(domain[part[0]])
        per_domain[d].append(float(y[top].sum() / best) if best > 0 else 1.0)
        rho[d].append(spearman(s, y))
    if not any(per_domain.values()):
        return dict(capture=None, capture_r2r=None, capture_rxr=None, spearman=None, pools=0)
    capture = float(np.mean([np.mean(v) for v in per_domain.values() if v]))
    return dict(capture=capture, capture_r2r=float(np.mean(per_domain[0])) if per_domain[0] else None,
                capture_rxr=float(np.mean(per_domain[1])) if per_domain[1] else None,
                spearman=float(np.mean([np.mean(v) for v in rho.values() if v])),
                pools=sum(len(v) for v in per_domain.values()))


# ---------------------------------------------------------------- training
def standardize_fit(x):
    mean = x.double().mean(0)
    std = x.double().std(0, unbiased=False).clamp_min(STD_FLOOR)
    return mean.float(), std.float()


class LayerData:
    """Fit/dev tensors of one (arm, attention layer); fit-only constants, built once for the LR grid."""

    def __init__(self, arm, data, layer, device):
        self.fit = np.flatnonzero(data.mask('ranker_fit', layer))
        self.dev = np.flatnonzero(data.mask('ranker_dev', layer))
        k = data.layer_column(layer)
        self.raw_fit, self.raw_dev = data.rows['near'][self.fit, k], data.rows['near'][self.dev, k]
        self.kappa, self.degenerate = kappa_for(self.raw_fit)
        x_fit = learned_features(data, arm, layer, self.fit)
        self.mean, self.std = standardize_fit(x_fit)
        self.z_fit = ((x_fit - self.mean) / self.std).to(device)
        del x_fit
        self.z_dev = ((learned_features(data, arm, layer, self.dev) - self.mean) / self.std).to(device)
        self.y_fit = torch.from_numpy(transform_target(self.raw_fit, self.kappa)).float().to(device)
        self.y_dev = torch.from_numpy(transform_target(self.raw_dev, self.kappa)).float().to(device)
        self.w_fit = torch.from_numpy(data.weights(self.fit)).float().to(device)
        self.w_dev = torch.from_numpy(data.weights(self.dev)).float().to(device)
        self.base_fit = self.base_dev = self.kappa_s0 = None
        if arm == 'L0_S0res':
            self.kappa_s0, _ = kappa_for(data.rows['s0'][self.fit, k])
            self.base_fit = s0_baseline(data, self.fit, k, self.kappa_s0).to(device)
            self.base_dev = s0_baseline(data, self.dev, k, self.kappa_s0).to(device)


def s0_baseline(data, index, k, kappa_s0):
    return torch.from_numpy(np.log1p(np.maximum(data.rows['s0'][index, k], 0) / kappa_s0)).float()


def train_layer(arm, data, layer, device, lr, d):
    z_fit, z_dev, y_fit, y_dev, w_fit, w_dev = d.z_fit, d.z_dev, d.y_fit, d.y_dev, d.w_fit, d.w_dev
    base_fit, base_dev, dev, raw_dev = d.base_fit, d.base_dev, d.dev, d.raw_dev
    torch.manual_seed(SEED)
    head = head_for(arm, z_fit.shape[1]).to(device)
    if arm == 'L0_S0res':                                       # zero-initialized residual correction
        torch.nn.init.zeros_(head.out.weight)
        torch.nn.init.zeros_(head.out.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device='cpu').manual_seed(SEED)

    def predict(z, base):
        with torch.no_grad():
            out = torch.cat([head(chunk) for chunk in z.split(65536)])
        return out if base is None else out + base

    best, history, stale = None, [], 0
    started = time.time()
    for epoch in range(EPOCHS + 1):
        if epoch:
            head.train()
            perm = torch.randperm(z_fit.shape[0], generator=generator).to(device)
            for batch in perm.split(BATCH):
                pred = head(z_fit[batch]) + (0 if base_fit is None else base_fit[batch])
                loss = huber(pred, y_fit[batch], w_fit[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
                optimizer.step()
            head.eval()
        dev_pred = predict(z_dev, base_dev)
        dev_loss = float(huber(dev_pred, y_dev, w_dev))
        metrics = pool_metrics(data, dev, raw_dev, dev_pred.double().cpu().numpy())
        history.append(dict(epoch=epoch, dev_loss=dev_loss, **metrics))
        score = (metrics['capture'], -dev_loss)
        if best is None or score > best[0]:
            best, stale = (score, epoch, {n: p.detach().cpu().clone() for n, p in head.state_dict().items()}), 0
        elif epoch:
            stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(best[2])
    head = head.cpu().eval()
    return dict(head=head, mean=d.mean.cpu(), std=d.std.cpu(), kappa=d.kappa, kappa_degenerate=d.degenerate,
                kappa_s0=d.kappa_s0, best_epoch=best[1], best_dev_capture=best[0][0], best_dev_loss=-best[0][1],
                history=history, seconds=time.time() - started, fit_rows=int(d.fit.size), dev_rows=int(dev.size))


def evaluate(arm, data, layer, result, partition='ranker_holdout'):
    index = np.flatnonzero(data.mask(partition, layer))
    k = data.layer_column(layer)
    raw = data.rows['near'][index, k]
    if index.size == 0:
        return pool_metrics(data, index, raw, raw), raw
    if arm in FIXED_SCORES:
        scores = fixed_scores(data, arm, layer, index)
    else:
        x = learned_features(data, arm, layer, index)
        z = (x - result['mean']) / result['std']
        with torch.no_grad():
            scores = torch.cat([result['head'](c) for c in z.split(65536)]).double().numpy()
        if arm == 'L0_S0res':
            scores = scores + s0_baseline(data, index, k, result['kappa_s0']).double().numpy()
    return pool_metrics(data, index, raw, scores), scores


def summarize(per_layer):
    if any(m['capture'] is None for m in per_layer.values()):
        return dict(layers=per_layer, capture=None, spearman=None)
    return dict(layers=per_layer, capture=float(np.mean([m['capture'] for m in per_layer.values()])),
                spearman=float(np.mean([m['spearman'] for m in per_layer.values()])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arms', nargs='+', default=['L0', 'L1', 'L2', 'L3', *CONTROLS])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--source-commit', required=True)
    p.add_argument('--no-fixed', action='store_true', help='Skip the fixed S-score evaluation (another shard does it)')
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = RankerData(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = data.attention_layers
    report = dict(version='gdn_ranker_fit_v1', target='shadow_fixed_query_loo_near16_mean_v2', seed=SEED,
                  lrs=list(LRS), weight_decay=WEIGHT_DECAY, batch=BATCH, epochs=EPOCHS, clip=CLIP,
                  patience=PATIENCE, budget_rows=BUDGET_ROWS, dataset=data.manifest | {'episode_uids': None},
                  source_commit=args.source_commit, arms={}, fixed={})
    hidden_size = data.hidden(layers[0]).shape[1] if data.manifest['hidden'] else None
    if hidden_size is not None:
        weight = projection(hidden_size).numpy().astype('<f4')
        report['h_projection'] = dict(seed=PROJECTION_SEED, width=PROJECTION_WIDTH, hidden_size=hidden_size,
                                      sha256=hashlib.sha256(weight.tobytes()).hexdigest())
    for name in () if args.no_fixed else FIXED_SCORES:
        per_layer = {}
        for layer in layers:
            for part in ('ranker_dev', 'ranker_holdout'):
                metrics, _ = evaluate(name, data, layer, None, part)
                per_layer.setdefault(part, {})[str(layer)] = metrics
        report['fixed'][name] = {part: summarize(v) for part, v in per_layer.items()}
        print('FIXED', name, {part: r['capture'] for part, r in report['fixed'][name].items()}, flush=True)
    for arm in args.arms:
        trials = {lr: {} for lr in LRS}
        for layer in layers:                  # one layer's tensors resident at a time
            layer_data = LayerData(arm, data, layer, args.device)
            for lr in LRS:
                trials[lr][layer] = train_layer(arm, data, layer, args.device, lr, layer_data)
                print('TRIAL', arm, layer, lr, round(trials[lr][layer]['best_dev_capture'], 4),
                      'epoch', trials[lr][layer]['best_epoch'], flush=True)
            del layer_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        chosen = max(LRS, key=lambda lr: (np.mean([trials[lr][l]['best_dev_capture'] for l in layers]), -lr))
        results = trials[chosen]
        holdout, dev = {}, {}
        for layer in layers:
            holdout[str(layer)], _ = evaluate(arm, data, layer, results[layer], 'ranker_holdout')
            dev[str(layer)], _ = evaluate(arm, data, layer, results[layer], 'ranker_dev')
        record = dict(
            chosen_lr=chosen,
            trials={str(lr): {str(l): dict(best_epoch=t['best_epoch'], best_dev_capture=t['best_dev_capture'],
                                           best_dev_loss=t['best_dev_loss'], seconds=t['seconds'],
                                           history=t['history'])
                              for l, t in trial.items()} for lr, trial in trials.items()},
            kappa={str(l): results[l]['kappa'] for l in layers},
            kappa_degenerate={str(l): results[l]['kappa_degenerate'] for l in layers},
            rows={str(l): dict(fit=results[l]['fit_rows'], dev=results[l]['dev_rows']) for l in layers},
            dev=summarize(dev), holdout=summarize(holdout),
            parameters=sum(sum(p.numel() for p in results[l]['head'].parameters()) for l in layers))
        if arm in ARMS:
            if record['parameters'] != parameter_count(arm):
                raise RuntimeError(f'{arm} parameter count {record["parameters"]} != {parameter_count(arm)}')
            provenance = dict(source_commit=args.source_commit, dataset_manifest_sha256=hashlib.sha256(
                (Path(args.dataset) / 'manifest.json').read_bytes()).hexdigest(), seed=SEED, lr=chosen,
                best_epochs={str(l): results[l]['best_epoch'] for l in layers},
                fit_partition='ranker_fit', early_stopping='ranker_dev capture@96')
            artifact = build_artifact(arm, {l: results[l]['head'] for l in layers},
                                      {l: (results[l]['mean'].numpy(), results[l]['std'].numpy()) for l in layers},
                                      {l: results[l]['kappa'] for l in layers}, provenance)
            sha = canonical_sha256(artifact)
            (out / f'{arm}_artifact.json').write_text(json.dumps(artifact, sort_keys=True) + '\n')
            record['artifact_sha256'] = sha
        else:
            torch.save({str(l): dict(state=results[l]['head'].state_dict(), mean=results[l]['mean'],
                                     std=results[l]['std'], kappa=results[l]['kappa'],
                                     kappa_s0=results[l]['kappa_s0']) for l in layers},
                       out / f'{arm}_control.pt')
        report['arms'][arm] = record
        print('ARM', arm, 'lr', chosen, 'dev', record['dev']['capture'], 'dev_spearman', record['dev']['spearman'],
              'holdout', record['holdout']['capture'], 'holdout_spearman', record['holdout']['spearman'], flush=True)
        (out / 'fit_report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print('RANKERS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
