"""Quick sanity report over collected label episodes (coverage, scales, reconstruction, timing)."""
import argparse
import glob
import json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    args = p.parse_args()
    files = sorted(glob.glob(f'{args.dir}/episodes/*.npz'))
    rows = []
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        u = z['utility']
        res = z['resident']
        finite = np.isfinite(u)
        rec = meta['reconstruction']
        rows.append(dict(
            uid=meta['episode']['episode_uid'], logger=f"{meta['logger']}:{meta['logger_seed']}",
            T=meta['groups'], tokens=meta['tokens'], seconds=round(meta['seconds'], 1),
            timing={k: round(v, 1) for k, v in meta['timing'].items()},
            labels=int(finite.sum()), missing=meta['coverage']['missing'],
            resident_fraction=float((res[finite] == 1).mean()) if finite.any() else None,
            utility_median=float(np.median(u[finite])), utility_p99=float(np.quantile(u[finite], .99)),
            utility_zero=int((u[finite] == 0).sum()),
            recon_max=max(r['max_rel_error'] for r in rec.values()),
            recon_mean=float(np.mean([r['mean_rel_error'] for r in rec.values()])),
            ablated_mean=float(np.mean([r['bank_ablated_mean_rel_error'] for r in rec.values()
                                        if r['bank_ablated_mean_rel_error'] is not None])),
            ablation_detected={k: r.get('bank_ablation_detected') for k, r in rec.items()},
            fallbacks=meta['stable_fallbacks'], action_nll=round(meta['action_nll'], 4),
            dense=meta['dense'],
            dense_finite=meta['coverage'].get('dense_finite'),
            arrays={k: list(z[k].shape) for k in z.files if k != 'meta'}))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
