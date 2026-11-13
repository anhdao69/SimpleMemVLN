"""Paired comparison of two fixed RxR val_unseen results (rxr_fix_merge.py outputs), same statistics as
scripts/rl/compare_evals.py (exact two-sided McNemar on per-episode success, bootstrap 95% CI, same metric table).

python3 -I rxr_fixed_compare.py BASE_FIXED_DIR NEW_FIXED_DIR NAME_BASE NAME_NEW OUT.json

compare_evals.py cannot read these results directly (they have no single evaluation contract: the 99 oLBMNvg9in8
episodes come from a re-evaluation with the RxR-CE agent navmesh). This applies its checks to both halves instead: the
two base evaluations must share every protocol key and evaluation source, and so must the two re-evaluations (except
the episode list, which is the 99 or their 52 in sets A + B); both results must cover the same episodes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/nvme/public/SimpleMemVLN_dev/scripts/rl')
from compare_evals import EVAL_SOURCES, PROTOCOL_KEYS, bootstrap, category, episode_metrics, mcnemar_exact  # noqa: E402


def contract(d):
    for c in [Path(d) / 'merged/evaluation_contract.json', *sorted(Path(d).glob('shard*/evaluation_contract.json'))]:
        if c.exists():
            j = json.loads(c.read_text())
            if 'weights_sha256' in j:
                return j
    raise FileNotFoundError(d)


def load(d):
    d = Path(d)
    s = json.loads((d / 'merged/summary.json').read_text())
    p = json.loads((d / 'merged/provenance.json').read_text())
    assert s.get('complete'), d
    rows = {r['episode_id']: r for r in map(json.loads, (d / 'merged/episodes.jsonl').read_text().splitlines())}
    assert len(rows) == s['scheduled_episodes'] == s['episodes'], d
    return s, p, rows


def same(ca, cb, skip=()):
    diff = [k for k in PROTOCOL_KEYS if k not in skip and ca.get(k) != cb.get(k)]
    diff += [k for k in EVAL_SOURCES if ca['source_sha256'].get(k) != cb['source_sha256'].get(k)]
    return diff


def main():
    bdir, ndir, nb, nn, out = sys.argv[1:6]
    (sb, pb, rb), (sn, pn, rn) = load(bdir), load(ndir)
    assert set(rb) == set(rn), 'different episode sets'
    d1 = same(contract(pb['base']), contract(pn['base']), skip=('episode_ids',) if pb['mode'] != pn['mode'] else ())
    d2 = same(contract(pb['reevaluated']), contract(pn['reevaluated']), skip=('episode_ids',))
    if d1 or d2:
        raise SystemExit(f'protocol or sources differ: base {d1} re-evaluation {d2}')
    ids = [r for r in rb]
    eb = {i: episode_metrics(rb[i]) for i in ids}
    en = {i: episode_metrics(rn[i]) for i in ids}
    table = {}
    for key, name in (('sr', 'SR'), ('spl', 'SPL'), ('navigation_error', 'NE'), ('oracle_success', 'OSR'),
                      ('trajectory_length', 'TL'), ('mean_episode_steps', 'steps'), ('forced_stops', 'forced_stops'),
                      ('invalid_responses', 'invalid'), ('failures', 'failures'), ('ndtw', 'nDTW')):
        a, b = sb.get(key), sn.get(key)
        table[name] = dict(base=a, new=b, abs_delta=None if a is None or b is None else b - a,
                           rel_delta=None if not a or b is None else (b - a) / a)
    both = sum(bool(eb[i]['success'] and en[i]['success']) for i in ids)
    only_base = sum(bool(eb[i]['success'] and not en[i]['success']) for i in ids)
    only_new = sum(bool(en[i]['success'] and not eb[i]['success']) for i in ids)
    ci = {name: bootstrap([en[i][k] - eb[i][k] for i in ids]) for name, k in (('SR', 'success'), ('SPL', 'spl'))}
    cats = sorted({category(e) for e in list(eb.values()) + list(en.values())})
    result = dict(base=nb, new=nn, episodes=len(ids), metrics=table,
                  paired_success=dict(both=both, only_base=only_base, only_new=only_new,
                                      neither=len(ids) - both - only_base - only_new,
                                      mcnemar_exact_p=mcnemar_exact(only_base, only_new)),
                  bootstrap_95ci=ci,
                  failure_taxonomy={c: dict(base=sum(category(eb[i]) == c for i in ids),
                                            new=sum(category(en[i]) == c for i in ids)) for c in cats},
                  checkpoints=dict(base=pb['checkpoint'], new=pn['checkpoint'], base_weights=pb['weights_sha256'],
                                   new_weights=pn['weights_sha256']),
                  fixed_results=dict(base=str(bdir), new=str(ndir)))
    Path(out).write_text(json.dumps(result, indent=2) + '\n')
    m, p = table['SR'], result['paired_success']
    print(f"{nb} -> {nn} ({Path(bdir).name}): SR {100 * m['base']:.2f} -> {100 * m['new']:.2f} ({100 * m['abs_delta']:+.2f}), "
          f"only_base {p['only_base']} only_new {p['only_new']}, p {p['mcnemar_exact_p']:.3g}, CI {[round(100 * x, 2) for x in ci['SR']]}")


if __name__ == '__main__':
    main()
