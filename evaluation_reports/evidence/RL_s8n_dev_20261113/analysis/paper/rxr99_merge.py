"""Full RxR val_unseen result with the 99 re-evaluated episodes (2026-11-13; analysis/paper/rxr99_fix.md).

python3 -I rxr99_merge.py OLD_FULL_EVAL_DIR NEW_99_EVAL_DIR OUT_DIR
OLD_FULL_EVAL_DIR/merged/episodes.jsonl: the original 3,669-episode evaluation (R2R-agent navmesh). NEW_99_EVAL_DIR/merged:
the same checkpoint on the 99 episodes that were unscoreable there, evaluated with the RxR-CE agent navmesh. Writes
OUT_DIR/episodes.jsonl (original order; the 99 records replaced), summary.json (the evaluator's own summarize()), and
provenance.json. Checks: same checkpoint weights, the replaced IDs are exactly the old run's unscoreable episodes.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, '/nvme/public/svln/code/rxrce_agent_eval_20261113/src')
from qwen_vl.eval.metrics import summarize  # the evaluator's summary (frozen copy)

old_dir, new_dir, out = (Path(a) for a in sys.argv[1:4])
old = [json.loads(l) for l in open(old_dir / 'merged/episodes.jsonl')]
new = {r['episode_id']: r for r in (json.loads(l) for l in open(new_dir / 'merged/episodes.jsonl'))}
ids = set(json.load(open('/nvme/public/svln/analysis/paper/rxr99_episode_ids.json')))
unscored = {r['episode_id'] for r in old if r.get('failure_reason') == 'Required simulator metrics unavailable'}
assert set(new) == ids and unscored <= ids, (len(new), len(unscored))
other = {r['episode_id']: r.get('failure_reason') for r in old if r['episode_id'] in ids and r['episode_id'] not in unscored}
co = json.load(open(old_dir / 'merged/evaluation_contract.json')) if (old_dir / 'merged/evaluation_contract.json').exists() else \
     json.load(open(next(old_dir.glob('shard0/evaluation_contract.json'))))
cn = json.load(open(next(new_dir.glob('shard0/evaluation_contract.json'))))
assert co['weights_sha256'] == cn['weights_sha256'], 'different checkpoints'
assert len(old) == 3669 and len({r['episode_id'] for r in old}) == 3669
merged = [dict(new[r['episode_id']], rxrce_agent_navmesh=True) if r['episode_id'] in ids else r for r in old]
out.mkdir(parents=True, exist_ok=False)
with open(out / 'episodes.jsonl', 'w') as f:
    for r in merged: f.write(json.dumps(r) + '\n')
s_old, s_new = summarize(old), summarize(merged)
(out / 'summary.json').write_text(json.dumps(dict(s_new, scheduled_episodes=3669, complete=True), indent=2) + '\n')
n99 = [new[i] for i in sorted(ids)]
prov = dict(old=str(old_dir), new99_dir=str(new_dir), weights_sha256=co['weights_sha256'],
            new99_habitat_config_sha256=cn['habitat_config_sha256'], old_habitat_config_sha256=co['habitat_config_sha256'],
            replaced=len(ids), old_failures_in_99={'unscoreable': len(unscored & ids), 'other': other},
            new99=dict(failed=sum(r['failed'] for r in n99), success=sum(0 if r['failed'] else r['metrics'].get('success', 0) for r in n99),
                       failure_reasons=sorted({r.get('failure_reason') for r in n99 if r['failed']}, key=str)),
            note='99 RxR val_unseen episodes (scene oLBMNvg9in8) unreachable on the R2R-agent navmesh; re-evaluated with the '
                 'RxR-CE agent navmesh (height 0.88 m, radius 0.18 m), everything else unchanged.')
(out / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
f = lambda s, k: (round(100 * s[k], 2) if k in ('sr', 'spl', 'oracle_success', 'ndtw') else round(s[k], 3)) if s[k] is not None else None
print(out.name)
for k in ('sr', 'spl', 'oracle_success', 'ndtw', 'navigation_error'):
    print(f"  {k:17s} {f(s_old, k)} -> {f(s_new, k)}   (coverage {s_old[k + '_coverage'] if k + '_coverage' in s_old else ''} -> {s_new[k + '_coverage'] if k + '_coverage' in s_new else ''})")
print('  failures', s_old['failures'], '->', s_new['failures'], '| on the 99:', prov['new99'])
