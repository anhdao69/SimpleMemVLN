"""RxR val_unseen results with the 99 oLBMNvg9in8 episodes evaluated on the RxR-CE agent navmesh (2026-11-13).

The 99 episodes (rxr99_episode_ids.json) are unreachable on the R2R-agent navmesh that the evaluator used for every RxR
episode; on the navmesh Habitat builds for the RxR-CE agent (0.88 m / 0.18 m) they are solvable (rxr99_fix.md). Every
model's records of those episodes come from a re-evaluation of the same checkpoint with that navmesh; the other episodes
keep their original records.

replace:  python3 -I rxr_fix_merge.py replace OLD_EVAL_DIR REEVAL_DIR OUT_DIR
    OLD_EVAL_DIR/merged/episodes.jsonl: a complete evaluation (any episode set: full, A, B, C, 2k); every record of an
    episode in the 99 is replaced by REEVAL_DIR's record (REEVAL_DIR must contain all of them).
combine:  python3 -I rxr_fix_merge.py combine MAIN_EVAL_DIR REEVAL_DIR OUT_DIR
    MAIN_EVAL_DIR: the 3,570 other episodes (R2R-agent navmesh), REEVAL_DIR: the 99; output in the canonical full order.
Both check that the two evaluations used the same checkpoint weights, write OUT_DIR/merged/{episodes.jsonl, summary.json,
provenance.json} with the evaluator's own summarize(), and leave the inputs untouched.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SVLN_ROOT', '/nvme/public/svln'))
sys.path.insert(0, str(ROOT / 'code/rxrce_agent_eval_20261113/src'))
from qwen_vl.eval.metrics import summarize  # noqa: E402  (the evaluator's summary, frozen copy)

UNSCOREABLE = 'Required simulator metrics unavailable'


def records(d):
    return [json.loads(line) for line in open(d / 'merged/episodes.jsonl')]


def contract(d):
    for c in [d / 'merged/evaluation_contract.json', *sorted(d.glob('shard*/evaluation_contract.json'))]:
        if c.exists():
            j = json.loads(c.read_text())
            if 'weights_sha256' in j:
                return j
    raise FileNotFoundError(f'no evaluation contract with weights in {d}')


def main():
    mode, old_dir, re_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    fix_ids = set(json.loads((HERE / 'rxr99_episode_ids.json').read_text()))
    assert len(fix_ids) == 99
    old, new = records(old_dir), {r['episode_id']: r for r in records(re_dir)}
    co, cn = contract(old_dir), contract(re_dir)
    assert co['weights_sha256'] == cn['weights_sha256'], f'different checkpoints: {old_dir} {re_dir}'
    assert cn['habitat_config_sha256'] != co['habitat_config_sha256'], 're-evaluation did not use the RxR-CE agent config'
    assert not any(r['episode_id'] in fix_ids for r in new.values() if r.get('failure_reason') == UNSCOREABLE)
    if mode == 'replace':
        ids = [r['episode_id'] for r in old]
        assert len(set(ids)) == len(ids)
        unscoreable = {r['episode_id'] for r in old if r.get('failure_reason') == UNSCOREABLE}
        assert unscoreable <= fix_ids, sorted(unscoreable - fix_ids)[:5]
        assert fix_ids & set(ids) <= set(new), 'missing re-evaluated episodes'
        merged = [dict(new[r['episode_id']], rxrce_agent_navmesh=True) if r['episode_id'] in fix_ids else r for r in old]
        replaced = sorted(fix_ids & set(ids))
        other_old = {r['episode_id']: r.get('failure_reason') for r in old if r['episode_id'] in fix_ids and r['episode_id'] not in unscoreable}
    elif mode == 'combine':
        order = [json.loads(line)['episode_id'] for line in open(ROOT / 'evals/rewind_sfr_mix1k_s125_rxrfull_val_unseen/merged/episodes.jsonl')] \
            if (ROOT / 'evals/rewind_sfr_mix1k_s125_rxrfull_val_unseen').exists() else json.loads((HERE / 'rxr_full_order.json').read_text())
        main_by = {r['episode_id']: r for r in old}
        assert len(order) == 3669 and set(main_by) == set(order) - fix_ids and set(new) == fix_ids, (len(main_by), len(new))
        merged = [dict(new[i], rxrce_agent_navmesh=True) if i in fix_ids else main_by[i] for i in order]
        replaced, unscoreable, other_old = sorted(fix_ids), set(), {}
    else:
        raise SystemExit('mode: replace | combine')
    assert not any(r.get('failure_reason') == UNSCOREABLE for r in merged)
    (out / 'merged').mkdir(parents=True, exist_ok=False)
    with open(out / 'merged/episodes.jsonl', 'w') as f:
        for r in merged:
            f.write(json.dumps(r) + '\n')
    s = summarize(merged)
    (out / 'merged/summary.json').write_text(json.dumps(dict(s, scheduled_episodes=len(merged), complete=True), indent=2) + '\n')
    (out / 'merged/provenance.json').write_text(json.dumps(dict(
        mode=mode, base=str(old_dir), reevaluated=str(re_dir), weights_sha256=co['weights_sha256'], checkpoint=co.get('checkpoint'),
        base_habitat_config_sha256=co['habitat_config_sha256'], reevaluated_habitat_config_sha256=cn['habitat_config_sha256'],
        episodes=len(merged), replaced=len(replaced), base_unscoreable=len(unscoreable), base_other_failures_replaced=other_old),
        indent=2) + '\n')
    so = summarize(old) if mode == 'replace' else None
    pct = lambda d, k: None if d is None or d.get(k) is None else round(100 * d[k], 2) if k != 'navigation_error' else round(d[k], 3)
    print(f"{out.name}: n={len(merged)} replaced {len(replaced)} | " + ' '.join(
        f"{k} {pct(so, k)}->{pct(s, k)}" for k in ('sr', 'spl', 'oracle_success', 'ndtw', 'navigation_error')))


if __name__ == '__main__':
    main()
