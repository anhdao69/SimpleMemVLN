import json, gzip, math, statistics as st, sys
E = '/nvme/public/svln/evals'
eps = {str(e['episode_id']): e for e in json.load(gzip.open('/nvme/public/svln/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz'))['episodes']}
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f'{E}/{name}_r2r_val_unseen/merged/episodes.jsonl')]
    out = dict(within1=0, within1_nostop_there=0, steps_in_3m_before_leave=[], succ_d=[], oracle_fail=0, overshoot_after_1m=[], first_turns=[], n=len(rows))
    for r in rows:
        g = eps[str(r['episode_id'])]['goals'][0]['position']
        tr = r['trajectory']
        d = [math.dist((p[0], p[2]), (g[0], g[2])) if abs(p[1]-g[1]) < 1.5 else 99 for p in tr]
        m = r['metrics']
        if m['success'] > 0: out['succ_d'].append(m['distance_to_goal'])
        i1 = next((i for i, x in enumerate(d) if x < 1.0), None)
        if i1 is not None:
            out['within1'] += 1
            if m['success'] == 0:
                out['within1_nostop_there'] += 1
                out['overshoot_after_1m'].append(len(d) - 1 - i1)
        acts = [a['action_name'] for a in r['actions']]
        k = next((i for i, a in enumerate(acts) if a == 'MOVE_FORWARD'), len(acts))
        out['first_turns'].append(k)
    print('%-34s reached<1m %4d, of which failed %3d (%.1f%%), steps after first <1m on those: median %s; success stop d mean %.2f; initial turns before 1st forward mean %.1f' % (
        name, out['within1'], out['within1_nostop_there'], 100*out['within1_nostop_there']/max(1,out['within1']),
        st.median(out['overshoot_after_1m']) if out['overshoot_after_1m'] else '-', st.mean(out['succ_d']), st.mean(out['first_turns'])))
