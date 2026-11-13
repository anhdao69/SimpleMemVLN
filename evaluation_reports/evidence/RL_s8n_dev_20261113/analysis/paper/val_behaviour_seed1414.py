import json, sys
E = '/nvme/public/svln/evals/'
print(f"{'model':32s} {'SR':>6s} {'OS':>6s} {'SPL':>6s} {'turn':>6s} {'acts':>6s} {'TL':>6s} {'NE':>5s} {'stop<OS':>8s} {'forced':>6s}")
for n in sys.argv[1:]:
    eps = [json.loads(l) for l in open(E + n + '_r2r_val_unseen/merged/episodes.jsonl')]
    acts = [a['predicted'] for e in eps for a in e['actions']]
    turn = sum(a in (2, 3) for a in acts) / len(acts)
    sr = sum(e['metrics']['success'] for e in eps) / len(eps)
    os_ = sum(e['metrics']['oracle_success'] for e in eps) / len(eps)
    spl = sum(e['metrics']['spl'] for e in eps) / len(eps)
    tl = sum(e['metrics']['trajectory_length'] for e in eps) / len(eps)
    ne = sum(e['metrics']['distance_to_goal'] for e in eps) / len(eps)
    miss = sum(e['metrics']['oracle_success'] == 1 and e['metrics']['success'] == 0 for e in eps)
    forced = sum(e['actions'][-1]['predicted'] != 0 for e in eps)
    print(f"{n:32s} {100*sr:6.2f} {100*os_:6.2f} {100*spl:6.2f} {turn:6.3f} {len(acts)/len(eps):6.1f} {tl:6.2f} {ne:5.2f} {miss:8d} {forced:6d}")
