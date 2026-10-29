"""H2 proxy (pre-registered H2 names reference-path turning points; positions are not logged, so this uses
logged privileged geometry): is the S8N event score of o_t higher where the shortest path to the goal bends?
Decisions t >= 2 whose previous action was FORWARD (model action 0), so o_t is not a rotation of o_{t-1}.
bend_t = |wrap(bearing_error_t - bearing_error_{t-1})| (heading unchanged by a forward step, so a jump means
the shortest path changed direction: the agent passed a corner/junction). Also progress slope change.
event_pos = event - median event of the same RL step and position bucket (as analyze_event_forks)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts/rl')
from qwen_vl.rl.rollout import event_score, position_bucket
from analyze_event_forks import spearman
run, hi = Path(sys.argv[2]), int(sys.argv[3])
recs, pooled = [], {}
for p in sorted((run / 'rollouts').glob('step_*_rank*.jsonl')):
    step = int(p.name.split('_')[1])
    if step >= hi:
        continue
    for line in p.read_text().splitlines():
        e = json.loads(line)
        s = [event_score(d) if d else None for d in e['step_drift']]
        b, d, a = e['step_bearing'], e['step_d'], e['actions']
        for t in range(len(s)):
            if s[t] is not None:
                pooled.setdefault((step, position_bucket(t)), []).append(s[t])
        for t in range(2, len(s)):
            if s[t] is None or a[t - 1] != 0 or b[t] is None or b[t - 1] is None:
                continue
            bend = abs((b[t] - b[t - 1] + math.pi) % (2 * math.pi) - math.pi)
            slope_change = abs((d[t] - d[t - 1]) - (d[t - 1] - d[t - 2]))
            recs.append(dict(step=step, t=t, ev=s[t], bend=bend, dslope=slope_change, turn_next=a[t] in (1, 2)))
med = {k: float(np.median(v)) for k, v in pooled.items()}
for r in recs:
    r['ev_pos'] = r['ev'] - med[(r['step'], position_bucket(r['t']))]
ev = np.asarray([r['ev_pos'] for r in recs]); bend = np.asarray([r['bend'] for r in recs])
out = dict(decisions=len(recs))
for lo, hi_ in [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.2), (1.2, 4)]:
    m = (bend >= lo) & (bend < hi_)
    out[f'bend {math.degrees(lo):.0f}-{math.degrees(hi_):.0f} deg'] = dict(n=int(m.sum()), mean_event_pos=round(float(ev[m].mean()), 4),
                                                         p_event_pos_top20=round(float((ev[m] >= np.quantile(ev, 0.8)).mean()), 3))
out['spearman_eventpos_bend'] = round(spearman(ev, bend), 3)
out['spearman_eventpos_dslope'] = round(spearman(ev, [r['dslope'] for r in recs]), 3)
top = ev >= np.quantile(ev, 0.8)
out['p_bend_gt_35deg | top20 event_pos'] = round(float((bend[top] > 0.6).mean()), 3)
out['p_bend_gt_35deg | rest'] = round(float((bend[~top] > 0.6).mean()), 3)
out['p_agent_turns_next | top20'] = round(float(np.mean([r['turn_next'] for r, k in zip(recs, top) if k])), 3)
out['p_agent_turns_next | rest'] = round(float(np.mean([r['turn_next'] for r, k in zip(recs, top) if not k])), 3)
print(json.dumps(out, indent=1))
