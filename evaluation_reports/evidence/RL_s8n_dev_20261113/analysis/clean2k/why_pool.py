"""Why does the prompt pool matter? Training-signal decomposition of local Rewind runs (steps 0-124).

python3 -I why_pool.py RUN [RUN ...] > out.json
Per run: group outcome mix, GRPO advantage mass by group type and sign, advantage-weighted action push (which
actions the update raises / lowers), STOP calibration on the training prompts (policy STOP log-prob by distance
band, early vs late steps), failure taxonomy of the mains, rescue-branch mass and expert takeover action mix,
trajectory shape of successes vs failures.
"""
import collections
import glob
import json
import math
import sys
from pathlib import Path

RUNS = Path('/nvme/public/svln/runs')
FWD, LEFT, RIGHT, STOP = 0, 1, 2, 3


def rows_of(run, max_step=124):
    for f in sorted(glob.glob(str(RUNS / run / 'rollouts' / 'step_*_rank*.jsonl'))):
        step = int(Path(f).name.split('_')[1])
        if step > max_step:
            continue
        rank = Path(f).name.split('_')[2].split('.')[0]
        for line in open(f):
            r = json.loads(line)
            r['_step'], r['_rank'] = step, rank
            yield r


def fail_kind(r):
    if r['success']:
        return 'success'
    if r.get('truncated') or not r.get('stopped'):
        return 'no_stop'          # ran out of budget
    if r.get('oracle_success'):
        return 'passed_goal'      # was within 3 m, stopped elsewhere
    return 'stopped_far' if r['final_distance'] > 5 else 'stopped_near'


def analyse(run):
    groups = collections.defaultdict(list)
    for r in rows_of(run):
        r.pop('memory', None); r.pop('step_drift', None)
        groups[(r['_step'], r['_rank'], r['episode_id'])].append(r)
    out = dict(run=run, groups=len(groups))
    # A. group outcome mix
    s_hist = collections.Counter(sum(x['success'] for x in g) for g in groups.values())
    n = sum(s_hist.values())
    out['group_successes'] = {k: round(v / n, 3) for k, v in sorted(s_hist.items())}
    out['rerolled_groups'] = round(sum(g[0]['attempt'] > 0 for g in groups.values()) / n, 3)
    # B. advantage mass by group type (decision-weighted) and C. action push
    mass = collections.defaultdict(float)
    push = collections.defaultdict(float)       # sum adv * count(action)
    push_by = collections.defaultdict(lambda: collections.defaultdict(float))
    tot_abs = 0.0
    for g in groups.values():
        s = sum(x['success'] for x in g)
        typ = 'all_fail' if s == 0 else 'all_success' if s == len(g) else f'mixed_{s}of{len(g)}'
        for x in g:
            a = x['advantage']
            m = abs(a) * x['num_steps']
            mass[typ] += m
            tot_abs += m
            c = collections.Counter(x['actions'])
            for act in (FWD, LEFT, RIGHT, STOP):
                push[act] += a * c.get(act, 0)
                push_by[typ][act] += a * c.get(act, 0)
    names = {FWD: 'forward', LEFT: 'left', RIGHT: 'right', STOP: 'stop'}
    out['adv_mass_share'] = {k: round(v / tot_abs, 3) for k, v in sorted(mass.items())}
    out['action_push_per_1k_decision_mass'] = {names[k]: round(1000 * v / tot_abs, 2) for k, v in push.items()}
    out['action_push_by_group'] = {t: {names[k]: round(1000 * v / tot_abs, 2) for k, v in d.items()} for t, d in push_by.items()}
    # D. failure taxonomy of mains, early / late
    mains = [x for g in groups.values() for x in g]
    for lab, lo, hi in (('early_0_24', 0, 24), ('late_100_124', 100, 124), ('all', 0, 124)):
        sel = [x for x in mains if lo <= x['_step'] <= hi]
        k = collections.Counter(fail_kind(x) for x in sel)
        out[f'mains_{lab}'] = dict(n=len(sel), **{kk: round(v / len(sel), 3) for kk, v in sorted(k.items())},
                                   oracle=round(sum(bool(x.get('oracle_success')) for x in sel) / len(sel), 3),
                                   steps=round(sum(x['num_steps'] for x in sel) / len(sel), 1),
                                   turn_share=round(sum(a in (LEFT, RIGHT) for x in sel for a in x['actions']) /
                                                    sum(x['num_steps'] for x in sel), 3),
                                   steps_over_budget=round(sum(x['num_steps'] / x['budget'] for x in sel) / len(sel), 3))
    # E. STOP calibration on training prompts: mean P(stop) by distance band, early vs late
    def stop_cal(sel):
        bands = collections.defaultdict(list)
        for x in sel:
            for d, lp in zip(x['step_d'], x['step_stop_logp']):
                if d is None or lp is None:
                    continue
                b = '<1m' if d < 1 else '1-2m' if d < 2 else '2-3m' if d < 3 else '3-5m' if d < 5 else '>5m'
                bands[b].append(math.exp(lp))
        return {b: dict(mean_p_stop=round(sum(v) / len(v), 4), n=len(v)) for b, v in sorted(bands.items())}
    out['stop_calibration_early'] = stop_cal([x for x in mains if x['_step'] <= 24])
    out['stop_calibration_late'] = stop_cal([x for x in mains if x['_step'] >= 100])
    # F. rescue branches
    br = [b for x in mains for b in (x.get('fork_summary') or [])]
    nb = max(1, len(br))
    tk = collections.Counter(a for b in br for a in b.get('takeover_actions', []))
    imit_dec = sum(b['takeover'] for b in br if b.get('imitation', 0) > 0)
    imit_mass = sum(b['takeover'] * b.get('imitation', 0) for b in br)
    contrast_mass = sum(abs(b['coefficient']) * (b['num_steps'] - b['takeover']) for b in br)
    out['rescue'] = dict(branches_per_main=round(len(br) / len(mains), 3), success=round(sum(b['success'] for b in br) / nb, 3),
                         takeover_mean=round(sum(b['takeover'] for b in br) / nb, 2),
                         takeover_actions={names[k]: round(v / max(1, sum(tk.values())), 3) for k, v in tk.items()},
                         imitation_mass_vs_grpo=round(imit_mass / tot_abs, 3), contrast_mass_vs_grpo=round(contrast_mass / tot_abs, 3),
                         imitation_decisions_per_main=round(imit_dec / len(mains), 2))
    # G. trajectory shape: successes vs failures, relative to the shortest path (d0) and budget
    def shape(sel):
        if not sel:
            return {}
        return dict(n=len(sel), steps=round(sum(x['num_steps'] for x in sel) / len(sel), 1),
                    steps_per_d0=round(sum(x['num_steps'] / max(0.5, x['d0']) for x in sel) / len(sel), 2),
                    turn_share=round(sum(a in (LEFT, RIGHT) for x in sel for a in x['actions']) / sum(x['num_steps'] for x in sel), 3),
                    path_over_d0=round(sum(x['step_path'][-1] / max(0.5, x['d0']) for x in sel if x.get('step_path')) / len(sel), 2),
                    mean_adv=round(sum(x['advantage'] for x in sel) / len(sel), 3))
    out['shape_success'] = shape([x for x in mains if x['success']])
    out['shape_success_positive_adv'] = shape([x for x in mains if x['success'] and x['advantage'] > 0])
    out['shape_failure'] = shape([x for x in mains if not x['success']])
    # H. prompt mix
    out['rxr_share'] = round(sum(str(x['episode_id']).startswith('rxr-') for x in mains) / len(mains), 3)
    out['d0_mean'] = round(sum(x['d0'] for x in mains) / len(mains), 2)
    out['budget_mean'] = round(sum(x['budget'] for x in mains) / len(mains), 1)
    return out


if __name__ == '__main__':
    print(json.dumps([analyse(r) for r in sys.argv[1:]], indent=1))
