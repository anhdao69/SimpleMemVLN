"""Assemble the S8N RL report tables (s8n_rl_report.md) from finished journals and training records only.

python scripts/rl/s8n_report.py [--out-dir plans/RL/research/results/s8n]

* every complete evaluation journal in EVALS: SR / SPL / NE / OSR / TL / forced stops / failures;
* paired comparisons (compare_evals.py, which refuses mismatched protocols or evaluation sources) for PAIRS
  whose two journals both exist. F-B + start tilt (snapshot v4) against F-B (snapshot v2) is compared with
  --allow-source-diff, and only if base S8N gives the same journal on both snapshots (`code_path_check`);
* Holm-adjusted McNemar p over the three retention probe arms (decision D1);
* training compute from metrics.jsonl with the Window8/M5 report definitions (last record of a step wins;
  records of steps re-run after a resume are counted separately as rerun_*): simulator steps = main +
  sampled branch + sampled replay + twins, policy forward steps = main + sampled branch + twins;
* retention analyses on F-B's step 0-4 rollouts (policy ~ base S8N): per-layer frame sources kept at episode
  end, and how much tilted retention rules change S8N's selection (offline replay of the logged drift);
* SR by instruction-length tercile;
* the memory-twins pilot (per-step twin statistics and the learned tilt).
Journals or runs that do not exist are listed as pending; nothing is filled in for them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path('/blue/yu954774.ucf/an221229.ucf/svln_runs')
HERE = Path(__file__).resolve().parent
STEPS = (25, 50, 75, 100, 125)
PROBES = ('balanced', 'recency', 'start')
EVALS = {   # label: journal directory under R/evals (the merged journal is DIR/merged); seen = val_seen-300
    'unseen_base': 's8n_base/val_unseen',
    'unseen_grpo': 'activevln_grpo_s8n_r2r1k/step_0125/val_unseen',
    'unseen_fb': 'hico_grpo_fork_bs_s8n_r2r1k/step_0125/val_unseen',
    'unseen_twins_grpo': 'memory_twins_grpo_s8n_r2r1k/step_0125/val_unseen',
    'unseen_grpo_g5': 'activevln_grpo_g5_s8n_r2r1k/step_0125/val_unseen',
    'unseen_start': 'probes/s8n_tilt_start/val_unseen',
    'seen_base': 's8n_base/val_seen300_w6',
    'seen_base_v4': 's8n_base/val_seen300_v4',
    **{f'seen_grpo_s{s}': f'activevln_grpo_s8n_r2r1k/step_{s:04d}/val_seen300' for s in STEPS},
    **{f'seen_fb_s{s}': f'hico_grpo_fork_bs_s8n_r2r1k/step_{s:04d}/val_seen300' for s in STEPS},
    'seen_twins_grpo_s125': 'memory_twins_grpo_s8n_r2r1k/step_0125/val_seen300',
    **{f'seen_{a}': f'probes/s8n_tilt_{a}/val_seen300' for a in PROBES},
    'seen_fb_start': 'probes/fb125_tilt_start/val_seen300',
}
PAIRS = ([('unseen_base', x) for x in ('unseen_grpo', 'unseen_fb', 'unseen_twins_grpo', 'unseen_grpo_g5',
                                        'unseen_start')]
         + [('unseen_grpo', 'unseen_fb'), ('unseen_grpo', 'unseen_grpo_g5'), ('unseen_grpo', 'unseen_twins_grpo'),
            ('unseen_grpo_g5', 'unseen_twins_grpo'), ('unseen_fb', 'unseen_twins_grpo')]
         + [('seen_base', f'seen_{m}_s{s}') for m in ('grpo', 'fb') for s in STEPS]
         + [(f'seen_grpo_s{s}', f'seen_fb_s{s}') for s in STEPS]
         + [('seen_base', 'seen_twins_grpo_s125'), ('seen_grpo_s125', 'seen_twins_grpo_s125')]
         + [('seen_base_v4', f'seen_{a}') for a in PROBES]
         + [('seen_fb_s125', 'seen_fb_start')])
CROSS_SOURCE = {('seen_fb_s125', 'seen_fb_start')}   # only if code_path_check('seen_base', 'seen_base_v4') holds
RUNS = {   # label: run directory under R/rl
    'grpo': 'activevln_grpo_s8n_r2r1k',
    'fb': 'hico_grpo_fork_bs_s8n_r2r1k',
    'twins_fb_pilot': 'memory_twins_s8n_r2r1k',
    'twins_grpo': 'memory_twins_grpo_s8n_r2r1k',
    'grpo_g5': 'activevln_grpo_g5_s8n_r2r1k',
}
TILT_ARMS = [(0, 0), (0, 0.1), (0, 0.3), (0, 0.6), (0, 1.0), (0, 2.0), (0, -0.3), (0, -2.0), (1, 0), (1, 0.3)]


def eval_dir(label):
    return R / 'evals' / EVALS[label] / 'merged'


def complete(label):
    path = eval_dir(label) / 'summary.json'
    return path.exists() and json.loads(path.read_text()).get('complete') is True


def journal(label):
    return {r['episode_id']: r for r in map(json.loads, (eval_dir(label) / 'episodes.jsonl').read_text().splitlines())}


TIMING_KEYS = ('model_seconds', 'transport_and_model_seconds', 'environment_seconds')


def code_path_check(a, b):
    """Two journals of the same weights on two code snapshots: is every step record (response tokens, executed
    action, retained KV tokens, ...; wall-clock timings excluded) and every episode metric identical?"""
    ja, jb = journal(a), journal(b)
    steps = lambda e: [{k: v for k, v in x.items() if k not in TIMING_KEYS} for x in e['actions']]
    ids = sorted(ja)
    return dict(episodes=len(ids), same_episodes=set(ja) == set(jb), ignored_keys=list(TIMING_KEYS),
                identical_actions=sum(i in jb and steps(ja[i]) == steps(jb[i]) for i in ids),
                identical_metrics=sum(ja[i]['metrics'] == jb.get(i, {}).get('metrics') for i in ids))


def compute(label):
    path = R / 'rl' / RUNS[label] / 'metrics.jsonl'
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines()]
    by_step = {}
    for record in records:
        by_step[record['step']] = record
    kept = list(by_step.values())
    rerun = [r for r in records if by_step[r['step']] is not r]
    total = lambda rs, k: sum(r.get(k) or 0 for r in rs)
    sim = lambda rs: (total(rs, 'env_steps') + total(rs, 'sampled_branch_env_steps')
                      + total(rs, 'sampled_replay_env_steps') + total(rs, 'twin_env_steps'))
    window = lambda lo, hi, k: (float(np.mean([r[k] for r in kept if lo <= r['step'] < hi]))
                                if any(lo <= r['step'] < hi for r in kept) else None)
    return dict(updates=len(by_step), last_step=max(by_step), complete=(path.parent / 'COMPLETE').exists(),
                main_rollouts=total(kept, 'trajectories'), discarded=total(kept, 'discarded_trajectories'),
                trained_sequences=total(kept, 'trained_trajectories'), forks=total(kept, 'branches'),
                main_env_steps=total(kept, 'env_steps'),
                sampled_branch_env_steps=total(kept, 'sampled_branch_env_steps'),
                sampled_replay_env_steps=total(kept, 'sampled_replay_env_steps'),
                twin_env_steps=total(kept, 'twin_env_steps'), simulator_steps=sim(kept),
                policy_forward_steps=(total(kept, 'env_steps') + total(kept, 'sampled_branch_env_steps')
                                      + total(kept, 'twin_env_steps')),
                wall_hours=round(total(kept, 'step_seconds') / 3600, 2),
                rerun_records=len(rerun), rerun_simulator_steps=sim(rerun),
                rerun_wall_hours=round(total(rerun, 'step_seconds') / 3600, 2),
                train_sr_first25=window(0, 25, 'success_rate'), train_sr_last25=window(100, 125, 'success_rate'),
                truncation_first25=window(0, 25, 'truncation_rate'), truncation_last25=window(100, 125, 'truncation_rate'))


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def fb_rollouts(min_steps):
    run = R / 'rl' / RUNS['fb'] / 'rollouts'
    for path in sorted(run.glob('step_000[0-4]_rank*.jsonl')):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['num_steps'] >= min_steps:
                yield e


def bank_composition():
    """Frame sources each attention layer keeps at the end of F-B step 0-4 episodes with >= 30 decisions."""
    early, total, relative, n = {}, {}, {}, 0
    for e in fb_rollouts(30):
        banks = e['memory']['layer_banks']
        if not banks:
            continue
        n += 1
        for layer, sources in banks[max(banks, key=int)].items():
            total[layer] = total.get(layer, 0) + len(sources)
            early[layer] = early.get(layer, 0) + sum(s <= 3 for s in sources)
            relative.setdefault(layer, []).extend(s / e['num_steps'] for s in sources)
    return dict(episodes=n, layers={layer: dict(share_from_t_le_3=round(early[layer] / total[layer], 3),
                                                median_source_t_over_n=round(float(np.median(relative[layer])), 3))
                                    for layer in sorted(total, key=int)})


def tilt_sensitivity(recent=4, frames=4):
    """Offline: frames kept at episode end by priority (1-a) log p^l + a mean_l log p + b log(1+s) vs S8N's."""
    logs = [np.log(np.asarray(e['step_drift'], dtype=float)) for e in fb_rollouts(20) if all(e['step_drift'])]

    def banks(logp, a, b):
        eligible = logp.shape[0] - recent
        priority = ((1 - a) * logp[:eligible] + a * logp[:eligible].mean(1, keepdims=True)
                    + b * np.log1p(np.arange(eligible))[:, None])
        return [set(np.lexsort((np.arange(eligible), -priority[:, l]))[:frames].tolist())
                for l in range(logp.shape[1])]

    arms = {}
    for a, b in TILT_ARMS:
        overlap, early, position = [], [], []
        for logp in logs:
            eligible = logp.shape[0] - recent
            for base, tilted in zip(banks(logp, 0, 0), banks(logp, a, b)):
                overlap.append(len(base & tilted) / frames)
                early.append(np.mean([s <= 3 for s in tilted]))
                position.append(np.mean([s / eligible for s in tilted]))
        arms[f'a={a} b={b:+.1f}'] = dict(overlap_with_s8n=round(float(np.mean(overlap)), 3),
                                         share_from_t_le_3=round(float(np.mean(early)), 3),
                                         mean_source_position=round(float(np.mean(position)), 3))
    return dict(episodes=len(logs), arms=arms)


def by_instruction_length(labels):
    """SR per instruction-word tercile (terciles of the first journal) on the episodes all journals share."""
    journals = {label: journal(label) for label in labels}
    ids = sorted(set.intersection(*(set(j) for j in journals.values())))
    first = journals[labels[0]]
    words = np.asarray([len(first[i]['instruction'].split()) for i in ids])
    q = np.quantile(words, [1 / 3, 2 / 3])
    bins = dict(short=words <= q[0], mid=(words > q[0]) & (words <= q[1]), long=words > q[1])
    out = dict(episodes=len(ids), tercile_bounds_words=[float(x) for x in q], sr={})
    for label, j in journals.items():
        s = np.asarray([0.0 if j[i]['failed'] else float(j[i]['metrics']['success']) for i in ids])
        out['sr'][label] = dict(all=round(100 * s.mean(), 2),
                                **{k: round(100 * float(s[m].mean()), 2) for k, m in bins.items()})
    return out


def twins_pilot(label):
    run = R / 'rl' / RUNS[label]
    path = run / 'metrics.jsonl'
    if not path.exists():
        return None
    by_step = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        by_step[record['step']] = record
    rows = [by_step[s] for s in sorted(by_step)]
    keys = ('twin_pairs', 'twin_diverged', 'twin_success_flip', 'twin_first_divergence', 'twin_reward_gap_abs',
            'tilt_birth_mean', 'tilt_consensus_mean', 'tilt_grad_norm', 'success_rate')
    pairs = sum(r['twin_pairs'] for r in rows)
    cfg = json.loads((run / 'run_config.json').read_text())['config']
    return dict(steps=len(rows), memory_tilt=cfg.get('memory_tilt'), pairs=pairs,
                diverged=round(sum(r['twin_diverged'] * r['twin_pairs'] for r in rows) / pairs, 3),
                success_flip=round(sum(r['twin_success_flip'] * r['twin_pairs'] for r in rows) / pairs, 3),
                diverged_range=[min(r['twin_diverged'] for r in rows), max(r['twin_diverged'] for r in rows)],
                per_step=[{'step': r['step'], **{k: r.get(k) for k in keys}} for r in rows],
                final_tilt=dict(birth=rows[-1]['tilt_birth'], consensus=rows[-1]['tilt_consensus']))


def pct(x):
    return f'{100 * x:.2f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=str(HERE.parents[1] / 'plans/RL/research/results/s8n'))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = [x for x in EVALS if complete(x)]
    report = dict(done=done, pending=[x for x in EVALS if x not in done], results={}, paired={}, compute={})
    for label in done:
        s = json.loads((eval_dir(label) / 'summary.json').read_text())
        report['results'][label] = dict(journal=str(eval_dir(label)),
                                        **{k: s.get(k) for k in ('sr', 'spl', 'navigation_error', 'oracle_success',
                                                                 'trajectory_length', 'forced_stops', 'failures',
                                                                 'mean_episode_steps', 'episodes')})
        failed = {i: e.get('failure_reason') for i, e in journal(label).items() if e['failed']}
        if failed:
            report['results'][label]['failure_reasons'] = failed
    if {'seen_base', 'seen_base_v4'} <= set(done):
        report['code_path_check'] = check = code_path_check('seen_base', 'seen_base_v4')
        same_code_path = check['same_episodes'] and check['identical_actions'] == check['episodes']
    else:
        same_code_path = False
    for base, new in PAIRS:
        if base not in done or new not in done:
            continue
        extra = []
        if (base, new) in CROSS_SOURCE:
            if not same_code_path:
                continue
            extra = ['--allow-source-diff']
        target = out / f'{new}_vs_{base}_comparison.json'
        subprocess.run([sys.executable, str(HERE / 'compare_evals.py'), '--base', str(eval_dir(base)),
                        '--new', str(eval_dir(new)), '--name-base', base, '--name-new', new, '--out', str(target),
                        *extra], check=True, stdout=subprocess.DEVNULL)
        c = json.loads(target.read_text())
        report['paired'][f'{new} vs {base}'] = dict(
            episodes=c['episodes'], dSR=c['metrics']['SR']['abs_delta'], dSPL=c['metrics']['SPL']['abs_delta'],
            dNE=c['metrics']['NE']['abs_delta'], dOSR=c['metrics']['OSR']['abs_delta'],
            ci_SR=c['bootstrap_95ci']['SR'], ci_SPL=c['bootstrap_95ci']['SPL'],
            gained=c['paired_success']['only_new'], lost=c['paired_success']['only_base'],
            mcnemar_p=c['paired_success']['mcnemar_exact_p'], source_diff=c.get('source_diff', []))
    family = {k: v['mcnemar_p'] for k, v in report['paired'].items() if k.endswith(' vs seen_base_v4')}
    for k, adj in holm(family).items():
        report['paired'][k]['holm_p'] = adj
    for label in RUNS:
        c = compute(label)
        if c:
            report['compute'][label] = c
    report['retention'] = dict(bank_composition=bank_composition(), tilt_sensitivity=tilt_sensitivity())
    report['instruction_length'] = {
        split: by_instruction_length([x for x in labels if x in done])
        for split, labels in (('val_unseen', ['unseen_base', 'unseen_grpo', 'unseen_fb', 'unseen_twins_grpo']),
                              ('val_seen300', ['seen_base', 'seen_start', 'seen_grpo_s125', 'seen_fb_s125',
                                               'seen_fb_start']))}
    report['twins_pilot'] = twins_pilot('twins_fb_pilot')
    (out / 's8n_report.json').write_text(json.dumps(report, indent=2) + '\n')

    # summary.json stores SR / SPL / OSR as fractions; tables print percentages.
    lines = ['| Journal | episodes | SR | SPL | NE | OSR | TL | forced stops | failures |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, r in report['results'].items():
        lines.append(f"| {label} | {r['episodes']} | {pct(r['sr'])} | {pct(r['spl'])} | {r['navigation_error']:.2f} | "
                     f"{pct(r['oracle_success'])} | {r['trajectory_length']:.2f} | {r['forced_stops']} | "
                     f"{r['failures']} |")
    lines += ['', '| Run | updates | main rollouts (+discarded) | trained seqs | forks | main env steps | '
                  'simulator steps | policy forward steps | wall h | re-run (resume) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for label, c in report['compute'].items():
        rerun = (f"{c['rerun_records']} steps, {c['rerun_simulator_steps'] / 1e3:.1f} k sim, {c['rerun_wall_hours']} h"
                 if c['rerun_records'] else '—')
        lines.append(f"| {label} | {c['updates']}{'' if c['complete'] else ' (stopped)'} | {c['main_rollouts']:,} "
                     f"(+{c['discarded']:,}) | {c['trained_sequences']:,} | {c['forks']:,} | "
                     f"{c['main_env_steps'] / 1e3:.0f} k | {c['simulator_steps'] / 1e3:.0f} k | "
                     f"{c['policy_forward_steps'] / 1e3:.0f} k | {c['wall_hours']} | {rerun} |")
    lines += ['', '| new vs base | n | dSR [95% CI] | dSPL [95% CI] | dNE | dOSR | gained / lost | McNemar p | Holm p |',
              '|---|---:|---|---|---:|---:|---|---:|---:|']
    for key, d in report['paired'].items():
        holm_p = f"{d['holm_p']:.3g}" if 'holm_p' in d else '—'
        note = ' (sources differ, see code_path_check)' if d['source_diff'] else ''
        lines.append(f"| {key}{note} | {d['episodes']} | {100 * d['dSR']:+.2f} "
                     f"[{100 * d['ci_SR'][0]:+.1f}, {100 * d['ci_SR'][1]:+.1f}] | {100 * d['dSPL']:+.2f} "
                     f"[{100 * d['ci_SPL'][0]:+.1f}, {100 * d['ci_SPL'][1]:+.1f}] | {d['dNE']:+.2f} | "
                     f"{100 * d['dOSR']:+.2f} | {d['gained']} / {d['lost']} | {d['mcnemar_p']:.3g} | {holm_p} |")
    if 'code_path_check' in report:
        lines += ['', f"code_path_check seen_base (v2) vs seen_base_v4 (v4): {report['code_path_check']}"]
    lines += ['', 'pending: ' + (', '.join(report['pending']) or 'none')]
    (out / 's8n_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
