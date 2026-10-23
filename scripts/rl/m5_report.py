"""Assemble the M5 RL report tables from finished runs only (memfork_report.md §3-§4).

python scripts/rl/m5_report.py [--out-dir plans/RL/research/results/m5]

For every run in RUNS whose step-125 val_unseen journal is complete:
  * the val_unseen row (SR / SPL / NE / OSR / TL / forced stops / failures);
  * compute from the training records (last record of a step wins), with the Window8 report's definitions:
    simulator steps = main + sampled branch + sampled replay, policy forward steps = main + sampled branch
    (replay is teacher-forced, no sampling);
  * paired comparisons (compare_evals.py, which refuses mismatched protocols or evaluation sources) for
    PAIRS whose two journals both exist, with Holm-adjusted McNemar p over the MemFork family;
  * analyze_memfork.py for MemFork-family runs.
Runs without a complete journal are listed as pending; nothing is filled in for them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

R = Path('/data/anh/rl_baseline')
HERE = Path(__file__).resolve().parent
RUNS = {   # label: (run name, eval step)
    'm5_e1': ('m5_e1_base', 0),
    'grpo': ('grpo_m5_r2r1k', 125),
    'grpo_fb': ('grpo_fork_bs_m5_r2r1k', 125),
    'memfork': ('memfork_m5_r2r1k', 125),
    'a3_actfork': ('memfork_actfork_m5_r2r1k', 125),
    'a2_memfork_grpo': ('memfork_grpo_m5_r2r1k', 125),
    'a1_frame_greedy_fb': ('frame_greedy_fb_m5_r2r1k', 125),
    'grpo_g8': ('grpo_g8_m5_r2r1k', 125),
}
MEMFORK_FAMILY = ('memfork', 'a3_actfork', 'a2_memfork_grpo')
PAIRS = ([('m5_e1', x) for x in RUNS if x != 'm5_e1']
         + [('grpo', 'grpo_fb'), ('grpo', 'grpo_g8'), ('grpo_g8', 'grpo_fb')]
         + [(b, 'memfork') for b in ('grpo', 'grpo_fb', 'grpo_g8', 'a1_frame_greedy_fb', 'a2_memfork_grpo',
                                      'a3_actfork')])


def eval_dir(label):
    run, step = RUNS[label]
    return R / 'evals' / f'{run}-step{step:04d}' / 'val_unseen'


def complete(label):
    path = eval_dir(label) / 'summary.json'
    return path.exists() and json.loads(path.read_text()).get('complete') is True


def compute(label):
    path = R / 'runs' / RUNS[label][0] / 'metrics.jsonl'
    if not path.exists():
        return None
    by_step = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        by_step[record['step']] = record
    total = lambda k: sum(r.get(k) or 0 for r in by_step.values())
    main = total('env_steps')
    return dict(updates=len(by_step), main_rollouts=total('trajectories'), discarded=total('discarded_trajectories'),
                trained_sequences=total('trained_trajectories'), main_env_steps=main,
                simulator_steps=main + total('sampled_branch_env_steps') + total('sampled_replay_env_steps'),
                policy_forward_steps=main + total('sampled_branch_env_steps'),
                trained_tokens=total('trained_tokens'),
                wall_hours=round(total('step_seconds') / 3600, 2))


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=str(HERE.parents[1] / 'plans/RL/research/results/m5'))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = [x for x in RUNS if complete(x)]
    report = dict(done=done, pending=[x for x in RUNS if x not in done], results={}, compute={}, paired={})
    for label in done:
        s = json.loads((eval_dir(label) / 'summary.json').read_text())
        report['results'][label] = {k: s.get(k) for k in ('sr', 'spl', 'navigation_error', 'oracle_success',
                                                          'trajectory_length', 'forced_stops', 'failures',
                                                          'mean_episode_steps', 'episodes')}
        retries = sorted(eval_dir(label).parent.glob('shard*/infra_retries.jsonl'))
        report['results'][label]['infra_retries'] = sum(len(f.read_text().splitlines()) for f in retries)
        if label != 'm5_e1':
            report['compute'][label] = compute(label)
    for base, new in PAIRS:
        if base not in done or new not in done:
            continue
        target = out / f'{new}_vs_{base}_comparison.json'
        subprocess.run([sys.executable, str(HERE / 'compare_evals.py'), '--base', str(eval_dir(base)),
                        '--new', str(eval_dir(new)), '--name-base', base, '--name-new', new, '--out', str(target)],
                       check=True, stdout=subprocess.DEVNULL)
        c = json.loads(target.read_text())
        report['paired'][f'{new}_vs_{base}'] = dict(
            dSR=c['metrics']['SR']['abs_delta'], dSPL=c['metrics']['SPL']['abs_delta'],
            dNE=c['metrics']['NE']['abs_delta'], dOSR=c['metrics']['OSR']['abs_delta'],
            ci_SR=c['bootstrap_95ci']['SR'], ci_SPL=c['bootstrap_95ci']['SPL'],
            gained=c['paired_success']['only_new'], lost=c['paired_success']['only_base'],
            mcnemar_p=c['paired_success']['mcnemar_exact_p'])
    family = {k: v['mcnemar_p'] for k, v in report['paired'].items() if k.startswith('memfork_vs_')}
    for k, adj in holm(family).items():
        report['paired'][k]['holm_p'] = adj
    for label in MEMFORK_FAMILY:
        run = R / 'runs' / RUNS[label][0]
        if (run / 'COMPLETE').exists():
            target = out / f'{label}_memory_behaviour.json'
            subprocess.run([sys.executable, str(HERE / 'analyze_memfork.py'), str(run), '--out', str(target)],
                           check=True, stdout=subprocess.DEVNULL)
            report.setdefault('memory_behaviour', {})[label] = str(target)
    (out / 'm5_report.json').write_text(json.dumps(report, indent=2) + '\n')

    # summary.json stores SR / SPL / OSR as fractions; tables print percentages.
    lines = ['| Policy | SR | SPL | NE | OSR | TL | forced stops | failures | infra retries |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, r in report['results'].items():
        lines.append(f"| {label} | {100 * r['sr']:.2f} | {100 * r['spl']:.2f} | {r['navigation_error']:.2f} | "
                     f"{100 * r['oracle_success']:.2f} | {r['trajectory_length']:.2f} | {r['forced_stops']} | "
                     f"{r['failures']} | {r['infra_retries']} |")
    lines += ['', '| Run | updates | main rollouts (+discarded) | trained seqs | simulator steps | '
                  'policy forward steps | wall h |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, c in report['compute'].items():
        if c:
            lines.append(f"| {label} | {c['updates']} | {c['main_rollouts']:,} (+{c['discarded']:,}) | "
                         f"{c['trained_sequences']:,} | {c['simulator_steps'] / 1e3:.0f} k | "
                         f"{c['policy_forward_steps'] / 1e3:.0f} k | {c['wall_hours']} |")
    lines += ['', '| new vs base | dSR [95% CI] | dSPL [95% CI] | dNE | dOSR | gained / lost | McNemar p | Holm p |',
              '|---|---|---|---:|---:|---|---:|---:|']
    for key, d in report['paired'].items():
        holm_p = f"{d['holm_p']:.3g}" if 'holm_p' in d else '—'
        lines.append(f"| {key.replace('_vs_', ' vs ')} | {100 * d['dSR']:+.2f} [{100 * d['ci_SR'][0]:+.1f}, "
                     f"{100 * d['ci_SR'][1]:+.1f}] | {100 * d['dSPL']:+.2f} [{100 * d['ci_SPL'][0]:+.1f}, "
                     f"{100 * d['ci_SPL'][1]:+.1f}] | {d['dNE']:+.2f} | {100 * d['dOSR']:+.2f} | "
                     f"{d['gained']} / {d['lost']} | {d['mcnemar_p']:.3g} | {holm_p} |")
    lines += ['', 'pending: ' + (', '.join(report['pending']) or 'none')]
    (out / 'm5_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
