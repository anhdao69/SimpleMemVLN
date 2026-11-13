"""Generate every RxR number of the consolidated report from the fixed RxR results (rxr_fix_merge.py outputs).

python3 -I rxr_fixed_report_tables.py GPU2_SUMMARIES.json OUT.md
  local fixed results:  /nvme/public/svln/evals/<model>_<set>_fixed_val_unseen/merged/summary.json
  gpu2 fixed results:   GPU2_SUMMARIES.json (collected on gpu2 by collect_gpu2.py from the same kind of directories)
  paired tests:         rxr_fixed_compare.py, local results only, outputs in /nvme/public/svln/compare/*_fixed_val_unseen.json
Sets: rxr = A (first 1,000), rxrb_rxr = B (next 1,000), rxrc_rxr = C (remaining 1,669), rxr2k = A+B, rxrfull = all 3,669.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

E = Path('/nvme/public/svln/evals')
CMP = Path('/nvme/public/svln/compare')
HERE = Path(__file__).resolve().parent
SETS = ('rxr', 'rxrb_rxr', 'rxrc_rxr', 'rxr2k', 'rxrfull')
ROWS = [  # model, label, machine (Table R5 order)
    ('sfr', 'Stage 2', 'local'),
    ('rewind_sfr_s125', 'CoRe, R2R-only prompts (B)', 'local'),
    ('rewind_path_sfr_mix1k_s125', 'path-aware rewind, mix', 'local'),
    ('grpo_sfr_mix8k_aplus_s125', 'GRPO run 1, mix', 'local'),
    ('rewind_sfr_mix1k_s125', 'CoRe run 1, mix', 'local'),
    ('rewind_sfr_mix8k_aplus_s125', 'CoRe run 2 (gpu2), 125', 'gpu2'),
    ('grpo_sfr_mix1k_seed1414_s125', 'GRPO seed 20261414, mix', 'local'),
    ('rewind_sfr_mix1k_seed1414_s125', 'CoRe seed 20261414, mix', 'local'),
    ('rewind_sfr_mix1k_seed2_s125', 'CoRe run 3 (seed 20261111), mix', 'local'),
]
# paired tests of the report (base, new, set): the pairs of the old appendix A and section 4.1, same orientation
PAIRS = [(b, n, s) for s in ('rxr', 'rxr2k') for b, n in (
    ('rewind_path_sfr_mix1k_s125', 'grpo_sfr_mix8k_aplus_s125'), ('rewind_sfr_mix1k_s125', 'grpo_sfr_mix8k_aplus_s125'),
    ('rewind_sfr_mix1k_s125', 'rewind_path_sfr_mix1k_s125'), ('rewind_sfr_s125', 'grpo_sfr_mix8k_aplus_s125'),
    ('rewind_sfr_s125', 'rewind_path_sfr_mix1k_s125'), ('rewind_sfr_s125', 'rewind_sfr_mix1k_s125'),
    ('sfr', 'grpo_sfr_mix8k_aplus_s125'), ('sfr', 'rewind_path_sfr_mix1k_s125'), ('sfr', 'rewind_sfr_mix1k_s125'),
    ('sfr', 'rewind_sfr_s125'))] + [('grpo_sfr_mix8k_aplus_s125', 'rewind_sfr_mix1k_s125', 'rxrfull')]
KEYS = ('episodes', 'failures', 'sr', 'spl', 'oracle_success', 'ndtw', 'navigation_error', 'trajectory_length',
        'forced_stops', 'invalid_responses', 'mean_episode_steps')


def fixed(model, s, gpu2):
    if model in gpu2.get('summaries', {}) and s in gpu2['summaries'][model]:
        return gpu2['summaries'][model][s]
    f = E / f'{model}_{s}_fixed_val_unseen/merged/summary.json'
    if f.exists():
        d = json.loads(f.read_text())
        return {k: d.get(k) for k in KEYS}
    return None


def pct(x):
    return f'{100 * x:.2f}'


def compare(b, n, s):
    out = CMP / f'{b}_vs_{n}_{s}_fixed_val_unseen.json'
    if not out.exists():
        subprocess.run([sys.executable, '-I', str(HERE / 'rxr_fixed_compare.py'), str(E / f'{b}_{s}_fixed_val_unseen'),
                        str(E / f'{n}_{s}_fixed_val_unseen'), b, n, str(out)], check=True, cwd='/tmp',
                       stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main():
    gpu2 = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 2 else {}
    out = []
    w = out.append
    w('## Table R5 rows (SR / SPL; full: SR / SPL / nDTW)\n')
    w('| Model | A SR / SPL | B SR / SPL | C SR / SPL | 2k SR / SPL | full SR / SPL / nDTW | full OS |')
    w('|---|---|---|---|---|---|---|')
    for m, label, machine in ROWS:
        cells = []
        for s in SETS[:4]:
            d = fixed(m, s, gpu2)
            cells.append(f"{pct(d['sr'])} / {pct(d['spl'])}" if d else '-')
        d = fixed(m, 'rxrfull', gpu2)
        cells.append(f"{pct(d['sr'])} / {pct(d['spl'])} / {pct(d['ndtw'])}" if d else '-')
        cells.append(pct(d['oracle_success']) if d else '-')
        if any(c != '-' for c in cells):
            w(f"| `{m}` ({label}) | " + ' | '.join(cells) + ' |')
    w('\n## Full-set details (NE, TL, forced stops, failures, episodes)\n')
    for m, label, machine in ROWS:
        d = fixed(m, 'rxrfull', gpu2)
        if d:
            w(f"- `{m}`: SR {pct(d['sr'])} SPL {pct(d['spl'])} nDTW {pct(d['ndtw'])} OS {pct(d['oracle_success'])} "
              f"NE {d['navigation_error']:.3f} TL {d['trajectory_length']:.2f} forced {d['forced_stops']} "
              f"failures {d['failures']} invalid {d['invalid_responses']} episodes {d['episodes']} ({machine})")
    w('\n## CoRe runs 1 + 2 (seed 20261010), full RxR: mean +- sd (sample sd)\n')
    runs = [fixed('rewind_sfr_mix1k_s125', 'rxrfull', gpu2), fixed('rewind_sfr_mix8k_aplus_s125', 'rxrfull', gpu2)]
    if all(runs):
        for k in ('sr', 'spl', 'ndtw', 'oracle_success', 'navigation_error'):
            v = [r[k] * (1 if k == 'navigation_error' else 100) for r in runs]
            w(f'- {k}: {v[0]:.2f}, {v[1]:.2f} -> mean {statistics.mean(v):.3f} sd {statistics.stdev(v):.3f}')
    w('\n## Paired tests (local; base -> new)\n')
    w('| base | new | split | SR base | SR new | Delta SR | Delta SPL | only base | only new | McNemar p | 95% CI |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for b, n, s in PAIRS:
        if not all((E / f'{x}_{s}_fixed_val_unseen/merged/summary.json').exists() for x in (b, n)):
            w(f'| `{b}` | `{n}` | {s} | (missing) |')
            continue
        c = compare(b, n, s)
        m, p = c['metrics'], c['paired_success']
        ci = [round(100 * x, 2) for x in c['bootstrap_95ci']['SR']]
        w(f"| `{b}` | `{n}` | {s} | {pct(m['SR']['base'])} | {pct(m['SR']['new'])} | {100 * m['SR']['abs_delta']:+.2f} | "
          f"{100 * m['SPL']['abs_delta']:+.2f} | {p['only_base']} | {p['only_new']} | {p['mcnemar_exact_p']:.3g} | "
          f"[{ci[0]}, {ci[1]}] |")
    Path(sys.argv[-1]).write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
