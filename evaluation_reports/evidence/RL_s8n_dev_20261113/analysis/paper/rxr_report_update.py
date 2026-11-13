"""Write the fixed RxR results into the consolidated report (s8n_rl_full_report_20261113.md), every value from the
fixed summaries (rxr_fixed_report_tables.py) and paired tests (rxr_fixed_compare.py). Each edit is an exact replacement
that must match exactly once.

python3 -I rxr_report_update.py GPU2_SUMMARIES.json REPORT.md TABLES.md
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rxr_fixed_report_tables as T  # noqa: E402

gpu2 = json.loads(Path(sys.argv[1]).read_text())
report = Path(sys.argv[2])
tables = Path(sys.argv[3]).read_text()
s = report.read_text()


def F(m, st='rxrfull'):
    d = T.fixed(m, st, gpu2)
    assert d, (m, st)
    return d


def p2(x):
    return f'{100 * x:.2f}'


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


C1, C2, G1 = F('rewind_sfr_mix1k_s125'), F('rewind_sfr_mix8k_aplus_s125'), F('grpo_sfr_mix8k_aplus_s125')
g2 = {125: F('rewind_sfr_mix8k_aplus_s125')}
cf = T.compare('grpo_sfr_mix8k_aplus_s125', 'rewind_sfr_mix1k_s125', 'rxrfull')
cA = T.compare('rewind_sfr_mix1k_s125', 'grpo_sfr_mix8k_aplus_s125', 'rxr')
c2k = T.compare('rewind_sfr_mix1k_s125', 'grpo_sfr_mix8k_aplus_s125', 'rxr2k')
d_full = 100 * cf['metrics']['SR']['abs_delta']
p_full = cf['paired_success']['mcnemar_exact_p']
ci_full = [round(100 * x, 2) for x in cf['bootstrap_95ci']['SR']]
sr12 = [100 * C1['sr'], 100 * C2['sr']]
mean12, sd12 = statistics.mean(sr12), statistics.stdev(sr12)
spl12 = statistics.mean([100 * C1['spl'], 100 * C2['spl']])
ndtw12 = statistics.mean([100 * C1['ndtw'], 100 * C2['ndtw']])


def pfmt(p):
    return f'{p:.2g}' if p >= 0.001 else f'{p:.0e}'.replace('e-0', 'e-')


# 1 summary
rep('On RxR-CE the gap is small: full RxR 71.08 vs 69.96 (+1.12, p 0.18).',
    f"On RxR-CE the gap is small: full RxR {p2(C1['sr'])} vs {p2(G1['sr'])} ({d_full:+.2f}, p {pfmt(p_full)}).")
# user 2026-11-13 22:5x: "Dont need to eval the 4k, 6k, etc. Just keep the 1k checkpoint eval": the 250-750 RxR numbers
# (old evaluations) are removed; those sentences keep their R2R numbers (Table R4)
rep('70.53 R2R; RxR full 70.59 /\n   71.19 / 71.22 / 71.00). GRPO reaches', '70.53 R2R). GRPO reaches')
rep('71.0 +- 1.3 R2R, 70.8 +- 0.3 RxR', f'71.0 +- 1.3 R2R, {mean12:.1f} +- {sd12:.1f} RxR')
# 2.4 evaluation
rep(', full\n  3,669. 98-99 RxR episodes have no finite goal distance in our evaluator and count as failures for every model.\n',
    ', full\n  3,669.\n')
# Table R5
rep('#### Table R5. RxR-CE val_unseen (English RxR15; unscorable episodes count as failures)',
    '#### Table R5. RxR-CE val_unseen (English RxR15)')
old_r5 = s[s.index('| Model | A SR / SPL | B SR / SPL |'):s.index('\nGreedy Stage-2 screens')]
new_r5 = tables[tables.index('| Model | A SR / SPL | B SR / SPL |'):tables.index('\n## Full-set details')].rstrip('\n') + '\n'
# keep only the rows of models the old table had (new seeds are added separately when their evaluations exist)
keep = [ln for ln in new_r5.splitlines() if ln.startswith('| Model') or ln.startswith('|---')
        or any(f'`{m}`' in ln for m in re.findall(r'^\| `([^`]+)`', old_r5, re.M))]
extra = [ln for ln in new_r5.splitlines() if ln not in keep]
rep(old_r5, '\n'.join(keep) + '\n' + ('\n'.join(extra) + '\n' if extra else ''))
# 4.1
m = cf['metrics']
rep('| mix1k (RxR full) | 69.96 / 57.67 | 71.08 / 57.50 | +1.12 | 0.18 | [-0.49, 2.70] |',
    f"| mix1k (RxR full) | {p2(m['SR']['base'])} / {p2(m['SPL']['base'])} | {p2(m['SR']['new'])} / {p2(m['SPL']['new'])} | "
    f"{d_full:+.2f} | {pfmt(p_full)} | [{ci_full[0]:.2f}, {ci_full[1]:.2f}] |")
for c, label, old in ((cA, 'RxR A', '| mix1k (RxR A) | 69.90 | 71.20 | +1.30 | 0.44 | - |'),
                      (c2k, 'RxR 2k', '| mix1k (RxR 2k) | 69.75 | 70.75 | +1.00 | 0.38 | - |')):
    mm = c['metrics']['SR']  # base = CoRe, new = GRPO in these files; the table lists GRPO | CoRe | CoRe - GRPO
    rep(old, f"| mix1k ({label}) | {p2(mm['new'])} | {p2(mm['base'])} | {-100 * mm['abs_delta']:+.2f} | "
             f"{pfmt(c['paired_success']['mcnemar_exact_p'])} | - |")
# 4.2
rep('RxR full 70.84 +- 0.35\n  (71.08, 70.59)', f"RxR full {mean12:.2f} +- {sd12:.2f}\n  ({p2(C1['sr'])}, {p2(C2['sr'])})")
rep('one run, 66.83 (RxR full 69.96)', f"one run, 66.83 (RxR full {p2(G1['sr'])})")
# 4.3
rep('70.53 R2R and 70.59 -> 71.19 -> 71.22 -> 71.00 RxR full: flat after 250.', '70.53 R2R: flat after 250.')
rep('SPL 63.91 and RxR nDTW 73.22 (vs 69.5 at 125), TL 15.68 m on RxR (vs 17.9)',
    'SPL 63.91 (vs 62.69 at 125) and TL 10.13 m on R2R (vs 11.11)')
# 6
mixA, bA = F('rewind_sfr_mix1k_s125', 'rxr'), F('rewind_sfr_s125', 'rxr')
mix2k, b2k = F('rewind_sfr_mix1k_s125', 'rxr2k'), F('rewind_sfr_s125', 'rxr2k')
rep('on RxR A 71.20 vs 71.80 and 2k\n  70.75 vs 70.70, i.e. no RxR loss',
    f"on RxR A {p2(mixA['sr'])} vs {p2(bA['sr'])} and 2k\n  {p2(mix2k['sr'])} vs {p2(b2k['sr'])}, i.e. no RxR loss")
# 10
rep('6k and RxR\n  full 71.19 / 71.22 / 71.00. Within that run', '6k. Within that run')
rep('Notes on Table R4: GRPO 250-750 were evaluated on R2R only (RxR deferred by the user);',
    'Notes on Table R4: the 250-750 checkpoints are reported on R2R only (RxR only at 125 updates, the user\'s choice);')
# 13
rep('RxR 70.8 +- 0.3 / 57.5 /\n    nDTW 69.5 (now all real)', f'RxR {mean12:.1f} +- {sd12:.1f} / {spl12:.1f} /\n    nDTW {ndtw12:.1f} (now all real)')
# appendix A: RxR rows
old_rows = [ln for ln in s.splitlines() if re.match(r'^\| `[^`]+` \| `[^`]+` \| rxr(2k|full)? \|', ln)]
assert len(old_rows) == 21, len(old_rows)
new_rows = [ln for ln in tables.splitlines() if re.match(r'^\| `[^`]+` \| `[^`]+` \| rxr(2k|full)? \|', ln)]
assert len(new_rows) == 21 and not any('missing' in ln for ln in new_rows), len(new_rows)
block_old = '\n'.join(old_rows)
assert block_old in s
rep(block_old, '\n'.join(new_rows))
# 15 files: where the RxR summaries are
rep('| merged journals | `/nvme/public/svln/evals/<model>_<split>/merged/` (gpu2: `~/workspace/anhdh35/svln/evals/`) |',
    '| merged journals | `/nvme/public/svln/evals/<model>_<split>/merged/`; RxR: `<model>_<set>_fixed_val_unseen/merged/` (gpu2: `~/workspace/anhdh35/svln/evals/`) |')
report.write_text(s)
print('report updated:', report)
