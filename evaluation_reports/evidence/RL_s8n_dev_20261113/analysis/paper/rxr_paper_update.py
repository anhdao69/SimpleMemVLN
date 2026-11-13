"""Write the fixed RxR results into the paper (/nvme/public/overleaf_glance, branch rl-revision). Every value is computed
from the fixed summaries; every edit is an exact replacement that must match exactly once.

python3 -I rxr_paper_update.py GPU2_SUMMARIES.json PAPER_DIR

Real values: CoRe run 1 (rewind_sfr_mix1k_s125), CoRe run 2 (gpu2 rewind_sfr_mix8k_aplus_s125), GRPO run 1
(grpo_sfr_mix8k_aplus_s125), full RxR val_unseen. Red placeholders that were set from the old RxR numbers keep their
offsets to the real numbers they were derived from (GRPO run 2 and the GRPO mean: from GRPO run 1; Table source: from CoRe
run 1; Stage 1/2 RxR: from Stage 2's sets A+B, shifted by Stage 2's own change on those 2,000 episodes).
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rxr_fixed_report_tables as T  # noqa: E402

gpu2 = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
files = {}


def get(rel):
    if rel not in files:
        files[rel] = (root / rel).read_text()
    return files[rel]


def rep(rel, old, new):
    s = get(rel)
    n = s.count(old)
    assert n == 1, (rel, n, old[:90])
    files[rel] = s.replace(old, new)


def F(m, st='rxrfull'):
    d = T.fixed(m, st, gpu2)
    assert d, (m, st)
    return d


def r1(x):  # percent, one decimal
    return f'{100 * x:.1f}'


C1, C2, G1 = F('rewind_sfr_mix1k_s125'), F('rewind_sfr_mix8k_aplus_s125'), F('grpo_sfr_mix8k_aplus_s125')
assert C1['failures'] == 0 and C2['failures'] == 0 and C1['episodes'] == C2['episodes'] == G1['episodes'] == 3669
sr = [100 * C1['sr'], 100 * C2['sr']]
core_sr, core_sd = statistics.mean(sr), statistics.stdev(sr)
core_spl = statistics.mean([100 * C1['spl'], 100 * C2['spl']])
core_ndtw = statistics.mean([100 * C1['ndtw'], 100 * C2['ndtw']])
# GRPO placeholders: old run 2 = run 1 - 0.4 (SR and SPL), old mean nDTW = run 1 nDTW - 0.09 (69.39 -> 69.3)
g1_sr, g1_spl, g1_ndtw = round(100 * G1['sr'], 1), round(100 * G1['spl'], 1), 100 * G1['ndtw']
g2_sr, g2_spl = round(g1_sr - 0.4, 1), round(g1_spl - 0.4, 1)
grpo_sr, grpo_spl, grpo_ndtw = (g1_sr + g2_sr) / 2, (g1_spl + g2_spl) / 2, g1_ndtw - 0.09
grpo_sd = abs(g1_sr - g2_sr) / 2 ** 0.5
# Stage 1/2 placeholders: shift by Stage 2's change on sets A+B (2,000 episodes)
old2k, new2k = json.loads(Path('/nvme/public/svln/evals/sfr_rxr2k_val_unseen/merged/summary.json').read_text()), F('sfr', 'rxr2k')
dsr, dspl, dndtw = (100 * (new2k[k] - old2k[k]) for k in ('sr', 'spl', 'ndtw'))
SACA = 60.3   # best previous RxR-CE SR without external data (SACA, Table main)


def f1(x):
    return f'{x:.1f}'


vals = dict(core=(f1(core_sr), f1(core_sd), f1(core_spl), f1(core_ndtw)), run1=(r1(C1['sr']), r1(C1['spl']), r1(C1['ndtw']),
            f"{C1['navigation_error']:.2f}"), run2=(r1(C2['sr']), r1(C2['spl'])), grpo1=(f1(g1_sr), f1(g1_spl)),
            grpo2_pseudo=(f1(g2_sr), f1(g2_spl)), grpo_mean_pseudo=(f1(grpo_sr), f1(grpo_sd), f1(grpo_spl), f1(grpo_ndtw)),
            stage_shift=(round(dsr, 2), round(dspl, 2), round(dndtw, 2)))
print(json.dumps(vals, indent=1))
cs, csd, cspl, cnd = vals['core']
s1sr, s1spl, s1nd, s1ne = vals['run1']

# ---------------- tab/main.tex ----------------
assert float(s1sr) > 73.4 and float(s1sr) < 75.6 and float(s1nd) > 70.0 and float(s1nd) < 75.2 and 3.18 < float(s1ne) < 4.09
assert float(s1spl) < 60.0
rep('tab/main.tex', r'& \chgcell\underline{3.62} & 71.1 & 57.5 & \chgcell 69.5 & 0\\',
    rf'& \chgcell\underline{{{s1ne}}} & \chgcell\underline{{{s1sr}}} & \chgcell {s1spl} & \chgcell\underline{{{s1nd}}} & 0\\')
rep('tab/main.tex', r'& 4.16 & \underline{73.4} & \textbf{63.5} & 69.9 & 15,600K\\',
    r'& 4.16 & \chgcell 73.4 & \textbf{63.5} & 69.9 & 15,600K\\')
rep('tab/main.tex', r'& 4.58 & 61.4 & 51.8 & \underline{70.0} & n/r\\', r'& 4.58 & 61.4 & 51.8 & \chgcell 70.0 & n/r\\')
rep('tab/main.tex', r'and 70.8\pmsd{0.3} on RxR-CE (\cref{tab:rl}). Yellow cells: corrected marks (RxR-CE NE: \method is second best) and citation key.',
    rf'and {cs}\pmsd{{{csd}}} on RxR-CE (\cref{{tab:rl}}). Yellow cells: changed RxR-CE values and marks, and citation key.')

# ---------------- tab/rl.tex ----------------
rep('tab/rl.tex', '%             GRPO run 1 (R2R 66.83/60.01; RxR 69.96/57.67/69.39);\n%             CoRe RxR run 1 71.08/57.50/69.51, run 2 70.59/57.44.',
    f"%             GRPO run 1 (R2R 66.83/60.01; RxR {100 * G1['sr']:.2f}/{100 * G1['spl']:.2f}/{100 * G1['ndtw']:.2f});\n"
    f"%             CoRe RxR run 1 {100 * C1['sr']:.2f}/{100 * C1['spl']:.2f}/{100 * C1['ndtw']:.2f}, run 2 "
    f"{100 * C2['sr']:.2f}/{100 * C2['spl']:.2f}/{100 * C2['ndtw']:.2f}.\n"
    f"%             Red RxR cells keep their offsets to the real values: GRPO mean from GRPO run 1;\n"
    f"%             Stages 1/2 shifted by Stage 2's change on RxR sets A+B ({dsr:+.2f} SR, {dspl:+.2f} SPL, {dndtw:+.2f} nDTW).")
sh = lambda x, d: f1(x + d)
rep('tab/rl.tex', r'Stage 1: episode-level SFT & 52.0 & 48.3 & \pseudo{52.9} & \pseudo{45.0} & \pseudo{60.1}\\',
    rf'Stage 1: episode-level SFT & 52.0 & 48.3 & \pseudo{{{sh(52.9, dsr)}}} & \pseudo{{{sh(45.0, dspl)}}} & \pseudo{{{sh(60.1, dndtw)}}}\\')
rep('tab/rl.tex', r'Stage 2: + DAgger & 62.6 & 55.6 & \pseudo{63.8} & \pseudo{52.4} & \pseudo{64.6}\\',
    rf'Stage 2: + DAgger & 62.6 & 55.6 & \pseudo{{{sh(63.8, dsr)}}} & \pseudo{{{sh(52.4, dspl)}}} & \pseudo{{{sh(64.6, dndtw)}}}\\')
gm = vals['grpo_mean_pseudo']
b = lambda mine, other: (r'\textbf{' + mine + '}') if float(mine) >= float(other) else mine
rep('tab/rl.tex', r'& \pseudo{69.8\pmsd{0.3}} & \pseudo{\textbf{57.5}} & \pseudo{69.3}\\',
    rf"& \pseudo{{{b(gm[0], cs)}\pmsd{{{gm[1]}}}}} & \pseudo{{{b(gm[2], cspl)}}} & \pseudo{{{b(gm[3], cnd)}}}\\")
rep('tab/rl.tex', r'& \textbf{70.8}\pmsd{0.3} & \textbf{57.5} & \textbf{69.5}\\',
    rf'& {b(cs, gm[0])}\pmsd{{{csd}}} & {b(cspl, gm[2])} & {b(cnd, gm[3])}\\')

# ---------------- sec/4_experiments.tex ----------------
assert float(s1sr) > 73.4  # only Efficient-VLN (75.6) is higher among single-RGB methods
rep('sec/4_experiments.tex',
    r'On RxR-CE, \method achieves 71.1\% SR, surpassing SACA trained without external data by 10.8 percentage points. Among single-RGB methods, only Efficient-VLN~\cite{zheng2025efficient} (75.6\%) and the 4B and 8B Qwen-RobotNav models~\cite{zhang2026qwen} (71.3\% and 73.4\%, respectively) achieve higher SR; all three use external training data.' + '\n',
    '\\begin{revised}\n\\noindent '
    + rf'On RxR-CE, \method achieves {s1sr}\% SR, surpassing SACA trained without external data by {float(s1sr) - SACA:.1f} percentage points. Among single-RGB methods, only Efficient-VLN~\cite{{zheng2025efficient}} (75.6\%), which uses external training data, achieves a higher SR.'
    + '\n\\end{revised}\n')
rep('sec/4_experiments.tex', r'and 70.8\pmsd{0.3} on RxR-CE (\cref{tab:rl}), 10.7 and 10.5 points above',
    rf'and {cs}\pmsd{{{csd}}} on RxR-CE (\cref{{tab:rl}}), 10.7 and {float(cs) - SACA:.1f} points above')
rep('sec/4_experiments.tex', r'and by \pseudo{1.0} on RxR-CE.', rf'and by \pseudo{{{float(cs) - float(gm[0]):.1f}}} on RxR-CE.')

# ---------------- sec/0_abstract.tex ----------------
rep('sec/0_abstract.tex', r'success rates of 71.9\% and 71.1\% on R2R-CE and RxR-CE val-unseen.',
    rf'success rates of 71.9\% and {s1sr}\% on R2R-CE and RxR-CE val-unseen.')

# ---------------- sec/X_suppl.tex ----------------
rep('sec/X_suppl.tex', '\\begin{revised}\n\\para{Evaluation.}\nIn our evaluator, 99 RxR-CE val-unseen episodes have no finite goal distance; they count as failures for every model.\n\\end{revised}\n',
    '\\para{Evaluation.}\n')
d_r2r_only, d_rxr_only = 70.4 - 71.1, 71.0 - 71.1   # old placeholder offsets to CoRe run 1 (71.1)
rep('sec/X_suppl.tex', r'R2R-CE only & 68.7 & \pseudo{70.4}\\', rf'R2R-CE only & 68.7 & \pseudo{{{f1(float(s1sr) + d_r2r_only)}}}\\')
rep('sec/X_suppl.tex', r'RxR-CE only & \pseudo{67.5} & \pseudo{71.0}\\', rf'RxR-CE only & \pseudo{{67.5}} & \pseudo{{{f1(float(s1sr) + d_rxr_only)}}}\\')
rep('sec/X_suppl.tex', r'Both, alternating (default) & \textbf{71.9} & \textbf{71.1}\\', rf'Both, alternating (default) & \textbf{{71.9}} & \textbf{{{s1sr}}}\\')
rep('sec/X_suppl.tex', r'GRPO & 1 & 66.8 & 60.0 & 70.0 & 57.7\\', rf"GRPO & 1 & 66.8 & 60.0 & {vals['grpo1'][0]} & {vals['grpo1'][1]}\\")
rep('sec/X_suppl.tex', r' & 2 & \pseudo{66.4} & \pseudo{59.6} & \pseudo{69.6} & \pseudo{57.3}\\',
    rf" & 2 & \pseudo{{66.4}} & \pseudo{{59.6}} & \pseudo{{{vals['grpo2_pseudo'][0]}}} & \pseudo{{{vals['grpo2_pseudo'][1]}}}\\")
rep('sec/X_suppl.tex', r'\core & 1 & 71.9 & 63.7 & 71.1 & 57.5\\', rf'\core & 1 & 71.9 & 63.7 & {s1sr} & {s1spl}\\')
rep('sec/X_suppl.tex', r' & 2 & 70.2 & 62.7 & 70.6 & 57.4\\', rf" & 2 & 70.2 & 62.7 & {vals['run2'][0]} & {vals['run2'][1]}\\")

if '--write' in sys.argv:
    for rel, s in files.items():
        (root / rel).write_text(s)
    print('written:', sorted(files))
else:
    print('dry run (pass --write):', sorted(files))
