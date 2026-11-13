"""STABLE-V1 checks on a run directory (2026-11-12).
  rollouts: every step's main advantages are (R - group mean) / s with ONE s per step (adv_scale: batch), s >= floor;
  lr (with --lr): metrics lr follows lr * min(1, (total - step) / decay) for the steps present.
Usage: python3 -I stable_check.py RUN_DIR [--lr 1e-6 --total 125 --decay 50] [--floor 0.75]"""
import argparse, collections, glob, json, statistics as st, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--lr', type=float); ap.add_argument('--total', type=int, default=125)
ap.add_argument('--decay', type=int, default=50); ap.add_argument('--floor', type=float, default=0.75)
a = ap.parse_args()
run = Path(a.run); bad = []
steps = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(str(run / 'rollouts' / 'step_*_rank*.jsonl')):
    s = int(f.split('step_')[1][:4])
    for l in open(f):
        r = json.loads(l)
        if 'sample_idx' in r and 'advantage' in r:
            steps[s][(r['episode_id'], r.get('attempt', 0))].append((r['reward'], r['advantage'], r['success']))
allsucc = []
for s in sorted(steps):
    scales = []
    for key, g in steps[s].items():
        if len(g) < 2: continue
        m = sum(x[0] for x in g) / len(g)
        for rw, adv, _ in g:
            if abs(adv) > 1e-3: scales.append((rw - m) / adv)  # near-zero advantages: float32 rounding, not a scale
        if all(x[2] for x in g) and len(g) == 4: allsucc.append(st.mean(abs(x[1]) for x in g))
    if not scales: continue
    lo, hi = min(scales), max(scales)
    ok = hi - lo <= 1e-3 * hi and lo >= a.floor - 1e-6
    print(f'step {s}: advantage scale {lo:.4f}..{hi:.4f} over {len(scales)} trajectories -> {"one batch scale" if ok else "NOT a single batch scale"}')
    if not ok: bad.append(f'step {s} scales {lo:.4f}..{hi:.4f}')
print(f'all-success groups: {len(allsucc)}, mean |advantage| {st.mean(allsucc) if allsucc else float("nan"):.3f} (group-std scaling gives ~0.74)')
if a.lr:
    rows = {}
    for l in open(run / 'metrics.jsonl'):
        x = json.loads(l); rows[x['step']] = x
    for s in sorted(rows):
        want = a.lr * min(1.0, (a.total - s) / a.decay)
        if abs(rows[s]['lr'] - want) > 1e-3 * want: bad.append(f'step {s} lr {rows[s]["lr"]} != {want}')
    show = [s for s in (0, 74, 75, 100, 124) if s in rows]
    print('lr ' + ', '.join(f'step {s}: {rows[s]["lr"]:.3g}' for s in show) + f' ({len(rows)} steps checked)')
if bad:
    sys.exit('STABLE CHECK FAILED: ' + '; '.join(bad[:5]))
print('STABLE CHECK OK')
