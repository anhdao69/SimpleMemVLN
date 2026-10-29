import glob, json, sys
import numpy as np
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1] + '/rollouts/step_0000_rank*.jsonl')) for l in open(f)]
main = [r for r in rows if r.get('step_drift')]
scores = [(t, float(np.mean(np.log(d))), r['num_steps']) for r in main for t, d in enumerate(r['step_drift'])]
s = np.array([x[1] for x in scores]); t = np.array([x[0] for x in scores]); n = np.array([x[2] for x in scores])
med = np.median(s)
print(f"{len(main)} trajectories, {len(s)} decisions; median event score {med:.4f} (initial_min_drift candidate)")
print("t-bucket  decisions  mean_score  P(score>=median)")
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25), (25, 50), (50, 1000)]:
    m = (t >= lo) & (t < hi)
    if m.any(): print(f"[{lo:>3},{hi:>4})  {m.sum():>9}  {s[m].mean():>10.3f}  {np.mean(s[m] >= med):>16.2f}")
rel = t / np.maximum(n, 1)
print("Spearman(score, t/N):", round(float(np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(rel)))[0, 1]), 3))
# Local surprise: score minus the median of the previous 8 decisions of the same trajectory.
loc = []
for r in main:
    e = [float(np.mean(np.log(d))) for d in r['step_drift']]
    loc += [(i, e[i] - np.median(e[max(0, i - 8):i])) for i in range(1, len(e))]
lt = np.array([x[0] for x in loc]); ls = np.array([x[1] for x in loc]); lm = np.median(ls)
print("local surprise P(>=median) by t:", {f"[{lo},{hi})": round(float(np.mean(ls[(lt >= lo) & (lt < hi)] >= lm)), 2) for lo, hi in [(1, 3), (3, 6), (6, 12), (12, 25), (25, 50), (50, 1000)] if ((lt >= lo) & (lt < hi)).any()})
