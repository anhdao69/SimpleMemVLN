"""Gram matrix of RL weight deltas (theta_run - theta_SFR) over all tensors that changed; per-group shares."""
import json, sys, time, torch
R = '/nvme/public/svln/runs'
base = '/nvme/public/svln/checkpoints/stage2_S8N/S8N_DAGGER_SF_REMAINING/pytorch_model.bin'
runs = sys.argv[1:]
t0 = time.time()
B = torch.load(base, mmap=True, weights_only=True, map_location='cpu')
M = [torch.load(f'{R}/{r}/exports/step_0125/pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu') for r in runs]
for r, m in zip(runs, M):
    assert set(m) == set(B), (r, len(set(m) ^ set(B)))
n = len(runs)
G = torch.zeros(n, n, dtype=torch.float64)
changed = unchanged = 0
per_group = {}
for k in B:
    b = B[k]
    if not b.is_floating_point():
        continue
    rows = b.shape[0] if b.dim() else 1
    step = max(1, (64 << 20) // max(1, b.numel() // max(rows, 1)))
    g = torch.zeros(n, n, dtype=torch.float64)
    for i in range(0, rows, step):
        sl = slice(i, i + step)
        bb = (b[sl] if b.dim() else b).float().reshape(-1)
        D = torch.stack([((m[k][sl] if b.dim() else m[k]).float().reshape(-1) - bb) for m in M])
        g += (D @ D.T).double()
    if g.diagonal().sum() == 0:
        unchanged += b.numel(); continue
    changed += b.numel()
    G += g
    grp = 'embed' if 'embed' in k else 'lm_head' if 'lm_head' in k else 'mlp' if 'mlp' in k else 'attn' if ('attn' in k or 'linear_attn' in k) else 'norm' if 'norm' in k else 'visual' if 'visual' in k else 'other'
    per_group[grp] = per_group.get(grp, torch.zeros(n, n, dtype=torch.float64)) + g
nrm = G.diagonal().sqrt()
C = G / nrm[:, None] / nrm[None, :]
print(f'changed params {changed/1e9:.3f} B, unchanged {unchanged/1e9:.3f} B, {time.time()-t0:.0f} s')
print('delta norms:', {r: round(float(x), 3) for r, x in zip(runs, nrm)})
print('cosine matrix:')
print(' ' * 26 + ' '.join(f'{r[:10]:>10s}' for r in runs))
for i, r in enumerate(runs):
    print(f'{r:26s}' + ' '.join(f'{float(C[i, j]):10.3f}' for j in range(n)))
for grp, g in per_group.items():
    d = g.diagonal().sqrt()
    print(f'group {grp:8s} norm share {float(d[0]**2/nrm[0]**2):.3f}  cos(run0, run1) {float(g[0,1]/d[0]/d[1]):.3f}')
json.dump(dict(runs=runs, gram=G.tolist()), open('/tmp/claude-1001/-nvme-public-SimpleMemVLN/dced37af-913d-422f-b0e4-f110c3584dd3/scratchpad/soup/gram.json', 'w'))
