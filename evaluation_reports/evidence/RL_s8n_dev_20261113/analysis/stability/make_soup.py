"""Uniform weight average (model soup) of RL exports that share one init (SFR): theta = mean_i theta_i, computed in
fp32 per tensor and rounded to the exports' dtype (bf16, round-to-nearest-even). Metadata files are copied from the
first ingredient; soup_provenance.json records the ingredients and their weight sha256.
Usage: make_soup.py OUT_DIR EXPORT_DIR [EXPORT_DIR ...]"""
import hashlib, json, shutil, sys, time
from pathlib import Path
import torch

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()

out, ings = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
assert len(ings) >= 2 and not out.exists()
t0 = time.time()
S = [torch.load(p / 'pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu') for p in ings]
keys = list(S[0])
assert all(list(s) == keys for s in S)
soup = {}
for k in keys:
    ts = [s[k] for s in S]
    if not ts[0].is_floating_point() or all(torch.equal(ts[0], t) for t in ts[1:]):
        soup[k] = ts[0].clone()
        continue
    acc = torch.zeros(ts[0].shape, dtype=torch.float32)
    for t in ts:
        acc += t.float()
    soup[k] = (acc / len(ts)).to(ts[0].dtype)
tmp = out.with_name(out.name + '.tmp')
tmp.mkdir(parents=True)
for f in ings[0].iterdir():
    if f.name not in ('pytorch_model.bin', 'rl_provenance.json'):
        shutil.copy2(f, tmp / f.name)
torch.save(soup, tmp / 'pytorch_model.bin')
prov = dict(kind='uniform weight average (model soup), fp32 accumulate, rounded to bf16', created=time.strftime('%FT%T'),
            ingredients=[dict(export=str(p), weights_sha256=sha(p / 'pytorch_model.bin'),
                              rl_provenance=json.loads((p / 'rl_provenance.json').read_text()) if (p / 'rl_provenance.json').exists() else None)
                         for p in ings],
            weights_sha256=sha(tmp / 'pytorch_model.bin'))
(tmp / 'soup_provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
tmp.rename(out)
print(out, prov['weights_sha256'][:16], f'{time.time() - t0:.0f} s')
