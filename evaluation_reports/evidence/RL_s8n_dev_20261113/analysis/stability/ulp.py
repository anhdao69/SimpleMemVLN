import torch
R = '/nvme/public/svln/runs'
B = torch.load('/nvme/public/svln/checkpoints/stage2_S8N/S8N_DAGGER_SF_REMAINING/pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu')
M1 = torch.load(f'{R}/rewind_sfr_mix1k/exports/step_0125/pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu')
M2 = torch.load(f'{R}/rewind_sfr_mix1k_seed2/exports/step_0125/pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu')
print('dtype', next(iter(B.values())).dtype)
keys = [k for k in B if B[k].is_floating_point() and B[k].dim() == 2 and 'visual' not in k and 'embed' not in k]
import random; random.seed(0)
tot = z1 = z2 = both0 = same = 0; ulps = []
for k in random.sample(keys, 20):
    b = B[k].float(); d1 = M1[k].float() - b; d2 = M2[k].float() - b
    ulp = torch.where(b == 0, torch.full_like(b, 2**-133), b.abs()) .log2().floor().exp2() * 2**-7   # bf16: 7 explicit mantissa bits
    u1 = (d1 / ulp).round(); u2 = (d2 / ulp).round()
    tot += b.numel(); z1 += (u1 == 0).sum().item(); z2 += (u2 == 0).sum().item(); both0 += ((u1 == 0) & (u2 == 0)).sum().item()
    same += ((u1 == u2) & (u1 != 0)).sum().item()
    ulps.append(u1.abs().flatten()[:200000])
u = torch.cat(ulps)
print(f'params sampled {tot/1e6:.0f} M: delta == 0 in mix1k {z1/tot:.3f}, seed2 {z2/tot:.3f}, both {both0/tot:.3f}; same nonzero ulp delta {same/tot:.3f}')
for q in (0.5, 0.75, 0.9, 0.99):
    print(f'  |delta| in bf16 ulps, quantile {q}: {u.quantile(q).item():.0f}')
print('  share |delta| == 1 ulp:', (u == 1).float().mean().item(), ' >= 3 ulp:', (u >= 3).float().mean().item())
