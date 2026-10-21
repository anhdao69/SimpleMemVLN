"""D-a native recall and deterministic state-probe feature primitives."""
import json
from pathlib import Path
import torch

LAGS=(0,*range(1,13),14,16,20,24,28,32)
REGIONS=('Q1','Q2','Q3','Q4','Global')


def sampled_pairs(frames):
    return [(t,t-lag) for t in range(0,frames,2) for lag in LAGS if lag<=t]


def lag_group(lag):
    if 1<=lag<=3:return 'A'
    if 4<=lag<=7:return 'B'
    if 8<=lag<=32:return 'C'
    return None


def lag_bucket(lag):
    if 0<=lag<=12:return str(lag)
    for low,high in ((13,16),(17,24),(25,32)):
        if low<=lag<=high:return f'{low}-{high}'
    raise ValueError('Lag outside registered range')


def native_error(state,keys,values):
    """Per-token mean-head normalized error; S orientation is [Hv,Dk,Dv]."""
    if state.ndim!=3 or keys.ndim!=3 or values.ndim!=3:
        raise ValueError('Expected state [Hv,Dk,Dv], keys/values [tokens,heads,dim]')
    if (values.shape[1:]!=(state.shape[0],state.shape[2]) or
        keys.shape[0]!=values.shape[0] or keys.shape[2]!=state.shape[1] or
        state.shape[0]%keys.shape[1]):
        raise ValueError('Incompatible state/key/value dimensions')
    repeated=keys.float().repeat_interleave(state.shape[0]//keys.shape[1],dim=1)
    read=torch.einsum('nhk,hkv->nhv',repeated,state.float())
    error=(values.float()-read).square().sum(-1)/(values.float().square().sum(-1)+1e-6)
    if not torch.isfinite(error).all():raise ValueError('Nonfinite recall error')
    return error.mean(-1)


def region_means(tokens):
    if tokens.shape[0]!=300:raise ValueError('Expected 15x20 visual grid')
    grid=tokens.reshape(15,20,*tokens.shape[1:])
    return torch.stack([grid[:7,:10].mean((0,1)),grid[:7,10:].mean((0,1)),
        grid[7:,:10].mean((0,1)),grid[7:,10:].mean((0,1)),grid.mean((0,1))])


@torch.no_grad()
def recall_rows(record,device='cpu'):
    states,keys,values=(record[name] for name in ('states','visual_keys','visual_values'))
    if len(states)!=len(keys) or len(keys)!=len(values):raise ValueError('Frame counts differ')
    baseline={}
    rows=[]
    for t,s in sampled_pairs(len(states)):
        key,value=keys[s].to(device),values[s].to(device)
        if s not in baseline:
            baseline[s]=region_means(native_error(states[s].to(device),key,value)).cpu()
        errors=region_means(native_error(states[t].to(device),key,value)).cpu()
        for r,name in enumerate(REGIONS):
            rows.append(dict(anchor=t,source=s,lag=t-s,lag_group=lag_group(t-s),
                lag_bucket=lag_bucket(t-s),region=name,epsilon=float(errors[r]),
                source_epsilon=float(baseline[s][r]),forgetting=float(errors[r]-baseline[s][r])))
    return rows


def prepare_projection_cache(path,heads=32,key_dim=128,value_dim=128,dimensions=4096,seed=0):
    """Generate immutable individual head blocks once, never full P in memory."""
    import numpy as np
    from scripts.diag.cache_teacher import sha256
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    contract=dict(heads=heads,key_dim=key_dim,value_dim=value_dim,dimensions=dimensions,seed=seed,
                  torch_version=torch.__version__,distribution='standard normal FP32 CPU, sequential head blocks')
    files=[]
    for head in range(heads):
        block=torch.randn(key_dim*value_dim,dimensions,generator=generator,dtype=torch.float32)
        destination=path/f'head_{head:02d}.npy'
        np.save(destination,block.numpy(),allow_pickle=False);del block
        destination.chmod(0o444)
        files.append(dict(file=destination.name,sha256=sha256(destination),bytes=destination.stat().st_size,
                          mtime_ns=destination.stat().st_mtime_ns))
    (path/'manifest.json').write_text(json.dumps(dict(contract=contract,files=files),indent=2)+'\n')
    (path/'manifest.json').chmod(0o444)


_VERIFIED_CACHES={}


def _projection_manifest(cache,states,dimensions,seed):
    from scripts.diag.cache_teacher import sha256
    path=Path(cache).resolve();identity=sha256(path/'manifest.json')
    if (str(path),identity) not in _VERIFIED_CACHES:
        manifest=json.loads((path/'manifest.json').read_text())
        for item in manifest['files']:
            if sha256(path/item['file'])!=item['sha256']:raise ValueError('Projection cache hash differs')
        _VERIFIED_CACHES[str(path),identity]=manifest
    manifest=_VERIFIED_CACHES[str(path),identity];contract=manifest['contract']
    expected=dict(heads=states.shape[1],key_dim=states.shape[2],value_dim=states.shape[3],dimensions=dimensions,seed=seed)
    if any(contract.get(k)!=v for k,v in expected.items()) or len(manifest['files'])!=states.shape[1]:
        raise ValueError('Projection cache shape/seed contract differs')
    for item in manifest['files']:
        stat=(path/item['file']).stat()
        if stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Projection cache changed after verification')
    return manifest


@torch.no_grad()
def gaussian_projection(states,dimensions=4096,seed=0,device='cpu',cache=None):
    """Shared standard Gaussian P, generated in FP32 CPU head-block order.

    Generator resets for every invocation, so every model and episode sees the
    identical P. Only one [Dk*Dv,4096] block exists at a time, never the full P.
    """
    if states.ndim!=4 or dimensions<1:raise ValueError('Expected [frames,Hv,Dk,Dv]')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    manifest=_projection_manifest(cache,states,dimensions,seed) if cache is not None else None
    result=torch.zeros(len(states),dimensions,dtype=torch.float32,device=device)
    for head in range(states.shape[1]):
        if manifest is None:
            block=torch.randn(states.shape[2]*states.shape[3],dimensions,generator=generator,dtype=torch.float32)
        else:
            import numpy as np
            block=torch.from_numpy(np.load(Path(cache)/manifest['files'][head]['file'],allow_pickle=False))
        result.add_(states[:,head].flatten(1).to(device=device,dtype=torch.float32)@block.to(device))
        del block
    if not torch.isfinite(result).all():raise ValueError('Nonfinite projected state')
    return result.cpu()


def lag_embedding(lags):
    lags=torch.as_tensor(lags,dtype=torch.float32)
    frequencies=10000**(-torch.arange(8,dtype=torch.float32)/8)
    phase=lags[...,None]*frequencies
    return torch.cat((phase.sin(),phase.cos()),-1)


def current_view_features(values):
    """All five pooled value regions, retaining every value head/dimension."""
    return torch.stack([region_means(frame.float()).flatten() for frame in values])
