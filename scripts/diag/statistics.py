"""Episode-unit paired inference. No token/pair pseudoreplication.

Input rows are one already-aggregated observation per episode. CI resampling draws
n_scene episodes WITH replacement independently inside each fixed scene, then
averages all sampled episodes (population scene proportions remain fixed). The
same resampled episode index carries every paired condition: callers form paired
contrasts before resampling. This conditions on the observed scenes; it does not
resample scenes or estimate between-scene population uncertainty. Percentile 95%
intervals use NumPy linear quantiles, 2000 draws and seed 20261019 by default.
None is explicitly undefined; NaN/Inf are numerical failures and raise errors.
"""
import math
import numpy as np


def finite(value):
    if value is None:return None
    value=float(value)
    if not math.isfinite(value):raise ValueError('Nonfinite measurement; use None only for explicitly undefined metrics')
    return value


def episode_bootstrap(rows, value_key='value', resamples=2000, seed=20261019):
    """Rows: episode_id, scene, value; returns estimate, CI and denominators."""
    if resamples<1:raise ValueError('Positive resample count required')
    seen=set();groups={};undefined=[]
    for row in sorted(rows,key=lambda r:str(r['episode_id'])):
        key=str(row['episode_id'])
        if key in seen:raise ValueError('Exactly one observation per episode required')
        seen.add(key)
        value=finite(row[value_key])
        if value is None:undefined.append(key);continue
        groups.setdefault(str(row['scene']),[]).append(value)
    n=sum(map(len,groups.values()))
    out={'estimate':None,'ci95':None,'n_episodes':n,'n_input':len(rows),
         'n_undefined':len(undefined),'undefined_episode_ids':undefined,
         'scene_counts':{k:len(v) for k,v in sorted(groups.items())},
         'resamples':resamples,'seed':seed,'unit':'episode',
         'method':'within-scene episode bootstrap; percentile linear quantiles',
         'undefined_reason':None if n else 'no_defined_episodes'}
    if not n:return out
    rng=np.random.default_rng(seed);draws=np.zeros(resamples,dtype=np.float64)
    for scene,values in sorted(groups.items()):
        a=np.array(values,dtype=np.float64)
        draws+=a[rng.integers(0,len(a),size=(resamples,len(a)))].sum(axis=1)
    draws/=n
    out.update(estimate=math.fsum(v for a in groups.values() for v in a)/n,
               ci95=np.quantile(draws,[.025,.975],method='linear').tolist())
    return out


def paired_bootstrap(rows, left='left', right='right', resamples=2000, seed=20261019):
    """Mean right-left on jointly defined episodes, paired before resampling.

    For D-e interaction provide left=delta_M5 and right=delta_G3_M5, both
    computed on the same episode; SR units remain fractions unless caller scales.
    """
    contrasts=[]
    for row in rows:
        a,b=finite(row[left]),finite(row[right])
        contrasts.append(dict(episode_id=row['episode_id'],scene=row['scene'],
                              value=None if a is None or b is None else b-a))
    result=episode_bootstrap(contrasts,resamples=resamples,seed=seed)
    result['contrast']=f'{right} - {left}'
    return result


def mcnemar(left,right):
    """Exact two-sided conditional binomial McNemar on paired binary success.

    p=min(1,2*sum(C(n,k),k=0..min(b,c))/2**n), without asymptotic correction.
    """
    if len(left)!=len(right):raise ValueError('Paired arrays must have equal lengths')
    b=c=same=missing=0
    for a,z in zip(left,right):
        a,z=finite(a),finite(z)
        if any(v is not None and v not in (0,1) for v in (a,z)):raise ValueError('Binary success required')
        if a is None or z is None:missing+=1;continue
        b+=a==1 and z==0;c+=a==0 and z==1;same+=a==z
    n=b+c
    p=min(1.,(2*sum(math.comb(n,k) for k in range(min(b,c)+1)))/(2**n)) if n else 1.
    return {'left_only':b,'right_only':c,'concordant':same,'discordant':n,
            'n_episodes':b+c+same,'n_undefined':missing,
            'p_two_sided':p if b+c+same else None,
            'undefined_reason':None if b+c+same else 'no_defined_pairs',
            'method':'exact conditional binomial, two-sided'}
