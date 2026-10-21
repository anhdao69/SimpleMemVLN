"""Literal registered R1–R5 thresholds; missing measurements are inconclusive.

Ambiguous 'not lower' is conservatively operationalized: R3 zero significant
negative layers; R4 a nonnegative point estimate. Partial negative-margin
patterns or negative R4 estimates with zero-crossing CIs remain inconclusive.
R1 Metric2 sign means raw state mean-five-region cosine, not state-control.
"""
import math
LAYERS=(2,6,10,14,18,22)
DRIFT_KEYS={f'{m}:{l}' for m in ('G3_M5','TRIPLE') for l in range(3,32,4)}

def valid(x):
    return bool(x and x.get('n_episodes',0)>0 and x.get('estimate') is not None and x.get('ci95') is not None and
                len(x['ci95'])==2 and all(math.isfinite(float(v)) for v in [x['estimate'],*x['ci95']]))

def decide(e):
    out={}
    def put(key,status,keys,reason):out[key]={'status':status,'evidence_keys':keys,'reason':reason}
    def ready(keys):return all(valid(e.get(k)) for k in keys)
    keys=['r1_b','r1_c','r1_probe_b','r1_probe_c'];status='Inconclusive'
    if ready(keys):
        if e['r1_b']['ci95'][0]>0 and e['r1_c']['ci95'][1]<0 and e['r1_probe_b']['estimate']<0 and e['r1_probe_c']['estimate']>0:status='Supported'
        elif e['r1_b']['estimate']<=0 and e['r1_b']['ci95'][1]<=0:status='Refuted'
    put('R1',status,keys,'B forgetting positive and C negative with CIs excluding zero; raw state cosine signs reversed. Refutation: B CI excludes positive values.')
    keys=['r2_target','r2_attention'];status='Inconclusive'
    if ready(keys):
        t,a=e['r2_target'],e['r2_attention']
        if t['ci95'][1]<0 and a['ci95'][0]>0:status='Supported'
        elif t['ci95'][0]<=0<=t['ci95'][1] and a['estimate']<=0:status='Refuted'
    put('R2',status,keys,'Targeted-control state cosine negative and combined33 lower-layer bank8–32 per-token mass positive, both CIs excluding zero.')
    margins=e.get('r3_margins',{});status='Inconclusive';n=None
    if set(margins)==set(map(str,LAYERS)) and all(valid(v) for v in margins.values()) and valid(e.get('r3_interaction')):
        n=sum(v['ci95'][1]<0 for v in margins.values());interaction=e['r3_interaction']['estimate']
        if n>=5 and interaction>=.015:status='Supported'
        elif n==0 and interaction<=0:status='Refuted'
    put('R3',status,['r3_margins','r3_interaction'],'At least five of six negative-margin CIs and interaction >=0.015 SR fraction; CI positivity of interaction is not required.')
    out['R3']['negative_margin_layers']=n
    status='Inconclusive'
    if valid(e.get('r4_mass')):
        if e['r4_mass']['ci95'][1]<0:status='Supported'
        elif e['r4_mass']['estimate']>=0:status='Refuted'
    put('R4',status,['r4_mass'],'Secondary: all-layer total bank mass, combined33 queries; negative CI supports, nonnegative estimate refutes.')
    overlaps=e.get('r5_overlap',{});status='Inconclusive'
    if set(overlaps)==DRIFT_KEYS and all(valid(v) for v in overlaps.values()):
        values=[v['estimate'] for v in overlaps.values()]
        if min(values)<.98:status='Supported'
        elif min(values)>=.995:status='Refuted'
    put('R5',status,['r5_overlap'],'Both models and every attention layer required; any mean <0.98 supports, all means >=0.995 refute.')
    return out
