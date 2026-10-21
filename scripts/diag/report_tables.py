"""Measured diagnostic tables; no manufactured observations or causal claims."""
import math

PREDICTIONS={
'D-a':'R1 predicts higher G3_M5−M5_E1 forgetting in B and lower in C, with opposite raw-state cosine signs; G3−G1 should improve C recall.',
'D-b':'R2 predicts negative targeted−control state cosine in C and a larger C-versus-A bank-off decodability drop for G3_M5 than M5_E1; native self-recall may change little.',
'D-c':'R2 predicts increased lower-layer bank8–32 mass per token; secondary R4 predicts reduced total bank mass, especially in upper layers.',
'D-d':'R3 predicts negative normalized-margin and CV differences at most of the six scorer layers below the probe, with weaker changes above it.',
'D-e':'R3 predicts a random-selector loss of at least two SR points for M5_E1, a smaller effect for G3_M5, and positive interaction; the registered support threshold is +1.5 interaction points.',
'D-f':'R5 predicts negligible selection drift if every model/layer mean is at least 0.995; any layer below 0.98 is material.'}

def table(headers,rows):
    return ['','| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]]

def describe(name,m):
    from scripts.diag.report_decisions import valid
    from scripts.diag.report import format_metric
    if not valid(m):return name+': unavailable; no direction is inferred.'
    lo,hi=m['ci95'];direction='CI strictly positive' if lo>0 else 'CI strictly negative' if hi<0 else 'CI contains zero'
    return name+': '+format_metric(m)+' ('+direction+').'

def diagnostic_tables(d,r):
    from scripts.diag.report import unique,format_metric
    f=format_metric;n=r['D_a_metric1']['summaries'];contrasts=r['D_a_metric1']['paired_contrasts'];p=r['D_a_metric2']['summaries'];e=r['decision_evidence'];out=['','**Pre-registered prediction:** '+PREDICTIONS[d]]
    if d=='D-a':
        rr=[]
        for model in ('M5_E1','G3_M5','TRIPLE','G3','G1'):
            for lag in ('A','B','C'):
                base=dict(model=model,layer=24,lag=lag)
                rr.append([model,lag,f(unique(n,**base,diagnostic='D-a',region='Global',metric='forgetting')),
                    f(unique(p,**base,feature='state',metric='cosine')),f(unique(p,**base,feature='control',metric='cosine')),
                    f(unique(p,**base,feature='memory_specific',metric='cosine')),
                    f(unique(p,**base,feature='state',metric='top1')),f(unique(p,**base,feature='state',metric='mrr'))])
        out+=table(['Model','Lag','Global forgetting','State cosine (five-region mean)','Current-view cosine','State−control cosine','State retrieval top-1','State retrieval MRR'],rr)
        out+=['','**Observed:** '+ ' '.join(describe(k,e.get(k)) for k in ('r1_b','r1_c','r1_probe_b','r1_probe_c'))]
        g=unique(contrasts,diagnostic='D-a',model='G3 - G1',layer=24,region='Global',metric='forgetting',lag='C')
        out+=['',describe('Window8 G3−G1 C forgetting',g)]
    elif d=='D-b':
        rr=[]
        for row in r['D_b']['contrasts']:
            if (row['feature'] in ('state','memory_specific') and row['metric']=='cosine') or (row['feature']=='native' and row['metric']=='epsilon_Global'):
                rr.append([row['model'],row['contrast'],row['group'],row['feature'],row['metric'],f(row)])
        out+=table(['Model','Contrast','Lag','Feature','Metric','Estimate [95% CI], denominators'],rr)
        out+=table(['C−A bank-off difference-in-differences','Metric','Estimate [95% CI], denominators'],
            [[row['feature'],row['metric'],f(row)] for row in r['D_b']['difference_in_differences'] if row['metric']=='cosine'])
        out+=table(['Model','Matched / selected pairs','Episodes with any match'],[[m,
            str(sum(x['matched_pairs'] for x in r['D_b']['coverage'] if x['model']==m))+' / '+str(sum(x['selected_pairs'] for x in r['D_b']['coverage'] if x['model']==m)),
            str(sum(x['matched_pairs']>0 for x in r['D_b']['coverage'] if x['model']==m))+' / '+str(sum(x['model']==m for x in r['D_b']['coverage']))] for m in ('M5_E1','G3_M5','TRIPLE')])
        out+=table(['Model','Global bank-off decision-token KL (nats)'],[[row['model'],f(row)] for row in r['D_b_decision_token_KL']['summaries']])
        out+=['','**Observed:** '+describe('G3_M5 targeted−control C state cosine',e.get('r2_target'))]
        for row in r['D_b']['difference_in_differences']:
            if row['feature']=='state' and row['metric']=='cosine':out.append(describe('C−A bank-off interaction',row))
    elif d=='D-c':
        rr=[]
        for model in ('M5_E1','G3_M5','TRIPLE'):
            rr.append([model,'3–23','bank8–32 per token',f(unique(n,diagnostic='D-c',model=model,layer='3-23',role='combined',category='bank_lag_8_32',metric='mass_per_token'))])
            for layer in range(3,32,4):rr.append([model,layer,'total bank mass',f(unique(n,diagnostic='D-c',model=model,layer=layer,role='combined',category='total_bank',metric='mass'))])
        out+=table(['Model','Attention layers','Combined33 metric','Estimate [95% CI], denominators'],rr)
        window8=[]
        for model in ('G3','G1'):
            for layer in range(3,32,4):
                exact=unique(n,diagnostic='D-c',model=model,layer=layer,role='combined',category='window8_exact_lag_4_7',metric='mass')
                absent=unique(n,diagnostic='D-c',model=model,layer=layer,role='combined',category='bank_lag_8_32',metric='mass')
                window8.append([model,layer,f(exact),f(absent),'not applicable: no bank tokens'])
        out+=table(['Window8 reference','Attention layer','Exact lag 4–7 mass','Structurally absent bank lag 8–32 mass','Bank mass per token'],window8)
        out+=['','**Observed:** '+describe('G3_M5−M5_E1 lower-bank mass/token',e.get('r2_attention'))+' '+describe('All-layer total bank mass difference (R4)',e.get('r4_mass'))]
    elif d=='D-d':
        out+=table(['Scorer layer','G3_M5−M5_E1 normalized cutoff margin','G3_M5−M5_E1 candidate CV'],[[layer,f(e['r3_margins'].get(str(layer))),f(unique(contrasts,diagnostic='D-d',model='G3_M5 - M5_E1',layer=layer,metric='candidate_cv',scope='admission_mean'))] for layer in (2,6,10,14,18,22)])
        value=r['decisions']['R3'].get('negative_margin_layers')
        out+=['','**Observed:** '+('The six-layer margin count is unavailable; no negative-layer count is inferred.' if value is None else f'{value} of six lower scorer layers have a normalized-margin CI strictly below zero.')+' The R3 decision also requires the separate closed-loop interaction.']
    elif d=='D-e':
        rr=[]
        for model,record in r['D_e']['models'].items():
            for condition in ('score','random'):
                rr.append([model,condition,*[f(record['metrics'][metric][condition]) for metric in ('sr','spl','os','ne','forced_stop')],record['forced_stop_counts'][condition],record['execution_failures'][condition]])
        out+=table(['Model','Selector','SR (fraction)','SPL','OS (fraction)','NE (m)','Forced-stop fraction','Forced-stop count','Execution failures'],rr)
        out+=table(['Model','ΔSR random−score (fraction)','Exact McNemar p','Success determinism','Metric determinism','Action determinism'],[[m,f(v['metrics']['sr']['delta']),finite_scalar(v['mcnemar']['p_two_sided']),
            agreement(v['determinism'],'success'),agreement(v['determinism'],'metric'),agreement(v['determinism'],'action')] for m,v in r['D_e']['models'].items()])
        out+=['','**Observed:** '+' '.join(describe(m+' random−score SR',v['metrics']['sr']['delta']) for m,v in r['D_e']['models'].items())+' '+describe('G3_M5−M5_E1 selector interaction',r['D_e']['interaction'])]
    elif d=='D-f':
        out+=table(['Attention layer','G3_M5 overlap','TRIPLE overlap'],[[layer,f(e['r5_overlap'].get('G3_M5:'+str(layer))),f(e['r5_overlap'].get('TRIPLE:'+str(layer)))] for layer in range(3,32,4)])
        out+=table(['Model','Logit RMS','Logit max absolute error','Argmax flip count (episode mean)','Argmax flip rate'],[[model,*[f(unique(n,diagnostic='D-f',model=model,layer='logits',metric=metric,scope='episode_metric_mean')) for metric in ('rms','max_abs','argmax_flips','argmax_flip_rate')]] for model in ('G3_M5','TRIPLE')])
        from scripts.diag.report_decisions import valid
        valid_rows=[(key,v) for key,v in e['r5_overlap'].items() if valid(v)]
        if len(valid_rows)==16:
            key,row=min(valid_rows,key=lambda kv:kv[1]['estimate']);out+=['','**Observed:** '+describe('Smallest model/layer mean overlap ('+key+')',row)+' R5 is '+r['decisions']['R5']['status']+'.']
        else:out+=['','**Observed:** At least one required model/layer overlap is unavailable; no all-layer boundary conclusion is made.']
    return out

def finite_scalar(v):
    if v is None:return 'unavailable'
    if not math.isfinite(float(v)):raise ValueError('Nonfinite report value')
    return f'{v:.6g}'

def agreement(r,kind):
    return f"{finite_scalar(r[kind+'_sequence_agreement' if kind=='action' else kind+'_agreement'])}; {r[kind+'_agree_count']}/{r['n_episodes']} episodes"

def baseline_table(baselines):
    return table(['Model','Successes / episodes','SR (%)','SPL','Inference'],[[model,f"{r['successes']} / {r['episodes']}",finite_scalar(100*r['sr']),finite_scalar(r.get('spl')),'Original full-run point estimate; no new CI inferred'] for model,r in baselines.items()])


def validate_baselines(baselines):
    if not isinstance(baselines,dict) or set(baselines)!={'M5_E1','G3_M5','TRIPLE','G3','G1'}:raise ValueError('Five archived baseline results required')
    for row in baselines.values():
        n,success=row['episodes'],row['successes']
        if type(n) is not int or type(success) is not int or n<=0 or not 0<=success<=n:raise ValueError('Invalid original success denominator')
        if not math.isfinite(float(row['sr'])) or not math.isclose(success/n,row['sr'],rel_tol=0,abs_tol=1e-12):raise ValueError('Original SR differs from success count')
        if row.get('spl') is not None and (not math.isfinite(float(row['spl'])) or not 0<=row['spl']<=1):raise ValueError('Invalid original SPL')
    return baselines
