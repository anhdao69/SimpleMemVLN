"""D-d summaries from frozen captured LayerSelectionPlan and visual scores.

selection_metrics(plan, visual_ids, visual_scores) consumes [G,300] IDs from
layout.visual_matrix() and [G,300] committed FP32 scores (CPU tensor or ndarray).
Scores are copied as float64 for statistics. Candidate admission at t is previous
resident bank plus visual group t-R, NOT all historical tokens. Population std
(ddof=0) is used for CV and cutoff normalization. Selected Gini is the standard
nonnegative sample Gini without finite-size correction. Undefined values are None
with reasons; their episode means carry defined/undefined admission counts.

plan_summary-equivalent fields describe FINAL occupancy/age/early share and
cumulative removed-token turnover. Per-step turnover is removed rows, not Jaccard
or the count of new admissions. Spatial counts use source frame's raster index,
not absolute token ID modulo 300. Empty-empty Jaccard is 1 by set identity but
has no selection information: cross-model output includes bank sizes.
"""
import math
import numpy as np


def array(x,dtype=None):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    return np.asarray(x,dtype=dtype)


def gini(values):
    v=np.sort(array(values,np.float64).reshape(-1))
    if not np.isfinite(v).all() or np.any(v<0):raise ValueError('Gini requires finite nonnegative scores')
    if not len(v) or v.sum()==0:return None
    return float(np.sum((2*np.arange(1,len(v)+1)-len(v)-1)*v)/(len(v)*v.sum()))


def jaccard(left,right):
    a,b=set(map(int,left)),set(map(int,right))
    return len(a&b)/len(a|b) if a|b else 1.


def ranks(v):
    order=np.argsort(v,kind='stable');out=np.empty(len(v),float);start=0
    while start<len(v):
        end=start+1
        while end<len(v) and v[order[end]]==v[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2+1;start=end
    return out


def spearman(left,right):
    """Average ranks for ties; None is missing, constant/short pairs undefined."""
    if len(left)!=len(right):raise ValueError('Same-token pairing requires equal lengths')
    pairs=[];missing=0
    for x,y in zip(left,right):
        for v in (x,y):
            if v is not None and not math.isfinite(float(v)):raise ValueError('Nonfinite correlation input')
        if x is None or y is None:missing+=1
        else:pairs.append((float(x),float(y)))
    result={'value':None,'n_tokens':len(pairs),'n_undefined':missing,'undefined_reason':None}
    if len(pairs)<2:result['undefined_reason']='fewer_than_two_pairs';return result
    a,b=np.array(pairs).T;a=ranks(a);b=ranks(b)
    if np.ptp(a)==0 or np.ptp(b)==0:result['undefined_reason']='constant_ranks';return result
    result['value']=float(np.corrcoef(a,b)[0,1]);return result


def selection_metrics(plan,visual_ids,visual_scores,grid_shape=(15,20),border_columns=(0,1,2,17,18,19)):
    plan.verify()
    ids=array(visual_ids,np.int64);scores=array(visual_scores,np.float64)
    if ids.shape!=scores.shape or ids.ndim!=2 or ids.shape!=(plan.groups,math.prod(grid_shape)):
        raise ValueError('Expected aligned group-by-visual-token IDs and scores')
    if len(np.unique(ids))!=ids.size or not np.isfinite(scores).all() or np.any(scores<0):raise ValueError('Unique IDs and finite nonnegative scores required')
    locations={int(token):(g,i) for g,row in enumerate(ids) for i,token in enumerate(row)}
    lookup={int(token):float(scores[g,i]) for token,(g,i) in locations.items()}
    rows=[];before=np.empty(0,np.int64)
    for t in range(plan.groups):
        selected=array(plan.group_ids(t),np.int64)
        if len(np.unique(selected))!=len(selected) or any(int(i) not in lookup for i in selected):raise ValueError('Invalid selected IDs')
        sources=np.array([locations[int(i)][0] for i in selected],dtype=int)
        if len(selected)>plan.budget_tokens or np.any(sources>t-plan.recent_groups):raise ValueError('Bank violates capacity/age')
        recorded=array(plan.token_sources)[plan.group_offsets[t]:plan.group_offsets[t+1]]
        if not np.array_equal(recorded,sources):raise ValueError('Captured token source mismatch')
        admitted=t-plan.recent_groups
        candidates=np.concatenate([before,ids[admitted]]) if admitted>=0 else np.empty(0,np.int64)
        candidate_set=set(map(int,candidates))
        if any(int(i) not in candidate_set for i in selected):raise ValueError('Selected row was not an admission candidate')
        v=np.array([lookup[int(i)] for i in candidates]);sv=np.array([lookup[int(i)] for i in selected])
        mean=float(v.mean()) if v.size else None;std=float(v.std()) if v.size else None
        margin=float(plan.cutoff_margins[t]);margin=None if math.isnan(margin) else margin
        if margin is not None and (not math.isfinite(margin) or margin<0):raise ValueError('Invalid cutoff margin')
        hist=np.zeros(grid_shape,dtype=int)
        for i in selected:
            position=locations[int(i)][1];hist[position//grid_shape[1],position%grid_shape[1]]+=1
        occupancy=len(selected);unique,count=np.unique(sources,return_counts=True)
        cv=std/mean if mean is not None and mean>0 else None
        normalized=margin/std if margin is not None and std is not None and std>0 else None
        row=dict(step=t,candidate_count=len(candidates),candidate_mean=mean,candidate_std=std,
                 candidate_cv=cv,cutoff_margin=margin,normalized_cutoff_margin=normalized,
                 selected_count=occupancy,selected_gini=gini(sv),distinct_source_groups=len(unique),
                 fraction_from_steps_0_2=float(np.mean(sources<3)) if occupancy else None,
                 mean_age_groups=float(np.mean(t-sources)) if occupancy else None,
                 turnover=len(set(map(int,before))-set(map(int,selected))),
                 spatial_histogram=hist.tolist(),border_share=float(hist[:,[c for c in border_columns if 0<=c<grid_shape[1]]].sum()/occupancy) if occupancy else None,
                 max_source_share=float(count.max()/occupancy) if occupancy else None,
                 undefined_reasons={})
        if cv is None:row['undefined_reasons']['candidate_cv']='no_admission' if not v.size else 'zero_candidate_mean'
        if normalized is None:row['undefined_reasons']['normalized_cutoff_margin']='no_cutoff' if margin is None else 'zero_candidate_std'
        if row['selected_gini'] is None:row['undefined_reasons']['selected_gini']='empty_bank' if not occupancy else 'zero_selected_sum'
        if not occupancy:
            for key in ('fraction_from_steps_0_2','mean_age_groups','border_share','max_source_share'):row['undefined_reasons'][key]='empty_bank'
        rows.append(row);before=selected
    last=rows[-1] if rows else None
    summary={'groups':plan.groups,'final_rows':last['selected_count'] if last else 0,
             'distinct_source_groups':last['distinct_source_groups'] if last else 0,
             'fraction_from_steps_0_2':last['fraction_from_steps_0_2'] if last else None,
             'mean_age_groups':last['mean_age_groups'] if last else None,
             'fraction_first_frame':float(np.mean(sources==0)) if last and len(sources) else None,
             'evicted_total':sum(r['turnover'] for r in rows),'digest':plan.source_digest[:16]}
    margins=[r['cutoff_margin'] for r in rows if r['cutoff_margin'] is not None]
    summary['margin_quantiles']=np.quantile(margins,[0,.1,.5]).tolist() if margins else None
    means={}
    for key in ('candidate_cv','normalized_cutoff_margin','selected_gini','mean_age_groups','border_share','max_source_share'):
        values=[r[key] for r in rows if r[key] is not None]
        means[key]={'value':math.fsum(values)/len(values) if values else None,'n_defined':len(values),'n_undefined':len(rows)-len(values)}
    return {'attention_layer':plan.layer_idx,'scorer_layer':plan.layer_idx-1,'steps':rows,'summary':summary,'episode_means':means}


def compare_models(left_plan,right_plan,visual_ids,left_scores,right_scores):
    """Same-token write-score correlation by source step and bank Jaccard by query step."""
    left_plan.verify();right_plan.verify()
    ids=array(visual_ids,np.int64);a=array(left_scores);b=array(right_scores)
    if ids.shape!=a.shape or a.shape!=b.shape or len(ids)!=left_plan.groups or left_plan.groups!=right_plan.groups or left_plan.layer_idx!=right_plan.layer_idx:
        raise ValueError('Cross-model inputs must align by episode, layer, group and token')
    return [{'step':t,'bank_jaccard':jaccard(left_plan.group_ids(t),right_plan.group_ids(t)),
             'left_bank_size':len(left_plan.group_ids(t)),'right_bank_size':len(right_plan.group_ids(t)),
             'write_score_spearman':spearman(a[t],b[t])} for t in range(len(ids))]
