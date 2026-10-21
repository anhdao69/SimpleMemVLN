"""Six publication-style PNG figure groups from real summary rows only.

Returned records retain every plotted value, interval and denominator for
results.json. Missing points are gaps; they are never replaced with zero.
"""
from collections import defaultdict
import csv
from pathlib import Path
import json
import numpy as np
from scripts.diag.cache_teacher import sha256

COLORS={'M5_E1':'#3366aa','G3_M5':'#cc5522','TRIPLE':'#9977bb','G3':'#228866','G1':'#777777'}
LAGS=[str(i) for i in range(13)]+['13-16','17-24','25-32']

def lag_position(value):
    value=str(value)
    if value in LAGS:return LAGS.index(value)
    # Native recall uses an en dash only if its registered labels do.
    value=value.replace('–','-')
    return LAGS.index(value) if value in LAGS else None

def series(ax,records,xkey,groupkey='model',scale=1.,categorical=None):
    grouped=defaultdict(list)
    for row in records:
        value=row.get('estimate');ci=row.get('ci95')
        if value is None or ci is None:continue
        x=categorical(row[xkey]) if categorical else float(row[xkey])
        if x is not None:grouped[str(row[groupkey])].append((x,value*scale,ci[0]*scale,ci[1]*scale))
    for name,points in sorted(grouped.items()):
        a=np.asarray(sorted(points));color=COLORS.get(name)
        ax.plot(a[:,0],a[:,1],marker='.',label=name,color=color)
        if len(a)==1:
            # A one-point fill_between polygon has zero area: draw its CI explicitly.
            ax.vlines(a[:,0],a[:,2],a[:,3],colors=color,linewidth=1.5)
        else:ax.fill_between(a[:,0],a[:,2],a[:,3],alpha=.15,color=color)
    ax.grid(alpha=.2)
    if grouped:ax.legend(fontsize=7)
    else:ax.text(.5,.5,'No defined measurements',transform=ax.transAxes,ha='center')

def spatial_maps(path):
    episode=defaultdict(lambda:np.zeros((15,20),dtype=np.float64));counts=defaultdict(int)
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            grid=np.asarray(json.loads(r['spatial_histogram_json']),dtype=np.float64)
            if grid.shape!=(15,20):raise ValueError('Spatial histogram shape differs')
            key=r['model'],int(r['scorer_layer']),r['episode_id'];episode[key]+=grid;counts[key]+=1
    groups=defaultdict(list)
    for (model,layer,eid),grid in episode.items():
        if grid.sum()>0:groups[model,layer].append((grid/grid.sum(),int(grid.sum()),counts[model,layer,eid]))
    return [dict(model=m,layer=l,mean_episode_occupancy_share=np.mean([v[0] for v in values],axis=0).tolist(),n_episodes=len(values),
                 n_token_visits=sum(v[1] for v in values),n_steps=sum(v[2] for v in values)) for (m,l),values in sorted(groups.items())]

def build_figures(result,output,selection_csv,*,include_selector=True):
    """Render completed non-selector figures early only by explicit opt-in.

    The default final-report path still requires D-e. This function neither
    fabricates missing selector data nor marks a partial study complete.
    """
    if include_selector:result['D_e']  # fail before creating partial output
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(output)/'figures';dest.mkdir();data={}
    native=result['D_a_metric1']['summaries'];probe=result['D_a_metric2']['summaries']
    def finish(name,fig,records):
        fig.suptitle(name+' — episode-unit estimates; bands/bars: 95% within-scene bootstrap CI',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.97));path=dest/(name+'.png');fig.savefig(path,dpi=180);plt.close(fig)
        data[name]=dict(file=str(path.relative_to(output)),sha256=sha256(path),data=records)
    fig,axes=plt.subplots(2,2,figsize=(13,8));used=[]
    for ax,metric in zip(axes[0],('epsilon','forgetting')):
        rr=[r for r in native if r.get('diagnostic')=='D-a' and r.get('layer')==24 and r.get('region')=='Global' and r.get('metric')==metric and lag_position(r.get('lag')) is not None]
        used+=rr;series(ax,rr,'lag',categorical=lag_position);ax.set_ylabel(metric+' (normalized error)');ax.set_xlabel('Source lag / bucket')
    for ax,metric in zip(axes[1],('cosine','top1')):
        rr=[r for r in probe if r['layer']==24 and r['feature']=='state' and r['metric']==metric and lag_position(r['lag']) is not None]
        used+=rr;series(ax,rr,'lag',categorical=lag_position);ax.set_ylabel('Mean region cosine' if metric=='cosine' else 'Retrieval top-1 (fraction)');ax.set_xlabel('Source lag / bucket')
    for ax in axes.flat:ax.set_xticks(range(len(LAGS)),LAGS,rotation=60,fontsize=7)
    finish('D-a',fig,used)
    fig,axes=plt.subplots(1,2,figsize=(12,5));used=[]
    for ax,contrast in zip(axes,('B0-baseline','targeted-control')):
        rr=[r for r in result['D_b']['contrasts'] if r['feature']=='state' and r['metric']=='cosine' and r['contrast']==contrast]
        used+=rr;ax.axhline(0,color='black',lw=.5)
        if contrast=='targeted-control':
            if any(r['group']!='C' for r in rr):raise ValueError('Targeted comparison requires registered lag group C')
            models=('M5_E1','G3_M5','TRIPLE')
            series(ax,rr,'model',categorical=models.index)
            ax.set_xticks(range(len(models)),models);ax.set_xlabel('Model; lag group C (8–32)')
            ax.margins(x=.2)
        else:
            series(ax,rr,'group',categorical=lambda g:{'A':0,'B':1,'C':2}[g])
            ax.set_xticks([0,1,2],['A (1–3)','B (4–7)','C (8–32)']);ax.set_xlabel('Lag group')
        ax.set_ylabel(contrast+' cosine difference');ax.set_title(contrast)
    finish('D-b',fig,used)
    fig,axes=plt.subplots(2,5,figsize=(21,8));used=[]
    for col,model in enumerate(('M5_E1','G3_M5','TRIPLE','G3','G1')):
        for row,metric in enumerate(('mass','mass_per_token')):
            rr=[r for r in native if r.get('diagnostic')=='D-c' and r['model']==model and r.get('role')=='combined' and r.get('metric')==metric and isinstance(r.get('layer'),int) and r.get('category') not in ('total_bank','bank_lag_8_32')]
            used+=rr;ax=axes[row,col];series(ax,rr,'layer',groupkey='category');ax.set_title(model);ax.set_xlabel('Attention layer');ax.set_ylabel('Probability mass' if metric=='mass' else 'Probability mass / visible token')
    finish('D-c',fig,used)
    maps=spatial_maps(selection_csv);fig,axes=plt.subplots(3,3,figsize=(14,12));used=[]
    for col,model in enumerate(('M5_E1','G3_M5','TRIPLE')):
        for row,metric in enumerate(('normalized_cutoff_margin','candidate_cv')):
            rr=[r for r in native if r.get('diagnostic')=='D-d' and r['model']==model and r['metric']==metric and r.get('scope')=='admission_mean']
            used+=rr;ax=axes[row,col];series(ax,rr,'layer');ax.set_xlabel('Scorer layer');ax.set_ylabel(metric+' (dimensionless)');ax.set_title(model)
        selected=[r for r in maps if r['model']==model];ax=axes[2,col]
        if selected:
            matrix=np.concatenate([np.asarray(r['mean_episode_occupancy_share']) for r in selected],axis=1)
            im=ax.imshow(matrix,aspect='auto',origin='upper',vmin=0);fig.colorbar(im,ax=ax,label='Mean episode token share')
            ax.set_xticks([10+20*i for i in range(len(selected))],[r['layer'] for r in selected]);ax.set_xlabel('Scorer layer (20 columns each)');ax.set_ylabel('Visual grid row')
        else:ax.text(.5,.5,'No defined occupancy',ha='center',transform=ax.transAxes)
    finish('D-d',fig,dict(metric_rows=used,spatial_maps=maps))
    if include_selector:
        fig,axes=plt.subplots(1,2,figsize=(11,5));used=[];names=[]
        for i,(model,value) in enumerate(result['D_e']['models'].items()):
            for j,condition in enumerate(('score','random')):
                r=value['metrics']['sr'][condition];used.append(dict(model=model,condition=condition,**r))
                x=i*3+j;axes[0].bar(x,100*r['estimate'],color=COLORS.get(model),alpha=1 if j==0 else .5)
                axes[0].errorbar(x,100*r['estimate'],yerr=np.asarray([[r['estimate']-r['ci95'][0]],[r['ci95'][1]-r['estimate']]])*100,fmt='none',color='black');names.append((x,model+' '+condition))
            r=value['metrics']['sr']['delta'];used.append(dict(model=model,condition='random-score',**r));axes[1].errorbar(i,100*r['estimate'],yerr=np.asarray([[r['estimate']-r['ci95'][0]],[r['ci95'][1]-r['estimate']]])*100,fmt='o',label=model)
        interaction=result['D_e']['interaction'];used.append(dict(model='interaction',condition='difference-in-differences',**interaction))
        axes[1].errorbar(2,100*interaction['estimate'],yerr=np.asarray([[interaction['estimate']-interaction['ci95'][0]],[interaction['ci95'][1]-interaction['estimate']]])*100,fmt='s')
        axes[0].set_xticks([x for x,_ in names],[n for _,n in names],rotation=30,ha='right');axes[0].set_ylabel('Success rate (%)');axes[1].set_xticks([0,1,2],['M5_E1 Δ','G3_M5 Δ','Interaction'],rotation=25);axes[1].set_ylabel('Success-rate difference (percentage points)');axes[1].axhline(0,color='black',lw=.5)
        finish('D-e',fig,used)
    fig,axes=plt.subplots(2,2,figsize=(13,9));used=[]
    for col,model in enumerate(('G3_M5','TRIPLE')):
        rr=[r for r in native if r.get('diagnostic')=='D-f' and r['model']==model and r['metric']=='selected_overlap' and r.get('scope')=='episode_mean']
        used+=rr;series(axes[0,col],rr,'layer',groupkey='length_cohort');axes[0,col].set_xlabel('Attention layer');axes[0,col].set_ylabel('Mean selected-ID overlap (fraction)');axes[0,col].set_title(model)
        rr=[r for r in native if r.get('diagnostic')=='D-f' and r['model']==model and r['metric']=='selected_overlap' and r.get('scope')=='per_step' and r['length_cohort']=='all']
        used+=rr;series(axes[1,col],rr,'step',groupkey='layer');axes[1,col].set_xlabel('Expert trajectory step');axes[1,col].set_ylabel('Selected-ID overlap (fraction)')
        for ax in axes[:,col]:ax.axhline(.995,color='green',ls=':',lw=.8);ax.axhline(.98,color='red',ls=':',lw=.8)
    finish('D-f',fig,used)
    return data
