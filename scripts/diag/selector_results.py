"""D-e paired closed-loop summaries with fixed denominators and exact McNemar."""
import argparse
import csv
import json
import math
from pathlib import Path
from scripts.diag.statistics import episode_bootstrap,paired_bootstrap,mcnemar
from scripts.diag.selector_eval import sha256,panel_ids

METRICS=('sr','spl','os','ne','forced_stop')


def effective_metric(record,name):
    if name=='forced_stop':return float(bool(record.get('forced_stop')))
    if name in ('sr','spl') and record['failed']:return 0.
    source={'sr':'success','spl':'spl','os':'oracle_success','ne':'distance_to_goal'}[name]
    value=record['metrics'].get(source,0. if name in ('sr','spl') else None)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def paired_rows(left,right,metric):
    if set(left)!=set(right):raise ValueError('Paired episode denominators differ')
    rows=[]
    for eid in sorted(left):
        a,b=left[eid],right[eid]
        if any(a[key]!=b[key] for key in ('episode_uid','scene_id','instruction')):
            raise ValueError('Paired UID/scene/instruction mismatch')
        rows.append(dict(episode_id=eid,scene=a['scene_id'],left=effective_metric(a,metric),right=effective_metric(b,metric)))
    return rows


def determinism(original,repeated):
    rows=paired_rows(original,repeated,'sr');metrics=actions=success=0
    for row in rows:
        a,b=original[row['episode_id']],repeated[row['episode_id']]
        success+=row['left']==row['right']
        agrees=True
        for metric in METRICS:
            x,y=effective_metric(a,metric),effective_metric(b,metric)
            agrees &= ((x is None and y is None) or (x is not None and y is not None and abs(x-y)<=1e-6))
        metrics+=agrees and a['failed']==b['failed']
        aa=[(v['predicted'],v['executed']) for v in a.get('actions',[])]
        bb=[(v['predicted'],v['executed']) for v in b.get('actions',[])]
        actions+=aa==bb
    n=len(rows)
    return dict(n_episodes=n,success_agreement=success/n,metric_agreement=metrics/n,
                action_sequence_agreement=actions/n,absolute_metric_tolerance=1e-6,
                success_agree_count=success,metric_agree_count=metrics,action_agree_count=actions)


def read_journal(path,wanted,exact=False):
    result={};seen=set()
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);eid=str(row['episode_id'])
        if eid in seen:raise ValueError('Duplicate journal episode')
        seen.add(eid)
        if eid in wanted:result[eid]=row
    if exact and seen!=set(wanted):raise ValueError('Worker journal contains unregistered episodes')
    if set(result)!=set(wanted):raise ValueError('Journal does not cover the fixed panel')
    return result


def read_workers(root,wanted,condition,*,panel_hash=None,baseline_hash=None):
    result={};evidence=[]
    for worker in sorted(Path(root).glob('shard-*')):
        contract=json.loads((worker/'evaluation_contract.json').read_text())
        summary=json.loads((worker/'summary.json').read_text())
        if (not contract['diagnostic'] or contract['qualified'] or contract['condition']!=condition or
            contract['smoke'] or not summary.get('complete') or
            summary.get('diagnostic') is not True or summary.get('qualified') is not False or
            summary.get('condition')!=condition or summary.get('smoke') is not False):
            raise ValueError('Incomplete/non-diagnostic/smoke worker cannot enter D-e results')
        if panel_hash is not None and contract['input_sha256']['panel']!=panel_hash:raise ValueError('Worker frozen panel hash differs')
        if baseline_hash is not None and contract['input_sha256']['baseline_journal']!=baseline_hash:raise ValueError('Worker baseline model/journal hash differs')
        if len(contract['episode_ids'])!=len(set(contract['episode_ids'])):raise ValueError('Duplicate scheduled worker episode')
        rows=read_journal(worker/'episodes.jsonl',set(contract['episode_ids']),exact=True)
        if summary.get('episodes')!=len(rows) or summary.get('scheduled_episodes')!=len(rows):raise ValueError('Worker completion denominator differs')
        if set(result)&set(rows):raise ValueError('Worker panels overlap')
        for row in rows.values():
            if not row.get('diagnostic') or row.get('qualified') is not False or row.get('condition')!=condition:
                raise ValueError('Row diagnostic labels differ')
            if row.get('smoke') is not False:raise ValueError('Smoke row cannot enter D-e results')
        result.update(rows)
        evidence.append(dict(root=str(worker),journal_sha256=sha256(worker/'episodes.jsonl'),
                             contract_sha256=sha256(worker/'evaluation_contract.json')))
    if set(result)!=set(wanted):raise ValueError('Workers do not cover the complete frozen panel')
    return result,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--baseline',action='append',required=True,help='MODEL=original/full/episodes.jsonl')
    parser.add_argument('--runs',action='append',required=True,help='MODEL=directory containing random/ and score/ shard-*')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    baseline=dict(item.split('=',1) for item in args.baseline);runs=dict(item.split('=',1) for item in args.runs)
    if set(baseline)!=set(runs) or set(runs)!={'M5_E1','G3_M5'}:raise ValueError('Require both registered models')
    panel=json.loads(args.panel.read_text());wanted=set(panel_ids(panel,'random'));repeat=set(panel_ids(panel,'score'))
    args.output.mkdir(parents=True,exist_ok=False)
    results=dict(diagnostic=True,qualified=False,panel_sha256=sha256(args.panel),models={},metric_units={'sr':'fraction','spl':'fraction','os':'fraction','ne':'meters','forced_stop':'fraction'},
                 sr_display_conversion='multiply fractions and CI bounds by 100 for percentage points')
    deltas={};csv_rows=[];model_originals={}
    for model in ('M5_E1','G3_M5'):
        original=read_journal(baseline[model],wanted)
        model_originals[model]=original
        random,evidence=read_workers(Path(runs[model])/'random',wanted,'random',panel_hash=sha256(args.panel),baseline_hash=sha256(baseline[model]))
        score,repeat_evidence=read_workers(Path(runs[model])/'score',repeat,'score',panel_hash=sha256(args.panel),baseline_hash=sha256(baseline[model]))
        result=dict(n_episodes=600,n_determinism=60,baseline_sha256=sha256(baseline[model]),workers=evidence,
                    determinism_workers=repeat_evidence,metrics={},
                    determinism=determinism({eid:original[eid] for eid in repeat},score))
        for metric in METRICS:
            rows=paired_rows(original,random,metric)
            result['metrics'][metric]=dict(score=episode_bootstrap([dict(r,value=r['left']) for r in rows]),
                random=episode_bootstrap([dict(r,value=r['right']) for r in rows]),delta=paired_bootstrap(rows))
            csv_rows.extend(dict(model=model,metric=metric,**r) for r in rows)
            if metric=='sr':
                result['mcnemar']=mcnemar([r['left'] for r in rows],[r['right'] for r in rows])
                deltas[model]={r['episode_id']:dict(scene=r['scene'],value=r['right']-r['left']) for r in rows}
        result['forced_stop_counts']={key:sum(bool(r.get('forced_stop')) for r in records.values())
                                      for key,records in [('score',original),('random',random)]}
        result['execution_failures']={key:sum(bool(r['failed']) for r in records.values())
                                     for key,records in [('score',original),('random',random)]}
        results['models'][model]=result
    paired_rows(model_originals['M5_E1'],model_originals['G3_M5'],'sr')  # UID/instruction as well as scene
    interaction=[]
    for eid in sorted(wanted):
        a,b=deltas['M5_E1'][eid],deltas['G3_M5'][eid]
        if a['scene']!=b['scene']:raise ValueError('Interaction scene pairing differs')
        interaction.append(dict(episode_id=eid,scene=a['scene'],left=a['value'],right=b['value']))
    results['interaction']=paired_bootstrap(interaction)
    results['interaction']['definition']='delta_SR(G3_M5 random-score) - delta_SR(M5_E1 random-score)'
    with (args.output/'episodes.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    with (args.output/'interaction.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(interaction[0]));writer.writeheader();writer.writerows(interaction)
    (args.output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
