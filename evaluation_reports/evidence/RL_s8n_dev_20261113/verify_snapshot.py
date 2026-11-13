import json,hashlib,csv
from pathlib import Path
base=Path('evaluation_reports/evidence/RL_s8n_dev_20261113')
m=json.loads((base/'snapshot_manifest.json').read_text());verified=[]
for rel,item in m['files'].items():assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==item['sha256'],rel
for item in m['evaluations']:
 d=base/'evaluations'/item['name'];s=json.loads((d/'summary.json').read_text());p=d/'episode_metrics.jsonl';rows=[json.loads(line) for line in p.open() if line.strip()]
 n=len(rows);success=sum(float(x.get('metrics',{}).get('success',0) or 0) for x in rows)
 assert n==s['episodes']==item['records'] and abs(success/n-s['sr'])<1e-9,item['name']
 ids=[x['episode_uid'] for x in rows];assert len(ids)==len(set(ids)),item['name']
 verified.append({'name':item['name'],'episodes':n,'successes':int(success),'sr_percent':success/n*100,'failed':sum(bool(x.get('failed')) for x in rows),'source_journal_sha256':item['journal_sha256']})
with (base/'verified_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
(base/'local_verification.json').write_text(json.dumps({'verified_evaluations':len(verified),'episode_records':sum(x['episodes'] for x in verified),'unique_ids_checked_within_each_run':True,'scope':'Counts and success rates recomputed from compact metrics; source file checksums verified. Stored GPU2 summary-only records and narrative statistical claims are not independently revalidated here. No training or simulations rerun.'},indent=2)+'\n')
print('Validated',len(verified),'evaluations,',sum(x['episodes'] for x in verified),'episode records and',len(m['files']),'copied source hashes.')
for x in verified:
 if x['name'] in ['rewind_sfr_mix1k_s125_rxrfull_fixed_val_unseen','grpo_sfr_mix8k_aplus_s125_rxrfull_fixed_val_unseen']:print(x)
