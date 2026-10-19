import json,pathlib,sys,itertools
r=pathlib.Path('/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019');sys.path.insert(0,str(r/'source/SimpleMemVLN-g3a1-877815e/src'))
from qwen_vl.research.vggt_sampling import collection_trajectory_key
from qwen_vl.research.vggt_g3 import sample_pairs
from qwen_vl.research.geoaddress_sampling import query_cells
import torch
original=r/'download/data/r2r_rxr15_train.jsonl';rebased=r/'production/data/r2r_rxr15_train.rebased.jsonl';alias=json.loads((r/'image_alias.json').read_text());n=actions=changed_naive=pair_checks=cell_checks=0
for i,(a,b) in enumerate(itertools.zip_longest(original.open(),rebased.open())):
 assert a is not None and b is not None
 old,new=json.loads(a),json.loads(b);oldkey,newkey=collection_trajectory_key(old),collection_trajectory_key(new)
 assert oldkey==newkey,(i,old['episode_uid'])
 assert old['episode_uid']==new['episode_uid'] and len(old['steps'])==len(new['steps'])
 normal=dict(old,steps=[dict(s,rgb_path=alias['target']+s['rgb_path'][len(alias['original_root']):]) for s in old['steps']])
 changed_naive+=int(collection_trajectory_key(normal)!=oldkey)
 if i%241==0:
  p,q=sample_pairs(oldkey,len(old['steps'])),sample_pairs(newkey,len(new['steps']));assert p==q;pair_checks+=1
  for source,read in p:
   assert torch.equal(query_cells(oldkey,source,read,429),query_cells(newkey,source,read,429));cell_checks+=1
 n+=1;actions+=len(old['steps'])
assert (n,actions)==(30815,3128624)
report=dict(passed=True,episodes=n,actions=actions,identical_collection_keys=n,naive_rebase_changed_keys=changed_naive,
            explicit_pair_lists_verified=pair_checks,explicit_query_cell_sets_verified=cell_checks,alias=alias,
            justification='All key/length/seed inputs identical across full corpus; pinned pure sampling functions therefore preserve every pair and query cell')
(r/'sampling_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('SAMPLING_AUDIT_PASS',report,flush=True)
