"""Fresh-process A0/A1 resume vs uninterrupted weights and adapter movement."""
import argparse,json
from pathlib import Path
import torch
from scripts.vln.check_vggt_recovery import compare
from qwen_vl.train.recovery import validate_complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--resumed',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    ref,res=Path(a.reference),Path(a.resumed)
    source=validate_complete(ref/'checkpoint-2');last=validate_complete(res/'checkpoint-3')
    assert source['world_size']==last['world_size']==4 and last['step']==3
    def load(path):return torch.load(path/'pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True)
    state,resumed,earlier=load(ref/'final'),load(res/'final'),load(ref/'checkpoint-2')
    prefix='backbone.model.language_model.layers.24.linear_attn.geoaddress.'
    weights=compare((state,resumed));adapter=compare((state,resumed),prefix)
    movement=compare((earlier,state),prefix)
    assert adapter['elements']==45056 and movement['changed_elements']>0
    changes={k:int((state[k]!=earlier[k]).count_nonzero()) for k in state if k.startswith(prefix)}
    assert all(changes[prefix+n+'.weight']>0 for n in ('q_up','k_up'))
    assert weights['relative_l2']<=1e-5 and adapter['relative_l2']<=1e-4
    logs=[json.loads((path/'trainer_state.json').read_text())['log_history'] for path in (ref,res)]
    losses=[next(r['loss'] for r in reversed(log) if r.get('step')==3 and 'loss' in r) for log in logs]
    assert abs(losses[0]-losses[1])<=1e-4
    report=dict(passed=True,weights=weights,adapters=adapter,movement=movement,changes=changes,
                losses=losses,checkpoint_files=len(last['files']),world_size=last['world_size'])
    Path(a.out).write_text(json.dumps(report,indent=2));print('ADDRESS_RECOVERY_PASS',report)

if __name__=='__main__':main()
