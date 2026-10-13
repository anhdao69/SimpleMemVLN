"""Compare a fresh-process G1 continuation with its uninterrupted reference.

Report bitwise equality separately. Predeclared BF16 numerical gate: global
relative weight L2 <=1e-5, head relative L2 <=1e-4, final logged loss <=1e-4.
Optimizer/RNG inventories must be complete; tensors are not compared filewise.
"""
import argparse
import json
from pathlib import Path
import torch
from qwen_vl.train.recovery import validate_complete


def compare(states,prefix=None):
    before,after=states;assert before.keys()==after.keys()
    error=scale=0.;changed=count=0;maximum=0.
    for key in before:
        if prefix is not None and not key.startswith(prefix):continue
        a,b=before[key],after[key];assert a.shape==b.shape and a.dtype==b.dtype,key
        if not a.is_floating_point():assert torch.equal(a,b);continue
        d=(a.float()-b.float());error+=float(d.double().square().sum())
        scale+=float(a.double().square().sum());maximum=max(maximum,float(d.abs().max()))
        changed+=int(d.count_nonzero());count+=a.numel()
    return dict(relative_l2=(error/max(scale,1e-30))**.5,max_abs=maximum,
                changed_elements=changed,elements=count,bitwise_equal=changed==0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True)
    p.add_argument('--resumed',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    ref,res=Path(a.reference),Path(a.resumed)
    source=validate_complete(ref/'checkpoint-2');last=validate_complete(res/'checkpoint-3')
    assert source['world_size']==last['world_size']==4 and last['step']==3
    def load(path):return torch.load(path/'pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True)
    state=load(ref/'final');resumed=load(res/'final');earlier=load(ref/'checkpoint-2')
    weights=compare((state,resumed));head=compare((state,resumed),'research_aux.')
    movement=compare((earlier,state),'research_aux.')
    assert movement['changed_elements']>0,'No saved head update at nonzero auxiliary weight'
    bias_updates={k:int((state[k]!=earlier[k]).count_nonzero()) for k in state
                  if k.startswith('research_aux.') and k.endswith('.bias')}
    assert sum(bias_updates.values())>0,'Head movement could be weight decay alone'
    assert weights['relative_l2']<=1e-5 and head['relative_l2']<=1e-4
    logs=[json.loads((path/'trainer_state.json').read_text())['log_history'] for path in (ref,res)]
    losses=[next(r['loss'] for r in reversed(log) if r.get('step')==3 and 'loss' in r) for log in logs]
    assert abs(losses[0]-losses[1])<=1e-4
    result=dict(passed=True,reference=str(ref),resumed=str(res),weights=weights,head=head,
                head_update_from_step2_to3=movement,step3_losses=losses,
                head_bias_updates=bias_updates,
                recovery_world_size=last['world_size'],checkpoint_files=len(last['files']))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print('G1_RECOVERY_COMPARE_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
