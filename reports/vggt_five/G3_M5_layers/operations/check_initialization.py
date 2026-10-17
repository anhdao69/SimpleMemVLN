"""Compare zero-LR checkpoint-1 tensors/order after reference saves finish.

Do not run against an actively rewritten diagnostic save: the strict inventory
validator intentionally rejects transient size mismatches.
"""
import argparse
import json
from pathlib import Path
import torch

ARMS={'l8':[8],'l16':[16],'l8_l16_l24':[8,16,24]}


def compare_states(states):
    reference=states['l8']
    backbone={k for k in reference if not k.startswith('research_aux.')}
    initial_head={k.removeprefix('research_aux.state_heads.8.'):v for k,v in reference.items()
                  if k.startswith('research_aux.state_heads.8.')}
    assert backbone and initial_head
    counts={}
    for arm,layers in ARMS.items():
        state=states[arm]
        assert {k for k in state if not k.startswith('research_aux.')}==backbone
        assert all(torch.equal(reference[k],state[k]) for k in backbone),f'{arm}: backbone initialization differs'
        for layer in layers:
            prefix=f'research_aux.state_heads.{layer}.'
            actual={k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)}
            assert actual.keys()==initial_head.keys(),f'{arm}/{layer}: head keys differ'
            assert all(torch.equal(initial_head[k],actual[k]) for k in initial_head),f'{arm}/{layer}: head initialization differs'
        counts[arm]=dict(backbone_tensors=len(backbone),head_layers=layers,bitwise_equal=True)
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);states={};orders={};data_contracts=[]
    from qwen_vl.train.recovery import validate_complete
    for arm,layers in ARMS.items():
        train=root/arm/'experiments/recovery-reference/train';checkpoint=train/'checkpoint-1'
        record=validate_complete(checkpoint);assert record['step']==1 and record['world_size']==4
        navigation=json.loads((checkpoint/'navigation.json').read_text())
        assert navigation['config']['research']['student_layers']==layers
        history=json.loads((checkpoint/'trainer_state.json').read_text())['log_history']
        assert next(r['learning_rate'] for r in history if r.get('step')==1 and 'learning_rate' in r)==0
        states[arm]=torch.load(checkpoint/'pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True)
        data_contracts.append(json.loads((checkpoint/'resume_contract.json').read_text())['data'])
        orders[arm]=[[{k:row[k] for k in ('update','episode_uid','actions','tokens','aux_pair_count','update_aux_count')}
                      for row in map(json.loads,(train/f'exposures_rank{rank}.jsonl').read_text().splitlines()) if row['update']==0]
                     for rank in range(4)]
    assert all(d==data_contracts[0] for d in data_contracts)
    assert all(v==orders['l8'] for v in orders.values()),'Initial sample/target support order differs'
    result=dict(passed=True,zero_learning_rate=True,initialization=compare_states(states),
                identical_data_contract=True,identical_first_update_exposure_order=True)
    path=root/'initialization_check.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
