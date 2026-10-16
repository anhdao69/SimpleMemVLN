import json
from pathlib import Path
import pytest


def test_qualification_requires_gates_and_unchanged_source(tmp_path):
    from scripts.vln.queue_g3_m5 import source_files,check_qualification,write_seal,REQUIRED_GATES
    source=tmp_path/'source';(source/'src').mkdir(parents=True)
    module=source/'src/model.py';module.write_text('original')
    expected=source_files(source);seal=tmp_path/'seal.json'
    records=[dict(name=n,returncode=0) for n in REQUIRED_GATES]
    with pytest.raises(ValueError):write_seal(source,seal,records[:-1],expected)
    write_seal(source,seal,records,expected)
    assert check_qualification(source,seal)['qualified_arm']=='G3_M5'
    module.write_text('changed')
    with pytest.raises(ValueError):check_qualification(source,seal)


def test_predecessor_needs_success_complete_inventory_and_full_epoch(tmp_path):
    from scripts.vln.queue_g3_m5 import predecessor_status
    from qwen_vl.train.recovery import mark_complete
    assert predecessor_status(tmp_path)=='waiting'
    sup=tmp_path/'supervisor-1';sup.mkdir()
    report=sup/'resources.json';report.write_text(json.dumps({'status':'FAIL'}))
    assert predecessor_status(tmp_path)=='failed'
    report.write_text(json.dumps({'status':'PASS'}))
    final=tmp_path/'train/checkpoint-3852';final.mkdir(parents=True)
    for name in ['pytorch_model.bin','resume_contract.json','navigation.json','optimizer.pt','scheduler.pt']+[f'rng_state_{r}.pth' for r in range(4)]:
        (final/name).write_text('fixture')
    state=final/'trainer_state.json';state.write_text(json.dumps(dict(global_step=3852,epoch=.5)))
    mark_complete(final,4,True)
    with pytest.raises(ValueError):predecessor_status(tmp_path)
    state.write_text(json.dumps(dict(global_step=3852,epoch=1.0)));mark_complete(final,4,True)
    assert predecessor_status(tmp_path)=='complete'
    (final/'rng_state_2.pth').write_text('truncated')
    with pytest.raises(ValueError):predecessor_status(tmp_path)


def test_launch_is_single_task_four_gpus_worker2():
    from scripts.vln.queue_g3_m5 import step_command
    cmd=step_command('4925',Path('/source'),'g3-m5',['python','test.py'])
    assert '--ntasks=1' in cmd and '--gpus=4' in cmd and '--nodelist=worker-2' in cmd
    assert cmd[-2:]==['python','test.py']
