import json
from pathlib import Path
import pytest
from scripts.vln.queue_g3_layers import run_sequence, check_seal, write_seal, ARMS, GATES


def make_seal(tmp_path):
    source=tmp_path/'source';source.mkdir(parents=True);(source/'src').mkdir()
    (source/'src/a.py').write_text('x=1')
    seal=tmp_path/'seal.json'
    from scripts.vln.queue_g3_m5 import source_files
    records=[dict(name=g,returncode=0) for g in GATES]
    write_seal(source,seal,records,source_files(source))
    return source,seal


def test_seal_rejects_incomplete_and_modified_source(tmp_path):
    source,seal=make_seal(tmp_path)
    check_seal(source,seal)
    data=json.loads(seal.read_text());data['records'].pop();seal.write_text(json.dumps(data))
    with pytest.raises(ValueError):check_seal(source,seal)
    source,seal=make_seal(tmp_path/'next')
    (source/'src/a.py').write_text('x=2')
    with pytest.raises(ValueError):check_seal(source,seal)


def test_sequence_restart_and_failure_do_not_skip_arm(tmp_path):
    states={arm:False for arm in ARMS};calls=[];events=[]
    def complete(arm):return states[arm]
    def launch(arm):calls.append(arm);states[arm]=True
    run_sequence(launch,complete,lambda **data:events.append(data))
    assert calls==list(ARMS)
    run_sequence(launch,complete,lambda **data:events.append(data))
    assert calls==list(ARMS)
    states[ARMS[1]]=False;states[ARMS[2]]=False
    def fail(arm):calls.append(arm);raise RuntimeError('child failed')
    with pytest.raises(RuntimeError):run_sequence(fail,complete,lambda **data:events.append(data))
    assert calls[-1]==ARMS[1] and events[-1]['status']=='blocked'


def test_zero_exit_without_complete_checkpoint_blocks_next_arm():
    calls=[]
    with pytest.raises(RuntimeError,match='checkpoint'):
        run_sequence(lambda arm:calls.append(arm),lambda arm:False,lambda **data:None)
    assert calls==[ARMS[0]]


def test_run_binding_rejects_other_source_and_existing_unbound_run(tmp_path):
    from scripts.vln.queue_g3_layers import bind_run
    source,seal=make_seal(tmp_path/'first');run=tmp_path/'run'
    bind_run(run,'l8',source,seal)
    bind_run(run,'l8',source,seal)
    with pytest.raises(ValueError,match='identity'):bind_run(run,'l16',source,seal)
    second,secondseal=make_seal(tmp_path/'second')
    with pytest.raises(ValueError,match='identity'):bind_run(run,'l8',second,secondseal)
    unbound=tmp_path/'unbound';unbound.mkdir();(unbound/'train').mkdir()
    with pytest.raises(ValueError,match='unbound'):bind_run(unbound,'l8',source,seal)


def test_complete_wrong_arm_is_never_skipped(tmp_path,monkeypatch):
    import scripts.vln.queue_g3_layers as queue
    monkeypatch.setattr(queue,'predecessor_status',lambda root:'complete')
    final=tmp_path/'train/checkpoint-3852';final.mkdir(parents=True)
    (final/'navigation.json').write_text(json.dumps({'config':{'research':{'run_id':'G3_M5','student_layers':[16]}}}))
    with pytest.raises(ValueError,match='layer'):queue.arm_complete(tmp_path,'l8')
    assert queue.arm_complete(tmp_path,'l16')
