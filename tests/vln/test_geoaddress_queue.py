import json
import pytest


def qualification_records(arm):
    common=['full-suite-initial','teacher-qc','distributed-pair-normalization','full-suite-final']
    gates=['mixed','rxrce_longest','r2rce_longest','threshold','mixed_extreme',
           'recovery-reference','recovery-resumed','recovery-compare','export',
           'r2rce-timing','rxrce-timing']
    if arm=='A1':gates.append('mixed-empty')
    return [dict(name=name,returncode=0) for name in common+[arm+'-'+g for g in gates]]


def test_predecessor_does_not_pass_running_or_failed_supervisor(tmp_path):
    from scripts.vln.queue_geoaddress import predecessor_status
    assert predecessor_status(tmp_path,3852)=='waiting'
    supervisor=tmp_path/'supervisor-20260101';supervisor.mkdir()
    (supervisor/'resources.json').write_text(json.dumps({'status':'FAIL'}))
    assert predecessor_status(tmp_path,3852)=='failed'
    (supervisor/'resources.json').write_text(json.dumps({'status':'PASS'}))
    # A successful process without a full checkpoint is not successful training.
    with pytest.raises((ValueError,FileNotFoundError)):predecessor_status(tmp_path,3852)


def test_source_qualification_fails_closed(tmp_path):
    from scripts.vln.queue_geoaddress import check_qualification
    source=tmp_path/'source';(source/'src').mkdir(parents=True);(source/'src'/'x.py').write_text('a=1\n')
    path=tmp_path/'qualified.json';path.write_text(json.dumps({'passed':True,'source_files':{}}))
    with pytest.raises(ValueError):check_qualification(source,path)


def test_qualification_record_roundtrip_and_modified_source_rejected(tmp_path):
    from scripts.vln.queue_geoaddress import check_qualification,source_files
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    source=tmp_path/'source';(source/'src').mkdir(parents=True);(source/'src'/'x.py').write_text('a=1\n')
    record=tmp_path/'seal.json'
    write_qualification_seal(source,record,qualification_records('A1'),source_files(source),arms=['A1'])
    assert check_qualification(source,record)['passed']
    (source/'src'/'x.py').write_text('a=2\n')
    with pytest.raises(ValueError):check_qualification(source,record)


def test_every_queued_step_pins_worker2_and_one_slurm_task():
    from scripts.vln.queue_geoaddress import step_command
    cmd=step_command('4925','/source','A0',['bash','/source/train/launch_a_omega_joint_e1.sh','A0','/run'])
    assert '--nodelist=worker-2' in cmd and '--ntasks=1' in cmd and '--gpus=4' in cmd
    assert '--jobid=4925' in cmd


def test_source_changed_during_qualification_rejected(tmp_path):
    from scripts.vln.queue_geoaddress import source_files
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    (tmp_path/'src').mkdir();p=tmp_path/'src/x.py';p.write_text('a=1')
    initial=source_files(tmp_path);p.write_text('a=2')
    with pytest.raises(ValueError,match='changed'):
        write_qualification_seal(tmp_path,tmp_path/'seal.json',qualification_records('A1'),initial,arms=['A1'])


def test_a0_qualification_cannot_authorize_a1(tmp_path):
    from scripts.vln.queue_geoaddress import check_qualification,source_files
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    (tmp_path/'src').mkdir();(tmp_path/'src/x.py').write_text('x=1')
    seal=tmp_path/'seal.json'
    write_qualification_seal(tmp_path,seal,qualification_records('A0'),source_files(tmp_path),arms=['A0'])
    assert check_qualification(tmp_path,seal,arm='A0')['qualified_arms']==['A0']
    with pytest.raises(ValueError,match='A1'):
        check_qualification(tmp_path,seal,arm='A1')


@pytest.mark.parametrize('mutation',['missing-longest','failed-recovery','duplicate-gate'])
def test_incomplete_or_failed_arm_evidence_cannot_be_sealed(tmp_path,mutation):
    from scripts.vln.queue_geoaddress import source_files
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    (tmp_path/'src').mkdir();(tmp_path/'src/x.py').write_text('x=1')
    records=qualification_records('A1')
    if mutation=='missing-longest':records=[r for r in records if r['name']!='A1-rxrce_longest']
    elif mutation=='failed-recovery':
        next(r for r in records if r['name']=='A1-recovery-compare')['returncode']=1
    else:records.append(records[-1].copy())
    with pytest.raises(ValueError):
        write_qualification_seal(tmp_path,tmp_path/'seal.json',records,source_files(tmp_path),arms=['A1'])


def test_launch_rechecks_arm_evidence_not_just_pass_boolean(tmp_path):
    from scripts.vln.queue_geoaddress import check_qualification,source_files
    (tmp_path/'src').mkdir();(tmp_path/'src/x.py').write_text('x=1')
    seal=tmp_path/'seal.json'
    seal.write_text(json.dumps(dict(passed=True,qualified_arms=['A1'],source_files=source_files(tmp_path),
                                   records=[dict(name='A1-mixed',returncode=0)])))
    with pytest.raises(ValueError):check_qualification(tmp_path,seal,arm='A1')


@pytest.mark.parametrize('arms',[[],['A1','A1'],['A2']])
def test_invalid_qualification_arm_selection_rejected(tmp_path,arms):
    from scripts.vln.queue_geoaddress import source_files
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    (tmp_path/'src').mkdir();(tmp_path/'src/x.py').write_text('x=1')
    with pytest.raises(ValueError):
        write_qualification_seal(tmp_path,tmp_path/'seal.json',qualification_records('A1'),source_files(tmp_path),arms=arms)


def test_clean_partial_checkpoint_waits(tmp_path,monkeypatch):
    from scripts.vln.queue_geoaddress import predecessor_status
    from qwen_vl.train import recovery
    supervisor=tmp_path/'supervisor-20260101';supervisor.mkdir()
    (supervisor/'resources.json').write_text(json.dumps({'status':'PASS'}))
    checkpoint=tmp_path/'train/checkpoint-2500';checkpoint.mkdir(parents=True)
    monkeypatch.setattr(recovery,'validate_complete',lambda p:dict(step=2500,world_size=4))
    assert predecessor_status(tmp_path,3852)=='waiting'


def test_a1_only_queue_qualifies_then_trains_without_replaying_a0(tmp_path,monkeypatch):
    from scripts.vln import queue_geoaddress as queue
    from scripts.vln.qualify_geoaddress import write_qualification_seal
    source=tmp_path/'source';(source/'src').mkdir(parents=True);(source/'src/x.py').write_text('x=1')
    root=tmp_path/'continuation';root.mkdir();seal=root/'qualification/qualification_pass.json'
    a0=tmp_path/'existing_a0';a1=root/'a1_joint_e1'
    states={a0:'complete'};calls=[]
    monkeypatch.setattr(queue,'predecessor_status',lambda p,n:states.get(p,'waiting'))
    def run_step(name,command):
        calls.append((name,command))
        if name=='address-qualification':
            assert command[-2:]==['--arms','A1']
            seal.parent.mkdir()
            write_qualification_seal(source,seal,qualification_records('A1'),queue.source_files(source),arms=['A1'])
        else:
            assert name=='a1-joint-e1'
            assert command[-4:]==['A1',str(a1),str(seal),str(a0)]
            states[a1]='complete'
    queue.execute_sequence(['A1'],source,root,a0,3852,seal,run_step)
    assert [name for name,cmd in calls]==['address-qualification','a1-joint-e1']
    queue.execute_sequence(['A1'],source,root,a0,3852,seal,run_step)
    assert len(calls)==2  # Restart does not replay qualification or completed training.


@pytest.mark.parametrize('status',['waiting','failed'])
def test_a1_sequence_never_uses_gpus_before_successful_a0(tmp_path,monkeypatch,status):
    from scripts.vln import queue_geoaddress as queue
    monkeypatch.setattr(queue,'predecessor_status',lambda p,n:status)
    def forbidden(*args):raise AssertionError('GPU step must not launch')
    with pytest.raises(RuntimeError,match='predecessor'):
        queue.execute_sequence(['A1'],tmp_path,tmp_path,tmp_path/'a0',3852,tmp_path/'seal.json',forbidden)


def test_failed_qualification_cannot_launch_training(tmp_path,monkeypatch):
    from scripts.vln import queue_geoaddress as queue
    monkeypatch.setattr(queue,'predecessor_status',lambda p,n:'complete')
    calls=[]
    def fail(name,command):
        calls.append(name);raise RuntimeError('Qualification failed')
    with pytest.raises(RuntimeError,match='Qualification failed'):
        queue.execute_sequence(['A1'],tmp_path,tmp_path,tmp_path/'a0',3852,tmp_path/'seal.json',fail)
    assert calls==['address-qualification']


def test_allocation_is_busy_until_previous_gpu_step_exits(monkeypatch):
    from scripts.vln import queue_geoaddress as queue
    import subprocess
    rows='4925.0|train_vln\n4925.41|a0-joint-e1\n'
    def listing(cmd,**kwargs):return subprocess.CompletedProcess(cmd,0,stdout=rows)
    monkeypatch.setattr(queue.subprocess,'run',listing)
    assert not queue.allocation_idle('4925')
    rows='4925.0|train_vln\n4925.extern|extern\n'
    assert queue.allocation_idle('4925')
