import importlib
import json
import pytest


def queue():return importlib.import_module('scripts.vln.queue_g3a1_then_g3_fixed')


def test_sequence_order_restart_and_failure():
    names=['g3a1','g3_fixed'];done=set();calls=[]
    def launch(name):calls.append(name);done.add(name)
    queue().run_sequence(names,launch,lambda n:n in done,lambda **kw:None)
    assert calls==names
    queue().run_sequence(names,launch,lambda n:n in done,lambda **kw:None)
    assert calls==names
    done.clear();calls.clear()
    def failed(name):calls.append(name);raise RuntimeError('failure')
    with pytest.raises(RuntimeError):queue().run_sequence(names,failed,lambda n:False,lambda **kw:None)
    assert calls==['g3a1']


def test_zero_exit_before_epoch_end_never_starts_future():
    calls=[]
    with pytest.raises(RuntimeError,match='complete'):
        queue().run_sequence(['g3a1','g3_fixed'],calls.append,lambda n:False,lambda **kw:None)
    assert calls==['g3a1']


def test_source_and_config_seal_rejects_mutation(tmp_path):
    source=tmp_path/'src';source.mkdir();(source/'a.py').write_text('pass')
    files=queue().source_files(tmp_path)
    queue().check_source(tmp_path,files)
    (source/'a.py').write_text('changed')
    with pytest.raises(ValueError,match='source'):queue().check_source(tmp_path,files)


def test_completion_requires_correct_epoch_model_and_contract(tmp_path):
    run=tmp_path/'run';train=run/'train';cp=train/'checkpoint-3852';cp.mkdir(parents=True)
    nav={'config':{'research':{'run_id':'G3A1'}}}
    def write(path,obj):path.write_text(json.dumps(obj))
    write(cp/'navigation.json',nav)
    write(cp/'trainer_state.json',dict(global_step=3852,max_steps=3852,epoch=1.))
    write(cp/'RECOVERY_COMPLETE.json',dict(step=3852,world_size=4,epoch_boundary=True,
        files={'navigation.json':(cp/'navigation.json').stat().st_size,'trainer_state.json':(cp/'trainer_state.json').stat().st_size}))
    stage=dict(run=str(run),navigation=nav,steps=3852)
    assert not queue().stage_complete(stage)
    final=train/'final';final.mkdir();(final/'pytorch_model.bin').write_bytes(b'weights')
    write(final/'export_info.json',dict(global_step=3852,max_steps=3852,epoch=1.))
    write(final/'reload_reference.json',dict(loss_sum=1.))
    write(train/'trainer_state.json',dict(global_step=3852,max_steps=3852,epoch=1.))
    assert queue().stage_complete(stage)
    stage['navigation']={'config':{'research':{'run_id':'G1'}}}
    with pytest.raises(ValueError,match='contract'):queue().stage_complete(stage)


def test_restarted_controller_waits_for_surviving_launcher(tmp_path):
    import subprocess,sys
    lock=tmp_path/'launcher.lock';done=tmp_path/'done'
    child=subprocess.Popen([sys.executable,'-c',
        'import fcntl,pathlib,sys,time;'
        'f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX);'
        'print("locked",flush=True);time.sleep(.3);pathlib.Path(sys.argv[2]).write_text("done")',
        str(lock),str(done)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        launches=[]
        queue().launch_after_existing(lock,done.exists,lambda:launches.append('duplicate'))
        assert done.exists() and launches==[]
        child.wait(timeout=5)
        done.unlink()
        queue().launch_after_existing(lock,done.exists,lambda:launches.append('resume'))
        assert launches==['resume']
    finally:
        if child.poll() is None:child.kill()
        child.wait()
