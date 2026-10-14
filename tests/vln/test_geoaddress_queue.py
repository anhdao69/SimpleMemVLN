import json
import pytest


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
    write_qualification_seal(source,record,[{'returncode':0}])
    assert check_qualification(source,record)['passed']
    (source/'src'/'x.py').write_text('a=2\n')
    with pytest.raises(ValueError):check_qualification(source,record)


def test_every_queued_step_pins_worker2_and_one_slurm_task():
    from scripts.vln.queue_geoaddress import step_command
    cmd=step_command('4925','/source','A0',['bash','/source/train/launch_a_omega_joint_e1.sh','A0','/run'])
    assert '--nodelist=worker-2' in cmd and '--ntasks=1' in cmd and '--gpus=4' in cmd
    assert '--jobid=4925' in cmd
