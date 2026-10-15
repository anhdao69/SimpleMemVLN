import json
import pytest


def test_r1_seal_requires_every_gate_arm_and_exact_source(tmp_path):
    from scripts.vln.queue_vggt_r1 import REQUIRED_GATES,check_qualification,write_seal
    from scripts.vln.queue_geoaddress import source_files
    source=tmp_path/'source';(source/'src').mkdir(parents=True)
    code=source/'src/x.py';code.write_text('x=1')
    seal=tmp_path/'seal.json';records=[dict(name=n,returncode=0) for n in REQUIRED_GATES]
    write_seal(source,seal,records,source_files(source))
    assert check_qualification(source,seal)['qualified_arm']=='R1'
    for bad in (records[:-1],records+[records[0]],[dict(r,returncode=1) for r in records]):
        with pytest.raises(ValueError):write_seal(source,seal,bad,source_files(source))
    data=json.loads(seal.read_text());data['qualified_arm']='G3';seal.write_text(json.dumps(data))
    with pytest.raises(ValueError):check_qualification(source,seal)
    write_seal(source,seal,records,source_files(source));code.write_text('x=2')
    with pytest.raises(ValueError):check_qualification(source,seal)


def test_registry_gate_rejects_changed_baseline_artifact(tmp_path):
    import hashlib
    from scripts.vln.queue_vggt_r1 import check_baseline
    from test_vggt_r1_contract import config
    cfg=config();record=cfg['research']['baseline'];record['artifact']=str(tmp_path)
    nav=tmp_path/'navigation.json';weight=tmp_path/'pytorch_model.bin'
    nav.write_text(json.dumps(record['metadata']));weight.write_bytes(b'baseline')
    record['navigation_sha256']=hashlib.sha256(nav.read_bytes()).hexdigest()
    record['weights_sha256']=hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest=tmp_path/'joint.jsonl';manifest.write_text('rxr15\n')
    record['manifest']=str(manifest)
    record['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    check_baseline(cfg)
    check_baseline(cfg,manifest)
    with pytest.raises(ValueError,match='manifest path'):check_baseline(cfg,tmp_path/'other.jsonl')
    manifest.write_text('wrong data\n')
    with pytest.raises(ValueError,match='manifest'):check_baseline(cfg)
    manifest.write_text('rxr15\n')
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='baseline'):check_baseline(cfg)
