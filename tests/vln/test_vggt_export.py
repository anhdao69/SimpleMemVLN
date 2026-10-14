import json
from pathlib import Path
import pytest
import torch
from scripts.vln.export_vggt_g1 import export_checkpoint
from qwen_vl.contracts import validate_resume_contract


def test_export_and_resume_keep_separate_contracts(tmp_path):
    source=tmp_path/'checkpoint-2';source.mkdir()
    config={'research':{'run_id':'G1','teacher_identity':{'weights_sha256':'fixed'}},'model':{'output_mode':'qwen_text'}}
    metadata={'config':config,'serializer':'unchanged'}
    (source/'navigation.json').write_text(json.dumps(metadata))
    state={'backbone.layer.weight':torch.arange(3.),'research_aux.current_projector.fc1.weight':torch.ones(3)}
    torch.save(state,source/'pytorch_model.bin')
    dest=export_checkpoint(source,tmp_path/'export')
    exported=torch.load(dest/'pytorch_model.bin',weights_only=True)
    assert set(exported)=={'backbone.layer.weight'}
    assert torch.equal(exported['backbone.layer.weight'],state['backbone.layer.weight'])
    assert 'research' not in json.loads((dest/'navigation.json').read_text())['config']
    assert json.loads((source/'navigation.json').read_text())==metadata
    changed=json.loads(json.dumps(metadata));changed['config']['research']['teacher_identity']['weights_sha256']='different'
    with pytest.raises(ValueError): validate_resume_contract({'navigation':metadata,'data':{}},changed,{})
    with pytest.raises(FileExistsError): export_checkpoint(source,dest)


def test_g3_export_drops_state_queries_and_decoder(tmp_path):
    source=tmp_path/'checkpoint';source.mkdir()
    metadata={'config':{'research':{'run_id':'G3'}}}
    (source/'navigation.json').write_text(json.dumps(metadata))
    torch.save({'backbone.weight':torch.ones(2),'research_aux.state_head.query.1.weight':torch.ones(2),
                'research_aux.state_head.decoder.0.weight':torch.ones(2)},source/'pytorch_model.bin')
    target=export_checkpoint(source,tmp_path/'serving')
    assert list(torch.load(target/'pytorch_model.bin',weights_only=True))==['backbone.weight']
    assert 'research' not in json.loads((target/'navigation.json').read_text())['config']


def test_export_audit_records_bounded_policy_failure_without_hiding_runtime_errors():
    from scripts.vln.check_vggt_export import greedy_outcome
    class Session:
        def reset(self,*args):pass
        def observe(self,*args):raise ValueError('Response limit reached without assistant terminator')
    result=greedy_outcome(Session(),'id','instruction',None)
    assert result=={'valid_action':False,'error':'Response limit reached without assistant terminator'}
    class Broken(Session):
        def observe(self,*args):raise ValueError('Invalid, busy, or mismatched session')
    with pytest.raises(ValueError,match='mismatched'):
        greedy_outcome(Broken(),'id','instruction',None)


@pytest.mark.parametrize('arm',['A0','A1'])
def test_address_export_preserves_runtime_without_auxiliary_head(tmp_path,arm):
    source=tmp_path/'source';source.mkdir()
    config={'research':{'run_id':arm},'memory':{'mode':'window8'},'model':{'output_mode':'qwen_text',
        'geoaddress':dict(version='geoaddress_v1',layer=24,rank=8,heads=[0,1],init_seed=429)}}
    (source/'navigation.json').write_text(json.dumps({'config':config}))
    prefix='backbone.model.language_model.layers.24.linear_attn.geoaddress.'
    state={'backbone.weight':torch.ones(2)}
    for name,shape in [('q_down',(8,2560)),('k_down',(8,2560)),('q_up',(256,8)),('k_up',(256,8))]:
        state[prefix+name+'.weight']=torch.randn(shape)
    torch.save(state,source/'pytorch_model.bin')
    target=export_checkpoint(source,tmp_path/'serving')
    exported=torch.load(target/'pytorch_model.bin',weights_only=True)
    assert exported.keys()==state.keys()
    for key in state:assert torch.equal(exported[key],state[key])
    saved=json.loads((target/'navigation.json').read_text())['config']
    assert 'research' not in saved and saved['model']['geoaddress']==config['model']['geoaddress']
    del state[prefix+'q_up.weight'];torch.save(state,source/'pytorch_model.bin')
    with pytest.raises(ValueError,match='GeoAddress'):export_checkpoint(source,tmp_path/'broken')
