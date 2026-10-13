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
