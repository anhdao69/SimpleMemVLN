import json
import pytest
import torch
from test_vggt_r1_contract import config


def test_r1_export_keeps_every_installed_lane_and_strips_training_head(tmp_path):
    from scripts.vln.export_vggt_g1 import export_checkpoint
    cfg=config();source=tmp_path/'source';source.mkdir()
    (source/'navigation.json').write_text(json.dumps({'config':cfg}))
    state={f'backbone.model.language_model.layers.{i}.linear_attn.step_lane.out_proj.weight':torch.ones(2,2)*i
           for i in (16,20,24,28)}
    state['research_aux.state_head.query.weight']=torch.ones(2,2)
    torch.save(state,source/'pytorch_model.bin')
    out=export_checkpoint(source,tmp_path/'export')
    actual=torch.load(out/'pytorch_model.bin',weights_only=True)
    assert set(actual)==set(state)-{'research_aux.state_head.query.weight'}
    for k,v in actual.items():assert torch.equal(v,state[k])
    metadata=json.loads((out/'navigation.json').read_text())
    assert metadata['config']['model']['step_lane']==cfg['model']['step_lane']
    assert 'research' not in metadata['config']
    del state['backbone.model.language_model.layers.28.linear_attn.step_lane.out_proj.weight']
    torch.save(state,source/'pytorch_model.bin')
    with pytest.raises(ValueError,match='lane'):export_checkpoint(source,tmp_path/'bad')


def test_profiler_accepts_r1_then_checks_real_fixture(monkeypatch,tmp_path):
    from scripts.vln.profile_vggt_g1 import main
    monkeypatch.setattr('sys.argv',['profile','--arm','R1','--root',str(tmp_path),'--name','fixture','--fixture',str(tmp_path/'missing')])
    with pytest.raises(FileNotFoundError):main()


@pytest.mark.parametrize('initial', [None, 32])
@pytest.mark.parametrize('fail', [False, True])
def test_canonical_offline_restores_attention_configuration(initial, fail):
    from scripts.vln.check_vggt_export import canonical_offline
    class Model:
        navigation_config = {'runtime': {}}
        def __call__(self, **data):
            assert self.navigation_config['runtime']['window_attention_batch_steps'] == 1
            if fail:
                raise RuntimeError('forward failed')
            return {'logits': torch.ones(2)}
    model = Model()
    if initial is not None:
        model.navigation_config['runtime']['window_attention_batch_steps'] = initial
    before = dict(model.navigation_config['runtime'])
    if fail:
        with pytest.raises(RuntimeError, match='forward failed'):
            canonical_offline(model, {})
    else:
        assert torch.equal(canonical_offline(model, {}), torch.ones(2))
    assert model.navigation_config['runtime'] == before


def test_replay_comparison_reports_failure_without_loosening_tolerance():
    from scripts.vln.check_vggt_export import replay_comparison
    reference = torch.tensor([1.8125])
    streamed = torch.tensor([2.625])
    record = replay_comparison(streamed, reference)
    assert not record['passed'] and record['violations'] == 1
    assert record['max_excess'] > 0
    assert replay_comparison(reference, reference)['passed']
