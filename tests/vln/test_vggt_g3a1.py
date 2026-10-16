"""G3+A1 contracts: config, shared pair layout, targets and serving export (CPU)."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from qwen_vl.research.vggt_g1 import parse_g1
from qwen_vl.research.vggt_g3 import sample_pairs, region_targets
from qwen_vl.research.vggt_sampling import collection_trajectory_key

GEO = dict(version='geoaddress_v1', layer=24, rank=8, heads=[0, 1], init_seed=429)


def g3a1_config(**research):
    value = dict(run_id='G3A1', student_layer=24, teacher_source='/s', teacher_weights='/w',
                 teacher_identity={'x': 1}, head_lr=1e-4, lambda_max=.1, ramp_updates=771,
                 adapter_lr=5e-5, relation_projection='direct-native-qk-cosine-v1')
    value.update(research)
    return {'research': value, 'memory': {'mode': 'window8'},
            'model': {'output_mode': 'qwen_text', 'geoaddress': dict(GEO)}}


def test_config_requires_address_relation_and_both_specified_weights():
    assert parse_g1(g3a1_config())['run_id'] == 'G3A1'
    no_address = g3a1_config(); del no_address['model']['geoaddress']
    with pytest.raises(ValueError, match='GeoAddress'): parse_g1(no_address)
    for bad in (dict(adapter_lr=1e-4), dict(relation_projection='mlp'), dict(lambda_max=.2),
                dict(head_lr=5e-5), dict(student_layer=23), dict(ramp_updates=100)):
        with pytest.raises(ValueError): parse_g1(g3a1_config(**bad))
    # Historical arms are unchanged by the new arm.
    g3 = g3a1_config(run_id='G3'); del g3['model']['geoaddress']
    assert parse_g1(g3)['run_id'] == 'G3'
    with pytest.raises(ValueError): parse_g1(g3a1_config(run_id='G3'))


def episode_fixture(steps=40, image_id=7):
    """Each step: 300 image tokens, 4 text tokens; first response target is the last text token."""
    episode = {'dataset': 'r2r', 'scene_id': 'scene', 'episode_uid': 'id', 'instruction': 'go',
               'steps': [{'rgb_path': f'/trajectory_data/a/{i}.png'} for i in range(steps)]}
    width = 304
    ids = torch.zeros(1, steps * width, dtype=torch.long)
    spans = []
    for t in range(steps):
        ids[0, t * width:t * width + 300] = image_id
        spans.append((t * width, (t + 1) * width))
    targets = torch.tensor([t * width + 303 for t in range(steps)])
    encoded = {'input_ids': ids, 'response_target_positions': targets,
               'response_action_index': torch.arange(steps), 'num_actions': steps,
               'step_plan': (None, spans)}
    return episode, encoded


def dataset_for(run_id, episode, encoded, captured, monkeypatch):
    from qwen_vl.data.episode_dataset import EpisodeDataset
    dataset = object.__new__(EpisodeDataset)
    dataset.episodes = [episode]
    dataset.encoded_lengths = [encoded['input_ids'].shape[1]]
    dataset.serializer = SimpleNamespace(config={'runtime': {}, 'research': {'run_id': run_id}},
                                         encode_episode=lambda ep: copy.deepcopy(encoded), image_id=7)

    def images(paths):
        captured.append(list(paths))
        return torch.zeros(len(paths), 3, 448, 592)
    monkeypatch.setattr('qwen_vl.data.vggt_online.preprocess_teacher_images', images)
    return dataset


def test_dataset_g3a1_is_exactly_g3_plus_a1_on_one_pair_list(monkeypatch):
    episode, encoded = episode_fixture()
    seen = {}
    for arm in ('G3', 'A1', 'G3A1'):
        captured = []
        seen[arm] = (dataset_for(arm, episode, encoded, captured, monkeypatch)[0], captured[0])
    (g3, g3_paths), (a1, a1_paths), (both, both_paths) = seen['G3'], seen['A1'], seen['G3A1']
    pairs = sample_pairs(collection_trajectory_key(episode), 40)
    assert pairs and both['aux_pair_count'] == g3['aux_pair_count'] == a1['aux_pair_count'] == len(pairs)
    # G3 fields are identical to the G3 arm.
    assert torch.equal(both['aux_anchors'], g3['aux_anchors'])
    assert torch.equal(both['aux_lags'], g3['aux_lags'])
    # A1 fields are identical to the A1 arm, including teacher frame order.
    for key in ('aux_query_positions', 'aux_key_positions', 'aux_teacher_sources', 'aux_teacher_reads', 'aux_query_cells'):
        assert torch.equal(both[key], a1[key]), key
    assert both_paths == a1_paths
    # G3 regional targets index the union frames at exactly G3's source images.
    assert [both_paths[i] for i in both['aux_teacher_sources']] == [g3_paths[i] for i in g3['aux_source_indices']]
    assert 'aux_source_indices' not in both
    # Same pair on both objectives: source cells < read cells < read anchor.
    assert (both['aux_key_positions'].max(-1).values < both['aux_query_positions'].min(-1).values).all()
    assert (both['aux_query_positions'].max(-1).values < both['aux_anchors']).all()


def test_dataset_g3a1_short_episode_has_empty_support(monkeypatch):
    episode, encoded = episode_fixture(steps=8)
    captured = []
    result = dataset_for('G3A1', episode, encoded, captured, monkeypatch)[0]
    assert result['aux_pair_count'] == 0 and captured == [[]]
    assert result['aux_anchors'].shape == (0,) and result['aux_query_positions'].shape == (0, 32)


def test_targets_match_single_arm_constructions():
    from qwen_vl.research.g3a1_teacher import g3a1_targets
    from qwen_vl.models.geoaddress_loss import teacher_relations
    torch.manual_seed(3)
    desc = torch.randn(5, 300, 16)
    sources, reads = torch.tensor([0, 1, 1]), torch.tensor([2, 3, 4])
    cells = torch.randint(0, 300, (3, 32))
    regions, relations = g3a1_targets(desc, sources, reads, cells)
    torch.testing.assert_close(regions, region_targets(desc)[sources], atol=0, rtol=0)
    current = desc[reads].gather(1, cells[:, :, None].expand(-1, -1, 16))
    torch.testing.assert_close(relations, teacher_relations(current, desc[sources]), atol=0, rtol=0)
    empty = g3a1_targets(desc, sources[:0], reads[:0], cells[:0])
    assert empty[0].shape == (0, 5, 16) and empty[1].shape == (0, 32, 300)


def test_g3a1_teacher_identity_rejects_changed_target_code(tmp_path):
    from qwen_vl.research import g3a1_teacher
    base = {'weights_sha256': 'w', 'position': 'p', 'position_sha256': 'x', 'teacher_layer': 23}
    identity = g3a1_teacher._g3a1_identity(base)
    assert 'position' not in identity['base'] and 'position_sha256' not in identity['base']
    assert identity['target'].startswith('g3a1-') and identity['regions_sha256'] and identity['relations_sha256']
    assert identity != g3a1_teacher._g3a1_identity(dict(base, weights_sha256='other'))


def address_state(head=True):
    prefix = 'backbone.model.language_model.layers.24.linear_attn.geoaddress.'
    state = {'backbone.weight': torch.ones(2)}
    for name, shape in [('q_down', (8, 2560)), ('k_down', (8, 2560)), ('q_up', (256, 8)), ('k_up', (256, 8))]:
        state[prefix + name + '.weight'] = torch.randn(shape)
    if head:
        state['research_aux.state_head.query.1.weight'] = torch.ones(3)
        state['research_aux.state_head.decoder.0.weight'] = torch.ones(3)
    return state


def test_g3a1_export_keeps_adapters_and_drops_state_head(tmp_path):
    from scripts.vln.export_vggt_g1 import export_checkpoint
    source = tmp_path / 'source'; source.mkdir()
    config = g3a1_config()
    (source / 'navigation.json').write_text(json.dumps({'config': config}))
    state = address_state()
    torch.save(state, source / 'pytorch_model.bin')
    exported = torch.load(export_checkpoint(source, tmp_path / 'serving') / 'pytorch_model.bin', weights_only=True)
    assert set(exported) == {k for k in state if not k.startswith('research_aux.')}
    for key in exported: assert torch.equal(exported[key], state[key])
    saved = json.loads((tmp_path / 'serving' / 'navigation.json').read_text())['config']
    assert 'research' not in saved and saved['model']['geoaddress'] == GEO
    torch.save(address_state(head=False), source / 'pytorch_model.bin')
    with pytest.raises(ValueError): export_checkpoint(source, tmp_path / 'no-head')
    other = address_state(); other['research_aux.current_projector.fc1.weight'] = torch.ones(1)
    torch.save(other, source / 'pytorch_model.bin')
    with pytest.raises(ValueError): export_checkpoint(source, tmp_path / 'wrong-head')
