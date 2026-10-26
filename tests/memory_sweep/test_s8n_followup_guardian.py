import json

from scripts.memory_sweep import s8n_followup_guardian as g


def predecessor(tmp_path, status, rows=()):
    (tmp_path / 'state.json').write_text(json.dumps(dict(status=status)))
    (tmp_path / 'history.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return dict(predecessor=dict(state=str(tmp_path / 'state.json'), step='4925.193', final_status='lane-complete',
                                 step_name_prefix='s8n-lane-', history=str(tmp_path / 'history.jsonl'),
                                 requires_arm_complete='S8N_MUON'))


def test_continuation_lane_waits_for_its_parent_arm(tmp_path):
    done = dict(status='arm-complete', arm='S8N_MUON')
    other = [('4925.0', 'train_vln')]
    spec = predecessor(tmp_path, 'training')
    assert g.predecessor_state('B', spec, {}, [*other, ('4925.193', 's8n-lane-B')], True) == 'predecessor step alive'
    spec = predecessor(tmp_path, 'lane-complete', [dict(status='arm-complete', arm='S8N_E2'), done])
    assert g.predecessor_state('B', spec, {}, other, True) == 'finished'
    # A relaunched predecessor controller has a new step id but the same name.
    assert g.predecessor_state('B', spec, {}, [*other, ('4925.201', 's8n-lane-B')], True).startswith('a s8n-lane-')
    # Our own step name is not the predecessor's.
    assert g.predecessor_state('B', spec, {}, [*other, ('4925.202', 's8n-muon-e2-lane-B')], True) == 'finished'
    spec = predecessor(tmp_path, 'lane-complete', [dict(status='arm-error', arm='S8N_MUON')])
    assert 'without completing S8N_MUON' in g.predecessor_state('B', spec, {}, other, True)


def test_abandoned_predecessor_without_the_parent_arm_is_never_overridden(tmp_path):
    spec = predecessor(tmp_path, 'blocked')
    record = dict(predecessor_absent_since=1.0)
    assert 'without completing S8N_MUON' in g.predecessor_state('B', spec, record, [], True)
    assert not (g.ROOT / 'lanes/B/OVERRIDE_PREDECESSOR').exists()


def test_lane_names_and_watchers_default_to_the_first_follow_up_campaign():
    assert g.step_name('B', {}) == 's8n-lane-B'
    assert g.step_name('B', dict(step_name='s8n-muon-e2-lane-B')) == 's8n-muon-e2-lane-B'
    assert g.campaign_watchers({}) == g.DEFAULT_WATCHERS
    declared = dict(watchers=dict(M5_S_muon_e2=dict(ops='/ops', script='watch_publish_followup.py')))
    assert g.campaign_watchers(declared) == {'M5_S_muon_e2': (g.Path('/ops'), 'watch_publish_followup.py')}
    command = g.launch_lane('B', dict(job='4925', step_name='s8n-muon-e2-lane-B'), dry_run=True)['command']
    assert command[1:3] == ['--jobid=4925', '--overlap'] and '--job-name=s8n-muon-e2-lane-B' in command
