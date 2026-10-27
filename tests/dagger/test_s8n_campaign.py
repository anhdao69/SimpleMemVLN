import json
import sys
import pytest
from s8n_dagger.campaign import active_collectors


def completed_campaign(tmp_path, monkeypatch, shards):
    """Real completed, rejected episodes; no model or simulator process is needed."""
    from s8n_dagger import campaign
    jobs = []
    for (dataset, index), ids in shards.items():
        jobs.append(dict(dataset=dataset, shard=index))
        shard = tmp_path / 'shards' / dataset / f'{index:03}.json'
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(ids))
        out = tmp_path / 'full' / f'{dataset}_{index:03}'
        for eid in set(ids):
            episode = out / 'episodes' / eid
            episode.mkdir(parents=True)
            (episode / 'record.json').write_text(json.dumps(dict(
                episode_id=eid, episode_uid=f'{dataset}:train:scene:{eid}:s8n_recovery2',
                dataset=dataset, accepted=False, failed=False, termination='max_steps',
                rgb_retained=False, steps=[])))
        (out / 'contract.json').write_text(json.dumps(dict(episode_ids=ids)))
        (out / 'training.jsonl').write_text('')
        (out / 'status.json').write_text(json.dumps(dict(
            complete=True, attempted=len(set(ids)), total=len(set(ids)), accepted=0,
            accepted_steps=0, failures=0)))
    site = tmp_path / 'site.json'
    site.write_text(json.dumps(dict(root=str(tmp_path), repo=str(tmp_path))))
    assignment = tmp_path / 'assignment.json'
    assignment.write_text(json.dumps(jobs))
    monkeypatch.setattr(sys, 'argv', ['campaign', '--site', str(site), '--assignment', str(assignment), '--workers', '1'])
    monkeypatch.setattr(campaign.signal, 'signal', lambda *args: None)
    return campaign


def test_completed_shard_must_match_requested_ordered_ids(tmp_path, monkeypatch):
    campaign = completed_campaign(tmp_path, monkeypatch, {('r2r', 0): ['1', '2']})
    (tmp_path / 'shards/r2r/000.json').write_text('["2", "1"]')
    with pytest.raises(ValueError, match='episode IDs'):
        campaign.main()


def test_resume_rejects_changed_shard_bytes_even_with_same_ids(tmp_path, monkeypatch):
    campaign = completed_campaign(tmp_path, monkeypatch, {('r2r', 0): ['1']})
    campaign.main()
    (tmp_path / 'shards/r2r/000.json').write_text('[\n  "1"\n]\n')
    with pytest.raises(ValueError, match='resume contract'):
        campaign.main()


def test_jobs_cannot_share_dataset_episode_ids(tmp_path, monkeypatch):
    campaign = completed_campaign(tmp_path, monkeypatch,
                                  {('r2r', 0): ['1'], ('r2r', 1): ['1']})
    with pytest.raises(ValueError, match='Duplicate.*episode'):
        campaign.main()


def test_same_episode_id_in_distinct_datasets_is_allowed(tmp_path, monkeypatch):
    campaign = completed_campaign(tmp_path, monkeypatch,
                                  {('r2r', 0): ['1'], ('rxr15', 0): ['1']})
    campaign.main()
    status = json.loads((tmp_path / 'campaign_status.json').read_text())
    assert status['phase'] == 'complete' and status['attempted'] == 2


def test_interrupted_controller_records_stopped_status(tmp_path, monkeypatch):
    import os
    campaign = completed_campaign(tmp_path, monkeypatch, {('r2r', 0): ['1']})
    (tmp_path / 'full/r2r_000/status.json').unlink()
    # Exercise real process-group cleanup using an inert CPU subprocess.
    monkeypatch.setattr(campaign, 'command', lambda *args: [sys.executable, '-c', 'import time; time.sleep(60)'])
    monkeypatch.setattr(campaign, 'environment', lambda site: os.environ.copy())
    def interrupt(_seconds):
        raise KeyboardInterrupt('test controller interruption')
    monkeypatch.setattr(campaign.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.main()
    status = json.loads((tmp_path / 'campaign_status.json').read_text())
    assert status['phase'] == 'interrupted'
    assert status['running'] == {} and status['jobs_pending'] == 1


def test_restart_detects_only_collectors_owned_by_this_campaign(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()
    for pid, output in [('12', '/campaign/full/r2r_000'), ('13', '/other/full/r2r_000')]:
        directory = proc / pid
        directory.mkdir()
        (directory / 'cmdline').write_bytes(b'python\0-m\0s8n_dagger.collect\0--out\0' + output.encode() + b'\0')
    assert active_collectors('/campaign', proc) == [12]


def test_validation_rejects_wrong_exported_frame_and_duplicate_rows(tmp_path):
    import hashlib
    import numpy as np
    import pytest
    from PIL import Image
    from s8n_dagger.common import atomic_json, training_row
    from s8n_dagger.validate import validate
    ep = tmp_path / 'episodes/1'
    (ep / 'rgb').mkdir(parents=True)
    pixels = np.zeros((480, 640, 3), np.uint8)
    for name in ['0000.png', 'other.png']:
        Image.fromarray(pixels).save(ep / 'rgb' / name)
    record = dict(episode_id='1',episode_uid='r2r:train:s:1',dataset='r2r',
                  dataset_version='hash',scene_id='s',instruction_id='1',instruction='Go.',
                  accepted=True, failed=False,termination='oracle_stop',distance_to_goal=.2,
                  rgb_retained=True,steps=[dict(step_id=0,executed_action=0,
                  rgb_path='rgb/0000.png',rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                  committed_response_token_ids=[48964,248046])])
    atomic_json(tmp_path / 'contract.json',dict(episode_ids=['1']))
    atomic_json(ep / 'record.json',record)
    training = training_row(record,ep)
    atomic_json(ep / 'training_episode.json',training)
    (tmp_path / 'training.jsonl').write_text(json.dumps(training)+'\n')
    assert validate(tmp_path)['complete']
    training['steps'][0]['rgb_path'] = str(ep / 'rgb/other.png')
    atomic_json(ep / 'training_episode.json',training)
    with pytest.raises(AssertionError):
        validate(tmp_path)
    training=training_row(record,ep)
    atomic_json(ep / 'training_episode.json',training)
    (tmp_path / 'training.jsonl').write_text((json.dumps(training)+'\n')*2)
    with pytest.raises(AssertionError):
        validate(tmp_path)
