import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_joint_manifest_preserves_available_rxr_and_reports_missing(tmp_path):
    root = tmp_path / 'data'
    meta = root / 'datasets/rxr/train/train_guide.json.gz'
    meta.parent.mkdir(parents=True)
    records = [dict(episode_id=str(i), scene_id='scene', instruction=dict(
        instruction_id=str(i + 10), instruction_text='Walk forward.', language='en-US'
    )) for i in (1, 2)]
    with gzip.open(meta, 'wt') as f:
        json.dump({'episodes': records}, f)
    frames = root / 'trajectory_data/RxR/train/1'
    frames.mkdir(parents=True)
    (frames / 'step_0000_MOVE_FORWARD.png').touch()
    (frames / 'step_0001_STOP.png').touch()
    r2r = tmp_path / 'r2r.jsonl'
    r2r.write_text('')
    out = tmp_path / 'joint.jsonl'
    script = Path(__file__).resolve().parents[2] / 'scripts/vln/prepare_joint.py'
    result = subprocess.run([sys.executable, str(script), '--data-root', str(root),
        '--r2r-manifest', str(r2r), '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(s) for s in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['dataset'] == 'rxrce'
    assert rows[0]['instruction_id'] == '11'
    assert [s['action_name'] for s in rows[0]['steps']] == ['MOVE_FORWARD', 'STOP']
    audit = json.loads(out.with_suffix('.audit.json').read_text())
    assert audit['missing_english_episode_ids'] == ['2']
    (frames / 'step_0001_STOP.png').unlink()
    result = subprocess.run([sys.executable, str(script), '--data-root', str(root),
        '--r2r-manifest', str(r2r), '--out', str(tmp_path/'invalid.jsonl')], capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'invalid.jsonl').exists()


def test_15deg_uses_matching_annotations_and_frames(tmp_path):
    root = tmp_path / 'data'
    meta = root / 'RxR_CE_15_deg_Trajectory_data/train/train_guide.json.gz'
    meta.parent.mkdir(parents=True)
    with gzip.open(meta, 'wt') as f:
        json.dump({'episodes': [dict(episode_id='1', scene_id='scene15', instruction=dict(
            instruction_id='42', instruction_text='Turn left.', language='en-US'))]}, f)
    frames = root / 'trajectory_data/RxR_15deg/train/1'
    frames.mkdir(parents=True)
    (frames / 'step_0000_TURN_LEFT.png').touch()
    (frames / 'step_0001_STOP.png').touch()
    r2r = tmp_path / 'r2r.jsonl'
    r2r.write_text('')
    out = tmp_path / 'joint.jsonl'
    script = Path(__file__).resolve().parents[2] / 'scripts/vln/prepare_joint.py'
    result = subprocess.run([sys.executable, str(script), '--data-root', str(root),
        '--rxr-variant', '15deg', '--r2r-manifest', str(r2r), '--out', str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    row = json.loads(out.read_text())
    assert row['scene_id'] == 'scene15'
    assert row['instruction_id'] == '42'
    assert row['trajectory_variant'] == '15deg'
    assert '/RxR_15deg/' in row['steps'][0]['rgb_path']
