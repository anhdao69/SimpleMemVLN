"""Independent saved-data checks, including lossless pre-action RGB identity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from s8n_dagger.common import ACTION_NAMES, atomic_json, training_row


def validate(root):
    contract = json.loads((root / 'contract.json').read_text())
    wanted = set(contract['episode_ids'])
    seen, accepted, steps = set(), 0, 0
    expected_exports = {}
    for directory in sorted((root / 'episodes').iterdir()):
        row = json.loads((directory / 'record.json').read_text())
        eid = row['episode_id']
        assert eid in wanted and eid not in seen and directory.name == eid
        seen.add(eid)
        if row['accepted']:
            assert row['termination'] == 'oracle_stop' and not row['failed']
            assert row['distance_to_goal'] < .5
            assert row['steps'][-1]['executed_action'] == 0
            assert all(s['executed_action'] != 0 for s in row['steps'][:-1])
            training = json.loads((directory / 'training_episode.json').read_text())
            from qwen_vl.contracts import validate_episode
            validate_episode(training)
            assert training == training_row(row, directory), 'Training export differs from captured trajectory'
            expected_exports[row['episode_uid']] = training
            assert len(training['steps']) == len(row['steps'])
            assert [s['action_name'] for s in training['steps']] == [ACTION_NAMES[s['executed_action']] for s in row['steps']]
            accepted += 1
        for index, step in enumerate(row['steps']):
            assert step['step_id'] == index
            if index:
                assert step['pose_before'] == row['steps'][index - 1]['pose_after']
            assert step['executed_action'] in ACTION_NAMES
            assert step['committed_response_token_ids'][-1] == 248046
            if row['rgb_retained']:
                frame = np.asarray(Image.open(directory / step['rgb_path']))
                assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
                assert hashlib.sha256(frame.tobytes()).hexdigest() == step['rgb_sha256']
            steps += 1
    exported = [json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines() if line.strip()]
    assert len(exported) == len(expected_exports), 'Export count mismatch'
    assert len({r['episode_uid'] for r in exported}) == len(exported), 'Duplicate export'
    assert {r['episode_uid']: r for r in exported} == expected_exports, 'Shard manifest differs from captured trajectories'
    return dict(passed=True, episodes=len(seen), accepted=accepted, steps=steps,
                expected=len(wanted), complete=seen == wanted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = validate(args.root)
    atomic_json(args.root / 'validation.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
