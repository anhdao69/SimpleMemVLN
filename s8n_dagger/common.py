"""Pure collection policy, provenance and training export contracts."""
import hashlib
import json
import os
from pathlib import Path

ACTION_NAMES = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def bind_contract(root, contract):
    path = Path(root) / 'contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Collection resume contract changed')
    else:
        atomic_json(path, contract)


def partition(episodes, index, count):
    if count <= 0 or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    ids = [str(ep['episode_id']) for ep in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate episode IDs')
    ordered = sorted(episodes, key=lambda ep: (ep['scene_id'], str(ep['episode_id'])))
    return ordered[len(ordered) * index // count:len(ordered) * (index + 1) // count]


class RecoveryControl:
    """Janus beta=0 recovery; error counts persist across waypoint advances."""
    def __init__(self, reference_actions, waypoints):
        self.segment_length = max(1, int(reference_actions / max(1, waypoints - 1)))
        self.errors = 0
        self.recovering = False
        self.model_success = True

    def advance(self):
        self.recovering = False

    def choose(self, proposed, oracle, distance):
        if oracle not in ACTION_NAMES or proposed not in ACTION_NAMES:
            raise ValueError('Invalid action')
        if oracle == 0:
            return 0, 'oracle_stop'
        if self.recovering:
            return oracle, 'recovery'
        self.errors += int(proposed != oracle)
        trigger = ((proposed == 0 and distance >= 3.0)
                   or self.errors / self.segment_length > .8 or self.errors > 12)
        if trigger:
            self.model_success = False
            self.recovering = True
            self.errors = 0
            return oracle, 'recovery'
        if proposed == 0:
            return oracle, 'stop_override'
        return proposed, 'model'


def accepted_trajectory(termination, distance, efficiency, model_success):
    return (termination == 'oracle_stop' and distance < .5
            and ((not model_success and efficiency < .93) or efficiency < .85))


def training_row(record, episode_root):
    if not record['accepted'] or record['termination'] != 'oracle_stop':
        raise ValueError('Only accepted, truly terminated trajectories may be exported')
    actions = [s['executed_action'] for s in record['steps']]
    if not actions or actions[-1] != 0 or 0 in actions[:-1]:
        raise ValueError('Exactly one real final STOP is required')
    return dict(schema_version=1, episode_uid=record['episode_uid'],
                dataset='r2rce' if record['dataset'] == 'r2r' else 'rxrce',
                dataset_version=record['dataset_version'], split='train',
                scene_id=record['scene_id'], episode_id=record['episode_id'],
                instruction_id=record['instruction_id'], instruction=record['instruction'],
                trajectory_variant='s8n_janus_recovery_15deg_v2',
                language=record.get('language', 'en'),
                observation_action_alignment='observation_before_action',
                alignment_evidence='copied simulator RGB before executed action; raw record.json',
                supervision='executed_recovery_actions',
                steps=[dict(step_id=i, rgb_path=str((Path(episode_root) / s['rgb_path']).resolve()),
                            action_name=ACTION_NAMES[s['executed_action']], is_valid=True)
                       for i, s in enumerate(record['steps'])])
