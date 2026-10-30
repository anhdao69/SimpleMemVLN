"""Weights-only policy adaptation; deliberately separate from exact Trainer resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def validate_parent_contract(parent, target):
    lane = target['model'].get('step_lane', {})
    if not lane.get('enabled', False):
        raise ValueError('Adaptation target must enable step_lane')
    if parent['model'].get('step_lane', {}).get('enabled', False):
        raise ValueError('Adaptation parent already has a lane; use strict resume')
    inherited = deepcopy(target)
    inherited['model'].pop('step_lane', None)
    expected = deepcopy(parent)
    expected['model'].pop('step_lane', None)
    # Only training settings and the new lane may change during adaptation.
    inherited.pop('training', None)
    expected.pop('training', None)
    if inherited != expected:
        raise ValueError('Target changes protected parent policy/serializer/visibility fields')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
    if not files:
        raise ValueError('No parent weight files')
    metadata = json.loads((path / 'navigation.json').read_text())
    return dict(path=str(path), navigation_sha256=file_sha256(path/'navigation.json'),
                weight_sha256={p.name: file_sha256(p) for p in files},
                parent_config=metadata['config'])


def initialize_step_lane_from_policy(parent_checkpoint, base_model_path, target_config, *, lane_seed):
    from qwen_vl.train.vln_runtime import load_checkpoint
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.models.install_step_lane import install_step_lanes
    from qwen_vl.research.lane_contract import parse_step_lane_spec

    metadata = json.loads((Path(parent_checkpoint)/'navigation.json').read_text())
    validate_parent_contract(metadata['config'], target_config)
    # Strictly load the actual wrapper, including a trained copied-linear head.
    model, serializer = load_checkpoint(parent_checkpoint, base_model_path)
    inherited = {name: id(p) for name, p in model.named_parameters()}
    model.navigation_config = deepcopy(target_config)
    spec = parse_step_lane_spec(model.navigation_config, model.config.text_config)
    install_step_lanes(model, spec, init_seed=lane_seed)
    current = dict(model.named_parameters())
    if any(id(current[name]) != identity for name, identity in inherited.items()):
        raise RuntimeError('Lane installation replaced inherited parameters')
    serializer = EpisodeSerializer(serializer.processor, model.navigation_config)
    report = dict(initialization='strict_wrapper_weights_only', parent=str(Path(parent_checkpoint).resolve()),
                  lane_seed=lane_seed, inherited_parameters_preserved=len(inherited),
                  lane_spec=spec.to_dict(), optimizer='fresh', scheduler='fresh')
    return model, serializer, report


# The epoch length of a different manifest; nothing the model, serializer or loss reads.
SCHEDULE_KEYS = ('expected_total_steps', 'warmup_steps', 'expected_warmup_steps')


def schedule_changes(parent, target):
    """Schedule keys the target changes relative to the parent; any other difference raises."""
    parent, target = deepcopy(parent), deepcopy(target)
    changes = {}
    for key in SCHEDULE_KEYS:
        old, new = parent['training'].pop(key, None), target['training'].pop(key, None)
        if old != new:
            changes[key] = [old, new]
    if parent != target:
        raise ValueError('Weights initialization requires the parent navigation config to equal the target '
                         'outside the epoch schedule')
    return changes


def initialize_weights_from_checkpoint(checkpoint, weights_sha256, base_model_path, target_config,
                                       new_schedule=False):
    """Strict trained weights for another epoch of the identical recipe; optimizer and schedule start fresh.

    ``new_schedule``: the target trains one epoch of a different manifest, so only its epoch schedule
    (SCHEDULE_KEYS) may differ; the model and serializer then carry the target config.
    """
    from qwen_vl.train.vln_runtime import load_checkpoint
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    path = Path(checkpoint).resolve()
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
    if len(files) != 1:
        raise ValueError(f'Expected exactly one weights file in {path}')
    digest = file_sha256(files[0])
    if digest != weights_sha256:
        raise ValueError(f'{files[0]}: SHA-256 {digest} differs from the declared {weights_sha256}')
    metadata = json.loads((path / 'navigation.json').read_text())
    changes = schedule_changes(metadata['config'], target_config) if new_schedule else {}
    if not new_schedule and metadata['config'] != target_config:
        raise ValueError('Weights initialization requires the parent navigation config to equal the target')
    model, serializer = load_checkpoint(path, base_model_path)
    if serializer.metadata() != metadata:
        raise ValueError('Parent tokenizer/serializer contract differs')
    if changes:
        model.navigation_config = deepcopy(target_config)
        serializer = EpisodeSerializer(serializer.processor, model.navigation_config)
        if dict(serializer.metadata(), config=metadata['config']) != metadata:
            raise ValueError('Schedule change altered the serializer contract')
    report = dict(initialization='strict_weights_only', parent=str(path), weights_file=files[0].name,
                  weights_sha256=digest, navigation_sha256=file_sha256(path / 'navigation.json'),
                  parameters=sum(p.numel() for p in model.parameters()), optimizer='fresh', scheduler='fresh')
    if new_schedule:
        report['schedule_changes'] = changes
    return model, serializer, report
