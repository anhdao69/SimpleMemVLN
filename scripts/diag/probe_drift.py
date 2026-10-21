"""D-f: probe-row selection drift with a strict two-dummy-head gate.

Invoke as python -m scripts.diag.probe_drift (avoid script-dir module shadowing).
"""
from contextlib import contextmanager
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.diag.replay_gate import compare_logits, compare_plans, sha256


def bitwise_invariant(first, second, first_plans, second_plans):
    return (first.shape == second.shape and first.dtype == second.dtype
            and first.tobytes() == second.tobytes() and first_plans == second_plans)


def drift_metrics(plain, probed, plain_plans, probed_plans):
    numerical = compare_logits(probed, plain)
    # D-f measures drift; G-2's qualification threshold is not its decision rule.
    for key in ('passed', 'atol', 'rtol', 'rms_limit', 'outside_tolerance'):
        numerical.pop(key)
    numerical.update(compare_plans(probed_plans, plain_plans))
    return numerical


def attach_dummy_heads(model, research, seed):
    import torch
    from qwen_vl.models.vggt_g3_heads import NativeStateHead, build_state_heads
    from qwen_vl.research.vggt_g1 import parse_g1
    config = copy.deepcopy(model.navigation_config)
    config['research'] = copy.deepcopy(research)
    spec = parse_g1(config)
    if spec['run_id'] != 'G3_M5':
        raise ValueError('D-f requires G3_M5 or its TRIPLE variant')
    parameter = next(model.backbone.parameters())
    heads = (torch.nn.ModuleDict({'state_heads': build_state_heads(spec['student_layers'], seed=seed)})
             if 'student_layers' in spec else
             torch.nn.ModuleDict({'state_head': NativeStateHead(seed=seed)}))
    model.g1_spec = spec
    model.research_aux = heads.to(device=parameter.device, dtype=parameter.dtype).eval().requires_grad_(False)


@contextmanager
def temporary_dummy_heads(model, research, seed):
    """Restore exact prior module/config references on attachment or forward error."""
    absent = object()
    previous_spec = getattr(model, 'g1_spec', absent)
    previous_heads = getattr(model, 'research_aux', absent)
    try:
        attach_dummy_heads(model, research, seed)
        yield
    finally:
        for name, previous in [('research_aux', previous_heads), ('g1_spec', previous_spec)]:
            if previous is absent:
                if hasattr(model, name):
                    delattr(model, name)
            else:
                setattr(model, name, previous)


def episode_measurement(model, serializer, episode, research, *, encoded=None, baseline=None):
    import torch
    from qwen_vl.research.vggt_g3 import sample_pairs, observation_anchors
    from qwen_vl.research.vggt_sampling import collection_trajectory_key
    device = next(model.backbone.parameters()).device
    data = (encoded if encoded is not None else
            {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in serializer.encode_episode(episode).items()})
    if baseline is not None and encoded is None:
        raise ValueError('Reused baseline requires its identical encoded inputs')
    pairs = sample_pairs(collection_trajectory_key(episode), len(episode['steps']), 429)
    if not pairs:
        raise ValueError('Frozen panel episode has no eligible training pairs')
    anchors = observation_anchors(data['response_target_positions'],
                                  data['response_action_index'], data['num_actions'])
    selected = anchors[torch.tensor([t for s, t in pairs], device=anchors.device)]
    auxiliary = dict(aux_anchors=selected,
                     aux_lags=torch.tensor([t-s for s, t in pairs], device=anchors.device),
                     aux_pair_count=len(pairs),
                     aux_targets=torch.zeros(len(pairs), 5, 2048, device=anchors.device))

    def forward(extra):
        diagnostics = {}
        with torch.inference_mode():
            logits = model(**data, **extra, episodic_diagnostics=diagnostics)['logits'].float().cpu().numpy()
        plans = {str(layer): [plan.group_ids(t).tolist() for t in range(len(episode['steps']))]
                 for layer, plan in diagnostics['plans'].items()}
        return logits, plans

    plain, plain_plans = forward({}) if baseline is None else baseline
    with temporary_dummy_heads(model, research, 429):
        first, first_plans = forward(auxiliary)
    with temporary_dummy_heads(model, research, 430):
        second, second_plans = forward(auxiliary)
    invariant = bitwise_invariant(first, second, first_plans, second_plans)
    result = dict(episode_uid=episode['episode_uid'], episode_id=episode['episode_id'],
                  scene_id=episode['scene_id'], frames=len(episode['steps']), pairs=pairs,
                  dummy_seeds=[429, 430], dummy_invariance_passed=invariant,
                  probe_layers=research.get('student_layers', [research.get('student_layer', 24)]))
    if invariant:
        result['drift'] = drift_metrics(plain, first, plain_plans, first_plans)
    else:
        result['failure'] = 'Dummy heads altered real-token outputs or plans; D-f unavailable with these heads'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=Path(__file__).resolve().parents[2] / 'reports/diagnostics/g3m5_failure_20261019/panel_ids.json')
    parser.add_argument('--research-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g2-result', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    from qwen_vl.train.vln_runtime import load_checkpoint
    gate = json.loads(args.g2_result.read_text())
    if not gate.get('passed') or Path(gate['checkpoint']).resolve() != Path(args.checkpoint).resolve():
        raise ValueError('Matching passed G-2 evidence required')
    weight = Path(args.checkpoint) / 'pytorch_model.bin'
    if not weight.exists():
        weight = Path(args.checkpoint) / 'model.safetensors'
    if sha256(weight) != gate['weights_sha256']:
        raise ValueError('Checkpoint identity changed since G-2')
    research = yaml.safe_load(args.research_config.read_text())['research']
    episodes = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not episodes or len({e['episode_uid'] for e in episodes}) != len(episodes):
        raise ValueError('Manifest must have unique nonempty episodes')
    from scripts.diag.cache_teacher import validate_panel
    validate_panel(episodes, json.loads(args.panel.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    if model.g1_spec is not None or hasattr(model, 'research_aux'):
        raise ValueError('Runner expects immutable serving export without original heads')
    model.eval().requires_grad_(False)
    versions = {name: p._version for name, p in model.named_parameters()}
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), weights_sha256=gate['weights_sha256'],
                    manifest_sha256=sha256(args.manifest), panel_sha256=sha256(args.panel), research_sha256=sha256(args.research_config),
                    g2_sha256=sha256(args.g2_result), dummy_heads=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (args.output / 'episodes.jsonl').open('x') as stream:
        for episode in episodes:
            record = episode_measurement(model, serializer, episode, research)
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if not record['dummy_invariance_passed']:
                raise RuntimeError(record['failure'])
            if versions != {name: p._version for name, p in model.named_parameters()}:
                raise RuntimeError('Serving parameter mutation detected')
            print(json.dumps(dict(episode=episode['episode_id'], frames=len(episode['steps']),
                                  rms=record['drift']['rms'])), flush=True)
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(episodes=len(episodes),
        parameter_versions_unchanged=True, weights_sha256_after=sha256(weight)), indent=2)+'\n')


if __name__ == '__main__':
    main()
