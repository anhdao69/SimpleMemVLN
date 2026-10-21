"""Cache verified online Omega regional targets without changing teacher math.

Invoke as python -m scripts.diag.cache_teacher (avoid script-dir module shadowing).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


@torch.inference_mode()
def collect_targets(paths, teacher, preprocess):
    if not paths:
        raise ValueError('Cannot cache an empty episode')
    chunks = []
    for start in range(0, len(paths), 8):
        batch = paths[start:start + 8]
        targets = teacher(preprocess(batch))
        if targets.dtype != torch.float32 or targets.shape != (len(batch), 5, 2048):
            raise ValueError('Expected FP32 teacher targets [frames,5,2048]')
        if not torch.isfinite(targets).all():
            raise ValueError('Nonfinite teacher targets')
        cached = targets.detach().cpu().half()
        if not torch.isfinite(cached).all():
            raise ValueError('FP16 teacher cache overflow')
        chunks.append(cached)
    return torch.cat(chunks)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panel(episodes, panel):
    """Reject changed population, identity, or cropped trajectories before compute."""
    expected = {str(row['episode_id']): row for row in panel['episodes']}
    actual = {str(row['episode_id']): row for row in episodes}
    if len(expected) != 240 or len(panel['episodes']) != 240 or len(episodes) != 240 or actual.keys() != expected.keys():
        raise ValueError('Manifest must cover exactly the frozen 240-episode panel')
    if len({row['episode_uid'] for row in episodes}) != 240:
        raise ValueError('Manifest must have unique episode UIDs')
    for eid, row in actual.items():
        ref = expected[eid]
        if Path(row['scene_id']).stem != ref['scene']:
            raise ValueError(f'Frozen panel scene differs for {eid}')
        if len(row['steps']) != ref['gt_action_count']:
            raise ValueError(f'Full GT trajectory required for {eid}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=Path(__file__).resolve().parents[2] / 'reports/diagnostics/g3m5_failure_20261019/panel_ids.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-source', required=True)
    parser.add_argument('--teacher-weights', required=True)
    parser.add_argument('--config', default='configs/vggt/g3_omega_h4.yaml')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import yaml
    from qwen_vl.data.vggt_online import preprocess_teacher_images
    from qwen_vl.research.vggt_g3_teacher import OnlineG3Teacher

    episodes = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    uids = [episode['episode_uid'] for episode in episodes]
    if not uids or len(set(uids)) != len(uids):
        raise ValueError('Manifest must contain unique nonempty episodes')
    validate_panel(episodes, json.loads(args.panel.read_text()))
    spec = yaml.safe_load(Path(args.config).read_text())['research']
    spec.update(teacher_source=args.teacher_source, teacher_weights=args.teacher_weights)
    args.output.mkdir(parents=True, exist_ok=False)
    teacher = OnlineG3Teacher(spec, args.device)
    provenance = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                      config_sha256=sha256(args.config), panel_sha256=sha256(args.panel), teacher_identity=teacher.identity,
                      cache_dtype='float16', compute_dtype='float32', episodes=len(episodes))
    (args.output / 'identity.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with (args.output / 'index.jsonl').open('x') as index:
        for number, episode in enumerate(episodes):
            paths = [step['rgb_path'] for step in episode['steps']]
            inputs = [dict(path=path, sha256=sha256(path)) for path in paths]
            targets = collect_targets(paths, teacher, preprocess_teacher_images)
            destination = args.output / f'{number:04d}.pt'
            torch.save(targets, destination)
            entry = dict(episode_uid=episode['episode_uid'], episode_id=episode['episode_id'],
                         file=destination.name, sha256=sha256(destination),
                         shape=list(targets.shape), inputs=inputs)
            index.write(json.dumps(entry) + '\n')
            index.flush()
            print(json.dumps(dict(completed=number + 1, total=len(episodes),
                                  episode_id=episode['episode_id'])), flush=True)
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(episodes=len(episodes),
        index_sha256=sha256(args.output / 'index.jsonl')), indent=2) + '\n')


if __name__ == '__main__':
    main()
