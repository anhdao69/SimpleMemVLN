"""Prepare the frozen expert panel from archive images or unqualified rendering.

Original rendering parity failure is retained. This helper neither reruns that
gate nor changes its threshold. Rendering requires an explicit --mode render;
archive mode requires complete ground-truth-matching frames already extracted.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.diag.render_panel import ACTIONS, render_episode, sha256, write_json


def write_subset(source, ids, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Subset must never replace original dataset')
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Require nonempty unique panel IDs')
    before = sha256(source)
    with gzip.open(source, 'rt') as f:
        data = json.load(f)
    lookup = {str(e['episode_id']): e for e in data['episodes']}
    if len(lookup) != len(data['episodes']) or set(ids) - lookup.keys():
        raise ValueError('Duplicate dataset IDs or missing panel IDs')
    subset = copy.deepcopy(data)
    subset['episodes'] = [copy.deepcopy(lookup[eid]) for eid in ids]
    raw = json.dumps(subset, sort_keys=True, separators=(',', ':')).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(raw, mtime=0))
    if sha256(source) != before:
        raise ValueError('Original dataset mutated')
    return {'parent_path': str(source.resolve()), 'parent_sha256': before,
            'subset_path': str(destination.resolve()), 'subset_sha256': sha256(destination),
            'episode_ids': ids, 'episode_count': len(ids)}


def check_frames(root, ids, gt):
    from PIL import Image
    root = Path(root)
    records = []
    for eid in ids:
        actions = gt[eid]['actions']
        if (not actions or actions[-1] != 0 or 0 in actions[:-1]
                or any(type(a) is not int or a not in ACTIONS for a in actions)):
            raise ValueError('Invalid GT actions for ' + eid)
        expected = [f'step_{i:04d}_{ACTIONS[a]}.png' for i, a in enumerate(actions)]
        actual = sorted(p.name for p in (root/eid).glob('*.png'))
        if actual != expected:
            raise ValueError(f'Missing, extra or mislabeled frames for {eid}')
        frames = []
        for filename in expected:
            path = root/eid/filename
            with Image.open(path) as im:
                im.load()
                if im.mode != 'RGB' or im.size != (640, 480):
                    raise ValueError('Require unconverted RGB 640x480: ' + str(path))
            frames.append({'filename': filename, 'sha256': sha256(path)})
        records.append({'episode_id': eid, 'frame_count': len(frames), 'frames': frames})
    return {'total_frames': sum(e['frame_count'] for e in records), 'episodes': records}


def render_frozen_panel(args, subset, ids, gt, report):
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    from omegaconf import OmegaConf
    from PIL import Image
    if (habitat.__version__, habitat_sim.__version__) != ('0.2.4', '0.2.4'):
        raise ValueError('Require Habitat-Lab/Sim 0.2.4')
    config = get_config(str(args.habitat_config.resolve()))
    with habitat.config.read_write(config):
        config.habitat.dataset.split = 'val_unseen'
        config.habitat.dataset.scenes_dir = str(args.scenes_dir.resolve())
        config.habitat.dataset.data_path = str(subset.resolve())
        config.habitat.seed = 42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
        config.habitat.environment.iterator_options.shuffle = False
    sim = config.habitat.simulator
    agent = sim.agents.main_agent
    camera = agent.sim_sensors.rgb_sensor
    if (camera.width, camera.height, camera.hfov, sim.forward_step_size,
        sim.turn_angle, list(camera.position), agent.height, agent.radius) != (
            640, 480, 79, .25, 15, [0., 1.25, 0.], 1.5, .1):
        raise ValueError('Fixed evaluation sensor/action/agent contract mismatch')
    if (config.habitat.environment.max_episode_steps != 500
            or config.habitat.task.measurements.success.success_distance != 3.0
            or config.habitat.dataset.type != 'R2RVLN-v1'):
        raise ValueError('Fixed evaluator environment/task contract mismatch')
    config_path = args.output/'resolved_config.yaml'
    config_path.write_text(OmegaConf.to_yaml(config))
    report['rendering'] = {'qualified': False, 'condition': 'fixed_evaluation_camera_original_training_pixel_parity_failed',
                           'habitat_lab': habitat.__version__, 'habitat_sim': habitat_sim.__version__,
                           'config_path': str(config_path), 'config_sha256': sha256(config_path),
                           'seed': 42, 'observation_action_alignment': 'observation_before_action'}
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    lookup = {str(e.episode_id): e for e in dataset.episodes}
    dataset.episodes = [lookup[eid] for eid in ids]
    report['rendered_episode_ids'] = []
    with habitat.Env(config=config, dataset=dataset) as env:
        for eid in ids:
            folder = args.frames_root/eid
            folder.mkdir(parents=True, exist_ok=False)
            def consume(step, action, rgb):
                if rgb.shape != (480, 640, 3) or str(rgb.dtype) != 'uint8':
                    raise ValueError('Unexpected RGB sensor output')
                Image.fromarray(rgb).save(folder/f'step_{step:04d}_{ACTIONS[action]}.png')
            render_episode(env, lookup[eid], gt[eid]['actions'], consume)
            report['rendered_episode_ids'].append(eid)
            write_json(args.output/'PREPARATION.json', report)
            print(json.dumps({'episode_id': eid, 'completed': len(report['rendered_episode_ids']), 'total': len(ids)}), flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['archive', 'render'])
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--ground-truth', required=True, type=Path)
    p.add_argument('--panel', required=True, type=Path)
    p.add_argument('--frames-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--rendering-gate', type=Path, default=root/'reports/diagnostics/g3m5_failure_20261019/RENDERING_GATE.json')
    p.add_argument('--scope-update', type=Path, default=root/'reports/diagnostics/g3m5_failure_20261019/SCOPE_UPDATE.md')
    p.add_argument('--habitat-config', type=Path, default=root/'configs/eval/vln_r2r.yaml')
    p.add_argument('--scenes-dir', type=Path, default=Path('/storage/anhdh35/JanusVLN/data/scene_datasets_full'))
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    # All scientific identities are recorded before any image generation.
    from scripts.diag.panels import DATASET_SHA256
    if sha256(args.source) != DATASET_SHA256:
        raise ValueError('Original official dataset identity mismatch')
    panel = json.loads(args.panel.read_text())
    ids = [str(e['episode_id']) for e in panel['episodes']]
    if len(ids) != 240 or len(set(ids)) != 240:
        raise ValueError('Require unchanged 240-episode panel')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'mode': args.mode,
              'command': [sys.executable, '-m', 'scripts.diag.prepare_expert'] + sys.argv[1:],
              'qualified': False, 'original_rendering_gate': json.loads(args.rendering_gate.read_text()),
              'scope_update': {'path': str(args.scope_update), 'sha256': sha256(args.scope_update)},
              'panel': {'path': str(args.panel), 'sha256': sha256(args.panel), 'episode_ids': ids},
              'ground_truth': {'path': str(args.ground_truth), 'sha256': sha256(args.ground_truth)},
              'frames_root': str(args.frames_root.resolve()),
              'limitation': 'Original training rendering parity failed. User authorized continuation; thresholds unchanged. Archive filename labels alone do not establish pixel timing.'}
    try:
        subset = args.output/'val_unseen_subset.json.gz'
        report['dataset'] = write_subset(args.source, ids, subset)
        with gzip.open(args.ground_truth, 'rt') as f: gt = json.load(f)
        with gzip.open(subset, 'rt') as f: selected = json.load(f)['episodes']
        for record, item in zip(selected, panel['episodes']):
            eid = str(record['episode_id'])
            iid = str(record['instruction'].get('instruction_id', eid))
            if (record['scene_id'] != item['scene_id'] or iid != str(item['instruction_id'])
                    or len(gt[eid]['actions']) != item['gt_action_count']):
                raise ValueError('Frozen panel metadata mismatch: ' + eid)
        if args.mode == 'render':
            args.frames_root.mkdir(parents=True, exist_ok=False)
            render_frozen_panel(args, subset, ids, gt, report)
        report['frames'] = check_frames(args.frames_root, ids, gt)
        manifest = args.output/'episodes.jsonl'
        command = [sys.executable, str(root/'scripts/vln/prepare_r2r.py'),
            '--data-root', str(args.source.parent), '--out', str(manifest),
            '--split', 'val_unseen', '--source', str(subset), '--frames-root', str(args.frames_root.resolve())]
        env = dict(os.environ, PYTHONPATH=str(root/'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        (args.output/'prepare_r2r.log').write_text(result.stdout + result.stderr)
        report['converter'] = {'command': command, 'source_sha256': sha256(root/'scripts/vln/prepare_r2r.py'), 'exit_code': result.returncode}
        result.check_returncode()
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if [e['episode_id'] for e in rows] != ids or any(e['dataset_version'] != report['dataset']['subset_sha256'] for e in rows):
            raise ValueError('Manifest IDs or dataset_version diverged')
        if sha256(args.source) != report['dataset']['parent_sha256'] or sha256(args.panel) != report['panel']['sha256']:
            raise ValueError('Original source/panel mutated')
        report.update(status='complete', manifest=str(manifest), manifest_sha256=sha256(manifest), episode_count=len(rows))
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.output/'PREPARATION.json', report)

if __name__ == '__main__': main()
