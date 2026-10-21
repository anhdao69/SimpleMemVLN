"""Fixed training-reference rendering gate; no model or diagnostic measurements.

Configuration construction follows unchanged qwen_vl.eval.habitat_r2r.main.
All image comparisons use observations BEFORE the filename's action, including
STOP. A failed gate is recorded and exits nonzero; no alternate timing is tried.
"""
import argparse
import gzip
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

ACTIONS = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
THRESHOLD = 1 / 255
SHAPE = (480, 640, 3)


def render_episode(env, episode, actions, consume):
    """Consume reset obs then each prior action's result; execute terminal STOP."""
    if (not actions or actions[-1] != 0 or 0 in actions[:-1]
            or any(type(a) is not int or a not in ACTIONS for a in actions)):
        raise ValueError('Require valid actions with exactly one terminal STOP')
    env.current_episode = episode
    observation = env.reset()
    for step, action in enumerate(actions):
        if env.episode_over:
            raise ValueError('Episode ended before all reference actions')
        consume(step, action, observation['rgb'])
        observation = env.step(action)
    if not env.episode_over:
        raise ValueError('Terminal STOP did not end episode')


def compare_rgb(reference, rendered):
    """Strict raw RGB, normalized MAE by channel; no resizing or conversion."""
    for rgb in (reference, rendered):
        if rgb.shape != SHAPE or rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB shape (480, 640, 3)')
    delta = np.abs(reference.astype(np.int16) - rendered.astype(np.int16))
    mae = delta.mean(axis=(0, 1), dtype=np.float64) / 255
    return {'mae_per_channel': mae.tolist(),
            'mean_absolute_error': float(mae.mean()),
            'max_abs_pixel': int(delta.max()),
            'absolute_error_sum_per_channel': delta.sum(axis=(0, 1), dtype=np.int64).tolist(),
            'nonidentical_channel_values': int(np.count_nonzero(delta))}


def summarize_frames(frames):
    if not frames:
        raise ValueError('Cannot pass an empty comparison')
    sums = np.array([r['absolute_error_sum_per_channel'] for r in frames], dtype=np.int64).sum(axis=0)
    pixels_per_channel = len(frames) * SHAPE[0] * SHAPE[1]
    mae = sums / (pixels_per_channel * 255)
    frame_mae = np.array([r['mae_per_channel'] for r in frames])
    return {'frame_count': len(frames), 'pixels_per_channel': pixels_per_channel,
            'mae_per_channel': mae.tolist(), 'mean_absolute_error': float(mae.mean()),
            'max_abs_pixel': max(r['max_abs_pixel'] for r in frames),
            'nonidentical_channel_values': sum(r['nonidentical_channel_values'] for r in frames),
            'frame_mae_percentiles_per_channel': {
                str(q): np.percentile(frame_mae, q, axis=0).tolist() for q in (0, 50, 90, 95, 99, 100)},
            'passed': bool(np.all(mae <= THRESHOLD))}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_gate(args, report):
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    from omegaconf import OmegaConf
    from PIL import Image

    report['environment'] = {'python': sys.executable, 'python_version': platform.python_version(),
                             'habitat_lab': habitat.__version__, 'habitat_sim': habitat_sim.__version__}
    if (habitat.__version__, habitat_sim.__version__) != ('0.2.4', '0.2.4'):
        raise ValueError('Expected unchanged Habitat-Lab/Sim 0.2.4')
    recovery = json.loads(args.reference_report.read_text())
    if len(recovery['episodes']) != 3:
        raise ValueError('Exactly three predetermined reference episodes required')
    with gzip.open(args.ground_truth, 'rt') as f:
        gt = json.load(f)
    config = get_config(str(args.habitat_config.resolve()))
    # Same overrides as habitat_r2r.main; only split/data path select training.
    with habitat.config.read_write(config):
        config.habitat.dataset.split = 'train'
        config.habitat.dataset.scenes_dir = str(args.scenes_dir.resolve())
        config.habitat.dataset.data_path = str(args.dataset.resolve())
        config.habitat.seed = 42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
        config.habitat.environment.iterator_options.shuffle = False
    sim = config.habitat.simulator
    agent = sim.agents.main_agent
    camera = agent.sim_sensors.rgb_sensor
    contract = (camera.width, camera.height, camera.hfov, sim.forward_step_size,
                sim.turn_angle, list(camera.position), agent.height, agent.radius)
    if contract != (640, 480, 79, .25, 15, [0., 1.25, 0.], 1.5, .1):
        raise ValueError('Unexpected camera/action/agent configuration: ' + repr(contract))
    if (config.habitat.environment.max_episode_steps != 500
            or config.habitat.task.measurements.success.success_distance != 3.0
            or config.habitat.dataset.type != 'R2RVLN-v1'):
        raise ValueError('Unexpected evaluation cap, success radius, or dataset type')
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'resolved_config.yaml'
    config_path.write_text(OmegaConf.to_yaml(config))
    source = Path(__file__).resolve().parents[2] / 'src/qwen_vl/eval/habitat_r2r.py'
    report['provenance'] = {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in {
        'dataset': args.dataset, 'ground_truth': args.ground_truth,
        'reference_recovery': args.reference_report, 'habitat_config': args.habitat_config,
        'resolved_config': config_path, 'evaluation_source': source, 'renderer': Path(__file__)}.items()}
    report['camera_action_contract'] = {'width': 640, 'height': 480, 'hfov': 79,
        'sensor_position': list(camera.position), 'agent_height': agent.height,
        'agent_radius': agent.radius, 'forward_step_size': .25, 'turn_angle': 15, 'seed': 42}
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    dataset.episodes = [by_id[e['episode_id']] for e in recovery['episodes']]
    # Hash exact scene and navmesh files before constructing the simulator.
    report['scene_assets'] = []
    for scene in sorted({ep.scene_id for ep in dataset.episodes}):
        p = Path(scene)
        report['scene_assets'].append({'path': scene, 'sha256': sha256(p),
            'navmesh_sha256': sha256(p.with_suffix('.navmesh'))})
    with habitat.Env(config=config, dataset=dataset) as env:
        for reference in recovery['episodes']:
            eid = reference['episode_id']
            actions = gt[eid]['actions']
            frames = reference['frames']
            if len(actions) != len(frames):
                raise ValueError('GT and reference frame counts disagree')
            rows = []
            directory = args.output / 'train' / eid
            directory.mkdir(parents=True, exist_ok=True)
            def consume(step, action, rgb):
                ref = frames[step]
                filename = f'step_{step:04d}_{ACTIONS[action]}.png'
                if (ref['filename'] != filename or sha256(ref['path']) != ref['sha256']):
                    raise ValueError('Reference labels or bytes changed')
                with Image.open(ref['path']) as im:
                    expected = np.asarray(im)
                row = compare_rgb(expected, rgb)
                destination = directory / filename
                Image.fromarray(rgb).save(destination)
                row.update(step=step, action_id=action, action_name=ACTIONS[action],
                           reference=ref['path'], rendered=str(destination),
                           rendered_sha256=sha256(destination))
                rows.append(row)
            render_episode(env, by_id[eid], actions, consume)
            result = dict(episode_id=eid, scene_id=by_id[eid].scene_id,
                          **summarize_frames(rows), frames=rows)
            report['episodes'].append(result)
            write_json(args.report, report)
            print(json.dumps({k: v for k, v in result.items() if k != 'frames'}), flush=True)
    report['passed'] = len(report['episodes']) == 3 and all(e['passed'] for e in report['episodes'])
    report['status'] = 'passed' if report['passed'] else 'failed_rendering_parity_stop'


def main():
    root = Path(__file__).resolve().parents[2]
    assets = Path('/data14t/anhdh35/diag_g3m5_20261019_assets')
    data = Path('/storage/anhdh35/JanusVLN/data')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--habitat-config', type=Path, default=root/'configs/eval/vln_r2r.yaml')
    p.add_argument('--dataset', type=Path, default=data/'datasets/r2r/train/train.json.gz')
    p.add_argument('--ground-truth', type=Path, default=data/'datasets/r2r/train/train_gt.json.gz')
    p.add_argument('--scenes-dir', type=Path, default=data/'scene_datasets_full')
    p.add_argument('--reference-report', type=Path, default=root/'reports/diagnostics/g3m5_failure_20261019/TRAINING_REFERENCE_RECOVERY.json')
    p.add_argument('--output', type=Path, default=assets/'rendering_gate')
    p.add_argument('--report', type=Path, default=root/'reports/diagnostics/g3m5_failure_20261019/RENDERING_GATE.json')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    report = {'status': 'running', 'passed': False, 'episodes': [],
              'threshold_per_channel': THRESHOLD,
              'definition': 'Mean absolute difference /255 over every pixel and frame separately for R,G,B; all channels in all 3 episodes must pass',
              'sequencing': 'Compare observation before its labeled action; include STOP frame and then execute STOP',
              'reproduction_command': [sys.executable, '-m', 'scripts.diag.render_panel'] + sys.argv[1:],
              'environment_selection': 'janusvln_host has no habitat; use spatialstack_dagger as recorded in prior combined evaluation campaign.py',
              'alternate_definitions_or_camera_tuning': False}
    try:
        run_gate(args, report)
    except Exception as exc:
        report.update(status='error_stop', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    write_json(args.report, report)
    if not report['passed']:
        print(json.dumps({'status': report['status'], 'error': report.get('error'), 'report': str(args.report)}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
