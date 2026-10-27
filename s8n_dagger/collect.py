"""Resumable Janus-style recovery collection for the frozen S8N policy."""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from s8n_dagger.common import (ACTION_NAMES, RecoveryControl, accepted_trajectory,
                               atomic_json, bind_contract, sha256, training_row)
from s8n_dagger.oracle import ReferenceOracle


class ModelClient:
    def __init__(self, python, checkpoint, base, timeout=600):
        self.process = subprocess.Popen(
            [python, '-u', '-m', 's8n_dagger.model_server', '--checkpoint', checkpoint,
             '--model-path', base], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = timeout
        try:
            self.runtime = self.read()
            if self.runtime.get('status') != 'ready':
                raise RuntimeError(f'Model initialization: {self.runtime}')
        except BaseException:
            self.close()
            raise

    def read(self):
        if not self.selector.select(self.timeout):
            raise RuntimeError('Model subprocess timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Model subprocess exited: {self.process.poll()}')
        return json.loads(line)

    def call(self, operation, **fields):
        self.process.stdin.write(json.dumps(dict(operation=operation, **fields)) + '\n')
        self.process.stdin.flush()
        response = self.read()
        if response.get('status') != 'ok':
            raise RuntimeError(f'Model protocol failure: {response}')
        return response

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


def validate_action(result, expected=None):
    action = result['habitat_action_id']
    if type(action) is not int or action not in ACTION_NAMES:
        raise ValueError('Invalid Habitat action ID')
    if result['action_name'] != ACTION_NAMES[action] or result['class_id'] != {1: 0, 2: 1, 3: 2, 0: 3}[action]:
        raise ValueError('Inconsistent model action metadata')
    if expected is not None and action != expected:
        raise ValueError('Committed response does not match executed action')
    return action


def pose(env):
    state = env.sim.get_agent_state()
    q = state.rotation
    return dict(position=[float(x) for x in state.position],
                rotation_xyzw=[float(q.x), float(q.y), float(q.z), float(q.w)])


def collect_episode(env, model, ep, raw, reference_actions, args):
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from qwen_vl.stream.transport import encode_rgb
    uid = f'{args.dataset}:train:{Path(ep.scene_id).parent.name}:{ep.episode_id}:s8n_recovery2'
    record = dict(schema_version=1, episode_uid=uid, episode_id=str(ep.episode_id),
                  dataset=args.dataset, dataset_version=args.dataset_hash, split='train',
                  scene_id=raw['scene_id'], instruction=ep.instruction.instruction_text.strip(),
                  instruction_id=str(raw['instruction'].get('instruction_id', ep.episode_id)),
                  language=raw['instruction'].get('language', 'en'),
                  trajectory_id=raw.get('trajectory_id'), reference_actions=reference_actions,
                  accepted=False, failed=False, termination='max_steps', steps=[])
    frames = []
    started = time.monotonic()
    control = RecoveryControl(reference_actions, len(ep.reference_path))
    travelled = 0.0
    distance = initial = None
    fatal = False
    try:
        env.current_episode = ep
        observation = env.reset()
        initial_value = float(env.get_metrics()['distance_to_goal'])
        if not math.isfinite(initial_value):
            raise ValueError('Nonfinite initial goal distance')
        initial = distance = initial_value
        model.call('reset', episode_uid=uid, instruction=record['instruction'])
        oracle = ReferenceOracle(ep.reference_path, lambda radius: ShortestPathFollower(
            env.sim, goal_radius=radius, return_one_hot=False, stop_on_error=False))
        for step in range(args.max_steps):
            rgb = np.asarray(observation['rgb']).copy()
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError('RGB contract violated')
            expert, advanced = oracle.action()
            if advanced:
                control.advance()
            before = pose(env)
            request = dict(episode_uid=uid, step_id=step, **encode_rgb(rgb))
            proposed = None
            proposal = None
            if expert == 0 or control.recovering or args.expert_only:
                executed = expert
                source = 'oracle_stop' if expert == 0 else 'recovery'
                committed = model.call('force', habitat_action_id=executed, **request)
            else:
                proposal = model.call('propose', **request)
                proposed = validate_action(proposal)
                executed, source = control.choose(proposed, expert, distance)
                committed = model.call('commit', episode_uid=uid, step_id=step,
                                       habitat_action_id=executed)
            validate_action(committed, executed)
            response_ids = committed.get('response_token_ids')
            if not response_ids or response_ids[-1] != args.navigation['eos']:
                raise ValueError('Missing committed response tokens/EOS')
            # Parse actual returned response: expert commits must use canonical IDs.
            if source != 'model':
                cls = {1: 0, 2: 1, 3: 2, 0: 3}[executed]
                if response_ids != args.navigation['action_token_ids'][cls]:
                    raise ValueError('Expert feedback differs from canonical serializer')
            tick = time.monotonic()
            next_observation = env.step(executed)
            after = pose(env)
            travelled += float(np.linalg.norm(np.subtract(after['position'], before['position'])))
            distance_after = float(env.get_metrics()['distance_to_goal'])
            if not math.isfinite(distance_after):
                raise ValueError('Nonfinite post-action goal distance')
            frames.append(rgb)
            row = dict(step_id=step, rgb_path=f'rgb/{step:04d}.png',
                       rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                       pose_before=before, pose_after=after,
                       oracle_action=expert, proposed_action=proposed, executed_action=executed,
                       proposed_response_token_ids=proposal.get('response_token_ids') if proposal else None,
                       action_source=source, waypoint_index=oracle.index,
                       distance_before=distance, distance_after=distance_after,
                       committed_response_token_ids=response_ids,
                       committed_separator_ids=args.navigation['separator_ids'],
                       retained_kv_tokens=committed.get('retained_kv_tokens'),
                       model_seconds=committed.get('model_seconds'),
                       environment_seconds=time.monotonic() - tick)
            record['steps'].append(row)
            observation, distance = next_observation, distance_after
            if executed == 0:
                if expert != 0:
                    raise ValueError('STOP executed before oracle completed route')
                record['termination'] = 'oracle_stop'
                break
            if env.episode_over:
                record['termination'] = 'environment_limit'
                break
    except Exception as exc:
        record.update(failed=True, termination='failure', error=f'{type(exc).__name__}: {exc}')
        # Protocol/model errors should stop a worker instead of poisoning later data.
        fatal = isinstance(exc, (RuntimeError, BrokenPipeError)) and ('Model' in str(exc) or isinstance(exc, BrokenPipeError))
    efficiency = initial / max(initial, travelled) if initial is not None and initial > 0 else 1.0
    record.update(distance_to_goal=distance, initial_distance=initial, path_length=travelled,
                  efficiency=efficiency, model_success=control.model_success,
                  wall_seconds=time.monotonic() - started, fatal=fatal)
    record['accepted'] = (not args.expert_only and not record['failed'] and distance is not None
                          and accepted_trajectory(record['termination'], distance, efficiency,
                                                  control.model_success))
    return record, frames


def write_episode(out, record, frames, keep_rejected=False):
    key = record['episode_id']
    target = out / 'episodes' / key
    if target.exists():
        raise ValueError(f'Duplicate completed episode {key}')
    staging = out / 'pending' / key
    if staging.exists():
        # Preserve evidence of interrupted writes; never regard them as complete.
        staging.rename(out / 'pending' / f'{key}.interrupted-{time.time_ns()}')
    staging.mkdir(parents=True)
    if record['accepted'] or keep_rejected:
        (staging / 'rgb').mkdir()
        for step, frame in enumerate(frames):
            Image.fromarray(frame).save(staging / f'rgb/{step:04d}.png', compress_level=1)
        record['rgb_retained'] = True
    else:
        record['rgb_retained'] = False
    if record['accepted']:
        atomic_json(staging / 'training_episode.json', training_row(record, target))
    atomic_json(staging / 'record.json', record)
    staging.rename(target)


def refresh_outputs(out, records, total):
    accepted = [r for r in records if r['accepted']]
    temporary = out / 'training.jsonl.tmp'
    with temporary.open('w') as handle:
        for record in accepted:
            row = training_row(record, out / 'episodes' / record['episode_id'])
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(out / 'training.jsonl')
    atomic_json(out / 'status.json', dict(pid=os.getpid(), attempted=len(records), total=total,
                accepted=len(accepted), accepted_steps=sum(len(r['steps']) for r in accepted),
                failures=sum(r['failed'] for r in records), complete=len(records) == total,
                total_steps=sum(len(r['steps']) for r in records), updated_unix=time.time(),
                last_episode=records[-1]['episode_id'] if records else None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['r2r', 'rxr15'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--gt-path', required=True)
    p.add_argument('--scenes-dir', required=True)
    p.add_argument('--episode-list', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-python', required=True)
    p.add_argument('--habitat-config', required=True)
    p.add_argument('--simulator-source', required=True)
    p.add_argument('--max-steps', type=int, default=675)
    p.add_argument('--out', required=True)
    p.add_argument('--keep-rejected-rgb', action='store_true')
    p.add_argument('--expert-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.max_steps <= 675:
        raise ValueError('Collection max_steps must fit the frozen S8N logical context')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'episodes').mkdir(exist_ok=True)
    (out / 'pending').mkdir(exist_ok=True)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.append(args.simulator_source)
    from habitat_extensions import task, measures  # noqa: F401
    import habitat
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    config = get_config(args.habitat_config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = 'train'
        config.habitat.dataset.data_path = args.data_path
        config.habitat.dataset.scenes_dir = args.scenes_dir
        config.habitat.seed = 42
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.environment.iterator_options.shuffle = False
        # nDTW is irrelevant to collection and expensive per action.
        if 'ndtw' in config.habitat.task.measurements:
            del config.habitat.task.measurements['ndtw']
    camera = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    sim = config.habitat.simulator
    if (camera.width, camera.height, camera.hfov, sim.forward_step_size, sim.turn_angle) != (640, 480, 79, .25, 15):
        raise ValueError('Incorrect camera/action contract')
    with gzip.open(args.data_path, 'rt') as handle:
        raw_list = json.load(handle)['episodes']
    raw = {str(ep['episode_id']): ep for ep in raw_list}
    if len(raw) != len(raw_list):
        raise ValueError('Duplicate source IDs')
    with gzip.open(args.gt_path, 'rt') as handle:
        gt = json.load(handle)
    wanted = [str(x) for x in json.loads(Path(args.episode_list).read_text())]
    if not wanted or len(set(wanted)) != len(wanted) or set(wanted) - raw.keys():
        raise ValueError('Invalid shard IDs')
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    lookup = {str(ep.episode_id): ep for ep in dataset.episodes}
    dataset.episodes = [lookup[x] for x in wanted]
    for ep in dataset.episodes:
        if not Path(ep.scene_id).is_file() or not Path(ep.scene_id).with_suffix('.navmesh').is_file():
            raise FileNotFoundError(ep.scene_id)
        if str(ep.episode_id) not in gt:
            raise ValueError('Missing reference actions')
    args.navigation = json.loads((Path(args.checkpoint) / 'navigation.json').read_text())
    if args.navigation['episodic']['score'] != 'S8N' or args.navigation['episodic']['budget_tokens'] != 1200 or args.navigation['episodic']['recent_groups'] != 4:
        raise ValueError('Expected original S8N R4/B1200')
    args.dataset_hash = sha256(args.data_path)
    from qwen_vl.eval.habitat_r2r import base_asset_hashes
    import qwen_vl
    qwen_root = Path(qwen_vl.__file__).parent
    from omegaconf import OmegaConf
    contract = dict(protocol='s8n_janus_recovery_v2', dataset=args.dataset, split='train',
                    dataset_sha256=args.dataset_hash, gt_sha256=sha256(args.gt_path),
                    episode_ids=wanted, max_steps=args.max_steps, seed=42,
                    beta=0, waypoint_radius=1.8, final_radius=.25,
                    recovery_ratio=.8, recovery_errors=12, save_recovery_efficiency=.93,
                    save_efficiency=.85, save_distance=.5, expert_only=args.expert_only,
                    keep_rejected_rgb=args.keep_rejected_rgb,
                    weights_sha256=sha256(Path(args.checkpoint) / 'pytorch_model.bin'),
                    navigation_sha256=sha256(Path(args.checkpoint) / 'navigation.json'),
                    base_assets=base_asset_hashes(args.model_path),
                    model_source={str(f.relative_to(qwen_root)): sha256(f) for f in sorted(qwen_root.rglob('*.py'))},
                    collector_source={f.name: sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
                    habitat_version=habitat.__version__, habitat_config=OmegaConf.to_container(config, resolve=True))
    if contract['weights_sha256'] != 'c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b':
        raise ValueError('Not the original S8N checkpoint')
    bind_contract(out, contract)
    records = []
    for directory in sorted((out / 'episodes').iterdir()):
        record = json.loads((directory / 'record.json').read_text())
        if record['episode_id'] not in wanted or record['episode_id'] != directory.name:
            raise ValueError('Unexpected completed episode')
        if record['accepted']:
            for step in record['steps']:
                if not (directory / step['rgb_path']).is_file():
                    raise ValueError('Accepted episode has missing RGB')
        records.append(record)
    refresh_outputs(out, records, len(wanted))
    done = {r['episode_id'] for r in records}
    pending = [ep for ep in dataset.episodes if str(ep.episode_id) not in done]
    if not pending:
        return
    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Collection interrupted')
    signal.signal(signal.SIGTERM, terminate)
    model = ModelClient(args.model_python, args.checkpoint, args.model_path)
    try:
        if model.runtime.get('episodic') != args.navigation['episodic']:
            raise ValueError('Model runtime memory does not match original S8N')
        teacher = model.runtime['teacher']
        for actual, expected in [('weights_sha256', 'weights_sha256'),
                                 ('navigation_sha256', 'navigation_sha256'),
                                 ('source_sha256', 'model_source')]:
            if teacher[actual] != contract[expected]:
                raise ValueError(f'Model subprocess provenance mismatch: {actual}')
        runtime_path = out / 'runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != model.runtime:
            raise ValueError('Serving runtime changed on resume')
        atomic_json(runtime_path, model.runtime)
        dataset.episodes = pending
        with habitat.Env(config=config, dataset=dataset) as env:
            for ep in pending:
                if shutil.disk_usage(out).free < 20 * 1024**3:
                    raise RuntimeError('Less than20GiB free; stop before next episode')
                record, frames = collect_episode(env, model, ep, raw[str(ep.episode_id)],
                                                  len(gt[str(ep.episode_id)]['actions']), args)
                write_episode(out, record, frames, args.keep_rejected_rgb)
                del frames
                records.append(record)
                refresh_outputs(out, records, len(wanted))
                print(json.dumps(dict(dataset=args.dataset, episode=ep.episode_id,
                      attempted=len(records), total=len(wanted), accepted=record['accepted'],
                      steps=len(record['steps']), termination=record['termination'],
                      distance=record['distance_to_goal'], efficiency=record['efficiency'],
                      seconds=round(record['wall_seconds'], 2), error=record.get('error'))), flush=True)
                if record['fatal']:
                    raise RuntimeError('Fatal model error; see saved record')
    finally:
        model.close()


if __name__ == '__main__':
    main()
