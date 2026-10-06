"""Closed-loop R2R evaluation with an independent persistent model process.

Camera/action settings follow StageVLN-v2. Streaming history and inference
failure semantics follow SimpleMemVLN's technical plan, sections 5 and 13.
"""
import argparse
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np
from qwen_vl.contracts import ACTIONS, HABITAT_IDS
from qwen_vl.eval.metrics import summarize
from qwen_vl.stream.transport import encode_rgb


class ModelProcessError(RuntimeError):
    """Broken transport/runtime: abort the run rather than failing future episodes."""


class ModelProcess:
    def __init__(self, python, checkpoint, model_path=None, timeout=600):
        cmd = [python, '-u', '-m', 'qwen_vl.stream.server', '--checkpoint', str(checkpoint)]
        if model_path:
            cmd.extend(['--model-path', str(model_path)])
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = timeout
        try:
            self.runtime = self.read()
            if self.runtime.get('status') != 'ready':
                raise ModelProcessError('Model process failed to initialize')
        except BaseException:
            self.close()
            raise

    def read(self):
        if not self.selector.select(self.timeout):
            raise ModelProcessError('Model request timed out')
        line = self.process.stdout.readline()
        if not line:
            raise ModelProcessError(f'Model process exited: {self.process.poll()}')
        try:
            return json.loads(line)
        except ValueError as exc:
            raise ModelProcessError('Corrupted model protocol') from exc

    def call(self, **request):
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ModelProcessError('Model pipe closed') from exc
        result = self.read()
        if result.get('status') == 'fatal':
            raise ModelProcessError(result.get('error', 'Fatal model failure'))
        if result.get('status') != 'ok':
            raise ValueError(result.get('error', 'Model failure'))
        return result

    def reset(self, uid, instruction):
        return self.call(operation='reset', episode_uid=uid, instruction=instruction)

    def observe(self, uid, step, rgb):
        return self.call(operation='observe', episode_uid=uid, step_id=step, **encode_rgb(rgb))

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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(template, split, role='guide'):
    """Resolve both R2R and role-specific RxR dataset path templates."""
    return template.format(split=split, role=role)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_episodes(episodes, wanted=None, limit=0):
    lookup = {str(ep.episode_id): ep for ep in episodes}
    if len(lookup) != len(episodes):
        raise ValueError('Dataset has duplicate episode IDs')
    if wanted is not None:
        wanted = [str(x) for x in wanted]
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError('Nonempty unique episode list required')
        if set(wanted) - lookup.keys():
            raise ValueError('Requested episodes missing from dataset')
        episodes = [lookup[key] for key in wanted]
    # Scene grouping avoids expensive simulator scene reloads; all epochs use
    # exactly the same deterministic order and selected episode identities.
    episodes = sorted(episodes, key=lambda ep: (ep.scene_id, str(ep.episode_id)))
    return episodes[:limit] if limit else episodes


def load_journal(path, wanted):
    path = Path(path)
    records, seen = [], set()
    if path.exists():
        content = path.read_bytes()
        lines = content.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b'\n'):
                    raise ValueError(f'Corrupt journal record {index + 1}')
                # Only an unterminated final write may be discarded. Preserve
                # it for diagnosis, under the caller's exclusive journal lock.
                path.with_suffix(path.suffix + '.incomplete-tail').write_bytes(line)
                with path.open('r+b') as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            key = str(row['episode_id'])
            if key in seen or key not in wanted:
                raise ValueError(f'Duplicate or unexpected journal episode: {key}')
            seen.add(key)
            records.append(row)
            offset += len(line)
        if records and lines and not lines[-1].endswith(b'\n') and offset == len(content):
            # Complete JSON followed by interruption before the newline.
            with path.open('ab') as handle:
                handle.write(b'\n')
                handle.flush()
                os.fsync(handle.fileno())
    return records


def base_asset_hashes(directory):
    if directory is None:
        return None  # Remote model is pinned by navigation.json's revision.
    directory = Path(directory)
    names = ('config.json', 'processor_config.json', 'preprocessor_config.json',
             'video_preprocessor_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json',
             'chat_template.jinja', 'vocab.json', 'merges.txt')
    return {name: sha256(directory / name) for name in names if (directory / name).is_file()}


def scalar_metrics(env):
    return {key: float(value) for key, value in env.get_metrics().items()
            if isinstance(value, numbers.Real) and math.isfinite(float(value))}


def validate_prediction(prediction):
    """Reject a protocol response whose class, label, and Habitat ID disagree."""
    cls = prediction.get('class_id')
    if type(cls) is not int or cls not in range(4):
        raise ValueError(f'Invalid model action class: {cls!r}')
    expected = (ACTIONS[cls], HABITAT_IDS[cls])
    actual = (prediction.get('action_name'), prediction.get('habitat_action_id'))
    if actual != expected:
        raise ValueError(f'Model action metadata inconsistent: {actual!r} for class {cls}')
    return HABITAT_IDS[cls]


def rollout_episode(env, model, episode, split, max_steps=500):
    uid = f'{split}:{episode.scene_id}:{episode.episode_id}'
    record = dict(episode_id=str(episode.episode_id), scene_id=episode.scene_id,
                  episode_uid=uid, instruction=episode.instruction.instruction_text.strip(),
                  failed=False, metrics={}, actions=[], forced_stop=False)
    positions, distances = [], []
    reset_ok = False
    started = time.perf_counter()
    try:
        env.current_episode = episode
        observation = env.reset()
        reset_ok = True
        positions.append([float(x) for x in env.sim.get_agent_state().position])
        initial = scalar_metrics(env)
        if 'distance_to_goal' in initial:
            distances.append(initial['distance_to_goal'])
        model.reset(uid, record['instruction'])
        for step in range(max_steps):
            tick = time.perf_counter()
            prediction = model.observe(uid, step, observation['rgb'])
            elapsed = time.perf_counter() - tick
            predicted = validate_prediction(prediction)
            executed = predicted
            if step == max_steps - 1 and executed != 0:
                executed = 0
                record['forced_stop'] = True
            tick = time.perf_counter()
            observation = env.step(executed)
            action = dict(prediction, predicted=predicted, executed=executed,
                          transport_and_model_seconds=elapsed,
                          environment_seconds=time.perf_counter()-tick)
            record['actions'].append(action)
            positions.append([float(x) for x in env.sim.get_agent_state().position])
            current = scalar_metrics(env)
            if 'distance_to_goal' in current:
                distances.append(current['distance_to_goal'])
            if env.episode_over:
                break
    except Exception as exc:
        record['failed'] = True
        record['failure_reason'] = f'{type(exc).__name__}: {exc}'
        record['fatal'] = isinstance(exc, ModelProcessError)
    if reset_ok:
        try:
            record['metrics'].update(scalar_metrics(env))
        except Exception as exc:
            record['metric_error'] = str(exc)
    if positions:
        record['trajectory'] = positions
        record['metrics']['trajectory_length'] = float(np.linalg.norm(
            np.diff(np.asarray(positions), axis=0), axis=1).sum())
    if distances:
        record['metrics']['oracle_success'] = float(min(distances) < 3.0)
        record['minimum_distance_to_goal'] = min(distances)
    if not {'success', 'spl'} <= record['metrics'].keys() and not record['failed']:
        record['failed'] = True
        record['failure_reason'] = 'Required simulator metrics unavailable'
    record['wall_seconds'] = time.perf_counter()-started
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--habitat-config', default=str(Path(__file__).resolve().parents[3] / 'configs/eval/vln_r2r.yaml'))
    p.add_argument('--data-path', help='R2R gzip path containing optional {split}')
    p.add_argument('--scenes-dir')
    p.add_argument('--data-root', help='Legacy JanusVLN data root')
    p.add_argument('--model-python', required=True)
    p.add_argument('--model-path')
    p.add_argument('--simulator-source')
    p.add_argument('--dataset-kind', choices=('r2r', 'rxr15'), default='r2r')
    p.add_argument('--ground-truth-path', help='RxR guide GT path, optionally with {split}/{role}')
    p.add_argument('--split', default='val_unseen', choices=['val_seen', 'val_unseen'])
    p.add_argument('--episode-list', help='Optional JSON list of official episode ID strings')
    p.add_argument('--max-episodes', type=int, default=0, help='0 evaluates the full split')
    p.add_argument('--gpu', type=int, default=0, help='Habitat physical GPU ID')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.max_episodes < 0:
        p.error('--max-episodes must be nonnegative')
    if args.simulator_source:
        sys.path.append(args.simulator_source)
    if args.dataset_kind == 'rxr15':
        if not args.simulator_source or not args.ground_truth_path:
            p.error('RxR requires --simulator-source and --ground-truth-path')
        from habitat_extensions import measures, task  # noqa: F401; register RxR Hydra components
    if args.data_root:
        args.scenes_dir = args.scenes_dir or str(Path(args.data_root) / 'scene_datasets')
        args.data_path = args.data_path or str(Path(args.data_root) / 'datasets/r2r/{split}/{split}.json.gz')
    if not args.data_path or not args.scenes_dir:
        p.error('--data-path and --scenes-dir are required (or --data-root)')
    import habitat
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset

    config = get_config(args.habitat_config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.scenes_dir = str(Path(args.scenes_dir).resolve())
        config.habitat.dataset.data_path = args.data_path
        config.habitat.seed = 42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
        config.habitat.environment.iterator_options.shuffle = False
    sim = config.habitat.simulator
    camera = sim.agents.main_agent.sim_sensors.rgb_sensor
    if (camera.width, camera.height, camera.hfov, sim.forward_step_size, sim.turn_angle) != (640, 480, 79, 0.25, 15):
        raise ValueError('Habitat camera/action contract differs from saved recipe')
    if config.habitat.environment.max_episode_steps != 500 or config.habitat.task.measurements.success.success_distance != 3.0:
        raise ValueError('Unexpected success distance or rollout cap')
    if args.dataset_kind == 'rxr15':
        if config.habitat.dataset.type != 'RxRVLNCE-v1':
            raise ValueError('RxR evaluation requires the RxRVLNCE-v1 dataset')
        if list(config.habitat.dataset.roles) != ['guide'] or set(config.habitat.dataset.languages) != {'en-US', 'en-IN'}:
            raise ValueError('RxR evaluation requires English guide en-US/en-IN episodes')
        with habitat.config.read_write(config):
            config.habitat.task.measurements.ndtw.gt_path = args.ground_truth_path
    elif config.habitat.dataset.type != 'R2RVLN-v1':
        raise ValueError('R2R evaluation requires the R2RVLN-v1 dataset')
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    wanted = json.loads(Path(args.episode_list).read_text()) if args.episode_list else None
    dataset.episodes = select_episodes(dataset.episodes, wanted, args.max_episodes)
    wanted = [str(ep.episode_id) for ep in dataset.episodes]
    if not wanted:
        raise ValueError('No selected evaluation episodes')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fail immediately on concurrent writers to one journal.
    import fcntl
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = Path(args.checkpoint)
    weight = checkpoint / ('model.safetensors' if (checkpoint / 'model.safetensors').exists() else 'pytorch_model.bin')
    sources = Path(__file__).resolve().parents[1]
    contract = dict(checkpoint=str(checkpoint.resolve()), weights_sha256=sha256(weight),
                    navigation_sha256=sha256(checkpoint / 'navigation.json'),
                    dataset_sha256=sha256(resolve_dataset_path(args.data_path, args.split)),
                    habitat_config_sha256=sha256(args.habitat_config),
                    source_sha256={str(f.relative_to(sources)): sha256(f) for f in sorted(sources.rglob('*.py'))},
                    base_assets_sha256=base_asset_hashes(args.model_path),
                    simulator_version=getattr(habitat, '__version__', 'unknown'),
                    split=args.split, episode_ids=wanted, seed=42,
                    scenes_dir=str(Path(args.scenes_dir).resolve()),
                    model_python=str(Path(args.model_python).resolve()),
                    model_path=str(Path(args.model_path).resolve()) if args.model_path else None,
                    protocol='R2R-VLNCE-0.2.4', max_steps=500,
                    failure_policy='zero_sr_spl_keep_partial_metrics', transport='raw_rgb_base64')
    if args.dataset_kind == 'rxr15':
        contract['ground_truth_sha256'] = sha256(resolve_dataset_path(args.ground_truth_path, args.split))
    contract_file = out / 'evaluation_contract.json'
    journal = out / 'episodes.jsonl'
    if contract_file.exists():
        if json.loads(contract_file.read_text()) != contract:
            raise ValueError('Evaluation resume contract changed')
    elif journal.exists():
        raise ValueError('Journal exists without its evaluation contract')
    atomic_json(contract_file, contract)
    records = load_journal(journal, set(wanted))
    done = {r['episode_id'] for r in records}

    def save_summary():
        if records:
            atomic_json(out / 'summary.json', dict(summarize(records), scheduled_episodes=len(wanted),
                        complete=len(records)==len(wanted)))

    save_summary()
    if done == set(wanted):
        return
    pending = [ep for ep in dataset.episodes if str(ep.episode_id) not in done]
    dataset.episodes = pending
    model = ModelProcess(args.model_python, args.checkpoint, args.model_path)
    try:
        runtime_file = out / 'runtime.json'
        if runtime_file.exists() and json.loads(runtime_file.read_text()) != model.runtime:
            raise ValueError('Model runtime changed; use a new output directory')
        atomic_json(runtime_file, model.runtime)
        with habitat.Env(config=config, dataset=dataset) as env:
            for ep in pending:
                row = rollout_episode(env, model, ep, args.split)
                records.append(row)
                with journal.open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                save_summary()
                print(json.dumps(dict(checkpoint=checkpoint.name, completed=len(records), total=len(wanted),
                                      episode=ep.episode_id, failed=row['failed'], steps=len(row['actions']),
                                      success=row['metrics'].get('success'), seconds=round(row['wall_seconds'],2))), flush=True)
                if row.get('fatal'):
                    raise RuntimeError('Evaluation stopped after fatal model failure; inspect journal before resuming')
    finally:
        model.close()
    if len(records) != len(wanted):
        raise RuntimeError('Evaluation denominator incomplete')


if __name__ == '__main__':
    main()
