"""Habitat-side driver: on-policy DAgger collection on R2R-VLNCE train (simulator interpreter).

    python -m qwen_vl.dagger.collect_r2r --checkpoint S8N --model-python .venv/bin/python \
        --model-path Qwen3.5-4B --data-path .../R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz \
        --scenes-dir .../scene_datasets --oracle-manifest joint.jsonl --out RUN \
        --shard-index 0 --num-shards 8 --gpu 0 --model-device 0

The camera, action, cap and success settings are asserted exactly as in the evaluator
(``qwen_vl.eval.habitat_r2r``); only the split (train) differs.  Each shard owns a fixed slice of
the scene-sorted episode list, its own journal, contract and runtime record, and resumes from
its journal.  ``qwen_vl.dagger.merge`` builds the training manifest from the journals.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from qwen_vl.contracts import ACTIONS
from qwen_vl.dagger.episode import collect_episode
from qwen_vl.dagger.expert import ExpertError, ReferencePathExpert, habitat_follower_factory
from qwen_vl.dagger.rules import DaggerRules, EpisodeController, path_efficiency, path_length, save_decision
from qwen_vl.dagger.store import CollectionStore, manifest_row
from qwen_vl.eval.habitat_r2r import (
    ModelProcess, ModelProcessError, atomic_json, base_asset_hashes, load_journal, resolve_dataset_path,
    scalar_metrics, select_episodes, sha256, validate_prediction)
from qwen_vl.stream.transport import encode_rgb

COLLECTOR_VERSION = 'simplememvln_dagger_r2r_v1'
SERVER_PROTOCOL = 'simplememvln_dagger_step_v1'


class DaggerModel(ModelProcess):
    """``ModelProcess`` running ``qwen_vl.dagger.server`` (optionally pinned to one CUDA device)."""

    def __init__(self, python, checkpoint, model_path=None, timeout=600, startup_timeout=900, device=None):
        cmd = [python, '-u', '-m', 'qwen_vl.dagger.server', '--checkpoint', str(checkpoint)]
        if model_path:
            cmd.extend(['--model-path', str(model_path)])
        env = dict(os.environ)
        if device is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(device)
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = startup_timeout
        try:
            self.runtime = self.read()
            if self.runtime.get('status') != 'ready':
                raise ModelProcessError('Model process failed to initialize')
            if self.runtime.get('protocol') != SERVER_PROTOCOL:
                raise ModelProcessError(f'Unexpected model protocol {self.runtime.get("protocol")!r}')
            self.timeout = timeout
        except BaseException:
            self.close()
            raise

    def step(self, uid, step_id, rgb, forced_class=None, intercept_with=None):
        result = self.call(operation='step', episode_uid=uid, step_id=step_id, forced_class=forced_class,
                           intercept_with=intercept_with, **encode_rgb(rgb))
        return check_response(result, uid, step_id, forced_class, intercept_with)


def check_response(result, uid, step_id, forced_class, intercept_with):
    """Reject a model response that is not the decision that was requested."""
    if result.get('episode_uid') != uid or result.get('step_id') != step_id:
        raise ModelProcessError('Model response belongs to another decision')
    validate_prediction(result)                      # class, action name and Habitat ID agree
    source = result.get('source')
    if forced_class is not None:
        if source != 'forced' or result['class_id'] != forced_class:
            raise ModelProcessError('Forced decision was not executed as requested')
    elif source == 'stop_intercepted':
        if result['class_id'] != intercept_with:
            raise ModelProcessError('Intercepted STOP did not execute the expert action')
    elif source != 'model':
        raise ModelProcessError(f'Unexpected decision source {source!r}')
    return result


class HabitatWorld:
    """``collect_episode`` world over one Habitat episode."""

    def __init__(self, env, episode):
        self.env, self.episode = env, episode

    def reset(self):
        self.env.current_episode = self.episode
        return self.env.reset()['rgb']

    def step(self, action):
        observation = self.env.step(action)
        return None if self.env.episode_over else observation['rgb']

    def distance_to_goal(self):
        value = scalar_metrics(self.env).get('distance_to_goal')
        if value is None:
            raise ExpertError('distance_to_goal unavailable')
        return value

    def position(self):
        return [float(x) for x in self.env.sim.get_agent_state().position]


def source_records(path):
    """episode_id -> scene/instruction fields exactly as ``scripts/vln/prepare_r2r.py`` writes them."""
    with gzip.open(path, 'rt') as handle:
        records = json.load(handle)['episodes']
    version = sha256(path)
    result = {}
    for record in records:
        eid = str(record['episode_id'])
        if eid in result:
            raise ValueError(f'Duplicate source episode {eid}')
        instruction = record['instruction']
        result[eid] = dict(scene_id=record['scene_id'],
                           instruction_id=str(instruction.get('instruction_id', eid)),
                           instruction=instruction['instruction_text'].strip(), dataset_version=version)
    return result


def oracle_lengths(path):
    """episode_id -> oracle step count of the R2R rows of a training manifest."""
    lengths = {}
    if path:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('dataset') == 'r2rce':
                lengths[str(row['episode_id'])] = len(row['steps'])
    return lengths


def build_config(args):
    import habitat
    from habitat.config.default import get_config

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
    if (config.habitat.environment.max_episode_steps != 500
            or config.habitat.task.measurements.success.success_distance != 3.0):
        raise ValueError('Unexpected success distance or rollout cap')
    if config.habitat.dataset.type != 'R2RVLN-v1':
        raise ValueError('DAgger collection requires the R2RVLN-v1 dataset')
    return config


def run_one(env, model, episode, rules, source, oracle_length, args, store):
    eid = str(episode.episode_id)
    started = time.perf_counter()
    record = dict(episode_id=eid, scene_id=source['scene_id'], saved=False, oracle_length=oracle_length,
                  oracle_length_known=oracle_length is not None)
    oracle_length = oracle_length or rules.default_oracle_length
    try:
        expert = ReferencePathExpert(episode.reference_path, habitat_follower_factory(env.sim), rules)
        controller = EpisodeController(rules, oracle_length, expert.waypoints)
        result = collect_episode(HabitatWorld(env, episode), model, expert, controller,
                                 uid=f'dagger:{args.split}:{source["scene_id"]}:{eid}',
                                 instruction=source['instruction'])
    except ModelProcessError:
        raise
    except (ExpertError, ValueError) as exc:          # ValueError: recoverable model-process failure
        record.update(reason='failure', failure=f'{type(exc).__name__}: {exc}',
                      wall_seconds=time.perf_counter() - started)
        return record
    travelled = path_length(result['positions'])
    efficiency = path_efficiency(result['start_distance'], travelled)
    steps = len(result['actions'])
    keep, reason = save_decision(rules, result['final_reached'], result['final_distance'], steps,
                                 efficiency, controller.model_failed, controller.clean)
    if result['failure']:
        keep, reason = False, 'failure'
    summary = dict(controller.summary(), steps=steps, start_distance=result['start_distance'],
                   final_distance=result['final_distance'], path_length=travelled,
                   path_efficiency=efficiency, waypoints=expert.waypoints, final_reached=result['final_reached'])
    record.update(summary, reason=reason, failure=result['failure'],
                  action_counts={name: result['actions'].count(c) for c, name in enumerate(ACTIONS)})
    if keep:
        paths = store.save_frames(eid, result['frames'], result['actions'])
        record['row'] = manifest_row(args.tag, args.split, source, eid, paths, result['actions'],
                                     result['steps'], dict(summary, save_reason=reason,
                                                           collector=COLLECTOR_VERSION))
        record['saved'] = True
    record['wall_seconds'] = time.perf_counter() - started
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-python', required=True)
    p.add_argument('--model-path')
    p.add_argument('--model-device', help='CUDA_VISIBLE_DEVICES for the model process only')
    p.add_argument('--model-startup-timeout', type=int, default=900)
    p.add_argument('--habitat-config',
                   default=str(Path(__file__).resolve().parents[3] / 'configs/eval/vln_r2r.yaml'))
    p.add_argument('--data-path', required=True, help='R2R-VLNCE gzip path, optionally with {split}')
    p.add_argument('--scenes-dir', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--oracle-manifest', help='Training manifest whose r2rce rows give oracle lengths')
    p.add_argument('--episode-list', help='Optional JSON list of episode IDs')
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0, help='Habitat GPU ID (as in the evaluator)')
    p.add_argument('--tag', default='s8n_r2r_it1', help='Collection name inside episode UIDs')
    p.add_argument('--out', required=True)
    for name, field in DaggerRules.__dataclass_fields__.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(field.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            p.add_argument(flag, type=type(field.default), default=field.default)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error('--shard-index must be in [0, --num-shards)')
    rules = DaggerRules(**{name: getattr(args, name) for name in DaggerRules.__dataclass_fields__})
    import habitat
    from habitat.datasets import make_dataset

    config = build_config(args)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    wanted = json.loads(Path(args.episode_list).read_text()) if args.episode_list else None
    episodes = select_episodes(dataset.episodes, wanted, args.max_episodes)[args.shard_index::args.num_shards]
    if not episodes:
        raise ValueError('This shard has no episodes')
    shard_ids = [str(ep.episode_id) for ep in episodes]
    source_path = resolve_dataset_path(args.data_path, args.split)
    sources = source_records(source_path)
    missing = sorted(set(shard_ids) - sources.keys())
    if missing:
        raise ValueError(f'Episodes missing from the source json: {missing[:10]}')
    lengths = oracle_lengths(args.oracle_manifest)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / f'.lock.shard{args.shard_index:03d}').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = Path(args.checkpoint)
    weight = checkpoint / ('model.safetensors' if (checkpoint / 'model.safetensors').exists() else 'pytorch_model.bin')
    sources_root = Path(__file__).resolve().parents[1]
    contract = dict(collector=COLLECTOR_VERSION, rules=rules.to_dict(), tag=args.tag, split=args.split,
                    shard_index=args.shard_index, num_shards=args.num_shards, episode_ids=shard_ids,
                    checkpoint=str(checkpoint.resolve()), weights_sha256=sha256(weight),
                    navigation_sha256=sha256(checkpoint / 'navigation.json'),
                    dataset_sha256=sha256(source_path), habitat_config_sha256=sha256(args.habitat_config),
                    oracle_manifest_sha256=sha256(args.oracle_manifest) if args.oracle_manifest else None,
                    source_sha256={str(f.relative_to(sources_root)): sha256(f)
                                   for f in sorted(sources_root.rglob('*.py'))},
                    base_assets_sha256=base_asset_hashes(args.model_path),
                    simulator_version=getattr(habitat, '__version__', 'unknown'),
                    scenes_dir=str(Path(args.scenes_dir).resolve()), seed=42, max_actions=500,
                    decoding='greedy', transport='raw_rgb_base64', frames='png_lossless')
    contract_file = out / f'contract.shard{args.shard_index:03d}.json'
    store = CollectionStore(out, args.shard_index)
    if contract_file.exists():
        if json.loads(contract_file.read_text()) != contract:
            raise ValueError('Collection contract changed; use a new --out')
    elif store.journal.exists():
        raise ValueError('Journal exists without its contract')
    atomic_json(contract_file, contract)
    records = load_journal(store.journal, set(shard_ids))
    done = {str(r['episode_id']) for r in records}
    removed = store.clean(shard_ids, {str(r['episode_id']) for r in records if r.get('saved')})
    if removed:
        print(json.dumps(dict(removed_unjournaled=removed)), flush=True)
    pending = [ep for ep in episodes if str(ep.episode_id) not in done]
    if not pending:
        return
    dataset.episodes = pending
    model = DaggerModel(args.model_python, args.checkpoint, args.model_path,
                        startup_timeout=args.model_startup_timeout, device=args.model_device)
    try:
        runtime_file = out / f'runtime.shard{args.shard_index:03d}.json'
        if runtime_file.exists() and json.loads(runtime_file.read_text()) != model.runtime:
            raise ValueError('Model runtime changed; use a new --out')
        atomic_json(runtime_file, model.runtime)
        saved = sum(1 for r in records if r.get('saved'))
        with habitat.Env(config=config, dataset=dataset) as env:
            for episode in pending:
                eid = str(episode.episode_id)
                record = run_one(env, model, episode, rules, sources[eid], lengths.get(eid), args, store)
                store.append(record)
                records.append(record)
                saved += bool(record['saved'])
                print(json.dumps(dict(shard=args.shard_index, completed=len(records), total=len(shard_ids),
                                      saved=saved, episode=eid, reason=record['reason'],
                                      steps=record.get('steps'), takeovers=record.get('takeovers'),
                                      pl=record.get('path_efficiency'),
                                      seconds=round(record['wall_seconds'], 1))), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
