"""ActiveVLN GRPO on a SimpleMemVLN text policy: colocated rollouts + DeepSpeed ZeRO-2.

Launch (one process per GPU; each rank renders its own Habitat server on its GPU):

    torchrun --nproc_per_node=2 -m qwen_vl.rl.train_grpo \
        --config configs/rl/activevln_grpo_r2r4k.yaml --output-dir $RUN

One RL step = 8 instructions x G=4 sampled episodes (ActiveVLN dynamic sampling
re-rolls all-fail groups up to twice), GRPO outcome advantages, one optimizer
update with verl's per-micro-batch token-mean policy loss. Trajectories with zero
advantage have zero policy-gradient (KL and entropy coefficients are 0), so they
enter only the token-mean denominators and are not re-forwarded.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from qwen_vl.rl.algos import (activevln_budget, grpo_advantages, loss_weights, micro_batches,
                              policy_loss_terms)
from qwen_vl.rl.checkpoint import load_rank_state, save_rank_state
from qwen_vl.rl.env_client import EnvClient, EnvironmentError_
from qwen_vl.rl.episode import encode_rollout, response_log_probs
from qwen_vl.rl.rollout import rollout_seed, run_episode
from qwen_vl.rl.sampling import SamplingSession

REPO = Path(__file__).resolve().parents[3]


def log(*parts):
    print(f'[{datetime.datetime.now():%H:%M:%S} rank{dist.get_rank() if dist.is_initialized() else 0}]',
          *parts, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool(cfg):
    path = REPO / cfg['data']['pool']
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != cfg['data']['pool_sha256']:
        raise ValueError('RL pool hash differs from the audited ActiveVLN 4K list')
    pool = [json.loads(line) for line in body.decode().splitlines()]
    if cfg['data']['shuffle']:
        raise ValueError('ActiveVLN trains in fixed data order')
    return pool


def load_base_mmap(path, model_path):
    """``vln_runtime.load_checkpoint`` with the same strict checks, but memory-mapped weights."""
    from qwen_vl.train.vln_runtime import load_model

    metadata = json.loads((Path(path) / 'navigation.json').read_text())
    model, serializer = load_model(metadata['config'], model_path)
    if serializer.metadata() != metadata:
        raise ValueError('Checkpoint tokenizer/serializer/codec contract differs')
    state = torch.load(Path(path) / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    return model, serializer


def enable_checkpointing(model, min_tokens):
    """SFT checkpointing without mutating the navigation config the export must keep."""
    from qwen_vl.models.activation_offload import checkpoint_with_cpu_offload

    lm = model.backbone.model.language_model
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lm._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=partial(
        checkpoint_with_cpu_offload, min_tokens=min_tokens, use_reentrant=False))


def build_engine(model, cfg, world):
    import deepspeed

    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    if any('.visual.' in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Vision tower must stay frozen')
    o = cfg['optim']
    # Identical hyperparameters in every group (AdamW is elementwise; clipping uses the global
    # norm), but ZeRO-2 stages each group's flat buffer through host RAM one group at a time:
    # one 4.2 B-parameter group needs ~17 GB of host RAM per rank and OOM-killed the job.
    groups, chunk, target = [], [], sum(p.numel() for p in trainable) / o.get('param_groups', 8)
    for parameter in trainable:
        chunk.append(parameter)
        if sum(p.numel() for p in chunk) >= target:
            groups.append(chunk)
            chunk = []
    if chunk:
        groups.append(chunk)
    optimizer = torch.optim.AdamW([{'params': g, 'weight_decay': o['weight_decay']} for g in groups],
                                  lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'])
    ds_config = {
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 1,
        'train_batch_size': world,
        'gradient_clipping': o['grad_clip'],
        'bf16': {'enabled': True},
        'zero_allow_untested_optimizer': True,
        'zero_optimization': {'stage': 2, 'overlap_comm': False, 'contiguous_gradients': True,
                              'reduce_bucket_size': 50_000_000, 'allgather_bucket_size': 50_000_000},
        'steps_per_print': 10 ** 9,
        'wall_clock_breakdown': False,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
    return engine, ds_config


def export(engine, serializer, base_navigation, path, provenance):
    """EpisodeTrainer._save format; navigation.json must be byte-identical to the base."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model = engine.module
    torch.save(model.state_dict(), tmp / 'pytorch_model.bin')
    model.config.save_pretrained(tmp)
    navigation = json.dumps(serializer.metadata(), indent=2) + '\n'
    if navigation.encode() != base_navigation:
        raise RuntimeError('Exported navigation.json differs from the base checkpoint')
    (tmp / 'navigation.json').write_bytes(base_navigation)
    serializer.processor.save_pretrained(tmp)
    provenance = dict(provenance, weights_sha256=sha256_file(tmp / 'pytorch_model.bin'))
    (tmp / 'rl_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if path.exists():
        shutil.rmtree(path)
    tmp.replace(path)
    return provenance['weights_sha256']


class DummyItem:
    """Zero-weight micro-step that keeps ZeRO-2 gradient collectives aligned across ranks."""

    def __init__(self, serializer):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.batch = encode_rollout(serializer, 'Stop.', [rgb],
                                    [dict(tokens=serializer.action_ids[3], complete=True)])


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def all_reduce(values, op=dist.ReduceOp.SUM, device=None):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return tensor.tolist()


def gather_objects(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


class RolloutWorker:
    """One simulator server + one streaming session + one CUDA stream (one rollout thread)."""

    def __init__(self, model, serializer, env_factory, device):
        self.env_factory, self.device = env_factory, device
        self.env = env_factory()
        self.session = SamplingSession(model, serializer)
        self.stream = torch.cuda.Stream(device)

    def run(self, item, *, step, attempt, sample_idx, budget, cfg):
        r = cfg['rollout']
        seed = rollout_seed(cfg['run']['seed'], step, item['episode_id'], attempt, sample_idx)
        for _ in range(3):
            try:
                with torch.cuda.stream(self.stream):
                    episode = run_episode(self.session, self.env, item, temperature=r['temperature'],
                                          budget=budget, seed=seed, sample_idx=sample_idx,
                                          reward_cfg=cfg['reward'], device=self.device)
                self.stream.synchronize()
                episode.update(attempt=attempt, budget=budget)
                return episode
            except EnvironmentError_ as exc:
                log('environment failure, restarting server:', exc)
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = self.env_factory()
        raise RuntimeError('Environment failed three times on one episode')

    def close(self):
        self.env.close()


def rollout_rank(workers, items, step, cfg):
    """All groups of this rank, G samples each, ActiveVLN dynamic sampling, K threads.

    Seeds depend only on (step, episode, attempt, sample), so results do not depend on
    which worker or in which order a sample runs.
    """
    import queue
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    r = cfg['rollout']
    attempts = 1 + (r['dynamic_sampling']['max_sample_attempts'] if r['dynamic_sampling']['enabled'] else 0)
    free = queue.Queue()
    for worker in workers:
        free.put(worker)

    def task(item, attempt, sample_idx, budget):
        worker = free.get()
        try:
            return worker.run(item, step=step, attempt=attempt, sample_idx=sample_idx,
                              budget=budget, cfg=cfg)
        finally:
            free.put(worker)

    budgets = []
    for item in items:
        budget = activevln_budget(item['N_star'], item['activevln_merged_len'], r['budget_factor'], r['budget_cap'])
        budgets.append(min(budget, r['budget_limit']) if r.get('budget_limit') else budget)
    groups, history = [None] * len(items), [[] for _ in items]
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        def submit(i, attempt):
            return attempt, [pool.submit(task, items[i], attempt, k, budgets[i]) for k in range(r['group_size'])]
        pending = {i: submit(i, 0) for i in range(len(items))}
        while pending:
            wait([f for _, fs in pending.values() for f in fs], return_when=FIRST_COMPLETED)
            for i in [i for i, (_, fs) in pending.items() if all(f.done() for f in fs)]:
                attempt, futures = pending.pop(i)
                group = [f.result() for f in futures]
                history[i].append([dict(success=e['success'], reward=e['reward'], num_steps=e['num_steps'])
                                   for e in group])
                groups[i] = group  # the last re-rolled group replaces earlier ones (ActiveVLN)
                if not any(e['success'] for e in group) and attempt + 1 < attempts:
                    pending[i] = submit(i, attempt + 1)
    return groups, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, help='Stop after this RL step (smoke tests)')
    p.add_argument('--prompts-per-step', type=int, help='Override for smoke tests only')
    p.add_argument('--group-size', type=int, help='Override for smoke tests only')
    p.add_argument('--budget-limit', type=int, help='Hard primitive cap, smoke tests only')
    p.add_argument('--export-steps', type=str, help='Comma list override')
    p.add_argument('--save-every', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load((REPO / args.config if not Path(args.config).is_absolute()
                          else Path(args.config)).read_text())
    smoke = any(v is not None for v in (args.prompts_per_step, args.group_size, args.budget_limit))
    if args.prompts_per_step:
        cfg['data']['prompts_per_step'] = args.prompts_per_step
    if args.group_size:
        cfg['rollout']['group_size'] = args.group_size
    if args.export_steps is not None:
        cfg['run']['export_steps'] = [int(x) for x in args.export_steps.split(',') if x]
    if args.save_every:
        cfg['run']['save_every'] = args.save_every
    if args.budget_limit:
        cfg['rollout']['budget_limit'] = args.budget_limit

    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(hours=4))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rollouts').mkdir(exist_ok=True)
    (out / 'exports').mkdir(exist_ok=True)
    random.seed(cfg['run']['seed'] + rank)
    np.random.seed(cfg['run']['seed'] + rank)
    torch.manual_seed(cfg['run']['seed'] + rank)

    pool = load_pool(cfg)
    per_step = cfg['data']['prompts_per_step']
    if per_step % world:
        raise ValueError('Prompts per step must divide evenly across ranks')
    total_steps = cfg['data']['epochs'] * (len(pool) // per_step)
    G = cfg['rollout']['group_size']
    a = cfg['algorithm']
    mb_size = a['micro_batch_trajectories']
    if (per_step * G) % (mb_size * world) or G % mb_size:
        raise ValueError('Micro-batches must tile groups and ranks')
    num_mb_global = per_step * G // mb_size
    T = cfg['rollout']['temperature']

    base = Path(cfg['base']['checkpoint'])
    base_navigation = (base / 'navigation.json').read_bytes()
    if rank == 0:
        actual = sha256_file(base / 'pytorch_model.bin')
        if actual != cfg['base']['weight_sha256']:
            raise ValueError(f'Base checkpoint hash {actual} differs from the configured SFT weights')
    dist.barrier()
    started = time.perf_counter()
    model, serializer = load_base_mmap(base, cfg['base']['model_path'])
    log(f'loaded base in {time.perf_counter() - started:.0f}s')
    if cfg['trainable'] != 'language_model':
        raise ValueError('Only the SFT trainable set is supported')
    enable_checkpointing(model, cfg['runtime']['activation_offload_min_tokens'])
    batch_steps = cfg['runtime'].get('window_attention_batch_steps', 1)
    if batch_steps != 1:
        # Training-only kernel packing of the same Window8 windows. The serializer keeps the
        # original config object, so the exported navigation.json stays byte-identical.
        import copy
        model.navigation_config = copy.deepcopy(model.navigation_config)
        model.navigation_config['runtime']['window_attention_batch_steps'] = batch_steps
    engine, ds_config = build_engine(model, cfg, world)
    model = engine.module
    trainable_names = [n for n, p_ in model.named_parameters() if p_.requires_grad]

    state_file = out / 'rl_state.json'
    start_step = 0
    if args.resume:
        rl_state = json.loads(state_file.read_text())
        # Per-rank, memory-mapped restore (rl/checkpoint.py); DeepSpeed's own ZeRO-2 resume
        # materializes every rank's fp32 partitions on the host and exhausted its RAM.
        load_rank_state(engine, out / 'resumable' / rl_state['tag'], rank, world)
        start_step = rl_state['next_step']
        if engine.global_steps != start_step:
            raise RuntimeError('Optimizer step counter disagrees with the RL state')
        log('resumed at step', start_step)
    elif state_file.exists():
        raise ValueError('Output exists; pass --resume')

    if rank == 0 and not args.resume:
        git = lambda *c: subprocess.run(['git', *c], cwd=REPO, capture_output=True, text=True).stdout.strip()
        (out / 'run_config.json').write_text(json.dumps(dict(
            config=cfg, args=vars(args), smoke=smoke, world_size=world, deepspeed=ds_config,
            total_steps=total_steps, git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
            git_dirty=git('status', '--porcelain'), python=platform.python_version(),
            torch=torch.__version__, deepspeed_version=deepspeed.__version__,
            transformers=__import__('transformers').__version__,
            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            trainable_parameters=sum(p_.numel() for p_ in model.parameters() if p_.requires_grad),
            trainable_tensors=len(trainable_names), command=sys.argv,
            started=datetime.datetime.now().isoformat()), indent=2) + '\n')

    episode_list = out / f'pool_ids_rank{rank}.json'
    episode_list.write_text(json.dumps([item['episode_id'] for item in pool]))
    env_root = os.environ
    env_factory = lambda: EnvClient(
        env_root['EVAL_PYTHON'], REPO / cfg['env']['habitat_config'], env_root['R2R_DATA'],
        env_root['MP3D_SCENES'], local_rank, episode_list, split=cfg['env']['split'],
        pythonpath=f"{REPO / 'src'}:{env_root.get('SIMULATOR_SOURCE', '')}",
        log_path=out / f'env_rank{rank}.log')
    workers = [RolloutWorker(model, serializer, env_factory, device)
               for _ in range(cfg['rollout'].get('workers_per_rank', 1))]
    log('environments ready', len(workers), workers[0].env.info)
    dummy = DummyItem(serializer)

    for step in range(start_step, total_steps):
        if args.max_steps is not None and step >= args.max_steps:
            break
        step_started = time.perf_counter()
        offset = (step * per_step) % len(pool)
        prompts = pool[offset:offset + per_step]
        local = prompts[rank * per_step // world:(rank + 1) * per_step // world]

        # ---- rollout phase: same weights as the learner (colocated, no sync) ----
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        groups, reroll = rollout_rank(workers, local, step, cfg)
        torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - tick
        rollout_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()

        # ---- advantages (each group lives on one rank) and verl weights ----
        trajectories = []
        for gi, group in enumerate(groups):
            adv = grpo_advantages([e['reward'] for e in group], a['epsilon'])
            for e, value in zip(group, adv.tolist()):
                e['advantage'] = value
                trajectories.append(e)
        for mb, keys in enumerate(micro_batches(range(len(trajectories)), mb_size)):
            for k in keys:
                trajectories[k]['micro_batch'] = rank * len(trajectories) // mb_size + mb
        weights = loss_weights([e['response_tokens'] for e in trajectories],
                               [e['micro_batch'] for e in trajectories], world, num_mb_global)
        items = [(e, w) for e, w in zip(trajectories, weights) if e['advantage'] != 0.0]
        n_max = int(all_reduce([len(items)], dist.ReduceOp.MAX, device)[0])
        steps_needed = max(n_max, 1)

        # ---- one optimizer update ----
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        stats = dict(pg_loss=0.0, clipfrac=0.0, tokens=0, mismatch_abs=0.0, mismatch_max=0.0,
                     train_entropy=0.0, forward_tokens=0)
        for i in range(steps_needed):
            engine.set_gradient_accumulation_boundary(i == steps_needed - 1)
            if i < len(items):
                e, w = items[i]
                batch = to_device(encode_rollout(serializer, e['instruction'], e['_frames'],
                                                 e['_responses']), device)
                logp, entropy = response_log_probs(model, batch, T, with_entropy=True)
                rollout_logp = torch.tensor(e['rollout_logps'], device=device)
                if logp.shape != rollout_logp.shape:
                    raise RuntimeError('Trainer and rollout response tokens differ')
                adv = torch.full_like(logp, e['advantage'])
                token_loss, clipped = policy_loss_terms(logp, logp.detach(), adv, a['clip_ratio_low'],
                                                        a['clip_ratio_high'], a['clip_ratio_c'])
                loss = token_loss.sum() * w
                diff = (logp.detach() - rollout_logp).abs()
                stats['mismatch_abs'] += float(diff.sum())
                stats['mismatch_max'] = max(stats['mismatch_max'], float(diff.max()))
                stats['pg_loss'] += float(loss.detach())
                stats['clipfrac'] += float(clipped.float().sum())
                stats['tokens'] += logp.numel()
                stats['train_entropy'] += float(entropy.detach().sum())
                stats['forward_tokens'] += batch['input_ids'].shape[1]
            else:
                batch = to_device(dummy.batch, device)
                logp, _ = response_log_probs(model, batch, T)
                loss = logp.sum() * 0.0
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite RL loss at step {step}')
            engine.backward(loss, scale_wrt_gas=False)
            del batch, logp, loss
        engine.step()
        grad_norm = engine.get_global_grad_norm()
        grad_norm = float(grad_norm) if grad_norm is not None else float('nan')
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f'Non-finite gradient norm at step {step}')
        if engine.global_steps != step + 1:
            raise RuntimeError(f'Expected one optimizer update per RL step, got {engine.global_steps}')
        train_seconds = time.perf_counter() - tick
        train_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()

        # ---- metrics and records ----
        records = [{k: v for k, v in e.items() if not k.startswith('_') and k != 'rollout_logps'}
                   for e in trajectories]
        with (out / 'rollouts' / f'step_{step:04d}_rank{rank}.jsonl').open('w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        gathered = gather_objects(dict(records=[{k: r[k] for k in (
            'reward', 'success', 'oracle_success', 'num_steps', 'truncated', 'format_error', 'stopped',
            'final_distance', 'spl', 'response_tokens', 'mean_entropy', 'greedy_agreement', 'advantage',
            'attempt', 'model_seconds', 'env_seconds')} for r in records],
            groups=[[e['success'] for e in g] for g in groups],
            rerolls=[len(h) - 1 for h in reroll], discarded=sum(len(h[:-1]) * G for h in reroll),
            stats=stats, items=len(items), rollout_seconds=rollout_seconds, train_seconds=train_seconds,
            rollout_peak=rollout_peak, train_peak=train_peak))
        if rank == 0:
            recs = [r for g in gathered for r in g['records']]
            grps = [x for g in gathered for x in g['groups']]
            tok = sum(g['stats']['tokens'] for g in gathered)
            mean = lambda key: sum(float(r[key]) for r in recs) / len(recs)
            metric = dict(
                step=step, global_steps=engine.global_steps, trajectories=len(recs),
                reward_mean=mean('reward'), success_rate=mean('success'),
                oracle_success_rate=mean('oracle_success'), spl=mean('spl'),
                truncation_rate=mean('truncated'), format_error_rate=mean('format_error'),
                mean_steps=mean('num_steps'), final_distance=mean('final_distance'),
                rollout_entropy=mean('mean_entropy'), greedy_agreement=mean('greedy_agreement'),
                all_fail_groups=sum(not any(x) for x in grps) / len(grps),
                all_success_groups=sum(all(x) for x in grps) / len(grps),
                rerolled_groups=sum(x > 0 for g in gathered for x in g['rerolls']),
                reroll_rounds=sum(sum(g['rerolls']) for g in gathered),
                discarded_trajectories=sum(g['discarded'] for g in gathered),
                trained_trajectories=sum(g['items'] for g in gathered),
                pg_loss=sum(g['stats']['pg_loss'] for g in gathered) / world,
                clipfrac=sum(g['stats']['clipfrac'] for g in gathered) / max(1, tok),
                logp_mismatch_mean=sum(g['stats']['mismatch_abs'] for g in gathered) / max(1, tok),
                logp_mismatch_max=max(g['stats']['mismatch_max'] for g in gathered),
                train_entropy=sum(g['stats']['train_entropy'] for g in gathered) / max(1, tok),
                trained_tokens=tok, response_tokens=sum(r['response_tokens'] for r in recs),
                forward_tokens=sum(g['stats']['forward_tokens'] for g in gathered),
                grad_norm=grad_norm, lr=engine.get_lr()[0],
                rollout_seconds=max(g['rollout_seconds'] for g in gathered),
                train_seconds=max(g['train_seconds'] for g in gathered),
                step_seconds=time.perf_counter() - step_started,
                rollout_peak_gib=max(g['rollout_peak'] for g in gathered),
                train_peak_gib=max(g['train_peak'] for g in gathered),
                env_steps=sum(r['num_steps'] for r in recs),
                time=datetime.datetime.now().isoformat())
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metric) + '\n')
            log('STEP', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metric.items()}))
        del groups, trajectories, items
        done = step + 1
        if done in cfg['run']['export_steps'] or done == total_steps:
            if rank == 0:
                git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                     text=True).stdout.strip()
                digest = export(engine, serializer, base_navigation, out / 'exports' / f'step_{done:04d}',
                                dict(base_checkpoint=str(base), base_weight_sha256=cfg['base']['weight_sha256'],
                                     rl_step=done, total_steps=total_steps, pool_sha256=cfg['data']['pool_sha256'],
                                     algorithm='activevln_grpo', config=cfg, git_commit=git, smoke=smoke))
                log('exported step', done, digest)
            dist.barrier()
        if (done % cfg['run']['save_every'] == 0 or done == total_steps
                or done in cfg['run'].get('extra_save_steps', [])):
            tag = f'step_{done:04d}'
            save_rank_state(engine, out / 'resumable' / tag, rank, world)
            dist.barrier()
            if rank == 0:
                state_file.write_text(json.dumps(dict(next_step=done, tag=tag, total_steps=total_steps)) + '\n')
                kept = sorted((out / 'resumable').glob('step_*'))
                for old in kept[:-cfg['run']['keep_resumable']]:
                    shutil.rmtree(old)
            dist.barrier()
    for worker in workers:
        worker.close()
    if rank == 0:
        (out / 'COMPLETE' if not smoke and engine.global_steps == total_steps else out / 'STOPPED').write_text(
            json.dumps(dict(global_steps=engine.global_steps, total_steps=total_steps,
                            finished=datetime.datetime.now().isoformat())) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
