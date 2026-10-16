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
from qwen_vl.rl.hico import (GeometricBaseline, action_terms, batch_scale, coefficients, episode_features,
                             fork_coefficients,
                             grpo_stop_coefficients, rloo_baselines, stop_values)
from qwen_vl.rl.placement import redistribute, release_cache, should_offload, start_hold, wait_for_free
from qwen_vl.rl.rollout import rollout_seed, run_episode
from qwen_vl.rl.sampling import SamplingSession

REPO = Path(__file__).resolve().parents[3]


def release_host_memory():
    """Return freed rollout frames to the OS (glibc keeps freed heap otherwise) on the shared host."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


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
    limit = cfg['data'].get('max_prompts')
    if limit:
        # A prefix of the fixed-order list: exactly the instructions ActiveVLN's first
        # limit/8 steps would consume.
        pool = pool[:limit]
        subset = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in pool)
        if hashlib.sha256(subset.encode()).hexdigest() != cfg['data']['subset_sha256']:
            raise ValueError('RL subset hash differs from the configured prefix')
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


HOST_OFFLOAD = {'on': False}  # set per training episode by rl/placement.should_offload


def checkpoint_with_host_offload(function, *args, min_tokens, **kwargs):
    """SFT ``checkpoint_with_cpu_offload`` with pageable (not pinned) host buffers.

    On the shared GPUs, long RL episodes need their per-layer checkpoint inputs moved off
    the device; pageable memory keeps that host RAM reclaimable-accounted under the unit cap.
    The non-reentrant checkpoint graph, and therefore every gradient, is unchanged.
    """
    from contextlib import nullcontext
    from torch.utils.checkpoint import checkpoint

    kwargs.pop('use_reentrant', None)
    hidden = args[0]
    context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
               if HOST_OFFLOAD['on'] or hidden.shape[-2] >= min_tokens else nullcontext())
    with context:
        return checkpoint(function, *args, use_reentrant=False, **kwargs)


def enable_checkpointing(model, min_tokens):
    """SFT checkpointing without mutating the navigation config the export must keep."""
    lm = model.backbone.model.language_model
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lm._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=partial(
        checkpoint_with_host_offload, min_tokens=min_tokens))


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

    def run(self, item, *, step, attempt, sample_idx, budget, cfg, fork=None):
        r = cfg['rollout']
        seed = rollout_seed(cfg['run']['seed'], step, item['episode_id'], attempt, sample_idx)
        for _ in range(3):
            try:
                with torch.cuda.stream(self.stream):
                    episode = run_episode(self.session, self.env, item, temperature=r['temperature'],
                                          budget=budget, seed=seed, sample_idx=sample_idx,
                                          reward_cfg=cfg['reward'], device=self.device, fork=fork)
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


def rollout_rank(workers, items, step, cfg, fork=None):
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
                              budget=budget, cfg=cfg, fork=fork)
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
                history[i].append([dict(success=e['success'], reward=e['reward'], num_steps=e['num_steps'],
                                        branch_steps=branch_steps(e)) for e in group])
                groups[i] = group  # the last re-rolled group replaces earlier ones (ActiveVLN)
                if not any(e['success'] for e in group) and attempt + 1 < attempts:
                    pending[i] = submit(i, attempt + 1)
    return groups, history


def rollout_rank_pool(pool, items, step, cfg, fork=None):
    """rollout_rank for multi-process IPC helpers (rl/rollout_pool.py); same scheduling."""
    r = cfg['rollout']
    attempts = 1 + (r['dynamic_sampling']['max_sample_attempts'] if r['dynamic_sampling']['enabled'] else 0)
    G = r['group_size']
    budgets = []
    for item in items:
        budget = activevln_budget(item['N_star'], item['activevln_merged_len'], r['budget_factor'], r['budget_cap'])
        budgets.append(min(budget, r['budget_limit']) if r.get('budget_limit') else budget)
    groups, history, pending = [None] * len(items), [[] for _ in items], {}

    def submit(i, attempt):
        for k in range(G):
            pool.submit((i, attempt, k), items[i], temperature=r['temperature'], budget=budgets[i],
                        seed=rollout_seed(cfg['run']['seed'], step, items[i]['episode_id'], attempt, k),
                        sample_idx=k, reward_cfg=cfg['reward'], fork=fork)
        pending[i] = (attempt, {})

    for i in range(len(items)):
        submit(i, 0)
    while pending:
        (i, attempt, k), episode = pool.get()
        if pending[i][0] != attempt:
            raise RuntimeError('Out-of-order rollout result')
        episode.update(attempt=attempt, budget=budgets[i])
        pending[i][1][k] = episode
        if len(pending[i][1]) == G:
            finished = pending.pop(i)[1]
            group = [finished[j] for j in range(G)]
            history[i].append([dict(success=e['success'], reward=e['reward'], num_steps=e['num_steps'],
                                    branch_steps=branch_steps(e)) for e in group])
            groups[i] = group  # the last re-rolled group replaces earlier ones (ActiveVLN)
            if not any(e['success'] for e in group) and attempt + 1 < attempts:
                submit(i, attempt + 1)
    return groups, history


def branch_steps(episode):
    """(branch decisions, replayed prefix decisions) of a grpo_fork main trajectory."""
    branches = episode.get('_branches') or []
    return [sum(b['num_steps'] for b in branches), sum(b['fork_t'] for b in branches)]


def fork_threshold(out, step, fork_cfg):
    """Entropy threshold for this step's fork points: a quantile of the previous step's first-token
    entropies (main trajectories, all ranks). It never depends on this step's samples."""
    values = []
    for path in sorted((out / 'rollouts').glob(f'step_{step - 1:04d}_rank*.jsonl')):
        for line in path.read_text().splitlines():
            values += [v for v in json.loads(line).get('step_entropy') or [] if v is not None]
    if not values:
        return float(fork_cfg['initial_min_entropy'])
    return float(np.quantile(values, fork_cfg['entropy_quantile']))


def fork_prepare(groups, weights_by_uid, cfg, step_scale=None):
    """grpo_fork: GRPO root advantages before the first fork, sibling-baselined credit after it.

    Expands every forked branch into a full training record (main prefix + branch suffix) whose
    prefix decisions carry no coefficient. Branch tokens get their parent's GRPO token weight.
    Sibling contrasts are divided by the group's reward std (``fork_scale: group``) or, with
    ``fork_scale: batch``, by ``step_scale``: one std of all main rewards of the RL step, so that
    near-zero-variance groups (all successes) do not amplify small final-distance differences.
    """
    a = cfg['algorithm']
    floor = a['scale_floor'] * cfg['reward']['success_reward_base']
    by_batch = a.get('fork_scale', 'group') == 'batch'
    if by_batch and not (step_scale is not None and step_scale > 0):
        raise ValueError('fork_scale: batch needs a positive step scale')
    items, stats = [], dict(forks=0, branches=0, branch_success=0, branch_same_first=0, local_nonzero=0,
                            branch_decisions=0, replay_decisions=0, mains=0, main_success=0,
                            branch_minus_main_at_fork=0.0, branch_c_abs=0.0)
    for group in groups:
        rewards = np.asarray([e['reward'] for e in group], dtype=np.float64)
        std = float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0
        scale = step_scale if by_batch else max(std + a['epsilon'], floor)
        for e in group:
            w = weights_by_uid[e['uid']]
            branches = e.pop('_branches', [])
            points = e.get('fork_points', [])
            by_fork = [[b for b in branches if b['fork_t'] == t] for t in points]
            stats['mains'] += 1
            if points:
                main_c, branch_c = fork_coefficients(e['num_steps'], points, e['reward'],
                                                     [[b['reward'] for b in bs] for bs in by_fork],
                                                     e['advantage'], scale)
            else:
                main_c, branch_c = np.full(e['num_steps'], e['advantage']), []
            e['_c_logpi'], e['_c_sigma'] = main_c.tolist(), [0.0] * e['num_steps']
            e['fork_summary'] = [dict(fork_t=b['fork_t'], branch=b['branch'], reward=b['reward'],
                                      success=b['success'], num_steps=b['num_steps'],
                                      same_first_action=b['same_first_action'], coefficient=c)
                                 for bs, cs in zip(by_fork, branch_c) for b, c in zip(bs, cs)]
            if any(e['_c_logpi']):
                items.append((e, w))
            stats['forks'] += len(points)
            for bs, cs in zip(by_fork, branch_c):
                for b, c in zip(bs, cs):
                    t = b['fork_t']
                    prefix_tokens = sum(len(r['tokens']) for r in e['_responses'][:t])
                    full = dict(
                        uid=f"{e['uid']}:fork{t}:{b['branch']}", episode_id=e['episode_id'],
                        instruction=e['instruction'], is_branch=True,
                        _frames=e['_frames'][:t] + b['_frames'], _responses=e['_responses'][:t] + b['_responses'],
                        rollout_logps=e['rollout_logps'][:prefix_tokens] + b['rollout_logps'],
                        step_stop_logp=e['step_stop_logp'][:t] + b['step_stop_logp'],
                        response_tokens=prefix_tokens + b['response_tokens'],
                        _c_logpi=[0.0] * t + [c] * b['num_steps'], _c_sigma=[0.0] * (t + b['num_steps']))
                    if c != 0.0:
                        items.append((full, w))
                        stats['local_nonzero'] += 1
                    stats['branches'] += 1
                    stats['branch_success'] += int(b['success'])
                    stats['branch_same_first'] += int(b['same_first_action'])
                    stats['branch_decisions'] += b['num_steps']
                    stats['replay_decisions'] += t
                    stats['branch_minus_main_at_fork'] += b['reward'] - e['reward']
                    stats['branch_c_abs'] += abs(c)
            stats['main_success'] += int(e['success'])
    return items, stats


def fork_metrics(gathered, threshold):
    if not gathered[0].get('fork'):
        return {}
    total = lambda k: sum(g['fork'][k] for g in gathered)
    branches = max(1, total('branches'))
    return dict(fork_min_entropy=threshold, forks_per_main=total('forks') / max(1, total('mains')),
                branches=total('branches'), branch_success_rate=total('branch_success') / branches,
                branch_same_first_action=total('branch_same_first') / branches,
                branch_nonzero_fraction=total('local_nonzero') / branches,
                branch_minus_main_reward=total('branch_minus_main_at_fork') / branches,
                branch_env_steps=total('branch_decisions'), replay_env_steps=total('replay_decisions'),
                branch_coefficient_abs=sum(g['fork'].get('branch_c_abs', 0.0) for g in gathered) / branches,
                fork_scale=gathered[0]['fork'].get('scale', 0.0))


def hico_metrics(gathered, critic):
    if not gathered[0]['hico']:
        return {}
    h = [g['hico'] for g in gathered]
    total = lambda k: sum(x[k] for x in h)
    near = max(1, total('near_goal_decisions'))
    decisions = max(1, total('decisions_local'))
    trained = max(1, sum(g['stats'].get('decisions', 0) for g in gathered))
    return dict(
        estimator_scale=h[0]['scale'], critic_used=h[0]['critic_used'],
        critic_diagnostics=critic.diagnostics if critic is not None else None,
        near_goal_decisions=total('near_goal_decisions'),
        near_goal_nonstop_fraction=total('near_goal_nonstop') / near,
        near_goal_sigma_mean=total('near_goal_sigma') / near,
        near_goal_c_sigma_mean=total('near_goal_c_sigma') / near,
        c_sigma_abs_mean=total('c_sigma_abs') / decisions,
        c_logpi_abs_mean=total('c_logpi_abs') / decisions,
        baseline_rmse=(total('baseline_resid_sq') / decisions) ** 0.5,
        sigma_mismatch_mean=sum(g['stats'].get('sigma_mismatch_abs', 0.0) for g in gathered) / trained,
        sigma_mismatch_max=max(g['stats'].get('sigma_mismatch_max', 0.0) for g in gathered))


def hico_prepare(groups, step, cfg, critic, n_star, world, device):
    """HiCo-PO coefficients for this rank's groups (rl/hico.py). Returns trajectories, items, stats."""
    a = cfg['algorithm']
    estimator = a['estimator']
    r_max = cfg['reward']['success_reward_base']
    trajectories = [e for group in groups for e in group]
    all_rewards = [r for g in gather_objects([e['reward'] for e in trajectories]) for r in g]
    scale = batch_scale(all_rewards, floor=a['scale_floor'] * r_max)
    use_critic = estimator == 'critic_stop' and critic is not None and critic.model is not None
    stats = dict(scale=scale, critic_used=bool(use_critic), near_goal_decisions=0, near_goal_nonstop=0,
                 near_goal_sigma=0.0, near_goal_c_sigma=0.0, c_sigma_abs=0.0, c_logpi_abs=0.0,
                 decisions_local=0, baseline_resid_sq=0.0)
    for group in groups:
        rloo = rloo_baselines([e['reward'] for e in group])
        for e, b_rloo in zip(group, rloo):
            e['N_star'] = n_star[e['episode_id']]
            q = stop_values(e['step_d'], r_max)
            baseline = critic.predict(episode_features(e)) if use_critic else np.full(len(q), b_rloo)
            c_logpi, c_sigma = coefficients(e['actions'], e['reward'], q, baseline, scale,
                                            analytic_stop=estimator in ('rloo_stop', 'critic_stop'))
            e['_c_logpi'], e['_c_sigma'] = c_logpi.tolist(), c_sigma.tolist()
            e['advantage'] = float((e['reward'] - float(np.mean(baseline))) / scale)
            near = np.asarray(e['step_d']) < 3.0
            sigma = np.exp(np.asarray(e['step_stop_logp'], dtype=np.float64))
            nonstop = np.asarray([act != 3 for act in e['actions']])
            stats['near_goal_decisions'] += int(near.sum())
            stats['near_goal_nonstop'] += int((near & nonstop).sum())
            stats['near_goal_sigma'] += float(sigma[near].sum())
            stats['near_goal_c_sigma'] += float(c_sigma[near].sum())
            stats['c_sigma_abs'] += float(np.abs(c_sigma).sum())
            stats['c_logpi_abs'] += float(np.abs(c_logpi).sum())
            stats['decisions_local'] += len(q)
            stats['baseline_resid_sq'] += float(((e['reward'] - baseline) ** 2).sum())
    n_global = int(all_reduce([stats['decisions_local']], device=device)[0])
    weight = world / n_global   # per-decision normalization over the whole RL step (SFT convention)
    items = [(e, weight) for e in trajectories if any(e['_c_logpi']) or any(e['_c_sigma'])]
    return trajectories, items, stats


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
    timeout = datetime.timedelta(hours=4)
    # DeepSpeed creates its data-parallel groups with dist.new_group, whose NCCL timeout is 10
    # minutes unless given; a rank slowed by the shared host must not abort the collective.
    _new_group = dist.new_group
    dist.new_group = lambda *a, **kw: _new_group(*a, **{**kw, 'timeout': kw.get('timeout') or timeout})
    deepspeed.init_distributed(dist_backend='nccl', timeout=timeout)
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
    estimator = a.get('estimator', 'grpo')
    if estimator not in ('grpo', 'grpo_stop', 'grpo_fork', 'rloo', 'rloo_stop', 'critic_stop'):
        raise ValueError(f'Unknown estimator {estimator}')
    if estimator not in ('grpo', 'grpo_stop', 'grpo_fork') and cfg['rollout']['dynamic_sampling']['enabled']:
        raise ValueError('HiCo-PO estimators require unfiltered on-policy rollouts (no dynamic sampling)')
    n_star = {item['episode_id']: item['N_star'] for item in pool}
    critic = None
    if estimator == 'critic_stop':
        c = a['critic']
        critic = GeometricBaseline(history=c['history_steps'], hidden=c['hidden'], iterations=c['iterations'],
                                   lr=c['lr'], r_max=cfg['reward']['success_reward_base'],
                                   seed=cfg['run']['seed'], min_rows=c['min_rows'])

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
        if critic is not None:
            critic.load(torch.load(out / 'resumable' / rl_state['tag'] / 'critic.pt', weights_only=False))
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
    backend = cfg['rollout'].get('backend', 'threads')
    workers, helpers = [], None
    if backend == 'processes':
        from qwen_vl.rl.rollout_pool import RolloutPool
        if 'expandable_segments:True' in os.environ.get('PYTORCH_ALLOC_CONF', '') + os.environ.get(
                'PYTORCH_CUDA_ALLOC_CONF', ''):
            raise RuntimeError('CUDA IPC with expandable segments needs ptrace rights; unset PYTORCH_ALLOC_CONF')
        helpers = RolloutPool(model, json.loads(base_navigation), cfg['base']['model_path'], dict(
            python=env_root['EVAL_PYTHON'], habitat_config=str(REPO / cfg['env']['habitat_config']),
            data_path=env_root['R2R_DATA'], scenes_dir=env_root['MP3D_SCENES'], gpu=local_rank,
            episode_list=str(episode_list), split=cfg['env']['split'],
            pythonpath=f"{REPO / 'src'}:{env_root.get('SIMULATOR_SOURCE', '')}",
            log_path=str(out / f'env_rank{rank}.log')), local_rank, cfg['rollout']['helpers_per_rank'],
            hold_bytes=cfg['runtime'].get('hold_helper_reserve_gib', 3) * 2 ** 30 if cfg['runtime'].get('hold_gpu_gib') else 0)
        log('rollout helpers ready', cfg['rollout']['helpers_per_rank'])
        # Every tensor the helpers map now lives in ordinary cudaMalloc segments, which CUDA IPC
        # shares without ptrace rights. All later (training) allocations may use expandable
        # segments, which the learner needs to avoid fragmentation on the shared GPU.
        torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    else:
        workers = [RolloutWorker(model, serializer, env_factory, device)
                   for _ in range(cfg['rollout'].get('workers_per_rank', 1))]
        log('environments ready', len(workers), workers[0].env.info)
    hold = cfg['runtime'].get('hold_gpu_gib')
    if hold:
        # Systems only: keep this rank's peak GPU memory reserved on the shared GPU (see placement.HOLD).
        start_hold(device, hold[local_rank] * 2 ** 30, cfg['runtime'].get('hold_helper_reserve_gib', 3) * 2 ** 30,
                   log, cfg['runtime'].get('hold_wait_minutes', 20) * 60)
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
        fork, threshold = None, None
        if estimator == 'grpo_fork':
            f = a['fork']
            threshold = fork_threshold(out, step, f)
            fork = dict(max_forks=f['max_forks'], branches=f['branches'], min_entropy=threshold,
                        window=f['window'])
        if helpers is not None:
            torch.cuda.synchronize()  # the last optimizer all-gather must land before helpers read
            helpers.check_storage()
            groups, reroll = rollout_rank_pool(helpers, local, step, cfg, fork)
        else:
            groups, reroll = rollout_rank(workers, local, step, cfg, fork)
        torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - tick
        rollout_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        release_cache()

        hico_stats = {}
        stop_token = serializer.action_ids[3][0]
        if critic is not None:
            # Fit only on previous RL steps (rank 0), so b(z) is independent of this step's actions.
            fitted = [critic.fit(step) if rank == 0 else None]
            payload = [critic.model if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            dist.broadcast_object_list(fitted, src=0)
            critic.model, critic.diagnostics = payload[0], fitted[0]
        fork_stats = {}
        if estimator in ('grpo', 'grpo_stop', 'grpo_fork'):
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
            if estimator == 'grpo_stop':
                for group in groups:
                    for e in group:
                        c_logpi, c_sigma = grpo_stop_coefficients(
                            e['actions'], e['reward'], stop_values(e['step_d'], cfg['reward']['success_reward_base']),
                            [x['reward'] for x in group], a['epsilon'],
                            a['scale_floor'] * cfg['reward']['success_reward_base'])
                        e['_c_logpi'], e['_c_sigma'] = c_logpi.tolist(), c_sigma.tolist()
                items = [(e, w) for e, w in zip(trajectories, weights) if any(e['_c_logpi']) or any(e['_c_sigma'])]
            elif estimator == 'grpo_fork':
                step_scale = None
                if a.get('fork_scale', 'group') == 'batch':
                    all_rewards = [r for g in gather_objects([e['reward'] for e in trajectories]) for r in g]
                    step_scale = batch_scale(all_rewards, floor=a['scale_floor'] * cfg['reward']['success_reward_base'])
                items, fork_stats = fork_prepare(groups, {e['uid']: w for e, w in zip(trajectories, weights)}, cfg,
                                                 step_scale)
                fork_stats['scale'] = step_scale or 0.0
            else:
                items = [(e, w) for e, w in zip(trajectories, weights) if e['advantage'] != 0.0]
        else:
            trajectories, items, hico_stats = hico_prepare(groups, step, cfg, critic, n_star, world, device)
        placement = dict(moved=0, free_gib=[])
        update_wait = 0
        if cfg['runtime'].get('update_min_free_gib'):
            update_wait = wait_for_free(device, cfg['runtime']['update_min_free_gib'] * 2 ** 30, log)
        if world > 1:
            items, placement = redistribute(items, device, cfg['runtime'].get('placement_margin_gib', 0) * 2 ** 30)
        placement['update_wait'] = update_wait
        n_max = int(all_reduce([len(items)], dist.ReduceOp.MAX, device)[0])
        steps_needed = max(n_max, 1)

        # ---- one optimizer update ----
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        stats = dict(pg_loss=0.0, clipfrac=0.0, tokens=0, mismatch_abs=0.0, mismatch_max=0.0,
                     train_entropy=0.0, forward_tokens=0, offloaded=0)
        for i in range(steps_needed):
            engine.set_gradient_accumulation_boundary(i == steps_needed - 1)
            if i < len(items):
                e, w = items[i]
                HOST_OFFLOAD['on'] = should_offload(e, i == 0, device)
                stats['offloaded'] += HOST_OFFLOAD['on']
                batch = to_device(encode_rollout(serializer, e['instruction'], e['_frames'],
                                                 e['_responses']), device)
                if estimator == 'grpo':
                    logp, entropy = response_log_probs(model, batch, T, with_entropy=True)
                else:
                    logp, entropy, stop_logp = response_log_probs(model, batch, T, with_entropy=True,
                                                                  probe_token=stop_token)
                rollout_logp = torch.tensor(e['rollout_logps'], device=device)
                if logp.shape != rollout_logp.shape:
                    raise RuntimeError('Trainer and rollout response tokens differ')
                if estimator == 'grpo':
                    adv = torch.full_like(logp, e['advantage'])
                    token_loss, clipped = policy_loss_terms(logp, logp.detach(), adv, a['clip_ratio_low'],
                                                            a['clip_ratio_high'], a['clip_ratio_c'])
                    loss = token_loss.sum() * w
                else:
                    logpi, sigma = action_terms(logp, stop_logp, batch['response_action_index'],
                                                batch['num_actions'])
                    c_logpi = torch.tensor(e['_c_logpi'], device=device, dtype=logpi.dtype)
                    c_sigma = torch.tensor(e['_c_sigma'], device=device, dtype=sigma.dtype)
                    loss = -(c_logpi * logpi + c_sigma * sigma).sum() * w
                    clipped = torch.zeros(1, device=device)
                    rollout_sigma = torch.tensor([math.exp(v) for v in e['step_stop_logp']], device=device)
                    gap = (sigma.detach() - rollout_sigma).abs()
                    stats['sigma_mismatch_abs'] = stats.get('sigma_mismatch_abs', 0.0) + float(gap.sum())
                    stats['sigma_mismatch_max'] = max(stats.get('sigma_mismatch_max', 0.0), float(gap.max()))
                    stats['decisions'] = stats.get('decisions', 0) + int(sigma.numel())
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
            HOST_OFFLOAD['on'] = False
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
        release_cache()

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
            stats=stats, items=len(items), placement=placement, hico=hico_stats, fork=fork_stats,
            # all sampled attempts, including groups later discarded by dynamic sampling
            sampled_branch_steps=[sum(x['branch_steps'][j] for h in reroll for grp in h for x in grp)
                                  for j in (0, 1)],
            rollout_seconds=rollout_seconds, train_seconds=train_seconds,
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
                trained_per_rank=[g['items'] for g in gathered],
                moved_trajectories=gathered[0]['placement']['moved'],
                free_gib_before_update=gathered[0]['placement']['free_gib'],
                update_wait_seconds=gathered[0]['placement'].get('update_wait', 0),
                offloaded_trajectories=sum(g['stats']['offloaded'] for g in gathered),
                **hico_metrics(gathered, critic),
                **fork_metrics(gathered, threshold),
                sampled_branch_env_steps=sum(g['sampled_branch_steps'][0] for g in gathered),
                sampled_replay_env_steps=sum(g['sampled_branch_steps'][1] for g in gathered),
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
        if critic is not None:
            rows = gather_objects([(episode_features(e), np.full(len(e['step_d']), e['reward']))
                                   for e in trajectories])
            z = np.concatenate([r[0] for g in rows for r in g])
            y = np.concatenate([r[1] for g in rows for r in g])
            critic.add(step, z, y)
        del groups, trajectories, items
        release_host_memory()
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
            if critic is not None and rank == 0:
                torch.save(critic.state(), out / 'resumable' / tag / 'critic.pt')
            dist.barrier()
            if rank == 0:
                state_file.write_text(json.dumps(dict(next_step=done, tag=tag, total_steps=total_steps)) + '\n')
                kept = sorted((out / 'resumable').glob('step_*'))
                for old in kept[:-cfg['run']['keep_resumable']]:
                    shutil.rmtree(old)
            dist.barrier()
    for worker in workers:
        worker.close()
    if helpers is not None:
        helpers.close()
    if rank == 0:
        (out / 'COMPLETE' if not smoke and engine.global_steps == total_steps else out / 'STOPPED').write_text(
            json.dumps(dict(global_steps=engine.global_steps, total_steps=total_steps,
                            finished=datetime.datetime.now().isoformat())) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
