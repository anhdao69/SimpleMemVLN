"""Closed-loop rollouts of the sampling policy in the Habitat RL environment."""
import hashlib
import math
import time

import numpy as np
import torch

from qwen_vl.contracts import HABITAT_IDS
from qwen_vl.rl.algos import episode_reward
from qwen_vl.rl.env_client import EnvironmentError_


def rollout_seed(*parts):
    digest = hashlib.sha256(':'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def _advance(session, env, obs, t0, budget, temperature, generator, keep_frames, on_decision=None,
             before_memory=None):
    """Sample decisions t0..budget-1 from the current session/environment state until the episode ends."""
    frames, responses, distances = [], [], [obs['distance_to_goal']]
    positions, bearings = [obs['position']], [obs.get('bearing_error')]
    model_seconds = env_seconds = 0.0
    outcome = dict(stopped=False, truncated=False, format_error=False)
    metrics = {}
    for t in range(t0, budget):
        if before_memory is not None:          # fork placement: memory -- snapshot before m_t
            before_memory(t, obs)
        # Promotion and the MemFork memory decision m_t; with action forks, before their snapshot.
        session.prepare_step(t, generator)
        if on_decision is not None:
            on_decision(t, obs)
        tick = time.perf_counter()
        response = session.sample_step(t, obs['rgb'], temperature, generator)
        model_seconds += time.perf_counter() - tick
        if keep_frames:
            frames.append(obs['rgb'])
        responses.append(response)
        if on_decision is not None:
            on_decision(t, obs, response)
        if not response['valid']:
            outcome['format_error'] = True
            outcome['error'] = response.get('error')
            break
        tick = time.perf_counter()
        obs = env.step(HABITAT_IDS[response['class_id']])
        env_seconds += time.perf_counter() - tick
        distances.append(obs['distance_to_goal'])
        positions.append(obs['position'])
        bearings.append(obs.get('bearing_error'))
        if response['class_id'] == 3:
            outcome['stopped'] = True
            metrics = obs['metrics']
            break
        if obs['episode_over']:
            # Habitat's own 500-step cap; unreachable with budget <= 500.
            metrics = obs['metrics']
            outcome['truncated'] = True
            break
    else:
        outcome['truncated'] = True
    return dict(frames=frames, responses=responses, distances=distances, positions=positions,
                bearings=bearings, outcome=outcome, metrics=metrics, model_seconds=model_seconds,
                env_seconds=env_seconds)


def _summary(run, item, reward_cfg, start_distance=None):
    """Reward and per-decision geometry of one (possibly forked) segment."""
    distances = run['distances']
    finite = [d for d in distances if d is not None and math.isfinite(d)]
    if len(finite) != len(distances):
        raise EnvironmentError_(f'Non-finite distance in episode {item["episode_id"]}')
    final_distance = finite[-1]
    outcome, metrics, responses = run['outcome'], run['metrics'], run['responses']
    success = bool(outcome['stopped'] and metrics.get('success', 0.0) == 1.0)
    if outcome['stopped'] and success != (final_distance < 3.0):
        raise RuntimeError('Habitat success disagrees with strict d < 3 at STOP')
    reward = episode_reward(success=success, distance=final_distance, **reward_cfg)
    return dict(
        num_steps=len(responses), reward=reward, success=success, final_distance=final_distance,
        min_distance=min(finite), spl=float(metrics.get('spl', 0.0)) if success else 0.0,
        actions=[r['class_id'] for r in responses], response_tokens=sum(len(r['tokens']) for r in responses),
        rollout_logps=[lp for r in responses for lp in r['logps']],
        step_d=finite[:len(responses)], step_bearing=run['bearings'][:len(responses)],
        step_stop_logp=[r.get('stop_logp') for r in responses],
        step_entropy=[r['entropies'][0] if r['entropies'] else None for r in responses],
        _frames=run['frames'], _responses=[dict(tokens=r['tokens'], complete=r['complete']) for r in responses],
        **outcome)


def replay(env, episode_id, actions):
    """Deterministic simulator replay of a decision prefix; returns the observation after it."""
    obs = env.reset(episode_id)
    for action in actions:
        obs = env.step(HABITAT_IDS[action])
        if obs['episode_over']:
            raise RuntimeError('Replay prefix ended the episode')
    return obs


def memory_record(session):
    """MemFork bank record (decisions + realised banks per step) of the session's episode, or None."""
    state = getattr(session, 'episodic', None)
    if state is None or state.bank is None:
        return None
    return state.bank.export()


def run_episode(session, env, item, *, temperature, budget, seed, sample_idx, reward_cfg,
                device, keep_frames=True, fork=None, retention=None):
    """One sampled episode: format error or budget exhaustion ends it with reward 0.

    ``fork`` (estimator ``grpo_fork``): dict(max_forks, branches, min_entropy, window[, horizon]);
    the horizon defaults to the item's reference length N*.
    Before the episode, ``max_forks`` target decisions are drawn uniformly from [1, horizon) with an
    independent generator (the main trajectory's draws are unchanged). Fork k is placed at the first
    decision t in [target_k, target_k + window] whose first-token entropy is >= ``min_entropy`` (a
    property of pi(.|h_t), not of the sampled a_t); a target without such a decision is skipped. So
    whether t is a fork point never depends on a_t or later events. The streaming memory is
    snapshotted before each candidate decision and kept only at fork points; after the main episode,
    each fork gets ``branches`` on-policy continuations from the identical history (simulator prefix
    replayed deterministically and verified).

    ``retention`` (MemFork): dict(export=RetentionPolicy.export(), mode='sample'|'greedy'). Memory
    decisions draw from the same generator as the actions (a branch re-draws them with its own), and
    each record carries ``memory`` (decisions and realised banks, rl/rollout.memory_record).
    """
    if retention is not None:
        from qwen_vl.stream.retention import RetentionControl, RetentionPolicy
        session.retention = RetentionControl(RetentionPolicy.from_export(retention['export']), retention['mode'])
    else:
        session.retention = None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.perf_counter()
    obs = env.reset(item['episode_id'])
    scene_id, instruction = obs['scene_id'], obs['instruction']
    uid = f"rl:train:{obs['scene_id']}:{obs['episode_id']}:{sample_idx}"
    session.reset(uid, instruction)
    forks, pending = [], {}
    on_decision = before_memory = None
    if fork and fork.get('placement', 'action') == 'memory':
        # MemFork memory forks: target decisions are drawn uniformly from [min_t, horizon) (min_t = the
        # first step with a memory decision); fork k is placed at the first decision t in
        # [target_k, target_k + window] whose memory decision has entropy >= min_entropy, with the
        # snapshot taken *before* m_t (a property of q(. | pool_t), known before m_t is drawn), so the
        # branches re-draw m_t and everything after it.
        coin = torch.Generator().manual_seed(rollout_seed(seed, 'fork-targets'))
        low = int(fork['min_t'])
        horizon = max(low + 1, int(fork.get('horizon') or item['N_star']))
        targets = sorted(int(torch.randint(low, horizon, (), generator=coin)) for _ in range(fork['max_forks']))
        cursor = [0]

        def before_memory(t, obs_t):
            while cursor[0] < len(targets) and t > targets[cursor[0]] + fork['window']:
                cursor[0] += 1
            if cursor[0] == len(targets) or t < targets[cursor[0]]:
                return
            entropy = session.memory_entropy(t)
            if entropy is not None and entropy >= fork['min_entropy']:
                forks.append(dict(t=t, snapshot=session.snapshot(), obs=obs_t, entropy=entropy,
                                  target=targets[cursor[0]]))
                cursor[0] += 1
                while cursor[0] < len(targets) and targets[cursor[0]] <= t:
                    targets[cursor[0]] = t + 1
    elif fork:
        coin = torch.Generator().manual_seed(rollout_seed(seed, 'fork-targets'))
        horizon = max(2, int(fork.get('horizon') or item['N_star']))
        targets = sorted(int(torch.randint(1, horizon, (), generator=coin)) for _ in range(fork['max_forks']))
        cursor = [0]

        def on_decision(t, obs_t, response=None):
            while cursor[0] < len(targets) and t > targets[cursor[0]] + fork['window']:
                cursor[0] += 1                      # window passed without an uncertain decision
            if cursor[0] == len(targets) or t < targets[cursor[0]]:
                return
            if response is None:
                pending.update(t=t, snapshot=session.snapshot(), obs=obs_t)
                return
            entropy = response['entropies'][0] if response['entropies'] else 0.0
            if pending.get('t') == t and entropy >= fork['min_entropy']:   # never uses a_t itself
                forks.append(dict(pending, entropy=entropy, target=targets[cursor[0]]))
                cursor[0] += 1
                # later targets that fall before the next free decision move with it
                while cursor[0] < len(targets) and targets[cursor[0]] <= t:
                    targets[cursor[0]] = t + 1
            pending.clear()

    main = _advance(session, env, obs, 0, budget, temperature, generator, keep_frames, on_decision,
                    before_memory)
    record = _summary(main, item, reward_cfg)
    record['memory'] = memory_record(session)
    finite = record['step_d'] + [record['final_distance']]
    model_seconds, env_seconds = main['model_seconds'], main['env_seconds']
    branches = []
    for k, f in enumerate(forks):
        t = f['t']
        for m in range(fork['branches']):
            session.restore(f['snapshot'], episode_uid=f'{uid}:fork{t}:{m}')
            tick = time.perf_counter()
            obs_t = replay(env, item['episode_id'], record['actions'][:t])
            env_seconds += time.perf_counter() - tick
            if abs(obs_t['distance_to_goal'] - record['step_d'][t]) > 1e-4 or not np.array_equal(obs_t['rgb'], f['obs']['rgb']):
                raise RuntimeError(f'Replay diverged at decision {t} of episode {item["episode_id"]}')
            branch_generator = torch.Generator(device=device)
            branch_generator.manual_seed(rollout_seed(seed, 'branch', t, m))
            run = _advance(session, env, obs_t, t, budget, temperature, branch_generator, keep_frames)
            model_seconds += run['model_seconds']
            env_seconds += run['env_seconds']
            branch = _summary(run, item, reward_cfg)
            branch['memory'] = memory_record(session)
            branch.update(fork_t=t, branch=m, fork_index=k,
                          same_first_action=bool(branch['actions'][:1] == record['actions'][t:t + 1]))
            branches.append(branch)
        f.pop('snapshot')
    del forks, pending
    record.update(
        episode_id=item['episode_id'], scene_id=scene_id, sample_idx=sample_idx, seed=seed, uid=uid,
        instruction=instruction, budget=budget, oracle_success=bool(min(finite) < 3.0), d0=finite[0],
        mean_entropy=sum(e for r in main['responses'] for e in r['entropies'])
        / max(1, sum(len(r['entropies']) for r in main['responses'])),
        greedy_agreement=sum(g == s for r in main['responses'] for g, s in zip(r['greedy_tokens'], r['tokens']))
        / max(1, sum(len(r['tokens']) for r in main['responses'])),
        model_seconds=model_seconds, env_seconds=env_seconds, wall_seconds=time.perf_counter() - started,
        # Per-decision geometry at the pose where decision t was made (training-time only).
        step_path=[float(x) for x in np.concatenate(([0.0], np.cumsum(np.linalg.norm(
            np.diff(np.asarray(main['positions'], dtype=np.float64), axis=0), axis=1))))[:record['num_steps']]])
    if fork:
        record['_branches'] = branches
        record['fork_points'] = sorted({b['fork_t'] for b in branches})
    return record
