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


def run_episode(session, env, item, *, temperature, budget, seed, sample_idx, reward_cfg,
                device, keep_frames=True):
    """One sampled episode: format error or budget exhaustion ends it with reward 0."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.perf_counter()
    obs = env.reset(item['episode_id'])
    scene_id, instruction = obs['scene_id'], obs['instruction']
    uid = f"rl:train:{obs['scene_id']}:{obs['episode_id']}:{sample_idx}"
    session.reset(uid, instruction)
    frames, responses, distances = [], [], [obs['distance_to_goal']]
    positions, bearings = [obs['position']], [obs.get('bearing_error')]
    model_seconds = env_seconds = 0.0
    outcome = dict(stopped=False, truncated=False, format_error=False)
    metrics = {}
    for t in range(budget):
        tick = time.perf_counter()
        response = session.sample_step(t, obs['rgb'], temperature, generator)
        model_seconds += time.perf_counter() - tick
        if keep_frames:
            frames.append(obs['rgb'])
        responses.append(response)
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
    finite = [d for d in distances if d is not None and math.isfinite(d)]
    if len(finite) != len(distances):
        raise EnvironmentError_(f'Non-finite distance in episode {item["episode_id"]}')
    final_distance = finite[-1]
    success = bool(outcome['stopped'] and metrics.get('success', 0.0) == 1.0)
    if outcome['stopped'] and success != (final_distance < 3.0):
        raise RuntimeError('Habitat success disagrees with strict d < 3 at STOP')
    reward = episode_reward(success=success, distance=final_distance, **reward_cfg)
    return dict(
        episode_id=item['episode_id'], scene_id=scene_id,
        sample_idx=sample_idx, seed=seed, uid=uid, instruction=instruction,
        budget=budget, num_steps=len(responses), reward=reward, success=success,
        oracle_success=bool(min(finite) < 3.0), final_distance=final_distance,
        min_distance=min(finite), d0=finite[0], spl=float(metrics.get('spl', 0.0)) if success else 0.0,
        actions=[r['class_id'] for r in responses], response_tokens=sum(len(r['tokens']) for r in responses),
        mean_entropy=sum(e for r in responses for e in r['entropies'])
        / max(1, sum(len(r['entropies']) for r in responses)),
        greedy_agreement=sum(g == s for r in responses for g, s in zip(r['greedy_tokens'], r['tokens']))
        / max(1, sum(len(r['tokens']) for r in responses)),
        rollout_logps=[lp for r in responses for lp in r['logps']],
        model_seconds=model_seconds, env_seconds=env_seconds,
        wall_seconds=time.perf_counter() - started, **outcome,
        # Per-decision geometry at the pose where decision t was made (training-time only).
        step_d=finite[:len(responses)], step_bearing=bearings[:len(responses)],
        step_path=[float(x) for x in np.concatenate(([0.0], np.cumsum(np.linalg.norm(
            np.diff(np.asarray(positions, dtype=np.float64), axis=0), axis=1))))[:len(responses)]],
        step_stop_logp=[r.get('stop_logp') for r in responses],
        _frames=frames, _responses=[dict(tokens=r['tokens'], complete=r['complete']) for r in responses])
