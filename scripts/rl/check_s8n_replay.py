"""GPU gate for S8N RL rollouts (one GPU, real Habitat): realised-memory replay vs offline re-scoring.

For a few sampled training episodes (T = 1.2, one entropy fork with one branch), the trainer's
full-episode BF16 forward is run twice: with the rollout's realised per-layer frames injected
(``rl/sweep_memory.injected_layer_plans``, what training uses) and with S8N drift re-scored offline from
the training activations. Reports per-token |log pi_train - log pi_rollout| for both, and the
fraction of (layer, group) plans where re-scoring selects different frames than the rollout used.
Branch trajectories (main prefix + restored-snapshot suffix) are scored the same way, so a wrong
snapshot of the S8N memory shows up as a branch mismatch (exactness itself: tests/rl/test_s8n_rollout.py).

Usage: python scripts/rl/check_s8n_replay.py CHECKPOINT MODEL_PATH GPU OUTDIR [EPISODES]
Pass: injected mismatch mean < 0.02 (check_fork_smoke threshold) and finite outputs.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    checkpoint, model_path, gpu, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    count = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)
    from qwen_vl.rl.env_client import EnvClient
    from qwen_vl.rl.episode import encode_rollout, response_log_probs
    from qwen_vl.rl.rollout import event_score, run_episode
    from qwen_vl.rl.sampling import SamplingSession
    from qwen_vl.rl.train_grpo import load_base_mmap, to_device

    model, serializer = load_base_mmap(checkpoint, model_path)
    model = model.to(device).eval()
    spec = model.episodic_spec
    print('spec', spec.version, spec.score, 'R', spec.recent_groups, 'B', spec.budget_tokens, flush=True)
    items = [json.loads(line) for line in open('data/rl/activevln_r2r4k.jsonl')][:count]
    ids = out / 'ids.json'
    ids.write_text(json.dumps([i['episode_id'] for i in items]))
    env = EnvClient(python=os.environ['EVAL_PYTHON'], habitat_config='docs/evaluation_setup/vln_r2r.yaml',
                    data_path=os.environ['R2R_DATA'], scenes_dir=os.environ['MP3D_SCENES'], gpu=gpu,
                    episode_list=str(ids), pythonpath=f"{Path('src').resolve()}:{os.environ.get('SIMULATOR_SOURCE', '')}",
                    log_path=str(out / 'env.log'))
    session = SamplingSession(model, serializer)
    fork = dict(max_forks=1, branches=1, min_entropy=0.0, window=8)
    report = []
    for k, item in enumerate(items):
        e = run_episode(session, env, item, temperature=1.2, budget=80, seed=1000 + k, sample_idx=0,
                        reward_cfg=dict(success_reward_base=15.0), device=device, fork=fork)
        trajectories = [('main', e, e['_responses'], e['_frames'], e['rollout_logps'], e['memory'])]
        for b in e['_branches']:
            t = b['fork_t']
            prefix = sum(len(r['tokens']) for r in e['_responses'][:t])
            trajectories.append((f"branch@{t}", b, e['_responses'][:t] + b['_responses'], e['_frames'][:t] + b['_frames'],
                                 e['rollout_logps'][:prefix] + b['rollout_logps'], b['memory']))
        for name, record, responses, frames, rollout_logps, memory in trajectories:
            row = dict(episode=item['episode_id'], trajectory=name, steps=len(responses),
                       reward=record['reward'], forks=e.get('fork_points'),
                       event_scores=[event_score(d) for d in record['step_drift']][:5])
            target = torch.tensor(rollout_logps, device=device)
            with torch.no_grad():
                for mode in ('injected', 'rescored'):
                    batch = to_device(encode_rollout(serializer, e['instruction'], frames, responses, episode_uid=e['uid'],
                                                     layer_banks=memory['layer_banks'] if mode == 'injected' else None), device)
                    logp, _ = response_log_probs(model, batch, 1.2)
                    if logp.shape != target.shape or not torch.isfinite(logp).all():
                        raise RuntimeError(f'{name}: bad trainer log-probs')
                    diff = (logp.float() - target).abs()
                    row[f'{mode}_mismatch_mean'] = float(diff.mean())
                    row[f'{mode}_mismatch_max'] = float(diff.max())
                    if mode == 'injected':
                        injected = batch['episodic_plans']
                    else:
                        diagnostics = {}
                        model.hidden(batch['input_ids'], batch['mm_token_type_ids'], batch['pixel_values'],
                                     batch['image_grid_thw'], batch['step_plan'], episodic_layout=batch['episodic_layout'],
                                     episode_uid=batch['episode_uid'], episodic_diagnostics=diagnostics)
                        rescored = diagnostics['plans']
            layout_groups = len(responses)
            differ = total = 0
            for layer in spec.attention_layers:
                a, b_ = injected[layer], rescored[layer]
                for g in range(layout_groups):
                    ia = a.token_ids[a.group_offsets[g]:a.group_offsets[g + 1]]
                    ib = b_.token_ids[b_.group_offsets[g]:b_.group_offsets[g + 1]]
                    total += 1
                    differ += not np.array_equal(ia, ib)
            row['rescored_plan_disagreement'] = differ / max(1, total)
            print(json.dumps(row), flush=True)
            report.append(row)
    env.close()
    injected = [r['injected_mismatch_mean'] for r in report]
    summary = dict(trajectories=len(report), injected_mismatch_mean=float(np.mean(injected)),
                   rescored_mismatch_mean=float(np.mean([r['rescored_mismatch_mean'] for r in report])),
                   rescored_plan_disagreement=float(np.mean([r['rescored_plan_disagreement'] for r in report])),
                   gpu=torch.cuda.get_device_name(device), passed=bool(max(injected) < 0.02))
    (out / 'replay_gate.json').write_text(json.dumps(dict(summary=summary, rows=report), indent=2) + '\n')
    print('SUMMARY', json.dumps(summary), flush=True)
    assert summary['passed'], 'injected-plan trainer/rollout mismatch above 0.02'


if __name__ == '__main__':
    main()
