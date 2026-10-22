"""Gate: an IPC rollout helper samples exactly what the learner process samples, and sees
in-place weight updates. Usage (one GPU): python scripts/rl/check_rollout_pool.py GPU OUTDIR"""
import json
import os
import sys
from pathlib import Path

import torch


def main():
    gpu, out = int(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(gpu)
    from qwen_vl.rl.env_client import EnvClient
    from qwen_vl.rl.rollout import run_episode
    from qwen_vl.rl.rollout_pool import RolloutPool
    from qwen_vl.rl.sampling import SamplingSession
    from qwen_vl.rl.train_grpo import load_base_mmap

    base = Path('/data/anh/checkpoints/simplemem/joint-window8-text-b/epoch-2')
    model_path = '/data/anh/checkpoints/simplemem/qwen-base'
    model, serializer = load_base_mmap(base, model_path)
    pool_items = [json.loads(l) for l in open('data/rl/activevln_r2r4k.jsonl')][:2]
    ids = out / 'ids.json'
    ids.write_text(json.dumps([i['episode_id'] for i in pool_items]))
    env_kwargs = dict(python=os.environ['EVAL_PYTHON'], habitat_config='docs/evaluation_setup/vln_r2r.yaml',
                      data_path=os.environ['R2R_DATA'], scenes_dir=os.environ['MP3D_SCENES'], gpu=gpu,
                      episode_list=str(ids), pythonpath=f"{Path('src').resolve()}:{os.environ.get('SIMULATOR_SOURCE', '')}",
                      log_path=str(out / 'env.log'))
    metadata = json.loads((base / 'navigation.json').read_text())
    pool = RolloutPool(model, metadata, model_path, env_kwargs, gpu, size=1)
    env = EnvClient(**env_kwargs)
    session = SamplingSession(model, serializer)
    kwargs = dict(temperature=1.2, budget=40, seed=1234, sample_idx=0, reward_cfg={})
    item = pool_items[0]

    def compare(tag):
        local = run_episode(session, env, item, device=torch.device('cuda', gpu), **kwargs)
        pool.submit(tag, item, **kwargs)
        _, remote = pool.get()
        same = (local['actions'] == remote['actions'] and local['rollout_logps'] == remote['rollout_logps'])
        print(tag, 'steps', local['num_steps'], remote['num_steps'], 'identical', same, flush=True)
        return same, local

    same1, first = compare('before')
    with torch.no_grad():  # in-place learner update, as ZeRO-2 does
        for name, p in model.named_parameters():
            if name.endswith('layers.10.mlp.down_proj.weight'):
                p.mul_(1.5)
    torch.cuda.synchronize()
    same2, second = compare('after')
    changed = first['rollout_logps'] != second['rollout_logps']
    print('helper follows in-place update', same2, 'policy changed', changed, flush=True)
    pool.check_storage()
    pool.close()
    env.close()
    assert same1 and same2 and changed


if __name__ == '__main__':
    main()
