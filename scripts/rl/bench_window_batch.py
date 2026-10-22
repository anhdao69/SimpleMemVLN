"""Training-forward speed and log-prob agreement: serial vs batched Window8 attention.
python scripts/rl/bench_window_batch.py GPU STEPS BATCHES (e.g. 1 80 1,8,16)"""
import copy
import sys
import time

import numpy as np
import torch


def main():
    gpu, steps, batches = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
    torch.cuda.set_device(gpu)
    from pathlib import Path
    from qwen_vl.rl.episode import encode_rollout, response_log_probs
    from qwen_vl.rl.train_grpo import enable_checkpointing, load_base_mmap

    model, serializer = load_base_mmap(Path('/data/anh/checkpoints/simplemem/joint-window8-text-b/epoch-2'),
                                       '/data/anh/checkpoints/simplemem/qwen-base')
    enable_checkpointing(model, 16384)
    model.train()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(steps)]
    acts = rng.integers(0, 3, steps - 1).tolist() + [3]
    batch = encode_rollout(serializer, 'Walk forward and stop at the door.', frames,
                           [dict(tokens=serializer.action_ids[a], complete=True) for a in acts])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    print('tokens', batch['input_ids'].shape[1], flush=True)
    base_cfg = model.navigation_config
    reference = None
    for b in batches:
        model.navigation_config = copy.deepcopy(base_cfg)
        model.navigation_config['runtime']['window_attention_batch_steps'] = b
        times = []
        for rep in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t = time.perf_counter()
            logp, _ = response_log_probs(model, batch, 1.2)
            logp.sum().backward()
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        grad = model.backbone.model.language_model.layers[3].self_attn.q_proj.weight.grad.float().clone()
        if reference is None:
            reference = (logp.detach().float(), grad)
        dl = (logp.detach().float() - reference[0]).abs().max().item()
        dg = ((grad - reference[1]).norm() / reference[1].norm()).item()
        print(f'batch_steps={b} fwd+bwd s={[round(x,2) for x in times]} max|dlogp|={dl:.2e} rel|dgrad|={dg:.2e} '
              f'peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB', flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    main()
