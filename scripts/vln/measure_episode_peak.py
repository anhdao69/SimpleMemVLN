"""Single-GPU forward+backward peak on one real complete episode (no optimizer).

Diagnostic for GPU headroom when the full multi-rank gate cannot run in the current
allocation. Uses the production model path (checkpointed decoder, activation offload
backend, episodic attention). Full BF16 parameter gradients are materialized; a ZeRO-2
rank instead holds a gradient partition plus its optimizer partition, so the multi-rank
peak must be re-measured by the real gate. Not a qualification.
"""
import argparse
import json
from pathlib import Path
import time
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--arms', nargs='+', default=['v0', 'm2'])
    p.add_argument('--runtime', default='configs/episodic/profile_h4.yaml')
    p.add_argument('--out', required=True)
    p.add_argument('--science', default='configs/episodic/science_common.yaml')
    p.add_argument('--batch-steps', type=int, nargs='+', help='Override window_attention_batch_steps (engineering sweep)')
    args = p.parse_args()
    from qwen_vl.train.vln_runtime import load_model, resolve_config
    from qwen_vl.data.episode_dataset import EpisodeDataset
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.models.activation_offload import arena_bytes_for, install_activation_arena
    from qwen_vl.research.episodic_contract import parse_episodic_spec
    report = []
    cfg = resolve_config(args.science, f'configs/episodic/arm_{args.arms[0]}.yaml', runtime=args.runtime)
    model, serializer = load_model(cfg, args.model_path)
    text = model.config.text_config
    for arm, batch in [(a, b) for a in args.arms for b in (args.batch_steps or [None])]:
        cfg = resolve_config(args.science, f'configs/episodic/arm_{arm}.yaml', runtime=args.runtime)
        if batch is not None:
            cfg['runtime']['window_attention_batch_steps'] = batch
        model.navigation_config = cfg
        model.episodic_spec = parse_episodic_spec(cfg, text)
        serializer = EpisodeSerializer(serializer.processor, cfg)
        dataset = EpisodeDataset(args.manifest, serializer)
        longest = max(dataset.encoded_lengths)
        install_activation_arena(arena_bytes_for(longest, text.hidden_size, text.num_hidden_layers))
        model.train()
        model.gradient_checkpointing_enable({'use_reentrant': False})
        sample = dataset[0]
        data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in sample.items()}
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); start = time.perf_counter()
        out = model(**data)
        forward = time.perf_counter() - start
        (out['loss_sum'] / data['num_actions']).backward()
        torch.cuda.synchronize()
        record = dict(arm=arm, batch_steps=cfg['runtime']['window_attention_batch_steps'], episode=sample['episode_uid'], tokens=int(data['input_ids'].numel()),
                      loss_sum=float(out['loss_sum']), forward_seconds=forward,
                      total_seconds=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      parameter_gib=sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30,
                      gradient_gib=sum(p.grad.numel() * p.grad.element_size() for p in model.parameters()
                                       if p.grad is not None) / 2**30)
        print(json.dumps(record), flush=True)
        report.append(record)
        del out, data
        model.zero_grad(set_to_none=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
