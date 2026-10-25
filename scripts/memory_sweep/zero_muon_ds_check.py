"""Real DeepSpeed ZeRO-2 CPU-offload check of ZeRO Muon (torchrun, 4 GPUs, small BF16 model).

The model has the training wrapper's parameter kinds and names (embeddings,
LM head, norms, 1D gates, a depthwise conv and 2D hidden matrices), and the
optimizer comes from the production ``QwenSFTTrainer.create_optimizer`` with
the production ZeRO-2 offload JSON (GAS 2, clip 1.0) and a warmup LR schedule.

1. Every update, each rank records the clipped fp32 gradient partition, the
   fp32 master partition before/after and the LR of each group as DeepSpeed
   steps it; rank 0 rebuilds full tensors and steps independent references
   (``torch.optim.Muon`` on cuda:0 for Muon groups, ``torch.optim.AdamW`` for
   the rest) from the same masters/gradients.  Each step's update must agree:
   Muon relative L2 <= 1e-3 (a layout error is O(1)); AdamW <= 1e-3.
2. A DeepSpeed checkpoint after update 2 is reloaded into a fresh engine,
   updates 3-4 rerun on the same inputs, and the final masters and Muon
   momentum buffers must match the uninterrupted run (relative L2 <= 1e-6).
"""
import argparse
import json
import os
from pathlib import Path
import types

import torch
import torch.distributed as dist
from torch import nn

SPEC = dict(name='muon_zero_offload_v1', scope='language_model_layers_2d', momentum=0.95, nesterov=True,
            ns_steps=5, ns_coefficients=[3.4445, -4.775, 2.0315], eps=1.0e-07, adjust_lr_fn='match_rms_adamw',
            group_numel=30000)
LR, WD, VOCAB, WIDTH, UPDATES, GAS = 2e-3, 0.01, 97, 64, 4, 2


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        lm = nn.Module()
        lm.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        layers = []
        for _ in range(3):
            layer = nn.Module()
            layer.input_layernorm = nn.LayerNorm(WIDTH, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.up_proj = nn.Linear(WIDTH, 176, bias=False)
            layer.mlp.down_proj = nn.Linear(176, WIDTH, bias=False)
            layer.linear_attn = nn.Module()
            layer.linear_attn.in_proj_a = nn.Linear(WIDTH, 8, bias=False)
            layer.linear_attn.out_proj = nn.Linear(8, WIDTH, bias=False)
            layer.linear_attn.conv1d = nn.Conv1d(WIDTH, WIDTH, 3, groups=WIDTH, bias=False, padding=1)
            layer.linear_attn.A_log = nn.Parameter(torch.zeros(8))
            layers.append(layer)
        lm.layers = nn.ModuleList(layers)
        lm.norm = nn.LayerNorm(WIDTH, bias=False)
        self.model = nn.Module()
        self.model.language_model = lm
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)
        self.navigation_config = dict(training=dict(cpu_optimizer_impl='deepspeed_cpu_adam', matrix_optimizer=SPEC))

    def forward(self, ids):
        lm = self.model.language_model
        h = lm.embed_tokens(ids)
        for layer in lm.layers:
            x = layer.input_layernorm(h)
            h = h + layer.mlp.down_proj(nn.functional.silu(layer.mlp.up_proj(x)))
            gate = torch.sigmoid(layer.linear_attn.in_proj_a(x) - layer.linear_attn.A_log.exp())
            h = h + layer.linear_attn.out_proj(gate) + layer.linear_attn.conv1d(x.transpose(1, 2)).transpose(1, 2)
        logits = self.lm_head(lm.norm(h)).float()
        return nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)) * 40


def ds_config(path):
    config = json.loads(Path(path).read_text())
    config.update(bf16=dict(enabled=True), train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=GAS,
                  train_batch_size=GAS * dist.get_world_size(), gradient_clipping=1.0,
                  zero_allow_untested_optimizer=True, steps_per_print=10**9)
    return config


def engine(config):
    import deepspeed
    from qwen_vl.train.trainer import QwenSFTTrainer
    torch.manual_seed(1234)
    model = Tiny().to(torch.bfloat16).cuda()            # as in training: BF16 on the GPU before the optimizer
    trainer = object.__new__(QwenSFTTrainer)
    trainer.model, trainer.optimizer = model, None
    trainer.args = types.SimpleNamespace(learning_rate=LR, mm_projector_lr=LR, weight_decay=WD, adam_beta1=0.9,
                                         adam_beta2=0.95, adam_epsilon=1e-8, deepspeed=config)
    optimizer = trainer.create_optimizer()
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 3))
    wrapped, _, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=schedule,
                                                    config=config)
    return wrapped, scheduler


def batch(update, micro):
    g = torch.Generator().manual_seed(10_000 * update + 100 * micro + dist.get_rank())
    return torch.randint(0, VOCAB, (1, 48), generator=g).cuda()


def install_recorder(wrapped):
    client = wrapped.optimizer.optimizer
    records, original = [], client.step

    def step(closure=None):
        group = client.param_groups[0]
        partition = group['params'][0]
        record = dict(key=('muon', group['muon_group']) if group.get('use_muon') else ('adam', group['weight_decay']),
                      lr=group['lr'], weight_decay=group['weight_decay'], before=partition.detach().clone(),
                      grad=partition.grad.detach().clone(), shapes=group.get('muon_shapes'))
        result = original(closure)
        record['after'] = partition.detach().clone()
        records.append(record)
        return result
    client.step = step
    return records


def train(wrapped, updates, records=None, save=None):
    norms = []
    for update in updates:
        if records is not None:
            records.clear()
        for micro in range(GAS):
            loss = wrapped(batch(update, micro))
            wrapped.backward(loss)
            wrapped.step()
        norms.append(float(wrapped.get_global_grad_norm()))
        if records is not None:
            yield update, list(records), norms[-1]
        if save and update == save[1]:
            wrapped.save_checkpoint(save[0], tag=f'update{update}')


def full(records, key, field):
    gathered = [None] * dist.get_world_size()
    local = [r[field] for r in records if r['key'] == key]
    dist.all_gather_object(gathered, local)
    return torch.cat([g[0] for g in gathered]) if dist.get_rank() == 0 else None


def masters(wrapped):
    client = wrapped.optimizer.optimizer
    out = {}
    for group in client.param_groups:
        key = ('muon', group['muon_group']) if group.get('use_muon') else ('adam', group['weight_decay'])
        partition = group['params'][0]
        state = client.state[partition]
        out[key] = (partition.detach().clone(), state.get('momentum_buffer', state.get('exp_avg')).clone())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, out)
    return {k: tuple(torch.cat([g[k][i] for g in gathered]) for i in range(2)) for k in out}


def relative(a, b):
    return float((a - b).double().norm() / b.double().norm().clamp_min(1e-30))


def unflatten(flat, shapes):
    out, offset = [], 0
    for shape in shapes:
        n = shape[0] * shape[1]
        out.append(flat[offset:offset + n].view(shape))
        offset += n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deepspeed', default='configs/deepspeed_zero2_cpu_optimizer.json')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    rank, out = dist.get_rank(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = ds_config(args.deepspeed)
    report = dict(steps=[], passed=False)
    wrapped, _ = engine(config)
    records = install_recorder(wrapped)
    references, device = {}, torch.device('cuda', torch.cuda.current_device())
    groups = sorted({r for r in [('muon', g['muon_group']) for g in wrapped.optimizer.optimizer.param_groups
                                 if g.get('use_muon')]})
    report['muon_groups'] = len(groups)
    report['muon_shapes'] = [g['muon_shapes'] for g in wrapped.optimizer.optimizer.param_groups if g.get('use_muon')]
    save = out / 'checkpoint'
    for update, step_records, norm in train(wrapped, range(1, UPDATES + 1), records, save=(save, 2)):
        keys = sorted({r['key'] for r in step_records})
        row = dict(update=update, grad_norm=norm, groups={})
        for key in keys:
            before, grad, after = (full(step_records, key, f) for f in ('before', 'grad', 'after'))
            record = next(r for r in step_records if r['key'] == key)
            if rank:
                continue
            if key[0] == 'muon':
                numel = sum(s[0] * s[1] for s in record['shapes'])
                if key not in references:
                    params = [t.clone().to(device) for t in unflatten(before[:numel], record['shapes'])]
                    references[key] = (params, torch.optim.Muon(
                        params, lr=record['lr'], weight_decay=record['weight_decay'], momentum=SPEC['momentum'],
                        nesterov=SPEC['nesterov'], ns_coefficients=tuple(SPEC['ns_coefficients']), eps=SPEC['eps'],
                        ns_steps=SPEC['ns_steps'], adjust_lr_fn=SPEC['adjust_lr_fn']))
                params, optimizer = references[key]
                for param, start, g in zip(params, unflatten(before[:numel], record['shapes']),
                                           unflatten(grad[:numel], record['shapes'])):
                    param.data.copy_(start.to(device))
                    param.grad = g.to(device)
                optimizer.param_groups[0]['lr'] = record['lr']
                optimizer.step()
                expected = torch.cat([p.detach().reshape(-1).cpu() for p in params]) - before[:numel]
                actual = after[:numel] - before[:numel]
            else:
                if key not in references:
                    param = before.clone().to(device)
                    references[key] = ([param], torch.optim.AdamW([param], lr=record['lr'], betas=(0.9, 0.95),
                                                                  eps=1e-8, weight_decay=record['weight_decay']))
                params, optimizer = references[key]
                params[0].data.copy_(before.to(device))
                params[0].grad = grad.to(device)
                optimizer.param_groups[0]['lr'] = record['lr']
                optimizer.step()
                expected, actual = params[0].detach().cpu() - before, after - before
            row['groups'][f'{key[0]}{key[1]}'] = dict(
                lr=record['lr'], relative_l2=relative(actual, expected), bitwise_fraction=float(
                    (actual == expected).double().mean()), update_norm=float(actual.double().norm()))
        if rank == 0:
            report['steps'].append(row)
    uninterrupted = masters(wrapped)
    dist.barrier()
    resumed, _ = engine(config)
    resumed.load_checkpoint(str(save), tag='update2', load_optimizer_states=True, load_lr_scheduler_states=True)
    for _ in train(resumed, range(3, UPDATES + 1), records=[]):
        pass
    restored = masters(resumed)
    if rank == 0:
        report['resume'] = {f'{k[0]}{k[1]}': dict(master_relative_l2=relative(restored[k][0], uninterrupted[k][0]),
                                                  state_relative_l2=relative(restored[k][1], uninterrupted[k][1]),
                                                  bitwise=bool(torch.equal(restored[k][0], uninterrupted[k][0])))
                            for k in uninterrupted}
        steps_ok = all(g['relative_l2'] <= 1e-3 and g['update_norm'] > 0
                       for row in report['steps'] for g in row['groups'].values())
        clipped = any(row['grad_norm'] > 1.0 for row in report['steps'])
        resume_ok = all(v['master_relative_l2'] <= 1e-6 and v['state_relative_l2'] <= 1e-6
                        for v in report['resume'].values())
        report.update(steps_ok=steps_ok, clipping_exercised=clipped, resume_ok=resume_ok,
                      passed=steps_ok and clipped and resume_ok and report['muon_groups'] > 1)
        (out / 'zero_muon_ds_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print('ZERO_MUON_DS_PASS' if report['passed'] else 'ZERO_MUON_DS_FAIL', json.dumps(
            dict(steps_ok=steps_ok, clipped=clipped, resume_ok=resume_ok, groups=report['muon_groups'])), flush=True)
    passed = [report['passed']]
    dist.broadcast_object_list(passed, src=0)
    dist.destroy_process_group()
    raise SystemExit(0 if passed[0] else 1)


if __name__ == '__main__':
    main()
