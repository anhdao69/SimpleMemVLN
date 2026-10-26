"""Optimizer state after the same update from an uninterrupted and a resumed run (ZeRO-1/2 checkpoints).

For every rank, the fp32 master partitions and every base-optimizer state tensor (Muon momentum, Adam
moments) of the two checkpoints must agree: masters to relative L2 <= 1e-5 (as for the weights) and state
tensors to <= 1e-2 (``--state-rtol``).  A state lost at resume would differ by O(1) (the state would hold
one gradient instead of three).  Muon groups must hold exactly a momentum buffer, AdamW groups exactly step/moments,
and step counters must be equal.
"""
import argparse
import json
from pathlib import Path

import torch

MASTER_RTOL, STATE_RTOL = 1e-5, 1e-2


def relative(actual, expected):
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))


def load(checkpoint, rank):
    files = sorted(Path(checkpoint).glob(f'global_step*/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'))
    if len(files) != 1:
        raise ValueError(f'{checkpoint}: expected one rank-{rank} optimizer file, found {len(files)}')
    # Our own DeepSpeed checkpoints; they hold non-tensor objects (loss scaler, config).
    return torch.load(files[0], map_location='cpu', weights_only=False)['optimizer_state_dict']


def compare_rank(reference, resumed):
    from deepspeed.checkpoint.constants import BASE_OPTIMIZER_STATE, SINGLE_PARTITION_OF_FP32_GROUPS
    masters = [relative(b, a) for a, b in zip(reference[SINGLE_PARTITION_OF_FP32_GROUPS],
                                              resumed[SINGLE_PARTITION_OF_FP32_GROUPS], strict=True)]
    ref, res = reference[BASE_OPTIMIZER_STATE], resumed[BASE_OPTIMIZER_STATE]
    groups = []
    for group_ref, group_res in zip(ref['param_groups'], res['param_groups'], strict=True):
        muon = bool(group_ref.get('use_muon'))
        if group_ref['params'] != group_res['params'] or muon != bool(group_res.get('use_muon')) \
                or len(group_ref['params']) != 1:
            raise ValueError('Optimizer groups differ between the checkpoints')
        a, b = ref['state'][group_ref['params'][0]], res['state'][group_res['params'][0]]
        expected = {'momentum_buffer'} if muon else {'step', 'exp_avg', 'exp_avg_sq'}
        if set(a) != expected or set(b) != expected:
            raise ValueError(f'Unexpected state keys {sorted(a)} / {sorted(b)} for a {"Muon" if muon else "AdamW"} group')
        row = dict(use_muon=muon)
        for key in sorted(expected):
            if key == 'step':
                if int(a[key]) != int(b[key]):
                    raise ValueError(f'Step counters differ: {a[key]} != {b[key]}')
                row[key] = int(a[key])
                continue
            if not bool(a[key].abs().sum() > 0):
                raise ValueError(f'Reference {key} is all zero')
            row[key] = relative(b[key], a[key])
        groups.append(row)
    return dict(masters=masters, groups=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--resumed', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--world', type=int, default=4)
    p.add_argument('--state-rtol', type=float, default=STATE_RTOL)
    args = p.parse_args()
    ranks = [compare_rank(load(args.reference, r), load(args.resumed, r)) for r in range(args.world)]
    master = max(m for r in ranks for m in r['masters'])
    state = max(v for r in ranks for g in r['groups'] for k, v in g.items() if k.startswith(('exp_', 'momentum')))
    muon_groups = [sum(g['use_muon'] for g in r['groups']) for r in ranks]
    passed = master <= MASTER_RTOL and state <= args.state_rtol and min(muon_groups) > 0
    summary = dict(passed=passed, max_master_relative_l2=master, max_state_relative_l2=state,
                   state_rtol=args.state_rtol,
                   muon_groups_per_rank=muon_groups, steps=sorted({g['step'] for r in ranks for g in r['groups']
                                                                  if 'step' in g}))
    Path(args.out).write_text(json.dumps(dict(summary=summary, ranks=ranks), indent=2) + '\n')
    print('OPTIMIZER_STATE_PASS' if passed else 'OPTIMIZER_STATE_FAIL', json.dumps(summary), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
