import copy
import os

import pytest
import torch

muon_module = pytest.importorskip('torch.optim._muon')

from qwen_vl.research.matrix_optimizer_contract import parse_matrix_optimizer
from qwen_vl.train import zero_muon as zm

SPEC = dict(name='muon_zero_offload_v1', scope='language_model_layers_2d', momentum=0.95, nesterov=True,
            ns_steps=5, ns_coefficients=[3.4445, -4.775, 2.0315], eps=1.0e-07, adjust_lr_fn='match_rms_adamw',
            group_numel=700)
SHAPES = [(24, 40), (40, 24), (17, 33), (8, 64), (33, 17)]   # tall and wide; partitions cut through matrices
WORLD, LR, WD = 4, 3e-3, 0.01


class MuonSGD(zm.ZeroMuonMixin, torch.optim.SGD):
    """Production uses DeepSpeedCPUAdam as the base; any torch optimizer exercises the same control flow."""


def training(**changes):
    return dict(cpu_optimizer_impl='deepspeed_cpu_adam', matrix_optimizer=dict(SPEC, **changes))


def test_contract_is_exact_and_absent_means_adamw():
    assert parse_matrix_optimizer({}) is None
    assert parse_matrix_optimizer(training()) == SPEC
    for bad in (dict(momentum=1), dict(momentum=1.0), dict(nesterov='yes'), dict(ns_steps=0),
                dict(ns_coefficients=[1.0, 2.0]), dict(eps=0.0), dict(adjust_lr_fn=None), dict(group_numel=0),
                dict(name='muon'), dict(scope='all')):
        with pytest.raises(ValueError):
            parse_matrix_optimizer(training(**bad))
    with pytest.raises(ValueError):
        parse_matrix_optimizer(dict(cpu_optimizer_impl='deepspeed_cpu_adam',
                                    matrix_optimizer={k: v for k, v in SPEC.items() if k != 'eps'}))
    with pytest.raises(ValueError):
        parse_matrix_optimizer(dict(training(), cpu_optimizer_impl='torch'))


def test_parameter_selection_and_chunking():
    decay = {'m.language_model.layers.0.mlp.up_proj.weight', 'm.language_model.embed_tokens.weight',
             'm.lm_head.weight', 'm.language_model.layers.0.linear_attn.conv1d.weight'}
    p2, p3, p1 = torch.zeros(4, 5, requires_grad=True), torch.zeros(4, 1, 2, requires_grad=True), torch.zeros(4, requires_grad=True)
    assert zm.is_muon_parameter('m.language_model.layers.0.mlp.up_proj.weight', p2, decay)
    assert not zm.is_muon_parameter('m.language_model.embed_tokens.weight', p2, decay)
    assert not zm.is_muon_parameter('m.lm_head.weight', p2, decay)
    assert not zm.is_muon_parameter('m.language_model.layers.0.linear_attn.conv1d.weight', p3, decay)
    assert not zm.is_muon_parameter('m.language_model.layers.0.input_layernorm.weight', p1, decay)
    frozen = torch.zeros(4, 5)
    assert not zm.is_muon_parameter('m.language_model.layers.0.mlp.up_proj.weight', frozen, decay)
    named = [(f'w{i}', torch.zeros(s)) for i, s in enumerate(SHAPES)]
    groups = zm.muon_groups(named, SPEC, LR, WD)
    assert [n for g in groups for n in g['muon_names']] == [n for n, _ in named]
    assert all(sum(p.numel() for p in g['params'][:-1]) < SPEC['group_numel'] for g in groups)
    assert [g['muon_group'] for g in groups] == list(range(len(groups)))
    assert all(g['lr'] == LR and g['weight_decay'] == WD and g['use_muon'] for g in groups)


def make_problem(seed=0, steps=4):
    g = torch.Generator().manual_seed(seed)
    params = [torch.randn(s, generator=g) * 0.1 for s in SHAPES]
    grads = [[torch.randn(s, generator=g) for s in SHAPES] for _ in range(steps)]
    return params, grads


def flat_partitions(tensors, world=WORLD):
    flat = torch.cat([t.reshape(-1) for t in tensors])
    padded = -(-flat.numel() // (2 * world)) * 2 * world
    flat = torch.cat([flat, flat.new_zeros(padded - flat.numel())])
    return list(flat.chunk(world))


def reference(params, grads, nesterov=True, adjust='match_rms_adamw'):
    full = [p.clone() for p in params]
    opt = torch.optim.Muon(full, lr=LR, weight_decay=WD, momentum=SPEC['momentum'], nesterov=nesterov,
                           ns_coefficients=tuple(SPEC['ns_coefficients']), eps=SPEC['eps'],
                           ns_steps=SPEC['ns_steps'], adjust_lr_fn=adjust)
    for step in grads:
        for p, gr in zip(full, step):
            p.grad = gr.clone()
        opt.step()
    return full


def simulated(params, grads, nesterov=True, adjust='match_rms_adamw', world=WORLD):
    """One optimizer per rank over its flat partition, gathered through a stand-in for all_gather."""
    spec = dict(SPEC, nesterov=nesterov, adjust_lr_fn=adjust)
    masters = flat_partitions(params, world)
    ranks = []
    for rank in range(world):
        group = zm.muon_group_dict([torch.zeros(s) for s in SHAPES], spec, LR, WD, 0)
        part = masters[rank].clone()
        group['params'] = [part]
        opt = MuonSGD([group], lr=LR).attach_muon({0: []})
        ranks.append((opt, part))
    for step in grads:
        for (opt, part), g in zip(ranks, flat_partitions(step, world)):
            part.grad = g.clone()
        locals_ = [opt.muon_local_update(opt.param_groups[0]).bfloat16() for opt, _ in ranks]
        gathered = torch.cat(locals_)
        for rank, (opt, _) in enumerate(ranks):
            opt.muon_apply(opt.param_groups[0], gathered.clone(), rank, world, torch.device('cpu'))
    flat = torch.cat([part for _, part in ranks])
    out, offset = [], 0
    for s in SHAPES:
        n = s[0] * s[1]
        out.append(flat[offset:offset + n].view(s))
        offset += n
    return out, ranks


@pytest.mark.parametrize('nesterov,adjust', [(True, 'match_rms_adamw'), (False, 'match_rms_adamw'),
                                             (True, 'original')])
def test_partitioned_muon_is_bitwise_torch_muon(nesterov, adjust):
    params, grads = make_problem()
    expected = reference(params, grads, nesterov, adjust)
    actual, _ = simulated(params, grads, nesterov, adjust)
    for e, a in zip(expected, actual):
        assert torch.equal(e, a)
    # and it differs from the start, so the comparison is not vacuous
    assert not torch.equal(actual[0], params[0])


def test_single_rank_matches_too():
    params, grads = make_problem(seed=3)
    actual, _ = simulated(params, grads, world=1)
    for e, a in zip(reference(params, grads), actual):
        assert torch.equal(e, a)


def test_state_dict_round_trip_continues_identically():
    params, grads = make_problem(seed=1, steps=4)
    full, ranks = simulated(params, grads[:2])
    resumed = []
    for opt, part in ranks:
        saved = copy.deepcopy(opt.state_dict())
        group = zm.muon_group_dict([torch.zeros(s) for s in SHAPES], SPEC, LR, WD, 0)
        fresh = part.clone()
        group['params'] = [fresh]
        new = MuonSGD([group], lr=LR).attach_muon({0: []})
        new.load_state_dict(saved)
        assert torch.equal(new.state[fresh]['momentum_buffer'], opt.state[part]['momentum_buffer'])
        resumed.append((new, fresh))
    for step in grads[2:]:
        for (opt, part), g in zip(resumed, flat_partitions(step)):
            part.grad = g.clone()
        gathered = torch.cat([opt.muon_local_update(opt.param_groups[0]).bfloat16() for opt, _ in resumed])
        for rank, (opt, _) in enumerate(resumed):
            opt.muon_apply(opt.param_groups[0], gathered.clone(), rank, WORLD, torch.device('cpu'))
    flat = torch.cat([p for _, p in resumed])
    assert torch.equal(flat[:sum(s[0] * s[1] for s in SHAPES)],
                       torch.cat([t.reshape(-1) for t in reference(params, grads)]))


def layout_optimizer(module, params):
    group = zm.muon_group_dict(module, SPEC, LR, WD, 0)
    partition = flat_partitions(params, 1)[0].clone()
    partition.grad = torch.zeros_like(partition)
    group['params'] = [partition]
    return MuonSGD([group], lr=LR).attach_muon({0: module}), group, partition


def test_layout_check_accepts_the_true_layout_and_rejects_others():
    params = [p.bfloat16().float() for p in make_problem()[0]]
    module = [p.bfloat16() for p in params]
    opt, group, partition = layout_optimizer(module, params)
    opt._verify_layout(group, partition, 0, 1, torch.device('cpu'))
    assert 0 in opt._muon_verified
    corrupted = [m.clone() for m in module]
    corrupted[2][0, 0] += 1
    opt, group, partition = layout_optimizer(corrupted, params)
    with pytest.raises(RuntimeError):
        opt._verify_layout(group, partition, 0, 1, torch.device('cpu'))
    opt, group, partition = layout_optimizer(module, params)
    opt._muon_parameters = {0: [module[1], module[0], *module[2:]]}
    with pytest.raises(RuntimeError):
        opt._verify_layout(group, partition, 0, 1, torch.device('cpu'))


def test_layout_rejects_wrong_partition_size():
    with pytest.raises(RuntimeError):
        zm.flat_layout(SHAPES, 10, WORLD)


def _gloo_worker(rank, world, port, params, grads, adam_param, adam_grads, queue):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.set_num_threads(1)
    muon_group = zm.muon_group_dict([p.bfloat16() for p in params], SPEC, LR, WD, 0)
    module = list(muon_group['params'])
    muon_group['params'] = [flat_partitions(params, world)[rank].clone()]
    adam_group = dict(params=[adam_param.clone()], lr=LR, weight_decay=WD)
    opt = MuonSGD([adam_group, muon_group], lr=LR, momentum=0.9).attach_muon({0: module})
    for step, adam_grad in zip(grads, adam_grads):
        opt.param_groups[1]['params'][0].grad = flat_partitions(step, world)[rank].clone()
        opt.param_groups[0]['params'][0].grad = adam_grad.clone()
        for index in range(2):                       # ZeRO steps one group at a time
            groups = opt.param_groups
            opt.param_groups = [groups[index]]
            opt.step()
            opt.param_groups = groups
    # numpy copies travel by value; a shared-memory tensor would vanish with this process
    queue.put((rank, opt.param_groups[1]['params'][0].numpy().copy(), opt.param_groups[0]['params'][0].numpy().copy()))
    dist.destroy_process_group()


def test_real_step_with_gloo_all_gather_and_adam_group(monkeypatch):
    import torch.multiprocessing as mp
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')        # workers orthogonalize on CPU, like the reference
    params, grads = make_problem(seed=7, steps=3)
    params = [p.bfloat16().float() for p in params]       # BF16-representable, like ZeRO's first masters
    adam_param = torch.randn(10)
    adam_grads = [torch.randn(10) for _ in grads]
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_gloo_worker, args=(r, WORLD, port, params, grads, adam_param, adam_grads, queue))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    results = sorted(queue.get(timeout=300) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    flat = torch.cat([torch.from_numpy(r[1]) for r in results])
    expected = torch.cat([t.reshape(-1) for t in reference(params, grads)])
    assert torch.equal(flat[:expected.numel()], expected)
    other = adam_param.clone()
    reference_other = torch.optim.SGD([other], lr=LR, weight_decay=WD, momentum=0.9)
    for g in adam_grads:
        other.grad = g.clone()
        reference_other.step()
    assert all(torch.equal(torch.from_numpy(r[2]), other) for r in results)
    assert not torch.equal(other, adam_param)
