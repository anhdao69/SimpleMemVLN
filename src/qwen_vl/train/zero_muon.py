"""Muon for hidden-layer matrices under DeepSpeed ZeRO-1/2 with CPU optimizer offload.

DeepSpeed 0.16 has no Muon.  ZeRO flattens each optimizer parameter group, keeps
one contiguous fp32 master partition per rank (on CPU with offload) and steps
the client optimizer once per group with ``param_groups == [group]`` and the
group's reduced, globally clipped gradient partition.  Partitions cut through
matrices, but Newton-Schulz orthogonalization needs whole matrices.  For a Muon
group every rank therefore

1. updates its slice of the momentum buffer and forms the Nesterov update,
2. all-gathers that update in BF16 (``_zeropower_via_newtonschulz`` casts its
   input to BF16 first, so gathering BF16 is exact relative to the reference),
3. orthogonalizes each matrix that overlaps its partition on its GPU, and
4. applies decoupled weight decay and the scaled update to its own slice only.

Every element follows ``torch.optim.Muon`` exactly: ``buf.lerp_(g, 1 - mu)``,
``g.lerp(buf, mu)``, torch's own ``_zeropower_via_newtonschulz`` and
``_adjust_lr``, then ``p.mul_(1 - lr * wd)`` and ``p.add_(update, alpha=-adjusted_lr)``.
Each element is owned by exactly one rank, so ranks never disagree.  All other
parameters (embeddings, LM head, norms, scalars, conv filters) keep the base
optimizer (DeepSpeedCPUAdam in training).  Momentum buffers live in
``state[partition]['momentum_buffer']`` and travel with DeepSpeed checkpoints.
"""
import torch
import torch.distributed as dist

from qwen_vl.research.matrix_optimizer_contract import MATRIX_OPTIMIZER, SCOPE, parse_matrix_optimizer  # noqa: F401


def is_muon_parameter(name, parameter, decay_names):
    """Trainable 2D hidden-layer weights of the language model (not embeddings, LM head, norms or biases)."""
    return (parameter.requires_grad and parameter.ndim == 2 and name in decay_names
            and '.language_model.layers.' in name)


def muon_groups(named_parameters, spec, lr, weight_decay):
    """Contiguous model-order chunks of about ``group_numel`` elements, split only at matrix boundaries.

    Smaller groups bound the transient GPU buffer that holds one gathered group.
    """
    groups, current, size = [], [], 0
    for name, parameter in named_parameters:
        current.append((name, parameter))
        size += parameter.numel()
        if size >= spec['group_numel']:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    return [muon_group_dict([p for _, p in members], spec, lr, weight_decay, index, [n for n, _ in members])
            for index, members in enumerate(groups)]


def muon_group_dict(parameters, spec, lr, weight_decay, index, names=None):
    return dict(params=list(parameters), lr=lr, weight_decay=weight_decay, use_muon=True, muon_group=index,
                muon_shapes=[list(p.shape) for p in parameters], muon_names=list(names or []),
                muon_momentum=spec['momentum'], muon_nesterov=spec['nesterov'], muon_ns_steps=spec['ns_steps'],
                muon_ns_coefficients=list(spec['ns_coefficients']), muon_eps=spec['eps'],
                muon_adjust_lr_fn=spec['adjust_lr_fn'])


def flat_layout(shapes, partition_numel, world):
    """(shape, start, end) of each matrix in the ZeRO flat buffer (model order, padding only at the end)."""
    layout, offset = [], 0
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Muon groups hold 2D matrices only')
        numel = shape[0] * shape[1]
        layout.append((tuple(shape), offset, offset + numel))
        offset += numel
    if not 0 <= world * partition_numel - offset < 2 * world * 512:
        raise RuntimeError(f'ZeRO partition {partition_numel} x {world} does not match a {offset}-element group')
    return layout


class ZeroMuonMixin:
    """Step Muon groups as above and every other group with the base optimizer, one group at a time."""

    def attach_muon(self, layout_parameters, gather=None):
        # group index -> the module parameters of that group in flat order (for the layout self-check)
        self._muon_parameters = layout_parameters
        self._muon_verified = set()
        self._muon_scratch = None
        self._muon_gather = gather or _all_gather
        return self

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported')
        groups = self.param_groups
        try:
            for group in groups:
                if group.get('use_muon'):
                    self._muon_step(group)
                else:
                    self.param_groups = [group]
                    super().step()
        finally:
            self.param_groups = groups
        return None

    # -- one Muon group (a ZeRO partition) --
    def _partition(self, group):
        if len(group['params']) != 1:
            raise RuntimeError('A Muon group must be flattened to one ZeRO partition before stepping')
        partition = group['params'][0]
        if partition.grad is None or partition.dtype != torch.float32 or partition.ndim != 1:
            raise RuntimeError('Muon expects a flat fp32 master partition with a gradient')
        return partition

    def muon_local_update(self, group):
        """Momentum/Nesterov on this rank's slice; returns the fp32 update slice."""
        partition = self._partition(group)
        state = self.state[partition]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(partition, memory_format=torch.preserve_format)
        buffer, gradient, momentum = state['momentum_buffer'], partition.grad, group['muon_momentum']
        buffer.lerp_(gradient, 1 - momentum)
        scratch = self._scratch(partition.numel(), partition.device)
        if group['muon_nesterov']:
            torch.lerp(gradient, buffer, momentum, out=scratch)
        else:
            scratch.copy_(buffer)
        return scratch

    def muon_apply(self, group, gathered, rank, world, device):
        """Weight decay and orthogonalized updates for the matrix pieces inside this rank's slice."""
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
        partition = self._partition(group)
        size = partition.numel()
        lo, hi = rank * size, (rank + 1) * size
        lr, coefficients = group['lr'], tuple(group['muon_ns_coefficients'])
        partition.mul_(1 - lr * group['weight_decay'])
        for shape, start, end in flat_layout(group['muon_shapes'], size, world):
            a, b = max(start, lo), min(end, hi)
            if a >= b:
                continue
            update = _zeropower_via_newtonschulz(gathered[start:end].view(shape), coefficients,
                                                 group['muon_ns_steps'], group['muon_eps'])
            adjusted = _adjust_lr(lr, group['muon_adjust_lr_fn'], torch.Size(shape))
            piece = update.reshape(-1)[a - start:b - start].to(partition.device)
            partition[a - lo:b - lo].add_(piece, alpha=-adjusted)

    def _muon_step(self, group):
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        partition = self._partition(group)
        if group['muon_group'] not in self._muon_verified:
            self._verify_layout(group, partition, rank, world, device)
        local = self.muon_local_update(group).to(device).bfloat16()
        gathered = self._muon_gather(local, world)
        self.muon_apply(group, gathered, rank, world, device)
        del local, gathered

    def _verify_layout(self, group, partition, rank, world, device):
        """The BF16 module weights must equal the gathered fp32 masters cast to BF16, matrix by matrix."""
        parameters = self._muon_parameters[group['muon_group']]
        layout = flat_layout(group['muon_shapes'], partition.numel(), world)
        if [list(p.shape) for p in parameters] != group['muon_shapes']:
            raise RuntimeError('Muon group shapes differ from its module parameters')
        masters = self._muon_gather(partition.to(device).bfloat16(), world)
        for parameter, (shape, start, end) in zip(parameters, layout):
            if not torch.equal(masters[start:end].view(shape), parameter.detach().to(device, torch.bfloat16)):
                raise RuntimeError(f'ZeRO flat layout differs from the Muon layout at {shape} [{start}, {end})')
        self._muon_verified.add(group['muon_group'])

    def _scratch(self, numel, device):
        if self._muon_scratch is None or self._muon_scratch.numel() < numel:
            pin = device.type == 'cpu' and torch.cuda.is_available()
            self._muon_scratch = torch.empty(numel, dtype=torch.float32, device=device, pin_memory=pin)
        return self._muon_scratch[:numel]


def _all_gather(local, world):
    if world == 1:
        return local
    gathered = torch.empty(world * local.numel(), dtype=local.dtype, device=local.device)
    dist.all_gather(list(gathered.chunk(world)), local)
    return gathered


def zero_muon_cpu_adam(groups, layout_parameters, **adam_kwargs):
    """DeepSpeedCPUAdam (identical to the AdamW arms for non-Muon groups) plus ZeRO Muon groups."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    class ZeroMuonCPUAdam(ZeroMuonMixin, DeepSpeedCPUAdam):
        pass

    return ZeroMuonCPUAdam(groups, **adam_kwargs).attach_muon(layout_parameters)
