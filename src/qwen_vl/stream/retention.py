"""MemFork frame-level retention policy for ``episodic_window_v1`` (plans/RL/research/memfork_design.md).

M5 keeps, per attention layer, the running top-B old visual rows by a fixed GDN committed-write
priority. MemFork replaces that rule by one *decision* per admission, shared by all attention
layers: the bank holds at most ``B / 300`` whole source frames; when a frame expires from the
recent window and the pool (bank + candidate) exceeds that size, the policy drops one frame

    q(drop j | pool) = softmax_j(-u(x_j)),     u(x) = kappa * (mean_l x_l + g(x)),

where ``x_j`` holds, per attention layer l, the log frame mean of frame j's 300 FP32 committed-write
scores (the M5 scorer's own values, fixed when the frame was observed). ``g`` is a small MLP whose
output layer starts at zero, so the initial policy is a tempered frame-level M5 rule and greedy
decoding keeps the running top frames by ``mean_l x_l``. Rows are never re-encoded or resurrected.

``FrameBank`` is plain numpy state (deep-copied by fork snapshots); the policy itself is stateless
and shared. Stochastic decisions draw one uniform from the episode's sampling generator, so a
restored branch with its own generator re-draws every later memory decision.
"""
import math

import numpy as np
import torch

from qwen_vl.research.episodic_contract import VISUAL_TOKENS_PER_GROUP

RETENTION_FILE = 'memory_retention.pt'
RETENTION_SCHEMA = 'memfork_frame_v1'


def frame_features(scores, attention_layers):
    """float64 [L]: log of the exactly rounded frame mean of each layer's 300 FP32 scores."""
    out = []
    for layer in attention_layers:
        values = np.asarray(scores[layer])
        if values.dtype != np.float32 or values.shape != (VISUAL_TOKENS_PER_GROUP,):
            raise TypeError('Frame features need the 300 FP32 committed-write scores of one frame')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite committed-write score')
        mean = math.fsum(values.astype(np.float64).tolist()) / VISUAL_TOKENS_PER_GROUP
        out.append(math.log(max(mean, 1e-30)))
    return np.asarray(out, dtype=np.float64)


class RetentionPolicy(torch.nn.Module):
    """u(x) = exp(log_kappa) * (mean_l x_l + g((x - shift) / scale)); drop logits are -u."""

    def __init__(self, layers=8, hidden=32, log_kappa=0.0):
        super().__init__()
        self.layers, self.hidden = layers, hidden
        self.log_kappa = torch.nn.Parameter(torch.tensor(float(log_kappa)))
        self.net = torch.nn.Sequential(torch.nn.Linear(layers, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)
        self.register_buffer('shift', torch.zeros(layers, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(layers, dtype=torch.float64))

    def utility(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        z = ((x - self.shift) / self.scale).to(self.net[0].weight.dtype)
        return self.log_kappa.double().exp() * (x.mean(-1) + self.net(z).squeeze(-1).double())

    def drop_log_probs(self, x):
        """[P, L] pool features -> float64 [P] log q(drop j)."""
        return torch.log_softmax(-self.utility(x), dim=-1)

    def config(self):
        return dict(schema=RETENTION_SCHEMA, layers=self.layers, hidden=self.hidden)

    def export(self):
        return dict(config=self.config(), state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_export(cls, payload):
        config = payload['config']
        if config.get('schema') != RETENTION_SCHEMA:
            raise ValueError(f'Retention schema {config.get("schema")!r} is not {RETENTION_SCHEMA}')
        policy = cls(config['layers'], config['hidden']).double()
        policy.load_state_dict(payload['state'], strict=True)      # exact float64 copy
        return policy.eval()


class RetentionControl:
    """How a session decides: ``greedy`` (evaluation) or ``sample`` (RL rollouts)."""

    def __init__(self, policy, mode):
        if mode not in ('greedy', 'sample'):
            raise ValueError(mode)
        self.policy, self.mode = policy, mode

    @torch.no_grad()
    def choose(self, features, generator):
        logq = self.policy.drop_log_probs(features)
        probs = logq.exp()
        if self.mode == 'greedy':
            # Lowest utility (highest drop probability); ties drop the newest frame, as M5's running
            # top-B keeps (priority descending, source ascending).
            index = int(torch.nonzero(logq == logq.max()).max())
        else:
            if generator is None:
                raise ValueError('Stochastic retention needs the episode sampling generator')
            draw = float(torch.rand((), generator=generator, device=generator.device))
            cdf = torch.cumsum(probs, 0)
            index = min(int(torch.searchsorted(cdf, torch.tensor(draw * float(cdf[-1]), dtype=cdf.dtype))),
                        len(probs) - 1)
        entropy = float(-(probs * logq).sum())
        return index, float(logq[index]), entropy


class FrameBank:
    """Retained whole frames (oldest first) and the record of every memory decision."""

    def __init__(self, frames):
        if frames < 1:
            raise ValueError('Frame retention needs a budget of at least one frame')
        self.frames = frames
        self.sources, self.starts, self.features = [], [], []
        self.decisions = []
        self.banks = {}          # step_id -> retained sources visible to that group (after promotion)

    @property
    def ids(self):
        if not self.starts:
            return np.empty(0, np.int64)
        return np.concatenate([np.arange(s, s + VISUAL_TOKENS_PER_GROUP, dtype=np.int64) for s in self.starts])

    def admit(self, source, visual_start, features, step_id, control, generator):
        self.sources.append(int(source))
        self.starts.append(int(visual_start))
        self.features.append(np.asarray(features, dtype=np.float64))
        if len(self.sources) <= self.frames:
            return
        pool = np.stack(self.features)
        index, logq, entropy = control.choose(pool, generator)
        self.decisions.append(dict(step=int(step_id), pool=list(self.sources), features=pool.tolist(),
                                   drop=index, logq=logq, entropy=entropy))
        for values in (self.sources, self.starts, self.features):
            values.pop(index)

    def record_bank(self, step_id):
        self.banks[int(step_id)] = list(self.sources)

    def export(self):
        return dict(decisions=list(self.decisions), banks={str(k): v for k, v in sorted(self.banks.items())})


def load_retention(checkpoint):
    """The checkpoint's retention policy (``memory_retention.pt``), or None for plain M5."""
    from pathlib import Path
    path = Path(checkpoint) / RETENTION_FILE
    if not path.exists():
        return None
    return RetentionPolicy.from_export(torch.load(path, map_location='cpu', weights_only=False))


def injected_plans(spec, layout, banks):
    """Per-layer ``LayerSelectionPlan`` reproducing the realised MemFork banks offline.

    ``banks``: step_id -> retained source frames visible to that group (``FrameBank.banks``).
    Group t sees the bank recorded at the latest promotion <= t (none before the first). The
    trainer's forward therefore attends exactly to the memory the rollout acted with; no score
    is recomputed, so BF16 near-ties can never change it.
    """
    from qwen_vl.stream.episodic import LayerSelectionPlan, _readonly, plan_digest
    banks = {int(k): [int(s) for s in v] for k, v in banks.items()}
    recent, frames = spec.recent_groups, spec.budget_tokens // VISUAL_TOKENS_PER_GROUP
    offsets, chunks, sources, current = [0], [], [], []
    for group in range(layout.groups):
        current = banks.get(group, current)
        if len(current) > frames or current != sorted(set(current)):
            raise ValueError(f'Invalid retained frames {current} at group {group}')
        if current and current[-1] > group - recent:
            raise ValueError('A retained frame is still inside the recent window')
        ids = [np.arange(layout.visual_starts[s], layout.visual_starts[s] + VISUAL_TOKENS_PER_GROUP,
                         dtype=np.int64) for s in current]
        chunks.append(np.concatenate(ids) if ids else np.empty(0, np.int64))
        sources.append(np.repeat(np.asarray(current, np.int64), VISUAL_TOKENS_PER_GROUP))
        offsets.append(offsets[-1] + chunks[-1].size)
    if set(banks) - set(range(layout.groups)):
        raise ValueError('Recorded banks for groups outside the layout')
    token_ids = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    token_sources = np.concatenate(sources) if sources else np.empty(0, np.int64)
    return {layer: LayerSelectionPlan(
        layer_idx=layer, recent_groups=recent, budget_tokens=spec.budget_tokens,
        group_offsets=_readonly(offsets), token_ids=_readonly(token_ids),
        token_sources=_readonly(token_sources),
        cutoff_margins=_readonly(np.full(layout.groups, np.nan)),
        source_digest=plan_digest(layer, recent, spec.budget_tokens, offsets, token_ids))
        for layer in spec.attention_layers}
