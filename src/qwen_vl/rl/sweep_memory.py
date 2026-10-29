"""RL rollouts of GDN-sweep frame policies (S8N): realised per-layer memory and fork snapshots.

S8N keeps, per attention layer, the running top-``B/300`` whole frames by a fixed FP32 normalised
GDN state-drift priority (``stream/gdn_sweep_state``). The serving state is used unchanged; this
module only adds what RL needs on top of it:

* ``SweepRolloutState`` records, after every promotion, the frame sources each attention layer
  retains (``realised``: step -> {layer: [sources]}), so the trainer's full-episode forward
  attends to exactly the memory the rollout acted with (``injected_layer_plans``). Re-scoring the
  drift offline from BF16 training activations could flip near-tied frames and train on a memory
  the policy never saw.
* ``clone_sweep_state``: an independent copy taken between steps (fork snapshots): running
  frame selections, resident row IDs, recent groups with their birth priorities, the scorers'
  carried FP32 recurrent state and conv tails, and the realised record.
"""
import copy

import numpy as np

from qwen_vl.research.episodic_contract import VISUAL_TOKENS_PER_GROUP
from qwen_vl.stream.gdn_sweep_state import GdnSweepSessionState

DRIFT_SCORES = ('S8N', 'S8W')


class SweepRolloutState(GdnSweepSessionState):
    bank = None        # no MemFork bank: rl/rollout.memory_record exports ``realised`` instead

    def __init__(self, spec, model, episode_uid, device):
        if not spec.is_frame:
            raise ValueError('RL replay of the GDN sweep is defined for whole-frame arms')
        super().__init__(spec, model, episode_uid, device)
        self.realised = {}

    def promote(self, cache, step_id, generator=None):
        """``GdnSweepSessionState.promote`` (deterministic: ``generator`` is unused), then record."""
        super().promote(cache, step_id)
        current = {layer: [int(s) for s in self.running[layer].frame_sources] for layer in self.attention_layers}
        latest = self.realised[max(self.realised)] if self.realised else None
        if current != (latest or {layer: [] for layer in self.attention_layers}):
            self.realised[int(step_id)] = current

    def pending_decision_entropy(self, step_id):
        return None

    def observation_drift(self):
        """Per-layer birth priorities of the newest observation (S8N: normalised state drift)."""
        if self.spec.score not in DRIFT_SCORES:
            return None
        group = self.groups[-1]
        return [float(group.scores[layer]) for layer in self.attention_layers]

    def export(self):
        return dict(layer_banks={str(step): {str(layer): sources for layer, sources in banks.items()}
                                 for step, banks in sorted(self.realised.items())})


def clone_sweep_state(state):
    if state.segment_kind is not None:
        raise ValueError('Sweep snapshot inside an append')
    twin = copy.copy(state)
    twin.running = copy.deepcopy(state.running)
    twin.resident = {layer: ids.copy() for layer, ids in state.resident.items()}
    twin.groups = copy.deepcopy(state.groups)
    twin.promotions = copy.deepcopy(state.promotions)
    twin.realised = copy.deepcopy(state.realised)
    twin.scorers = {}
    for gdn, scorer in state.scorers.items():
        if scorer.legacy is not None:
            raise ValueError('Sweep snapshots need a native GDN scorer backend')
        if scorer.pending:
            raise ValueError('Sweep snapshot with an unclosed response')
        copied = copy.copy(scorer)
        copied.state, copied.conv_tail, copied.pending = scorer.state.clone(), scorer.conv_tail.clone(), []
        twin.scorers[gdn] = copied
    return twin


def injected_layer_plans(spec, layout, layer_banks):
    """Per-layer ``LayerSelectionPlan`` reproducing the realised S8N frame memory offline.

    ``layer_banks``: step -> {attention layer: retained source frames} (``SweepRolloutState.export``).
    Group t of layer l sees the frames recorded at the latest promotion <= t (none before the first).
    """
    from qwen_vl.stream.episodic import LayerSelectionPlan, _readonly, plan_digest
    banks = {int(step): {int(layer): [int(s) for s in sources] for layer, sources in by_layer.items()}
             for step, by_layer in layer_banks.items()}
    if set(banks) - set(range(layout.groups)):
        raise ValueError('Recorded banks for groups outside the layout')
    recent, frames = spec.recent_groups, spec.budget_tokens // VISUAL_TOKENS_PER_GROUP
    plans = {}
    for layer in spec.attention_layers:
        offsets, chunks, sources, current = [0], [], [], []
        for group in range(layout.groups):
            if group in banks:
                if set(banks[group]) != set(spec.attention_layers):
                    raise ValueError(f'Bank record at group {group} does not cover every attention layer')
                current = banks[group][layer]
            if len(current) > frames or current != sorted(set(current)):
                raise ValueError(f'Invalid retained frames {current} at group {group}')
            if current and current[-1] > group - recent:
                raise ValueError('A retained frame is still inside the recent window')
            ids = [np.arange(layout.visual_starts[s], layout.visual_starts[s] + VISUAL_TOKENS_PER_GROUP,
                             dtype=np.int64) for s in current]
            chunks.append(np.concatenate(ids) if ids else np.empty(0, np.int64))
            sources.append(np.repeat(np.asarray(current, np.int64), VISUAL_TOKENS_PER_GROUP))
            offsets.append(offsets[-1] + chunks[-1].size)
        token_ids = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        plans[layer] = LayerSelectionPlan(
            layer_idx=layer, recent_groups=recent, budget_tokens=spec.budget_tokens,
            group_offsets=_readonly(offsets), token_ids=_readonly(token_ids),
            token_sources=_readonly(np.concatenate(sources) if sources else np.empty(0, np.int64)),
            cutoff_margins=_readonly(np.full(layout.groups, np.nan)),
            source_digest=plan_digest(layer, recent, spec.budget_tokens, offsets, token_ids))
    return plans
