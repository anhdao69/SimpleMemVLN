"""MemFork frame retention on a tiny real Qwen3.5 text model (FP32 CPU, torch fallbacks).

* stochastic streaming with retention == the offline training forward with the realised banks
  injected (``injected_plans``), so the trainer's action log-probs use the memory acted with;
* greedy retention == running top-F frames by the prior utility (``RunningTopFrames``);
* recorded log q equals the policy's log q recomputed from the recorded pool features;
* a fork branch with its own generator re-draws later memory decisions; the same generator replays.
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vln'))

from test_episodic_forward import _episode, _offline, _spec, tiny  # noqa: E402,F401  (fixture)

from qwen_vl.models.episodic_forward import episodic_text_forward  # noqa: E402
from qwen_vl.rl.sampling import clone_cache, clone_episodic  # noqa: E402
from qwen_vl.stream.cache import make_stream_cache_fp32  # noqa: E402
from qwen_vl.stream.episodic_cache import EpisodicSessionState  # noqa: E402
from qwen_vl.stream.gdn_selectors import RunningTopFrames  # noqa: E402
from qwen_vl.stream.retention import (FrameBank, RetentionControl, RetentionPolicy,  # noqa: E402
                                      injected_plans)


class Stream:
    def __init__(self, model, spec, layout, embeds, positions, control):
        self.model, self.spec, self.layout, self.embeds, self.positions = model, spec, layout, embeds, positions
        wrapper = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(language_model=model)))
        self.state = EpisodicSessionState(spec, wrapper, 'ep', 'cpu')
        self.cache = make_stream_cache_fp32(model.config)
        self.append(0, layout.prefix, 'prefix')
        self.state.set_retention(control)
        self.next_group = 0

    def append(self, start, end, kind, offset=None):
        self.state.begin_append(kind, start, end - start, offset)
        out = episodic_text_forward(self.model, self.embeds[:, start:end], self.positions[..., start:end],
                                    self.spec, past_key_values=self.cache, use_cache=True,
                                    stream_append=True, serving=self.state)
        self.state.end_append(self.cache)
        return out

    def group(self, generator):
        t, layout = self.next_group, self.layout
        (start, end), observed = layout.group_spans[t], layout.observation_ends[t]
        self.state.promote(self.cache, t, generator=generator)      # prepare_step (before a snapshot)
        self.state.promote(self.cache, t)                           # sample_step's promotion: no-op
        self.state.open_group(t, start, layout.visual_starts[t])
        outs = [self.append(start, observed, 'observation', layout.visual_starts[t] - start)]
        outs += [self.append(token, token + 1, 'response_token') for token in range(observed, end - 2)]
        outs.append(self.append(end - 2, end, 'response_close'))
        self.state.close_group(end)
        self.next_group += 1
        return torch.cat(outs, 1)

    def fork(self):
        twin = object.__new__(Stream)
        twin.__dict__.update(self.__dict__)
        twin.cache, twin.state = clone_cache(self.cache), clone_episodic(self.state)
        return twin


def _policy(log_kappa=-1.0, seed=0):
    torch.manual_seed(seed)
    policy = RetentionPolicy(layers=2, hidden=4, log_kappa=log_kappa).double()
    torch.nn.init.normal_(policy.net[2].weight, std=0.5)       # a non-trivial learned part
    return policy


def test_stochastic_streaming_equals_offline_with_injected_banks(tiny):
    spec = _spec(tiny, 2, 600, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(10, seed=3)
    policy = _policy()
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        stream = Stream(tiny, spec, layout, embeds, positions, RetentionControl(policy, 'sample'))
        online = torch.cat([stream.group(generator) for _ in range(layout.groups)], 1)
        plans = injected_plans(spec, layout, stream.state.bank.banks)
        offline = episodic_text_forward(tiny, embeds, positions, spec,
                                        step_plan=(layout.prefix, layout.group_spans), layout=layout,
                                        episode_uid='ep', window_attention_batch_steps=3, plans=plans)
        default = _offline(tiny, spec, layout, embeds, positions)
    prefix = layout.prefix
    torch.testing.assert_close(online, offline[:, prefix:], rtol=2e-4, atol=2e-5)
    bank = stream.state.bank
    assert len(bank.decisions) == layout.groups - spec.recent_groups - 2     # pools of 3 frames
    assert sorted(bank.banks) == list(range(spec.recent_groups, layout.groups))
    # recorded log q is reproducible from the recorded pool features (what the trainer uses)
    for d in bank.decisions:
        logq = policy.drop_log_probs(torch.tensor(d['features'], dtype=torch.float64)).detach()
        assert abs(float(logq[d['drop']]) - d['logq']) < 1e-12
    # the injected memory is genuinely different from M5's token rule on this episode
    assert not torch.allclose(default[:, prefix:], offline[:, prefix:], rtol=2e-4, atol=2e-5) or \
        all(len(v) == 0 for v in bank.banks.values())


def test_greedy_retention_is_running_top_frames():
    policy = RetentionPolicy(layers=3, hidden=4).double()
    control = RetentionControl(policy, 'greedy')
    rng = np.random.default_rng(0)
    bank, reference = FrameBank(4), RunningTopFrames(4)
    for source in range(20):
        features = rng.normal(size=3)
        start = 1000 + 400 * source
        bank.admit(source, start, features, source + 2, control, None)
        reference.admit(np.arange(start, start + 300), float(features.mean()), source)
        assert bank.sources == list(reference.frame_sources)
        np.testing.assert_array_equal(bank.ids, reference.ids)


def test_branch_redraws_memory_and_same_generator_replays(tiny):
    spec = _spec(tiny, 2, 300, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(12, seed=4)
    policy = _policy(log_kappa=-6.0)            # nearly uniform drops: branches disagree quickly
    with torch.no_grad():
        main = Stream(tiny, spec, layout, embeds, positions, RetentionControl(policy, 'sample'))
        generator = torch.Generator().manual_seed(1)
        for _ in range(5):
            main.group(generator)
        snapshot_bank = copy.deepcopy(main.state.bank.export())
        replay, other = main.fork(), main.fork()
        g_replay = torch.Generator().manual_seed(0)
        g_replay.set_state(generator.get_state())
        main_out = torch.cat([main.group(generator) for _ in range(7)], 1)
        assert replay.state.bank.export() == snapshot_bank            # snapshot untouched by main
        replay_out = torch.cat([replay.group(g_replay) for _ in range(7)], 1)
        drawn = []
        for seed in range(6):
            branch = other.fork()
            g = torch.Generator().manual_seed(100 + seed)
            for _ in range(7):
                branch.group(g)
            drawn.append(tuple(d['drop'] for d in branch.state.bank.decisions))
    assert torch.equal(main_out, replay_out)
    assert main.state.bank.decisions == replay.state.bank.decisions
    assert len(set(drawn)) > 1


def test_pending_decision_entropy_is_the_drawn_decisions_entropy(tiny):
    spec = _spec(tiny, 2, 600, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(9, seed=6)
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        stream = Stream(tiny, spec, layout, embeds, positions, RetentionControl(_policy(), 'sample'))
        checked = 0
        for t in range(layout.groups):
            pending = stream.state.pending_decision_entropy(t)
            before = len(stream.state.bank.decisions)
            stream.group(generator)
            made = len(stream.state.bank.decisions) - before
            assert (pending is None) == (made == 0)
            if made:
                assert abs(stream.state.bank.decisions[-1]['entropy'] - pending) < 1e-12
                checked += 1
    assert checked == layout.groups - spec.recent_groups - 2
