"""Stochastic streaming policy for RL rollouts.

``SamplingSession.sample_step`` is the exact twin of ``StreamSession.observe``'s
text decoder, except that each response token is drawn from the tempered
full-vocabulary softmax (as verl/vLLM rollouts do) instead of the argmax.
Validity follows the evaluator: a response must end with ``<|im_end|>`` within
16 tokens, contain no other special token, and decode to one canonical action.
An invalid response is a format error (ActiveVLN ends the episode with reward 0).
"""
import copy
import threading
from collections import deque

import torch

from qwen_vl.contracts import ACTIONS
from qwen_vl.stream.cache import kv_length, make_stream_cache_fp32
from qwen_vl.stream.positions import PositionLedger
from qwen_vl.stream.session import StreamSession

MAX_RESPONSE_TOKENS = 16
# The HF processor/fast tokenizer is shared by all rollout threads of a rank and is not
# safe under concurrent calls ("Already borrowed"); every serializer call takes this lock.
SERIALIZER_LOCK = threading.Lock()


def clone_cache(cache):
    """Independent copy of a stream cache: every tensor attribute of every layer is cloned."""
    out = copy.copy(cache)
    out.layers = []
    for layer in cache.layers:
        twin = copy.copy(layer)
        for name, value in vars(layer).items():
            if torch.is_tensor(value):
                setattr(twin, name, value.clone())
            elif isinstance(value, (list, tuple)) and any(torch.is_tensor(v) for v in value):
                setattr(twin, name, type(value)(v.clone() if torch.is_tensor(v) else v for v in value))
        out.layers.append(twin)
    return out


def clone_episodic(state):
    """Independent copy of an ``EpisodicSessionState`` taken between steps.

    Copies the bank selection (running top-B), resident row IDs, open recent groups with
    their source scores, and every committed-write scorer's carried FP32 state and conv
    tail. The scorers keep their reference to the (shared, read-only) GDN layer.
    """
    from qwen_vl.stream.episodic_cache import EpisodicSessionState

    if state is None:
        return None
    if type(state) is not EpisodicSessionState:
        raise ValueError(f'Snapshots are defined for EpisodicSessionState, not {type(state).__name__}')
    if state.segment_kind is not None:
        raise ValueError('Episodic snapshot inside an append')
    twin = copy.copy(state)
    twin.running = copy.deepcopy(state.running)
    twin.resident = {layer: ids.copy() for layer, ids in state.resident.items()}
    twin.groups = copy.deepcopy(state.groups)
    twin.promotions = copy.deepcopy(state.promotions)
    twin.bank = copy.deepcopy(state.bank)          # MemFork frame bank and decision record
    twin.scorers = {}
    for gdn, scorer in state.scorers.items():
        if scorer.pending:
            raise ValueError('Episodic snapshot with an unclosed response')
        copied = copy.copy(scorer)
        copied.state, copied.conv_tail, copied.pending = scorer.state.clone(), scorer.conv_tail.clone(), []
        twin.scorers[gdn] = copied
    return twin


def tempered_token(logits, temperature, generator, probe=None):
    """Sample one token; return (token, log-prob, entropy) under softmax(logits/T).

    With ``probe`` a token id, also return that token's tempered log-probability.
    """
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    token = int(torch.multinomial(logp.exp(), 1, generator=generator))
    entropy = float(-(logp.exp() * logp).sum())
    if probe is None:
        return token, float(logp[token]), entropy
    return token, float(logp[token]), entropy, float(logp[probe])


class SamplingSession(StreamSession):
    retention = None      # stream/retention.RetentionControl for MemFork rollouts; None = plain M5

    @torch.inference_mode()
    def reset(self, episode_uid, instruction):
        """StreamSession.reset for text policies, with the prefix encoded under the lock.

        It does not clear the shared ``rope_deltas`` attribute: streaming appends always pass
        explicit positions, so that attribute is never read, and concurrent sessions share it.
        """
        if getattr(self.model, 'step_lane_spec', None) is not None:
            raise ValueError('RL sampling does not support step-lane models')
        with SERIALIZER_LOCK:
            prefix = self.serializer.encode_prefix(instruction)
        self.valid = False
        self.episode_uid = episode_uid
        self.cache = make_stream_cache_fp32(self.model.config.text_config)
        self.positions = PositionLedger()
        self.resident = deque()
        self.next_step = 0
        self.last_result = None
        self.busy = False
        self.step_lane_cache = None
        self.episodic = None
        spec = getattr(self.model, 'episodic_spec', None)
        if spec is not None:
            from qwen_vl.research.episodic_contract import GDN_SWEEP_MODE
            from qwen_vl.stream.episodic_cache import EpisodicSessionState
            if spec.version == GDN_SWEEP_MODE:
                raise ValueError('RL sampling supports episodic_window_v1 (M5/M5c), not the GDN sweep')
            self.episodic = EpisodicSessionState(spec, self.model, episode_uid,
                                                 device=next(self.model.parameters()).device)
            if self.retention is not None:
                self.episodic.set_retention(self.retention)
        elif self.retention is not None:
            raise ValueError('Retention policies need an episodic_window_v1 checkpoint')
        self._append_with_role(prefix, 0, kind='prefix')
        self.prefix_length = self.positions.logical_token_count
        self.valid = True

    @torch.inference_mode()
    def snapshot(self):
        """Clone of the complete streaming state before the next observation (GDN + attention KV,
        and for M5 the episodic bank, recent groups and committed-write scorer state).

        A branch restored from it continues exactly as this session would; the memory is copied
        in O(state size), without re-encoding the history.
        """
        if not self.valid or self.step_lane_cache is not None:
            raise ValueError('Snapshot needs a valid text-policy session')
        return dict(cache=clone_cache(self.cache), positions=copy.copy(self.positions),
                    resident=deque(self.resident), next_step=self.next_step,
                    prefix_length=self.prefix_length, episode_uid=self.episode_uid,
                    episodic=clone_episodic(self.episodic))

    @torch.inference_mode()
    def restore(self, snapshot, episode_uid=None):
        """Continue from ``snapshot``; the snapshot itself stays reusable for further branches."""
        self.cache = clone_cache(snapshot['cache'])
        self.positions = copy.copy(snapshot['positions'])
        self.resident = deque(snapshot['resident'])
        self.next_step, self.prefix_length = snapshot['next_step'], snapshot['prefix_length']
        self.episode_uid = episode_uid or snapshot['episode_uid']
        # The bank keeps the root episode's identity (random-hash selectors key on it), so a
        # branch's training forward rebuilds its memory with the same UID as the root.
        self.episodic = clone_episodic(snapshot.get('episodic'))
        self.last_result, self.busy, self.step_lane_cache = None, False, None
        self.valid = True

    @torch.inference_mode()
    def prepare_step(self, step_id, generator):
        """Promotion (and the MemFork memory decision m_t) before observation ``step_id``.

        Runs before a fork snapshot, so m_t belongs to the shared history h_t of every branch at t
        and the fork rule (entropy of pi(.|h_t)) may depend on it; ``sample_step`` repeats the
        promotion as a no-op.
        """
        if not self.valid or step_id != self.next_step:
            raise ValueError('Invalid session or out-of-order observation')
        if self.episodic is not None:
            self.episodic.promote(self.cache, step_id, generator=generator)

    @torch.inference_mode()
    def sample_step(self, step_id, rgb, temperature, generator):
        if self.serializer.mode != 'qwen_text':
            raise ValueError('RL sampling is defined for text-action policies only')
        if not self.valid or step_id != self.next_step:
            raise ValueError('Invalid session or out-of-order observation')
        with SERIALIZER_LOCK:
            block = self.serializer.encode_observation(rgb, step_id)
        self._begin_step(step_id, block)
        start = self.positions.logical_token_count
        if self.episodic is not None:
            image = block['input_ids'][0].eq(self.serializer.image_id).nonzero().reshape(-1)
            self.episodic.open_group(step_id, start, start + int(image[0]))
        hidden = self._append_with_role(block, 1, kind='observation')
        forbidden = set(self.serializer.tokenizer.all_special_ids) - {self.serializer.eos}
        tokens, logps, entropies, greedy = [], [], [], []
        result = dict(valid=False, class_id=None, complete=False)
        for _ in range(MAX_RESPONSE_TOKENS):
            logits = self.model.backbone.lm_head(hidden[0, -1])
            greedy.append(int(logits.argmax()))
            if not tokens:  # decision token: also record sigma_t = pi_T(first token = STOP)
                token, logp, entropy, stop_logp = tempered_token(
                    logits, temperature, generator, probe=self.serializer.action_ids[3][0])
                result['stop_logp'] = stop_logp
            else:
                token, logp, entropy = tempered_token(logits, temperature, generator)
            tokens.append(token)
            logps.append(logp)
            entropies.append(entropy)
            if token == self.serializer.eos:
                with SERIALIZER_LOCK:
                    text = self.serializer.tokenizer.decode(
                        tokens[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                result.update(text=text, complete=True)
                if text.strip() in ACTIONS:
                    result.update(valid=True, class_id=ACTIONS.index(text.strip()))
                    self._append_with_role(
                        self.serializer.text_block([token] + self.serializer.separator), 2, complete=True,
                        kind='response_close')
                else:
                    result['error'] = f'Invalid navigation response: {text!r}'
                break
            if token in forbidden:
                result['error'] = f'Forbidden structural token {token}'
                break
            hidden = self._append_with_role(self.serializer.text_block([token]), 2, kind='response_token')
        else:
            result['error'] = 'Response limit reached without assistant terminator'
        if not result['valid']:
            # The episode ends here; the session cannot accept further steps.
            self.valid = False
        elif self.cfg['memory']['mode'] == 'window8':
            self.resident.append((step_id, self.positions.logical_token_count - start))
        elif self.episodic is not None:
            self.episodic.close_group(self.positions.logical_token_count)
        self.next_step += 1
        result.update(tokens=tokens, logps=logps, entropies=entropies, greedy_tokens=greedy,
                      retained_kv_tokens=kv_length(self.cache))
        return result

