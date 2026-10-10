"""Stochastic streaming policy for RL rollouts.

``SamplingSession.sample_step`` is the exact twin of ``StreamSession.observe``'s
text decoder, except that each response token is drawn from the tempered
full-vocabulary softmax (as verl/vLLM rollouts do) instead of the argmax.
Validity follows the evaluator: a response must end with ``<|im_end|>`` within
16 tokens, contain no other special token, and decode to one canonical action.
An invalid response is a format error (ActiveVLN ends the episode with reward 0).
"""
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


def tempered_token(logits, temperature, generator):
    """Sample one token; return (token, log-prob, entropy) under softmax(logits/T)."""
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    token = int(torch.multinomial(logp.exp(), 1, generator=generator))
    entropy = float(-(logp.exp() * logp).sum())
    return token, float(logp[token]), entropy


class SamplingSession(StreamSession):
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
        self._append_with_role(prefix, 0)
        self.prefix_length = self.positions.logical_token_count
        self.valid = True

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
        hidden = self._append_with_role(block, 1)
        forbidden = set(self.serializer.tokenizer.all_special_ids) - {self.serializer.eos}
        tokens, logps, entropies, greedy = [], [], [], []
        result = dict(valid=False, class_id=None, complete=False)
        for _ in range(MAX_RESPONSE_TOKENS):
            logits = self.model.backbone.lm_head(hidden[0, -1])
            greedy.append(int(logits.argmax()))
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
                        self.serializer.text_block([token] + self.serializer.separator), 2, complete=True)
                else:
                    result['error'] = f'Invalid navigation response: {text!r}'
                break
            if token in forbidden:
                result['error'] = f'Forbidden structural token {token}'
                break
            hidden = self._append_with_role(self.serializer.text_block([token]), 2)
        else:
            result['error'] = 'Response limit reached without assistant terminator'
        if not result['valid']:
            # The episode ends here; the session cannot accept further steps.
            self.valid = False
        elif self.cfg['memory']['mode'] == 'window8':
            self.resident.append((step_id, self.positions.logical_token_count - start))
        self.next_step += 1
        result.update(tokens=tokens, logps=logps, entropies=entropies, greedy_tokens=greedy,
                      retained_kv_tokens=kv_length(self.cache))
        return result

