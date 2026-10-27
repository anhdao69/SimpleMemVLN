"""Model-side streaming session for DAgger collection (model interpreter, torch).

``DaggerSession.step`` is ``StreamSession.observe`` for text-action checkpoints with two extra
outcomes (``qwen_vl.dagger.policy_step``): a forced expert action, and an intercepted policy STOP.
Every path appends exactly the blocks ``observe`` appends for the same tokens, in the same order
and with the same score-segment kinds, so the streaming memory (GDN/conv state, full-attention
KV, S8N/M5 episodic bank) after a forced action equals the memory after a policy that generated
that action.  ``scripts/dagger/check_session.py`` verifies this bitwise on a real checkpoint.
"""
import time

import torch

from qwen_vl.contracts import ACTIONS, HABITAT_IDS
from qwen_vl.dagger.policy_step import ResponseLayout, decode_response, forced_response
from qwen_vl.stream.cache import kv_length
from qwen_vl.stream.session import StreamSession


class DaggerSession(StreamSession):
    def __init__(self, model, serializer):
        super().__init__(model, serializer)
        if serializer.mode != 'qwen_text':
            raise ValueError('DAgger collection is defined for text-action (qwen_text) checkpoints')
        if getattr(model, 'step_lane_spec', None) is not None:
            raise ValueError('DAgger collection does not support step-lane checkpoints')
        forbidden = set(serializer.tokenizer.all_special_ids) - {serializer.eos}
        self.layout = ResponseLayout(serializer.action_ids, serializer.eos, serializer.separator, forbidden)

    @torch.inference_mode()
    def step(self, episode_uid, step_id, rgb, forced_class=None, intercept_with=None):
        if not self.valid or self.busy or episode_uid != self.episode_uid:
            raise ValueError('Invalid, busy, or mismatched session')
        if step_id != self.next_step:
            raise ValueError('Out-of-order observation')
        if forced_class is not None and intercept_with is not None:
            raise ValueError('A decision is either forced or decoded, not both')
        for value in (forced_class, intercept_with):
            if value is not None and (type(value) is not int or value not in range(len(ACTIONS))):
                raise ValueError(f'Invalid action class {value!r}')
        self.busy = True
        self.model._stage_timer = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            block = self.serializer.encode_observation(rgb, step_id)
            self._begin_step(step_id, block)
            start = self.positions.logical_token_count
            if self.episodic is not None:
                image = block['input_ids'][0].eq(self.serializer.image_id).nonzero().reshape(-1)
                self.episodic.open_group(step_id, start, start + int(image[0]))
            current = {'hidden': self._append_with_role(block, 1, kind='observation')}

            def next_token():
                return int(self.model.backbone.lm_head(current['hidden'][0, -1]).argmax())

            def append_token(token):
                current['hidden'] = self._append_with_role(
                    self.serializer.text_block([token]), 2, kind='response_token')

            def append_close():
                self._append_with_role(
                    self.serializer.text_block([self.serializer.eos] + self.serializer.separator),
                    2, complete=True, kind='response_close')

            def decode_text(tokens):
                return self.serializer.tokenizer.decode(
                    tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

            if forced_class is not None:
                result = forced_response(self.layout, forced_class, append_token, append_close)
            else:
                result = decode_response(self.layout, next_token, append_token, append_close,
                                         decode_text, intercept_with)
            if self.cfg['memory']['mode'] == 'window8':
                self.resident.append((step_id, self.positions.logical_token_count - start))
            if self.episodic is not None:
                self.episodic.close_group(self.positions.logical_token_count)
            torch.cuda.synchronize()
            cls = result['class_id']
            result.update(status='ok', episode_uid=episode_uid, step_id=step_id,
                          action_name=ACTIONS[cls], habitat_action_id=HABITAT_IDS[cls],
                          model_seconds=time.perf_counter() - started,
                          retained_kv_tokens=kv_length(self.cache))
            self.last_result = None          # no observe()-style idempotent replay for DAgger steps
            self.next_step += 1
            return result
        except Exception:
            self.valid = False
            raise
        finally:
            self.busy = False
