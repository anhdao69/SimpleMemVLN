"""Token-level control of one DAgger decision (no torch, no model).

``StreamSession.observe`` decodes the assistant response greedily and commits every response
token to the streaming memory as it is produced.  DAgger needs two more outcomes for the same
observation, and both keep the streaming history equal to the action that is executed:

* forced: the expert acts.  The canonical tokens of its action are committed exactly as the
  serializer writes a training target: the action tokens one by one (score segment kind
  ``response_token``, as greedy decoding appends them), then ``<|im_end|>`` + separator
  (``response_close``).
* intercepted STOP: the policy's first response token starts ``STOP``, which would end the
  Habitat episode.  That token is unique among the four canonical actions, so the decision is
  visible before any policy token is committed; the expert's action is committed instead and no
  rollback of the streaming state is needed.

Callbacks keep this module free of torch: ``next_token()`` returns the argmax at the latest
hidden state, ``append_token(t)`` commits one response token (and advances that hidden state),
``append_close()`` commits ``<|im_end|>`` + separator, and ``decode_text(ids)`` detokenizes.
"""
from qwen_vl.contracts import ACTIONS, parse_action

MAX_RESPONSE_TOKENS = 16          # StreamSession.observe / rl.sampling budget, EOS included
STOP = ACTIONS.index('STOP')


class ResponseLayout:
    """Canonical response tokens of the serializer (``EpisodeSerializer.action_ids``)."""

    def __init__(self, action_ids, eos, separator, forbidden):
        action_ids = [[int(token) for token in ids] for ids in action_ids]
        eos = int(eos)
        forbidden = frozenset(int(token) for token in forbidden)
        if len(action_ids) != len(ACTIONS):
            raise ValueError('One canonical token sequence per action is required')
        if eos in forbidden:
            raise ValueError('<|im_end|> must not be a forbidden token')
        for name, ids in zip(ACTIONS, action_ids):
            if len(ids) < 2 or ids[-1] != eos or eos in ids[:-1]:
                raise ValueError(f'{name}: canonical tokens must be followed by exactly one <|im_end|>')
            if len(ids) > MAX_RESPONSE_TOKENS:
                raise ValueError(f'{name}: canonical response exceeds the decode budget')
            if any(token in forbidden for token in ids[:-1]):
                raise ValueError(f'{name}: canonical response contains a structural token')
        if len({tuple(ids) for ids in action_ids}) != len(ACTIONS):
            raise ValueError('Canonical actions must be distinct')
        self.action_ids, self.eos, self.forbidden = action_ids, eos, forbidden
        self.separator = [int(token) for token in separator]
        self.stop_first = action_ids[STOP][0]
        shared = [ACTIONS[c] for c, ids in enumerate(action_ids) if c != STOP and ids[0] == self.stop_first]
        if shared:
            raise ValueError(f'STOP shares its first token with {shared}; interception is undefined')


def _check_class(class_id):
    if type(class_id) is not int or class_id not in range(len(ACTIONS)):
        raise ValueError(f'Invalid action class {class_id!r}')


def forced_response(layout, class_id, append_token, append_close, source='forced'):
    """Commit the canonical response of ``class_id`` without consulting the policy."""
    _check_class(class_id)
    ids = layout.action_ids[class_id]
    for token in ids[:-1]:
        append_token(token)
    append_close()
    return dict(class_id=class_id, source=source, response_token_ids=list(ids),
                response_text=ACTIONS[class_id], decode_tokens=0)


def decode_response(layout, next_token, append_token, append_close, decode_text, intercept_with=None):
    """Greedy decode identical to ``StreamSession.observe``; optionally intercept a policy STOP.

    With ``intercept_with`` set and a first token that starts STOP, the canonical response of
    ``intercept_with`` is committed instead (``source='stop_intercepted'``).  Without it, the
    policy's own STOP is decoded and returned like any other action.
    """
    if intercept_with is not None:
        _check_class(intercept_with)
    token = next_token()
    if intercept_with is not None and token == layout.stop_first:
        result = forced_response(layout, intercept_with, append_token, append_close,
                                 source='stop_intercepted')
        result['proposed_first_token'] = token
        return result
    response = []
    for index in range(MAX_RESPONSE_TOKENS):
        if index:
            token = next_token()
        if token == layout.eos:
            append_close()
            text = decode_text(response)
            return dict(class_id=parse_action(text), source='model', response_token_ids=response + [token],
                        response_text=text, decode_tokens=len(response) + 1)
        if token in layout.forbidden:
            raise ValueError(f'Forbidden structural token {token}')
        response.append(token)
        append_token(token)
    raise ValueError('Response limit reached without assistant terminator')
