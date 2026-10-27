"""Token-level DAgger decisions: greedy decode parity, forced actions, STOP interception."""
import unittest

from qwen_vl.contracts import ACTIONS
from qwen_vl.dagger.policy_step import MAX_RESPONSE_TOKENS, ResponseLayout, decode_response, forced_response

EOS, SEP = 99, [5]
# MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP; the two turns share their first token, as BPE may.
IDS = [[10, 11, EOS], [20, 21, EOS], [20, 22, EOS], [30, EOS]]
FORBIDDEN = {97, 98}
TEXT = {(10, 11): 'MOVE_FORWARD', (20, 21): 'TURN_LEFT', (20, 22): 'TURN_RIGHT', (30,): 'STOP'}


class FakePolicy:
    """Scripted greedy tokens; ``committed`` mirrors what reaches the streaming memory."""

    def __init__(self, script):
        self.script, self.committed, self.reads = list(script), [], 0

    def next_token(self):
        self.reads += 1
        return self.script.pop(0)

    def append_token(self, token):
        self.committed.append(token)

    def append_close(self):
        self.committed.extend([EOS] + SEP)

    @staticmethod
    def decode_text(ids):
        return TEXT.get(tuple(ids), 'garbage')

    def decode(self, layout, intercept_with=None):
        return decode_response(layout, self.next_token, self.append_token, self.append_close,
                               self.decode_text, intercept_with)


class PolicyStepTests(unittest.TestCase):
    def setUp(self):
        self.layout = ResponseLayout(IDS, EOS, SEP, FORBIDDEN)

    def test_greedy_decode_commits_exactly_the_generated_tokens(self):
        for cls, ids in enumerate(IDS):
            policy = FakePolicy(ids)
            result = policy.decode(self.layout)
            self.assertEqual(result['class_id'], cls)
            self.assertEqual(result['source'], 'model')
            self.assertEqual(policy.committed, ids + SEP)
            self.assertEqual(result['decode_tokens'], len(ids))
            self.assertEqual(policy.reads, len(ids))

    def test_forced_action_commits_what_greedy_decoding_of_it_would(self):
        for cls, ids in enumerate(IDS):
            forced, decoded = FakePolicy([]), FakePolicy(ids)
            result = forced_response(self.layout, cls, forced.append_token, forced.append_close)
            decoded.decode(self.layout)
            self.assertEqual(forced.committed, decoded.committed)
            self.assertEqual((result['class_id'], result['source'], forced.reads), (cls, 'forced', 0))

    def test_policy_stop_is_intercepted_before_any_token_is_committed(self):
        policy = FakePolicy([30, EOS])
        result = policy.decode(self.layout, intercept_with=0)
        self.assertEqual(result['source'], 'stop_intercepted')
        self.assertEqual(result['class_id'], 0)
        self.assertEqual(policy.committed, IDS[0] + SEP)      # the expert's action, nothing of STOP
        self.assertEqual(policy.reads, 1)

    def test_interception_leaves_other_actions_to_the_policy(self):
        policy = FakePolicy([20, 21, EOS])
        result = policy.decode(self.layout, intercept_with=0)
        self.assertEqual((result['class_id'], result['source']), (1, 'model'))
        self.assertEqual(policy.committed, IDS[1] + SEP)

    def test_without_interception_the_policy_may_stop(self):
        policy = FakePolicy([30, EOS])
        self.assertEqual(policy.decode(self.layout)['class_id'], ACTIONS.index('STOP'))

    def test_failures_match_the_evaluator(self):
        with self.assertRaisesRegex(ValueError, 'Forbidden'):
            FakePolicy([98]).decode(self.layout)
        policy = FakePolicy([10] * MAX_RESPONSE_TOKENS)
        with self.assertRaisesRegex(ValueError, 'Response limit'):
            policy.decode(self.layout)
        self.assertEqual(len(policy.committed), MAX_RESPONSE_TOKENS)
        with self.assertRaisesRegex(ValueError, 'Invalid navigation response'):
            FakePolicy([10, EOS]).decode(self.layout)
        with self.assertRaisesRegex(ValueError, 'Invalid navigation response'):
            FakePolicy([EOS]).decode(self.layout)
        for bad in (-1, 4, None, 1.0, True):
            with self.assertRaises(ValueError):
                forced_response(self.layout, bad, None, None)
        with self.assertRaises(ValueError):
            FakePolicy([10, 11, EOS]).decode(self.layout, intercept_with=7)

    def test_layout_rejects_ambiguous_or_malformed_actions(self):
        cases = [
            ([[30, 11, EOS], IDS[1], IDS[2], IDS[3]], FORBIDDEN),      # STOP first token shared
            ([[10, EOS, 11, EOS], IDS[1], IDS[2], IDS[3]], FORBIDDEN),  # terminator inside
            ([[10, 11], IDS[1], IDS[2], IDS[3]], FORBIDDEN),            # no terminator
            ([IDS[1], IDS[1], IDS[2], IDS[3]], FORBIDDEN),              # duplicate action
            (IDS[:3], FORBIDDEN),                                       # missing action
            ([[98, 11, EOS], IDS[1], IDS[2], IDS[3]], FORBIDDEN),      # structural token
            (IDS, FORBIDDEN | {EOS}),                                   # terminator forbidden
            ([[10] * MAX_RESPONSE_TOKENS + [EOS], IDS[1], IDS[2], IDS[3]], FORBIDDEN),
        ]
        for ids, forbidden in cases:
            with self.assertRaises(ValueError):
                ResponseLayout(ids, EOS, SEP, forbidden)


if __name__ == '__main__':
    unittest.main()
