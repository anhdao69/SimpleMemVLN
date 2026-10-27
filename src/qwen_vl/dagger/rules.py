"""Deviation-based DAgger rules (StreamVLN / JanusVLN, p = 0), as a testable state machine.

Reference implementations read for this module (both identical in their control logic):
``StreamVLN/streamvln/streamvln_dagger.py`` and ``JanusVLN/src/dagger.py``.  With ``dagger_p = 0``
the policy acts; the expert walks the instruction's reference path waypoint by waypoint
(ShortestPathFollower, radius 1.8 m, final point 0.25 m) and

1. replaces every policy STOP (the episode only ends when the expert reaches the final point);
   a STOP is a deviation (StreamVLN compares it with the expert action), and one issued >= 3 m
   from the goal additionally starts an expert takeover;
2. counts decisions where the policy's action differs from the expert's; a takeover starts when
   errors / (oracle actions per reference segment) > 0.8 or errors > 12 (the count then resets);
3. ends a takeover when the next reference waypoint is reached;
4. keeps an episode only if it ends within 0.5 m of the goal and its path efficiency
   PL = start geodesic / max(start geodesic, travelled length) is below 0.93 after a takeover,
   or below 0.85 otherwise -- near-oracle episodes duplicate the oracle data and are dropped.

Deliberate differences (``plans/dagger_s8n.md`` section 3):

a. StreamVLN replaces the action of the decision that crosses the error threshold.  A streaming
   text policy has already committed that decision's response tokens to its memory (only STOP is
   visible from the first token), so the takeover starts at the next decision; the deviating
   action is executed, recorded and labelled ``model``.  A deferred takeover counts (and sets
   ``model_failed``) only once at least one expert decision was forced; one cancelled by reaching
   the waypoint first is not counted.
b. Waypoints are advanced before the policy action is compared with the expert's.  StreamVLN
   compares first, which scores a spurious error whenever the agent enters a waypoint radius, and
   lets the expert finish that decision; here control returns to the policy at that decision.
c. A follower failure raises (``stop_on_error=False``) and drops the episode instead of being read
   as STOP, which would silently skip waypoints.
d. Saved episodes must contain a correction (``save_clean=False``): an episode whose policy agreed
   with the expert at every decision is an oracle-like duplicate even when a winding reference path
   puts its PL below 0.85.
"""
from dataclasses import asdict, dataclass

from qwen_vl.contracts import ACTIONS

STOP = ACTIONS.index('STOP')


class EscapedStop(RuntimeError):
    """A policy STOP was decoded without interception (it cannot be executed mid-episode)."""


@dataclass(frozen=True)
class DaggerRules:
    waypoint_radius: float = 1.8           # intermediate reference points (StreamVLN agent radius)
    goal_radius: float = 0.25              # final reference point (StreamVLN GOAL_RADIUS)
    far_stop_distance: float = 3.0         # policy STOP at least this far from the goal -> takeover
    error_ratio: float = 0.8               # takeover when errors / segment_actions exceeds this
    max_errors: int = 12                   # ... or when errors exceed this
    default_oracle_length: int = 60        # StreamVLN DEFAULT_EPISODE_LENGTH without an oracle record
    final_distance_max: float = 0.5        # StreamVLN MIDGOAL_RADIUS: saved episodes end this close
    save_pl_with_takeover: float = 0.93    # StreamVLN RELATIVE_PATH_LENGTH_THRESHOLD
    save_pl_without_takeover: float = 0.85  # StreamVLN SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD
    max_saved_steps: int = 183             # longest proven training episode (RxR, 58,290 tokens)
    save_clean: bool = False               # keep episodes without any deviation (StreamVLN: True)

    def __post_init__(self):
        for name in ('waypoint_radius', 'goal_radius', 'far_stop_distance', 'error_ratio',
                     'final_distance_max', 'save_pl_with_takeover', 'save_pl_without_takeover'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f'{name} must be positive')
        for name in ('max_errors', 'default_oracle_length', 'max_saved_steps'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.save_clean) is not bool:
            raise ValueError('save_clean must be a boolean')
        if self.save_pl_without_takeover > self.save_pl_with_takeover or self.save_pl_with_takeover > 1:
            raise ValueError('Expected 0 < save_pl_without_takeover <= save_pl_with_takeover <= 1')

    def to_dict(self):
        return asdict(self)


class EpisodeController:
    """Who acts at each decision, and the episode's deviation statistics."""

    def __init__(self, rules, oracle_length, waypoints):
        if type(waypoints) is not int or waypoints < 2:
            raise ValueError('A reference path needs at least a start and a goal')
        if type(oracle_length) is not int or oracle_length < 1:
            raise ValueError('Oracle length must be a positive integer')
        self.rules = rules
        # StreamVLN: max(1, int(ref_actions_len / (len(ref_path) - 1)))
        self.segment_actions = max(1, int(oracle_length / (waypoints - 1)))
        self.control = 'model'
        self.errors = 0
        self._pending = None              # kind of a takeover that has not forced a decision yet
        self.takeovers = self.error_takeovers = self.far_stop_takeovers = self.cancelled_takeovers = 0
        self.stop_overrides = self.model_steps = self.expert_steps = self.deviations = 0

    def waypoint_reached(self):
        """A reference waypoint was reached: a takeover ends; the error count persists (StreamVLN)."""
        if self._pending is not None:
            self.cancelled_takeovers += 1
            self._pending = None
        self.control = 'model'

    def request(self, expert_class):
        """Model-process arguments for the next decision."""
        if type(expert_class) is not int or expert_class not in range(len(ACTIONS)) or expert_class == STOP:
            raise ValueError('Expert STOP is executed by the collector, not requested from the model')
        if self.control == 'expert':
            return dict(forced_class=expert_class, intercept_with=None)
        return dict(forced_class=None, intercept_with=expert_class)

    def record(self, source, executed, expert_class, distance_to_goal):
        """Account for one completed decision; returns the step's provenance label."""
        if source == 'forced':
            if self.control != 'expert' or executed != expert_class:
                raise RuntimeError('Forced decision outside a takeover or not the expert action')
            self.expert_steps += 1
            if self._pending is not None:
                self._count(self._pending)
                self._pending = None
            return 'expert'
        if self.control != 'model':
            raise RuntimeError(f'Policy decision {source!r} during an expert takeover')
        if source == 'stop_intercepted':
            if executed != expert_class:
                raise RuntimeError('An intercepted STOP must execute the expert action')
            self.stop_overrides += 1
            self.expert_steps += 1
            self._deviation()
            if distance_to_goal >= self.rules.far_stop_distance:
                self._takeover('far_stop', realized=True)    # this decision already ran the expert
            else:
                self._check_errors()
            return 'stop_override'
        if source != 'model':
            raise ValueError(f'Unknown decision source {source!r}')
        if executed == STOP:
            raise EscapedStop('A policy STOP escaped interception and cannot be executed')
        self.model_steps += 1
        if executed != expert_class:
            self._deviation()
            self._check_errors()
        return 'model'

    def _deviation(self):
        self.errors += 1
        self.deviations += 1

    def _check_errors(self):
        if (self.errors / self.segment_actions > self.rules.error_ratio
                or self.errors > self.rules.max_errors):
            self._takeover('error', realized=False)          # starts at the next decision

    def _takeover(self, kind, realized):
        self.control = 'expert'
        self.errors = 0
        if realized:
            self._count(kind)
        else:
            self._pending = kind

    def _count(self, kind):
        self.takeovers += 1
        if kind == 'error':
            self.error_takeovers += 1
        else:
            self.far_stop_takeovers += 1

    @property
    def model_failed(self):
        """StreamVLN's ``not model_success``: at least one takeover forced the expert."""
        return self.takeovers > 0

    @property
    def clean(self):
        """The policy agreed with the expert at every decision it made."""
        return self.deviations == 0 and self.takeovers == 0

    def summary(self):
        return dict(segment_actions=self.segment_actions, takeovers=self.takeovers,
                    error_takeovers=self.error_takeovers, far_stop_takeovers=self.far_stop_takeovers,
                    cancelled_takeovers=self.cancelled_takeovers, stop_overrides=self.stop_overrides,
                    model_steps=self.model_steps, expert_steps=self.expert_steps,
                    deviations=self.deviations, model_failed=self.model_failed, clean=self.clean)


def path_length(positions):
    """Sum of straight-line displacements between consecutive agent positions (StreamVLN PL)."""
    total = 0.0
    for a, b in zip(positions, positions[1:]):
        total += sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5
    return total


def path_efficiency(start_distance, travelled):
    """StreamVLN ``pl`` measure: start geodesic / max(start geodesic, travelled length).

    A degenerate episode (no distance to cover, no movement) is 1.0, so it is never kept as a
    detour (StreamVLN's 0/0 is NaN, which no threshold keeps either).
    """
    denominator = max(float(start_distance), float(travelled))
    return 1.0 if denominator <= 0 else float(start_distance) / denominator


def save_decision(rules, final_reached, final_distance, steps, efficiency, model_failed, clean=False):
    """(keep, reason) for one collected episode."""
    if not final_reached:
        return False, 'goal_not_reached'
    if not final_distance < rules.final_distance_max:
        return False, 'final_distance'
    if steps > rules.max_saved_steps:
        return False, 'too_long'
    if model_failed and efficiency < rules.save_pl_with_takeover:
        keep, reason = True, 'takeover'
    elif efficiency < rules.save_pl_without_takeover:
        keep, reason = True, 'detour'
    else:
        return False, 'near_oracle'
    if clean and not rules.save_clean:
        return False, 'clean'
    return keep, reason
