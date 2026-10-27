"""Reference-path waypoint expert (StreamVLN / JanusVLN DAgger expert).

The expert follows the instruction's ``reference_path`` rather than the shortest path to the goal,
so its corrections stay consistent with the instruction (and with the oracle trajectories, which
follow the same reference path).  Waypoint ``i`` (from 1) is targeted with a follower of radius
``waypoint_radius``; the final point with ``goal_radius``.  ``advance`` skips every intermediate
waypoint the follower reports as reached and keeps the first non-reached answer; ``action``
returns that answer (STOP only at the final point).  The follower is therefore queried once per
target per decision: habitat-sim's GreedyFollower appends every answer to its action history, and
duplicated entries would defeat its thrashing (left/right oscillation) detection.

Follower objects need ``get_next_action(point) -> Habitat action id``.  ``habitat_follower_factory``
builds Habitat-Lab 0.2.4 ``ShortestPathFollower``s with ``stop_on_error=False`` when supported, so
a navmesh failure raises instead of silently reporting STOP (which would skip waypoints).
"""
HABITAT_STOP = 0
HABITAT_ACTIONS = (0, 1, 2, 3)


class ExpertError(RuntimeError):
    """The expert has no valid action (unreachable waypoint, follower failure)."""


class ReferencePathExpert:
    def __init__(self, reference_path, make_follower, rules):
        points = [[float(x) for x in point] for point in reference_path]
        if len(points) < 2 or any(len(point) != 3 for point in points):
            raise ExpertError('Reference path needs at least two 3D points')
        self.points = points
        self.index = 1
        self.final_index = len(points) - 1
        self._waypoint = make_follower(rules.waypoint_radius)
        self._goal = make_follower(rules.goal_radius)
        self._answer = None

    @property
    def waypoints(self):
        return len(self.points)

    def _next(self):
        follower = self._goal if self.index == self.final_index else self._waypoint
        try:
            action = follower.get_next_action(self.points[self.index])
        except ExpertError:
            raise
        except Exception as exc:          # habitat_sim GreedyFollowerError and friends
            raise ExpertError(f'Follower failed toward waypoint {self.index}: {type(exc).__name__}: {exc}') from exc
        if action is None:
            raise ExpertError(f'No expert action toward waypoint {self.index}')
        action = int(action)
        if action not in HABITAT_ACTIONS:
            raise ExpertError(f'Unexpected expert action {action!r}')
        return action

    def advance(self):
        """Skip reached intermediate waypoints at the current pose; returns how many were reached."""
        if self._answer is not None:
            raise ExpertError('advance() called twice without action()')
        reached = 0
        while True:
            answer = self._next()
            if answer == HABITAT_STOP and self.index < self.final_index:
                self.index += 1
                reached += 1
                continue
            self._answer = answer
            return reached

    def action(self):
        """Expert Habitat action at the current pose, as computed by ``advance`` (STOP: at the goal)."""
        if self._answer is None:
            raise ExpertError('action() requires advance() at the current pose')
        answer, self._answer = self._answer, None
        return answer


def habitat_follower_factory(sim):
    """``radius -> ShortestPathFollower`` for one Habitat simulator (Habitat interpreter only)."""
    import inspect
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    kwargs = dict(return_one_hot=False)
    if 'stop_on_error' in inspect.signature(ShortestPathFollower.__init__).parameters:
        kwargs['stop_on_error'] = False

    def make(radius):
        return ShortestPathFollower(sim=sim, goal_radius=float(radius), **kwargs)

    return make
