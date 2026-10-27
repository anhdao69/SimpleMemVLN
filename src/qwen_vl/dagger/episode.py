"""One DAgger collection episode against abstract world, model and expert objects.

``world``: ``reset() -> rgb``, ``step(habitat_action) -> rgb | None`` (None once the episode is
over), ``distance_to_goal() -> float``, ``position() -> [x, y, z]``.
``model``: ``reset(uid, instruction)``, ``step(uid, step_id, rgb, forced_class, intercept_with)``
returning at least ``class_id`` and ``source``.
``expert``: ``ReferencePathExpert`` (``advance``, ``action``, ``index``, ``waypoints``).

Recorded data follows the training contract ``observation_before_action``: frame ``t`` is the
observation the executed action ``t`` was chosen from, and the last frame carries the expert's
STOP at the goal.  The policy saw every recorded frame, including frames of expert takeovers
(forced decisions still stream the observation and the executed action into its memory).
"""
from qwen_vl.contracts import ACTIONS, HABITAT_IDS
from qwen_vl.dagger.expert import HABITAT_STOP, ExpertError
from qwen_vl.dagger.rules import STOP, EscapedStop

MAX_ACTIONS = 500                 # Habitat R2R-VLNCE episode cap used by the evaluator


class EpisodeAbort(RuntimeError):
    """The episode cannot produce a valid trajectory; it is journaled and skipped."""


def collect_episode(world, model, expert, controller, uid, instruction, max_actions=MAX_ACTIONS):
    rgb = world.reset()
    start_distance = float(world.distance_to_goal())
    positions = [world.position()]
    frames, actions, steps = [], [], []
    model.reset(uid, instruction)
    final_reached, failure = False, None
    try:
        for t in range(max_actions):
            if expert.advance():
                controller.waypoint_reached()
            expert_action = expert.action()
            distance = float(world.distance_to_goal())
            frames.append(rgb)
            if expert_action == HABITAT_STOP:          # final reference point reached
                final_reached = True
                actions.append(STOP)
                steps.append(dict(source='expert_goal', expert='STOP', executed='STOP',
                                  distance_to_goal=distance, waypoint=expert.index))
                world.step(HABITAT_STOP)
                break
            expert_class = HABITAT_IDS.index(expert_action)
            response = model.step(uid, t, rgb, **controller.request(expert_class))
            executed = response['class_id']
            if type(executed) is not int or executed not in range(len(ACTIONS)):
                raise EpisodeAbort(f'Invalid model class {executed!r}')
            label = controller.record(response['source'], executed, expert_class, distance)
            actions.append(executed)
            steps.append(dict(source=label, expert=ACTIONS[expert_class], executed=ACTIONS[executed],
                              distance_to_goal=distance, waypoint=expert.index))
            rgb = world.step(HABITAT_IDS[executed])
            positions.append(world.position())
            if rgb is None:
                failure = 'action_cap'
                break
        else:
            failure = 'action_cap'
    except ExpertError as exc:
        failure = f'expert_error: {exc}'
    except (EpisodeAbort, EscapedStop) as exc:
        failure = f'abort: {exc}'
    if final_reached and len(frames) != len(actions):
        raise RuntimeError('Frame/action alignment broken')
    return dict(frames=frames, actions=actions, steps=steps, positions=positions,
                start_distance=start_distance, final_distance=float(world.distance_to_goal()),
                final_reached=final_reached and failure is None, failure=failure)
