"""Audit a trained Window8 text DualLane export before closed-loop evaluation.

The repeated RGB fixture checks numerical/state behavior, not navigation quality.
Fixed-history replay controls decisions only; the second replay uses the actual
greedy vocabulary head. Neither replay changes trained parameters or the policy.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from qwen_vl.contracts import ACTIONS
from qwen_vl.stream.cache import assert_state_dtypes, kv_length
from qwen_vl.stream.session import StreamSession
from qwen_vl.train.vln_runtime import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    model.eval()
    cfg = model.navigation_config
    spec = model.step_lane_spec
    assert serializer.mode == 'qwen_text'
    assert cfg['memory']['mode'] == 'window8'
    assert cfg['memory']['kv_window_steps_including_current'] == 8
    assert cfg['observations']['append_action_tokens'] is True
    assert spec.layers == (16, 20, 24, 28) and spec.clock_mode == 'step_end'
    parameters = {n: p for n, p in model.named_parameters() if '.step_lane.' in n}
    count = sum(p.numel() for p in parameters.values())
    assert count == 21_054_000
    output_norms = {n: float(p.detach().float().norm()) for n, p in parameters.items()
                    if n.endswith('out_proj.weight')}
    assert len(output_norms) == 4 and all(v > 0 for v in output_norms.values())
    actions = (list(ACTIONS[:3]) * 6)[:16] + ['STOP']
    episode = dict(
        episode_uid='text-dual-audit', dataset='r2r', dataset_version='audit',
        split='val_unseen', scene_id='audit', episode_id='audit', instruction_id='audit',
        instruction='Walk forward, then turn left and stop.',
        observation_action_alignment='observation_before_action',
        steps=[dict(step_id=i, rgb_path=str(Path(args.image).resolve()),
                    action_name=action, is_valid=True) for i, action in enumerate(actions)],
    )
    device = next(model.parameters()).device
    data = {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in serializer.encode_episode(episode).items()}
    targets = data['input_ids'][0, data['response_target_positions']].tolist()
    scores, events, handles = [], [], []
    observation_max = {i: 0.0 for i in spec.layers}
    head = model.backbone.lm_head
    original = head.forward
    control = [False]
    cursor = [0]

    def capture(hidden):
        value = original(hidden)
        assert value.ndim == 1 and torch.isfinite(value).all()
        token = targets[cursor[0]] if control[0] else int(value.argmax())
        cursor[0] += 1
        scores.append(value.float().clone())
        events.append(('decision', token))
        if control[0]:
            result = torch.full_like(value, -10000.)
            result[token] = 10000.
            return result
        return value

    def hook(index):
        def record(module, positional, keyword, output):
            _, roles, initial = positional
            values = roles.flatten().tolist()
            writers = values.count(3)
            if writers:
                assert writers == 1 and values == [2, 3], values
            else:
                torch.testing.assert_close(output[1], initial, atol=0, rtol=0)
            if values[0] == 1:
                observation_max[index] = max(observation_max[index],
                                            float(output[0].float().abs().max()))
            events.append(('append', index, writers))
        return record

    def check_step(session, uid, step, image):
        events.clear()
        result = session.observe(uid, step, image)
        assert result['response_text'] == ACTIONS[result['class_id']]
        assert result['response_token_ids'] == serializer.action_ids[result['class_id']]
        eos = events.index(('decision', serializer.eos))
        writes = [i for i, e in enumerate(events) if e[0] == 'append' and e[2]]
        assert len(writes) == len(spec.layers) and all(i > eos for i in writes)
        cache = session.step_lane_cache
        assert set(cache._processed_tokens.values()) == {session.positions.logical_token_count}
        assert all(v.dtype == torch.float32 and not v.requires_grad for v in cache._states.values())
        assert sum(v.numel() * v.element_size() for v in cache._states.values()) == 1_048_576
        assert_state_dtypes(session.cache)
        assert len(session.resident) == min(step + 1, 8)
        assert [i for i, _ in session.resident] == list(range(max(0, step - 7), step + 1))
        assert kv_length(session.cache) == session.prefix_length + sum(n for _, n in session.resident)
        before = {i: v.clone() for i, v in cache._states.items()}
        events.clear()
        assert session.observe(uid, step, None) is result
        assert not events
        assert all(torch.equal(before[i], cache._states[i]) for i in before)
        return result

    with torch.inference_mode(), Image.open(args.image) as image:
        offline = model(**data)['logits'].float()
        del data
        for index in spec.layers:
            lane = model.backbone.model.language_model.layers[index].linear_attn.step_lane
            handles.append(lane.register_forward_hook(hook(index), with_kwargs=True))
        try:
            head.forward = capture
            session = StreamSession(model, serializer)
            session.reset(episode['episode_uid'], episode['instruction'])
            assert all(torch.count_nonzero(v) == 0 for v in session.step_lane_cache._states.values())
            control[0] = True
            controlled_results = []
            for step, action in enumerate(actions):
                result = check_step(session, episode['episode_uid'], step, image)
                assert result['action_name'] == action
                controlled_results.append(result)
            assert cursor[0] == len(targets)
            streamed = torch.stack(scores)
            diff = (streamed - offline).abs()
            # Predeclared branch tolerances; do not tune to measured results.
            torch.testing.assert_close(streamed, offline, atol=.75, rtol=.03)
            rms = float(diff.square().mean().sqrt())
            assert rms <= .15, rms
            comparison = dict(max_abs=float(diff.max()), rms=rms,
                              top1_matches=int((streamed.argmax(-1) == offline.argmax(-1)).sum()),
                              response_tokens=len(targets))
            del offline, streamed, diff
            scores.clear(); cursor[0] = 0; control[0] = False
            session.reset('greedy-audit', episode['instruction'])
            greedy_results = []
            for step in range(17):
                result = check_step(session, 'greedy-audit', step, image)
                if step == 0:
                    first_scores = torch.stack(scores).clone()
                    first_result = result
                greedy_results.append(result)
            assert all(torch.count_nonzero(v) > 0 for v in session.step_lane_cache._states.values())
            assert all(v > 0 for v in observation_max.values())
            scores.clear(); cursor[0] = 0
            session.reset('reset-audit', episode['instruction'])
            assert all(torch.count_nonzero(v) == 0 for v in session.step_lane_cache._states.values())
            reset_result = check_step(session, 'reset-audit', 0, image)
            assert reset_result['response_token_ids'] == first_result['response_token_ids']
            torch.testing.assert_close(torch.stack(scores), first_scores, atol=0, rtol=0)
        finally:
            head.forward = original
            for handle in handles:
                handle.remove()
    report = dict(
        passed=True, checkpoint=str(Path(args.checkpoint).resolve()),
        fixture='17 repeated real RGB frames; numerical/lifecycle checks, not navigation SR',
        step_lane=spec.to_dict(), parameter_count=count, output_projection_norms=output_norms,
        state_bytes=1_048_576, state_dtype='torch.float32', action_history=True,
        nonwriter_state_unchanged=True, one_writer_per_layer_after_eos=True,
        window8_eviction_checked_at_steps=[8, 16], retry_unchanged=True,
        reset_first_scores_exact=True, observation_lane_output_max_abs=observation_max,
        offline_stream=comparison, controlled_results=controlled_results,
        unmodified_greedy_results=greedy_results,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('results')}, indent=2))


if __name__ == '__main__':
    main()
