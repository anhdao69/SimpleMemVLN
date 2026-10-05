"""Audit a trained dual-lane export before Habitat evaluation.

Repeated real RGB input is a numerical/lifecycle fixture, not an SR measurement.
No lane weights, attention mode, or feedback settings are changed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import torch

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
    spec = model.step_lane_spec
    assert spec.layers == (16, 20, 24, 28) and spec.clock_mode == 'step_end'
    assert serializer.mode == 'candidate_logits'
    assert model.navigation_config['memory']['mode'] == 'full_context'
    assert [serializer.feedback_ids(i) for i in range(4)] == [[248046, 198]] * 4
    parameters = {n: p for n, p in model.named_parameters() if '.step_lane.' in n}
    assert sum(p.numel() for p in parameters.values()) == 21_054_000
    output_norms = {n: float(p.detach().float().norm()) for n, p in parameters.items()
                    if n.endswith('out_proj.weight')}
    assert len(output_norms) == 4 and all(v > 0 for v in output_norms.values())

    episode = dict(
        episode_uid='dual-audit', dataset='r2r', dataset_version='audit', split='val_unseen',
        scene_id='audit', episode_id='audit', instruction_id='audit',
        instruction='Walk forward, then turn left and stop.',
        observation_action_alignment='observation_before_action',
        steps=[dict(step_id=i, rgb_path=str(Path(args.image).resolve()),
                    action_name=a, is_valid=True)
               for i, a in enumerate(['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP'])],
    )
    data = serializer.encode_episode(episode)
    changed = deepcopy(episode)
    for step in changed['steps'][:-1]:
        step['action_name'] = 'TURN_RIGHT'
    other = serializer.encode_episode(changed)
    for key in ('input_ids', 'step_lane_roles', 'read_positions'):
        assert torch.equal(data[key], other[key]), key
    del other
    device = next(model.parameters()).device
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
    scores, events, handles = [], [], []
    observation_output_max = {index: 0.0 for index in spec.layers}
    readout = model.action_logits

    def capture(hidden):
        result = readout(hidden)
        events.append(('decision',))
        scores.append(result.detach().clone())
        return result

    def hook(index):
        def record(module, args, kwargs, output):
            _, roles, initial = args
            final = output[1]
            values = roles.flatten().tolist()
            writers = values.count(3)
            assert writers <= 1
            if writers:
                assert values == [2, 3], values
            else:
                # Read-only prefix/observation appends cannot alter lane state.
                torch.testing.assert_close(final, initial, atol=0, rtol=0)
            if values[0] == 1:
                observation_output_max[index] = max(
                    observation_output_max[index], float(output[0].float().abs().max())
                )
            events.append(('append', index, writers, roles.numel()))
        return record

    with torch.inference_mode():
        offline = model(**data)['logits'].float()
        del data
        for index in spec.layers:
            lane = model.backbone.model.language_model.layers[index].linear_attn.step_lane
            handles.append(lane.register_forward_hook(hook(index), with_kwargs=True))
        model.action_logits = capture
        try:
            session = StreamSession(model, serializer)
            session.reset(episode['episode_uid'], episode['instruction'])
            assert all(torch.count_nonzero(v) == 0 for v in session.step_lane_cache._states.values())
            results = []
            with Image.open(args.image) as rgb:
                for step in range(4):
                    events.clear()
                    result = session.observe(episode['episode_uid'], step, rgb)
                    assert result['feedback_token_ids'] == [248046, 198]
                    assert result['generated_tokens'] == 0
                    decision = events.index(('decision',))
                    writes = [i for i, event in enumerate(events) if event[0] == 'append' and event[2]]
                    assert len(writes) == 4 and all(i > decision for i in writes)
                    cache = session.step_lane_cache
                    assert set(cache._processed_tokens.values()) == {session.positions.logical_token_count}
                    assert all(v.dtype == torch.float32 and not v.requires_grad for v in cache._states.values())
                    assert sum(v.numel() * v.element_size() for v in cache._states.values()) == 1_048_576
                    before = {i: v.clone() for i, v in cache._states.items()}
                    events.clear()
                    assert session.observe(episode['episode_uid'], step, None) is result
                    assert not events
                    assert all(torch.equal(before[i], cache._states[i]) for i in before)
                    results.append(result)
                assert all(torch.count_nonzero(v) > 0 for v in cache._states.values())
                assert all(v > 0 for v in observation_output_max.values())
                streamed = torch.stack(scores).float()
                diff = (streamed - offline).abs()
                # Existing branch numerical gates, declared before measurement.
                torch.testing.assert_close(streamed, offline, atol=.75, rtol=.03)
                rms = float(diff.square().mean().sqrt())
                assert rms <= .15, rms
                first = streamed[0].clone()
                session.reset('fresh-episode', episode['instruction'])
                assert all(torch.count_nonzero(v) == 0 for v in session.step_lane_cache._states.values())
                session.observe('fresh-episode', 0, rgb)
                torch.testing.assert_close(scores[-1].float(), first, atol=0, rtol=0)
        finally:
            model.action_logits = readout
            for handle in handles:
                handle.remove()
    report = dict(
        passed=True, checkpoint=str(Path(args.checkpoint).resolve()),
        fixture='four repeated real RGB frames; not navigation quality',
        step_lane=spec.to_dict(), parameter_count=21_054_000,
        output_projection_norms=output_norms, state_bytes=1_048_576,
        observation_lane_output_max_abs=observation_output_max,
        state_dtype='torch.float32', no_action_history=True,
        nonwriter_state_unchanged=True, one_writer_per_layer_after_decision=True,
        retry_unchanged=True, reset_first_scores_exact=True,
        offline_stream_max_abs=float(diff.max()), offline_stream_rms=rms,
        offline_stream_top1_matches=int((streamed.argmax(-1) == offline.argmax(-1)).sum()),
        decisions=4, results=results,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
