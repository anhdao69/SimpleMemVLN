"""Exercise a checkpoint's 500-step cache horizon; this is NOT navigation scoring."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_checkpoint
from qwen_vl.stream.session import StreamSession
from qwen_vl.eval.metrics import quantile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    session = StreamSession(model, serializer)
    image = Image.open(args.image)
    torch.cuda.reset_peak_memory_stats()
    session.reset('resource-profile', args.instruction)
    rows = []
    for step in range(args.steps):
        result = session.observe('resource-profile', step, image)
        assert session.observe('resource-profile', step, image) is result
        assert session.next_step == step + 1
        if rows and model.navigation_config['memory']['mode'] == 'full_context':
            assert result['retained_kv_tokens'] > rows[-1]['retained_kv_tokens']
        rows.append(result)
        if (step + 1) % 50 == 0:
            print('PROFILE_STEPS', step + 1, flush=True)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    session.reset('resource-profile-reset', args.instruction)
    reset = session.observe('resource-profile-reset', 0, image)
    assert reset['response_token_ids'] == rows[0]['response_token_ids']
    report = dict(passed=True, kind='repeated_frame_resource_profile_not_navigation',
                  checkpoint=args.checkpoint, steps=args.steps,
                  peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
                  peak_kv_tokens=max(r['retained_kv_tokens'] for r in rows),
                  reset_response_identical=True, age_buckets=[])
    for first in range(0, args.steps, 100):
        times = [r['model_seconds'] for r in rows[first:first+100]]
        report['age_buckets'].append(dict(first_step=first, count=len(times),
              p50_seconds=quantile(times, .5), p95_seconds=quantile(times, .95)))
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
