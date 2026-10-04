"""Matched native versus freshly installed lane streaming latency, one process."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_checkpoint
from qwen_vl.models.install_step_lane import install_step_lanes
from qwen_vl.research.lane_contract import parse_step_lane_spec
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.stream.session import StreamSession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    model,serializer=load_checkpoint(args.checkpoint,args.model_path)
    if getattr(model,'step_lane_spec',None):
        raise ValueError('Benchmark requires the native parent')
    with Path(args.manifest).open() as f:
        episode=next(json.loads(line) for line in f if len(json.loads(line)['steps'])>=17)
    images=[]
    for step in episode['steps'][:17]:
        with Image.open(step['rgb_path']) as image:
            images.append(image.copy())
    def measure(serializer):
        times=[]
        actions=[]
        for repetition in range(3):
            session=StreamSession(model,serializer)
            session.reset('benchmark',episode['instruction'])
            run=[]
            for step,image in enumerate(images):
                result=session.observe('benchmark',step,image)
                if repetition:
                    times.append(result['model_seconds'])
                run.append(result['class_id'])
            actions.append(run)
        return dict(median_model_seconds=statistics.median(times),mean_model_seconds=statistics.mean(times),
                    timed_steps=len(times),warmup_episodes=1,measured_episodes=2,actions=actions)
    model.eval()
    native=measure(serializer)
    cfg=deepcopy(model.navigation_config)
    cfg['model']['step_lane']={'enabled':True,'init_seed':429}
    model.navigation_config=cfg
    install_step_lanes(model,parse_step_lane_spec(cfg,model.config.text_config),init_seed=429)
    model.eval()
    serializer=EpisodeSerializer(serializer.processor,cfg)
    lane=measure(serializer)
    if native['actions']!=lane['actions']:
        raise AssertionError('Zero lane changed matched inference actions')
    report=dict(status='PASS',native=native,lane=lane,
                median_ratio=lane['median_model_seconds']/native['median_model_seconds'],
                note='Same native parent, same 17 RGB observations, zero lane projection, one process sequentially. Excludes simulator/image-file IO; includes image preprocessing, model append/readout/closure. No diagnostics hooks.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
