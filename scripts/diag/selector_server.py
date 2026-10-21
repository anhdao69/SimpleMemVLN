"""Diagnostic selector server. Load stored checkpoint strictly, override afterward.

Invoke using policy Python -P -m scripts.diag.selector_server with frozen-source
PYTHONPATH. The serializer and parameters remain those of the original export.
"""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys


def effective_config(config,condition):
    from qwen_vl.research.episodic_contract import parse_episodic_spec
    original=parse_episodic_spec(config)
    if original is None or (original.recent_groups,original.budget_tokens,original.score)!=(4,1200,'committed_mean_l2_v1'):
        raise ValueError('D-e requires original R4/B1200 committed-score export')
    result=copy.deepcopy(config)
    if condition=='random':
        result['memory'].update(episodic_score='random_hash_v1',episodic_selector_seed=429,
                                episodic_score_backend='none',episodic_score_segmentation='none')
    elif condition!='score':raise ValueError('Unknown condition')
    return result,original,parse_episodic_spec(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','baseline-contract','checkpoint','model-path'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',choices=['random','score'],required=True);args=p.parse_args()
    protocol=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from scripts.diag.selector_eval import import_frozen,sha256
        contract=json.loads(args.baseline_contract.read_text());import_frozen(args.source,contract)
        weight=args.checkpoint/('model.safetensors' if (args.checkpoint/'model.safetensors').exists() else 'pytorch_model.bin')
        if sha256(weight)!=contract['weights_sha256'] or sha256(args.checkpoint/'navigation.json')!=contract['navigation_sha256']:
            raise ValueError('Original checkpoint identity mismatch')
        from qwen_vl.train.vln_runtime import load_checkpoint
        from qwen_vl.stream.session import StreamSession
        from qwen_vl.stream.transport import decode_rgb
        import torch
        model,serializer=load_checkpoint(str(args.checkpoint),str(args.model_path))
        model.eval().requires_grad_(False)
        original_configuration=copy.deepcopy(model.navigation_config)
        configuration,original,effective=effective_config(model.navigation_config,args.condition)
        if model.episodic_spec!=original:raise ValueError('Loaded model spec differs from stored config')
        if args.condition=='random':
            model.navigation_config=configuration;model.episodic_spec=effective
        session=StreamSession(model,serializer)
        parameter_versions={name:p._version for name,p in model.named_parameters()}
        import platform,transformers,importlib.metadata
        runtime={'status':'ready','diagnostic':True,'qualified':False,'condition':args.condition,
                 'original_spec':original.to_dict(),'effective_spec':effective.to_dict(),
                 'original_configuration':original_configuration,
                 'effective_configuration':model.navigation_config,'serializer':serializer.metadata(),
                 'checkpoint_sha256':contract['weights_sha256'],'python':platform.python_version(),
                 'torch':torch.__version__,'transformers':transformers.__version__,
                 'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
                 'cuda_runtime':torch.version.cuda,'torch_threads':torch.get_num_threads(),
                 'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','flash-attn','flash-linear-attention','fla-core','triton')},
                 'frozen_source':str(args.source.resolve()),'navigation_parameters_updated':False}
    protocol.write(json.dumps(runtime)+'\n');protocol.flush()
    for line in sys.stdin:
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
                if request['operation']=='reset':
                    torch.cuda.reset_peak_memory_stats();session.reset(request['episode_uid'],request['instruction']);response={'status':'ok'}
                elif request['operation']=='observe':
                    with decode_rgb(request) as rgb:response=session.observe(request['episode_uid'],request['step_id'],rgb)
                    if parameter_versions!={name:p._version for name,p in model.named_parameters()}:
                        raise RuntimeError('Navigation parameter version changed during diagnostic serving')
                    response={**response,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
                else:raise ValueError('Unknown operation')
        except Exception as exc:
            response={'status':'fatal' if isinstance(exc,(torch.OutOfMemoryError,RuntimeError)) else 'failure','error':f'{type(exc).__name__}: {exc}'}
        protocol.write(json.dumps(response)+'\n');protocol.flush()


if __name__=='__main__':main()
