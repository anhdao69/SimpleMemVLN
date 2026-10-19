"""G1 configuration/reduction/export contracts; no teacher dependency."""
import copy
import math


def parse_g1(config):
    value=config.get('research')
    if value is None: return None
    if value.get('run_id') not in ('G1','G3'): raise ValueError('This implementation permits only G1/G3')
    run=value['run_id']
    if config['memory']['mode'] != 'window8' or config['model']['output_mode'] != 'qwen_text':
        raise ValueError('G1 requires native Window8 text')
    if config['model'].get('step_lane',{}).get('enabled',False): raise ValueError('G1 has no lane')
    expected_layer=23 if run=='G1' else 24
    if value.get('student_layer',expected_layer) != expected_layer: raise ValueError(f'{run} requires decoder{expected_layer}')
    if config['model'].get('episodic',{}).get('enabled',False): raise ValueError('Native research forbids episodic bank')
    if value.get('sampling_identity','collected-rgb-paths-v1')!='collected-rgb-paths-v1':
        raise ValueError('Unsupported G1 trajectory sampling identity')
    for key in ('teacher_source','teacher_weights','teacher_identity'):
        if not value.get(key): raise ValueError(f'G1 requires {key}')
    for key in ('lambda_max','head_lr'):
        v=value[key]
        if not math.isfinite(v) or v<0: raise ValueError(f'Invalid {key}')
    ramp=value.get('ramp_updates')
    allowed=(0,771) if run=='G3' else (771,)
    if type(ramp) is not int or ramp not in allowed:
        raise ValueError('G1 requires ramp771; G3 permits ramp771 or constant weight (ramp0)')
    if 'head_seed' in value and (run!='G3' or type(value['head_seed']) is not int or value['head_seed']<0):
        raise ValueError('head_seed must be a nonnegative G3 integer')
    if value.get('teacher_residency','resident') not in ('resident','cpu_between'):
        raise ValueError('Unsupported online teacher residency')
    batch=value.get('teacher_batch_size',8)
    if type(batch) is not int or not 1<=batch<=16: raise ValueError('Teacher batch must be1..16')
    if 'cache_path' in value or 'cache_sha256' in value: raise ValueError('G1 is online; offline cache is forbidden')
    return dict(value)


def auxiliary_weight(update,ramp_updates,maximum):
    # Zero means constant supervision, including optimizer update zero.
    if ramp_updates==0:return maximum
    return maximum*min(1.,max(0.,update/ramp_updates))


def normalized_aux_loss(numerator,global_count,world_size,weight):
    return numerator*0 if global_count==0 else numerator*(world_size*weight/global_count)


def serving_state_dict(state):
    return {k:v for k,v in state.items() if not k.startswith('research_aux.')}


def serving_config(config):
    result=copy.deepcopy(config); result.pop('research',None); return result
