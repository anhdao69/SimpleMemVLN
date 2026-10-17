"""G1 configuration/reduction/export contracts; no teacher dependency."""
import copy
import math


def parse_g1(config):
    value=config.get('research')
    if value is None: return None
    if value.get('run_id') not in ('G1','G3','G3_M5'): raise ValueError('This implementation permits only G1/G3/G3_M5')
    run=value['run_id']
    if run=='G3_M5':
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        spec=parse_episodic_spec(config)
        if spec is None or (spec.recent_groups,spec.budget_tokens,spec.granularity,spec.score,
                            spec.selector_seed,spec.per_source_cap)!=(4,1200,'token','committed_mean_l2_v1',429,None):
            raise ValueError('G3_M5 requires the exact M5 memory contract')
    elif config['memory']['mode'] != 'window8' or config['model']['output_mode'] != 'qwen_text':
        raise ValueError('G1 requires native Window8 text')
    if config['model'].get('step_lane',{}).get('enabled',False): raise ValueError('G1 has no lane')
    expected_layer=23 if run=='G1' else 24
    if 'student_layers' in value:
        if run!='G3_M5' or 'student_layer' in value:
            raise ValueError('student_layers is exclusive to G3_M5 and cannot accompany student_layer')
        validate_student_layers(value['student_layers'])
    elif value.get('student_layer',expected_layer) != expected_layer:
        raise ValueError(f'{run} requires decoder{expected_layer}')
    if config['model'].get('episodic',{}).get('enabled',False): raise ValueError('Native research forbids episodic bank')
    if value.get('sampling_identity','collected-rgb-paths-v1')!='collected-rgb-paths-v1':
        raise ValueError('Unsupported G1 trajectory sampling identity')
    for key in ('teacher_source','teacher_weights','teacher_identity'):
        if not value.get(key): raise ValueError(f'G1 requires {key}')
    for key in ('lambda_max','head_lr'):
        v=value[key]
        if not math.isfinite(v) or v<0: raise ValueError(f'Invalid {key}')
    if value.get('ramp_updates') != 771: raise ValueError('Declared G1 ramp is 771 updates')
    if value.get('teacher_residency','resident') not in ('resident','cpu_between'):
        raise ValueError('Unsupported online teacher residency')
    batch=value.get('teacher_batch_size',8)
    if type(batch) is not int or not 1<=batch<=16: raise ValueError('Teacher batch must be1..16')
    if 'cache_path' in value or 'cache_sha256' in value: raise ValueError('G1 is online; offline cache is forbidden')
    return dict(value)


def auxiliary_weight(update,ramp_updates,maximum):
    return maximum*min(1.,max(0.,update/ramp_updates))


def normalized_aux_loss(numerator,global_count,world_size,weight):
    return numerator*0 if global_count==0 else numerator*(world_size*weight/global_count)


def serving_state_dict(state):
    return {k:v for k,v in state.items() if not k.startswith('research_aux.')}


def serving_config(config):
    result=copy.deepcopy(config); result.pop('research',None); return result


def validate_student_layers(layers, config=None):
    """Explicit zero-based native GDN layers; canonical order is checkpoint identity."""
    if (not isinstance(layers,list) or not layers or any(type(i) is not int for i in layers)
            or layers!=sorted(set(layers))):
        raise ValueError('student_layers must be a nonempty sorted list of unique integers')
    types=(config.layer_types if config is not None else
           ['full_attention' if i%4==3 else 'linear_attention' for i in range(32)])
    if any(not 0<=i<len(types) or types[i]!='linear_attention' for i in layers):
        raise ValueError('student_layers must select native GDN decoder layers')
    return tuple(layers)
