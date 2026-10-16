"""G1 configuration/reduction/export contracts; no teacher dependency."""
import copy
import math


def parse_g1(config):
    value=config.get('research')
    if value is None: return None
    if value.get('run_id') not in ('G1','G3','A0','A1','G3A1'):
        raise ValueError('This implementation permits only G1/G3/A0/A1/G3A1')
    run=value['run_id']
    from qwen_vl.research.geoaddress_contract import parse_geoaddress
    address = parse_geoaddress(config)
    if (run in ('A0','A1','G3A1')) != (address is not None):
        raise ValueError('A0/A1/G3A1 require runtime GeoAddress; G1/G3 forbid it')
    if run in ('A0','A1','G3A1') and value.get('adapter_lr') != 5e-5:
        raise ValueError('A0/A1/G3A1 adapter LR must be 5e-5')
    if run in ('A1','G3A1') and value.get('relation_projection')!='direct-native-qk-cosine-v1':
        raise ValueError('A1/G3A1 require the declared direct native Q/K objective')
    if run=='G3A1' and (value.get('lambda_max')!=.1 or value.get('head_lr')!=1e-4):
        # Each objective keeps its specified weight: lambda(G3)=lambda(A1)=0.1 on one shared
        # ramp and the shared global pair count, so their sum is applied with one lambda.
        raise ValueError('G3A1 keeps the specified lambda_max 0.1 for both objectives and head LR 1e-4')
    if run == 'A0' and value.get('lambda_max') != 0:
        raise ValueError('A0 is navigation-only')
    if config['memory']['mode'] != 'window8' or config['model']['output_mode'] != 'qwen_text':
        raise ValueError('G1 requires native Window8 text')
    if config['model'].get('step_lane',{}).get('enabled',False): raise ValueError('G1 has no lane')
    expected_layer=23 if run=='G1' else 24
    if value.get('student_layer',expected_layer) != expected_layer: raise ValueError(f'{run} requires decoder{expected_layer}')
    if config['model'].get('episodic',{}).get('enabled',False): raise ValueError('Native research forbids episodic bank')
    if value.get('sampling_identity','collected-rgb-paths-v1')!='collected-rgb-paths-v1':
        raise ValueError('Unsupported G1 trajectory sampling identity')
    for key in (() if run=='A0' else ('teacher_source','teacher_weights','teacher_identity')):
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


def inference_sample(sample,device):
    """Strip all research-only batch fields before teacher-free final/reference inference."""
    import torch
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in sample.items()
            if not k.startswith('aux_') and k not in ('teacher_images','update_aux_count')}
