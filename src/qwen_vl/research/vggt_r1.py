"""R1 binds the existing text lane baseline; it never loads its trained weights."""
from qwen_vl.research.lane_contract import parse_step_lane_spec


def validate_r1(config):
    record=config['research'].get('baseline')
    if not isinstance(record,dict):raise ValueError('R1 requires a verified R0 registry record')
    for key in ('artifact','source_commit','weights_sha256','navigation_sha256','manifest_sha256'):
        if not record.get(key):raise ValueError(f'R0 registry missing {key}')
    reference=record['metadata']['config']
    spec=parse_step_lane_spec(config);original=parse_step_lane_spec(reference)
    if spec is None or spec!=original or spec.to_dict()!=record['metadata']['step_lane']:
        raise ValueError('R1 must preserve the exact registered R0 lane architecture')
    if 24 not in spec.layers or spec.clock_mode!='step_end':
        raise ValueError('R1 requires the registered step-end lane24')
    for cfg in (config,reference):
        if (cfg['model']['output_mode']!='qwen_text' or cfg['memory']['mode']!='window8'
                or cfg['memory']['kv_window_steps_including_current']!=8
                or cfg['observations'].get('append_action_tokens') is not True):
            raise ValueError('R1/R0 require canonical Window8 text action history')
    for field in ('backbone','revision'):
        if config['model'][field]!=reference['model'][field]:raise ValueError('R0 base mismatch')
    if config['model']['step_lane'].get('init_seed',429)!=reference['model']['step_lane'].get('init_seed',429):
        raise ValueError('R0 lane initialization seed mismatch')
    for field in ('step_lane_lr','seed','backbone_lr'):
        if config['training'].get(field)!=reference['training'].get(field):raise ValueError(f'R0 {field} mismatch')
    return spec
