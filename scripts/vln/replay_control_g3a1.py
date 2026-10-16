"""Control: streaming-vs-offline BF16 replay of the SAME exported model, trained adapters vs zeroed adapter up-projections."""
import copy, json, sys
from pathlib import Path
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_checkpoint
from qwen_vl.stream.session import StreamSession
export, manifest, model_path = sys.argv[1:4]
selected = {}
for line in Path(manifest).open():
    ep = json.loads(line)
    if len(ep['steps']) >= 17: selected.setdefault(ep['dataset'], ep)
    if len(selected) == 2: break
for ep in selected.values():
    ep['steps'] = ep['steps'][:17]; ep['steps'][-1]['action_name'] = 'STOP'; ep['episode_uid'] += ':control'
model, serializer = load_checkpoint(export, model_path); model.eval()
adapter = model.backbone.model.language_model.layers[24].linear_attn.geoaddress
trained = {n: p.detach().clone() for n, p in adapter.named_parameters()}
report = {}
for mode in ('trained_adapters', 'zero_up_adapters'):
    with torch.no_grad():
        for n, p in adapter.named_parameters():
            p.copy_(trained[n] if mode == 'trained_adapters' or 'down' in n else torch.zeros_like(p))
    for dataset, source in selected.items():
        ep = copy.deepcopy(source)
        data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in serializer.encode_episode(ep).items()}
        targets = data['input_ids'][0, data['response_target_positions']].tolist()
        head = model.backbone.lm_head; original = head.forward; scores = []; cursor = [0]
        def capture(hidden):
            value = original(hidden); scores.append(value.float().clone())
            forced = torch.full_like(value, -10000.); forced[targets[cursor[0]]] = 10000.; cursor[0] += 1
            return forced
        with torch.inference_mode():
            offline = model(**data)['logits'].float(); del data
            head.forward = capture
            try:
                session = StreamSession(model, serializer); session.reset(ep['episode_uid'], ep['instruction'])
                for i, step in enumerate(ep['steps']):
                    with Image.open(step['rgb_path']) as rgb: session.observe(ep['episode_uid'], i, rgb)
            finally:
                head.forward = original
        streamed = torch.stack(scores); diff = streamed - offline
        over = (diff.abs() > .75 + .03 * offline.abs()).sum().item()
        report[f'{mode}/{dataset}'] = dict(max_abs=float(diff.abs().max()), rms=float(diff.square().mean().sqrt()),
            elements_over_tolerance=over, argmax_agreement=float((streamed.argmax(-1) == offline.argmax(-1)).float().mean()))
        print(mode, dataset, report[f'{mode}/{dataset}'], flush=True)
Path(export).parent.joinpath('replay_control.json').write_text(json.dumps(report, indent=2))
