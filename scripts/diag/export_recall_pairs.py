"""Build a hash-verified, long-form export of the captured native D-a pair rows."""
import argparse
import csv
import json
from pathlib import Path

from scripts.diag.cache_teacher import sha256
from scripts.diag.summarize_captures import MODELS

REQUIRED_LAYERS = {
    'M5_E1': (22, 24, 26),
    'G3_M5': (22, 24, 26),
    'TRIPLE': (8, 16, 22, 24, 26),
    'G3': (22, 24, 26),
    'G1': (22, 24, 26),
}
PAIR_FIELDS = ('anchor', 'source', 'lag', 'lag_group', 'lag_bucket', 'region',
               'epsilon', 'source_epsilon', 'forgetting')
OUTPUT_FIELDS = ('model', 'episode_id', 'instruction_id', 'episode_uid', 'scene_id',
                 'gt_action_count', 'layer', *PAIR_FIELDS)


def export_recall_pairs(capture_root, panel_path, output_path, models=MODELS,
                        required_layers=REQUIRED_LAYERS):
    capture_root = Path(capture_root)
    panel_path = Path(panel_path)
    output_path = Path(output_path)
    panel = json.loads(panel_path.read_text())
    panel_by_id = {str(r['episode_id']): r for r in panel['episodes']}
    if len(panel_by_id) != len(panel['episodes']):
        raise ValueError('panel has duplicate episode IDs')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_models = {}
    total_rows = 0
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    try:
        with tmp_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for model in models:
                expected_layers = tuple(required_layers[model])
                seen = set()
                completions = []
                model_rows = 0
                for complete_path in sorted((capture_root / model).glob('episode_*/COMPLETE.json')):
                    complete = json.loads(complete_path.read_text())
                    eid = str(complete['episode_id'])
                    if eid not in panel_by_id:
                        raise ValueError(f'{model}/{complete_path.parent.name}: episode not in panel: {eid}')
                    if eid in seen:
                        raise ValueError(f'{model}: duplicate capture for episode {eid}')
                    seen.add(eid)
                    episode = panel_by_id[eid]
                    if complete.get('scene_id') != episode['scene_id']:
                        raise ValueError(f'{model}/{eid}: scene differs from frozen panel')
                    if tuple(sorted(complete.get('layers', ()))) != tuple(sorted(expected_layers)):
                        raise ValueError(f'{model}/{eid}: captured layer set differs from required D-a layers')
                    files = complete.get('files_sha256', {})
                    completion_record = {'episode_id': eid, 'complete_sha256': sha256(complete_path),
                                         'layers': list(expected_layers), 'files': {}}
                    for layer in expected_layers:
                        name = f'recall_layer_{layer}.csv'
                        source_path = complete_path.parent / name
                        expected_hash = files.get(name)
                        if not expected_hash or sha256(source_path) != expected_hash:
                            raise ValueError(f'{model}/{eid}: capture hash mismatch for {name}')
                        completion_record['files'][name] = expected_hash
                        with source_path.open(newline='') as source:
                            reader = csv.DictReader(source)
                            if tuple(reader.fieldnames or ()) != PAIR_FIELDS:
                                raise ValueError(f'{model}/{eid}/{name}: unexpected pair schema')
                            n = 0
                            for pair in reader:
                                if int(pair['lag']) != int(pair['anchor']) - int(pair['source']):
                                    raise ValueError(f'{model}/{eid}/{name}: lag identity mismatch')
                                writer.writerow({
                                    'model': model, 'episode_id': eid,
                                    'instruction_id': episode.get('instruction_id', ''),
                                    'episode_uid': complete.get('episode_uid', ''),
                                    'scene_id': complete['scene_id'],
                                    'gt_action_count': episode.get('gt_action_count', ''),
                                    'layer': layer, **pair,
                                })
                                n += 1
                        model_rows += n
                    completions.append(completion_record)
                if seen != set(panel_by_id):
                    missing = sorted(set(panel_by_id) - seen)
                    raise ValueError(f'{model}: incomplete panel coverage; missing {missing[:8]} ({len(missing)} total)')
                manifest_models[model] = {'episodes': len(seen), 'layers': list(expected_layers),
                                          'rows': model_rows, 'captures': completions}
                total_rows += model_rows
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    manifest = {
        'status': 'verified', 'diagnostic': 'D-a native self-recall Metric 1 pair export',
        'panel_path': str(panel_path.resolve()), 'panel_sha256': sha256(panel_path),
        'capture_root': str(capture_root.resolve()), 'output_path': str(output_path.resolve()),
        'output_sha256': sha256(output_path), 'rows': total_rows,
        'columns': list(OUTPUT_FIELDS), 'models': manifest_models,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    manifest['manifest_path'] = str(manifest_path.resolve())
    manifest['manifest_sha256'] = sha256(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export_recall_pairs(args.capture_root, args.panel, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=2))
    print(json.dumps({k: {x: v[x] for x in ('episodes', 'layers', 'rows')}
                      for k, v in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
