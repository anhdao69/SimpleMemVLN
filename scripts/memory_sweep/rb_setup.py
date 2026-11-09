"""Create the S8N R/B ablation campaign root on newton (configs, fixture, campaign.json); never overwrites.

Reference provenance (``<root>/reference/{S8N,R4B0,M5_E1}``) is downloaded beforehand from the public Hub
folders anhdao69/M5_S and anhdao69/Streaming_VGGT with ``SOURCES.json`` recording repo, revision and SHA-256.
The family memory/runtime files are byte copies whose SHA-256 must equal the published run identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from scripts.memory_sweep.rb_ablation import ARMS, VERSION, arm_memory, memory_yaml

# Base-model small files of Qwen3.5-4B@851bf6e8 (R2R evaluation contract; the newton snapshot has no vocab/merges files,
# tokenization uses tokenizer.json). The safetensors blobs are verified against their content-addressed names.
BASE_ASSETS_SHA256 = {
    'config.json': 'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670',
    'preprocessor_config.json': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'video_preprocessor_config.json': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
}
LONGEST_RXR = 'rxrce:train:mp3d/82sE5b5pLXE/82sE5b5pLXE.glb:31149:22206'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, data):
    """Write bytes once; an existing file must already hold exactly these bytes."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f'{path} exists with different content')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    root, source = Path(args.root), Path(args.source).resolve()
    reference = root / 'reference'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Source worktree must be clean')
    sources = json.loads((reference / 'SOURCES.json').read_text())
    for name, record in sources.items():
        if sha(reference / name) != record['sha256']:
            raise SystemExit(f'{name} differs from its recorded download')
    configs = root / 'configs'
    families = {}
    for family in ('S8N', 'R4B0'):
        identity = json.loads((reference / family / 'run_identity.json').read_text())
        resolved = json.loads((reference / family / 'resolved_config.json').read_text())
        if identity['config'] != resolved:
            raise SystemExit(f'{family}: identity config differs from resolved_config.json')
        memory_file = reference / family / f'arm_{family}.yaml'
        if sha(memory_file) != identity['memory_sha256'] or sha(reference / family / 'runtime.yaml') != identity['runtime_sha256']:
            raise SystemExit(f'{family}: memory/runtime bytes differ from the published identity')
        if yaml.safe_load(memory_file.read_text())['memory'] != resolved['memory']:
            raise SystemExit(f'{family}: memory file differs from the resolved memory')
        put(configs / f'arm_{family}.yaml', memory_file.read_bytes())
        put(configs / 'runtime.yaml', (reference / family / 'runtime.yaml').read_bytes())
        families[family] = resolved
    for arm, spec in ARMS.items():
        resolved = families[spec['family']]
        memory = arm_memory(resolved['memory'], resolved['observations'], spec['recent'], spec['budget'])
        put(configs / f'arm_{arm}.yaml', memory_yaml(arm, memory, spec['family']).encode())
    line = None
    with open(args.manifest, 'rb') as f:
        for raw in f:
            if json.loads(raw)['episode_uid'] == LONGEST_RXR:
                line = raw if raw.endswith(b'\n') else raw + b'\n'
                break
    if line is None:
        raise SystemExit('Longest RxR episode not found')
    put(root / 'fixtures/rxrce_longest.jsonl', line)
    campaign = dict(
        version=VERSION, root=str(root), source=str(source), source_commit=commit, base=str(root),
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest), model=str(args.model),
        fixtures=str(root / 'fixtures'), seed=429, min_free_disk_gib=220,
        host_guard_gib=460.8,                  # 0.9 x the 512 GiB allocations (G3A1 newton setting)
        request_margin_seconds=1500,           # ask for a recovery checkpoint 25 min before the allocation ends
        kill_margin_seconds=300,               # hard stop 5 min before the end if the save has not finished
        min_resume_seconds=3600, min_fresh_seconds=7200, min_qualify_seconds=5400,
        # cgroup v1 counts active file cache in the guard's working set: drop clean cache near the guard
        cache_relief=dict(fraction=0.85, initial_gib=64, cooldown_seconds=600, threads=8,
                          trees=['/tmp/loopwam_rt10']),
        arms=list(ARMS), exclude_jobs=[],
        reference_sha256={name: record['sha256'] for name, record in sources.items()},
        base_assets_sha256=BASE_ASSETS_SHA256,
        created=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip())
    text = json.dumps(campaign, indent=2) + '\n'
    path = root / 'campaign.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('created', None)
        new = dict(campaign)
        new.pop('created')
        if old != new:
            raise SystemExit(f'{path} exists with different content')
    else:
        path.write_text(text)
    for sub in ('shared', 'runs', 'lanes', 'watcher'):
        (root / sub).mkdir(exist_ok=True)
    print(path)
    for arm in ARMS:
        print(arm, sha(configs / f'arm_{arm}.yaml'))


if __name__ == '__main__':
    main()
