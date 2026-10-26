"""CPU-only publication of one completed S8N follow-up arm to the Hub folder ``<ARM>/`` of M5_S.

Adapted from the S-arm publisher. The follow-up lane controller must have declared the arm complete; the
bound run identity is re-derived from the sealed source; all-rank coverage and finite losses are checked;
the data order is S8N's (LR/Muon arms) or a reordering of S8N's episodes (the continuations, S8N_MUON_E2
in exactly S8N_E2's order); serving files are copied from checkpoint-3852 (bitwise checked) and the real
navigation model strictly reloads on CPU.
One Hub commit adds only ``<ARM>/`` on an optimistic parent revision and never overwrites; every file is
then verified by SHA-256 and size at the immutable revision.

Runs from an operations directory outside the sealed source (which must stay clean):
    publish_followup.py ARM [--check-only]
The directory's ``publication.json`` names the campaign and its arms (the first follow-up campaign's
directory predates it).
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.pop(key, None)

BASE = Path('/mnt/data/vmo-ai-task/anhdh35')
OPS = Path(__file__).resolve().parent
SETTINGS = (json.loads((OPS / 'publication.json').read_text()) if (OPS / 'publication.json').exists() else
            dict(campaign=str(BASE / 'm5-s8n-followup-20261025/campaign.json'),
                 arms=['S8N_E2', 'S8N_LR1E5', 'S8N_MUON']))
CAMPAIGN = Path(SETTINGS['campaign'])
REPO = 'anhdao69/M5_S'
ARMS = tuple(SETTINGS['arms'])
SERVING = {'pytorch_model.bin', 'navigation.json', 'config.json', 'generation_config.json',
           'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json', 'tokenizer.json',
           'chat_template.jinja'}
REQUIRED = ('pytorch_model.bin', 'navigation.json', 'config.json', 'tokenizer.json', 'processor_config.json')

campaign_data = json.loads(CAMPAIGN.read_text())
SOURCE = Path(campaign_data['source'])
SOURCE_COMMIT = campaign_data['source_commit']
PARENT_RUN = Path(campaign_data['parent']['run'])
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE)]
from scripts.memory_sweep.s_campaign import EXPECTED_STEPS, WORLD, latest_resources, run_complete, sha, write  # noqa: E402
from scripts.memory_sweep.s8n_followup import ARMS as DECLARED, MUON, Campaign, exposure_plan  # noqa: E402

assert set(ARMS) == {arm for lane in campaign_data['lanes'].values() for arm in lane['arms']} <= set(DECLARED)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def declared_complete(arm):
    """The follow-up lane controller's own ``arm-complete`` record, or None."""
    for history in sorted((Path(campaign_data['root']) / 'lanes').glob('*/history.jsonl')):
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'arm-complete' and row.get('arm') == arm:
                return row
    return None


def validate_training(arm):
    from qwen_vl.train.recovery import validate_complete
    from scripts.vln.profile_episodic import verify
    campaign = Campaign(CAMPAIGN)
    run = campaign.run_dir(arm)
    identity = json.loads((run / 'run_identity.json').read_text())
    assert identity == campaign.run_identity(arm), 'bound identity differs from the sealed source'
    seal = json.loads((campaign.shared / 'QUALIFIED.json').read_text())
    assert seal['passed'] and seal['source_commit'] == SOURCE_COMMIT
    config = identity['config']
    assert config['memory']['score_id'] == 'S8N' and 'research' not in config
    assert config['training']['seed'] == 429 and config['training']['epochs'] == 1
    assert config['training']['expected_total_steps'] == EXPECTED_STEPS
    assert declared_complete(arm), f'{arm}: lane controller has not declared the arm complete'
    assert run_complete(run, config)
    resources = latest_resources(run)
    assert resources['status'] == 'PASS' and resources['new_oom_events'] == resources['new_oom_kill_events'] == 0
    checks = verify(run / 'train', WORLD, EXPECTED_STEPS, 8, 78.0)
    assert checks['passed'], checks['problems']
    for key in ('losses', 'grad_norms'):
        checks.pop(key, None)
    exposures, parent = exposure_plan(run / 'train'), exposure_plan(PARENT_RUN / 'train')
    bound = campaign.initialization(arm)
    if bound:
        assert exposures != parent, f'{arm}: repeats S8N\'s first-epoch order'
        uids = lambda plan: {u for r in plan.values() for e in r.values() for u in e}   # noqa: E731
        assert uids(exposures) == uids(parent), f'{arm}: episodes differ from S8N\'s'
        initialization = json.loads((run / 'train/weights_initialization.json').read_text())
        assert initialization['weights_sha256'] == bound[1] == identity['init_weights_sha256']
        assert initialization['optimizer'] == initialization['scheduler'] == 'fresh'
        checks['continuation'] = dict(parent_weights_sha256=initialization['weights_sha256'], sampler_epoch_offset=1,
                                      unique_episodes=len(uids(exposures)))
        if DECLARED[arm].get('continue_from'):
            assert sha(Path(bound[0]) / 'pytorch_model.bin') == bound[1], f'{bound[0]} changed'
            order = exposure_plan(campaign.continuation['order_reference'])
            assert exposures == order, f'{arm}: data order differs from S8N_E2 (the second-epoch order)'
            checks['continuation'].update(init_parent=campaign.continuation['arm'], init_weights_checkpoint=bound[0],
                                          exposures_identical_to_s8n_e2=sum(len(u) for r in exposures.values()
                                                                            for u in r.values()),
                                          init_weights_published=published_parent_weights(campaign, bound[1]))
    else:
        assert exposures == parent, f'{arm}: training exposures differ from S8N'
        checks['exposures_identical_to_s8n'] = sum(len(u) for r in exposures.values() for u in r.values())
    checkpoint = run / f'train/checkpoint-{EXPECTED_STEPS}'
    record = validate_complete(checkpoint)
    assert record['step'] == EXPECTED_STEPS and record['world_size'] == WORLD and record['epoch_boundary']
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert trainer['global_step'] == trainer['max_steps'] == EXPECTED_STEPS and trainer['epoch'] == 1
    navigation = json.loads((checkpoint / 'navigation.json').read_text())
    assert navigation['config'] == config
    return campaign, run, identity, checkpoint, record, resources, checks, navigation


def published_parent_weights(campaign, digest):
    """Where the continuation parent's weights were published (verified receipt), or None if not yet."""
    receipt = Path(SETTINGS.get('parent_publication', '/nonexistent')) / campaign.continuation['arm'] / \
        'UPLOAD_VERIFIED.json'
    if not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    name = f'{campaign.continuation["arm"]}/pytorch_model.bin'
    entry = next(f for f in saved['files'] if f['path'] == name)
    assert saved['passed'] and entry['sha256'] == digest, f'published {name} differs from the initial weights'
    return dict(repo=saved['repo'], revision=saved['revision'], path=name, sha256=entry['sha256'])


def stage(arm, checkpoint, navigation, root):
    """Exact serving copy, bitwise tensor check and strict CPU reload of the real navigation model."""
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
    export = root / 'model'
    export.mkdir(parents=True, exist_ok=True)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name in SERVING:
            destination = export / path.name
            if not destination.exists() or destination.stat().st_size != path.stat().st_size:
                partial = export / (path.name + '.partial')
                shutil.copy2(path, partial)
                partial.replace(destination)
    assert all((export / name).exists() for name in REQUIRED)
    assert not torch.cuda.is_available()
    metadata = json.loads((export / 'navigation.json').read_text())
    assert metadata == navigation
    before = torch.load(checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    after = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    assert not any(k.startswith('research_aux.') for k in before)
    assert all(torch.isfinite(t).all().item() for t in before.values() if t.is_floating_point()), 'nonfinite tensor'
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
    print('ALL_SERVING_TENSORS_BITWISE_EQUAL', len(before), flush=True)
    base_config = AutoConfig.from_pretrained(campaign_data['model'], local_files_only=True)
    with torch.device('meta'):
        backbone = Qwen3_5ForConditionalGeneration(base_config)
        model = SimpleMemVLNForNavigation(backbone, metadata['config'])
    model.load_state_dict(after, strict=True, assign=True)
    assert not hasattr(model, 'research_aux')
    assert all(p.device.type == 'cpu' for p in model.parameters())
    processor = AutoProcessor.from_pretrained(export, local_files_only=True)
    assert EpisodeSerializer(processor, metadata['config']).metadata() == metadata
    audit = dict(passed=True, cpu_only=True, strict_architecture_reload=True, exact_serving_tensors=len(before),
                 removed_training_tensors=[], serializer_metadata_exact=True, memory_metadata_exact=True,
                 all_model_tensors_finite=True, completed_model_gpu_replay=False)
    write(root / 'export_validation.json', audit)
    print('CPU_STRICT_EXPORT_VALIDATED', flush=True)
    return export


RECIPE = {
    'S8N_E2': lambda c: f"""S8N trained for **one more epoch**. The run starts from S8N's final weights
(`pytorch_model.bin` SHA-256 `{c.parent['weights_sha256']}`, the `S8N/` folder of this repository) with a
fresh AdamW state and the same schedule as S8N's epoch (116 warmup updates, cosine to 0.1 of peak LR 5e-6),
over the same 30,815 episodes in the seed's second-epoch order (sampler seed 429 + 1). With S8N's epoch it
has seen every episode twice (7,704 optimizer updates in total). The navigation config is S8N's exactly.""",
    'S8N_LR1E5': lambda c: """S8N's recipe from the original Qwen3.5-4B with peak LR **1e-5** (backbone and
classifier) instead of 5e-6. Everything else is S8N's, including warmup, cosine decay to 0.1 of peak, and
the exact episodes in the exact order on every rank.""",
    'S8N_MUON': lambda c: f"""S8N's recipe from the original Qwen3.5-4B at peak LR 5e-6 with the **Muon** optimizer on
the 2D weight matrices of the language-model layers (attention, Gated DeltaNet and MLP projections):
momentum {MUON['momentum']} (Nesterov), {MUON['ns_steps']} Newton-Schulz steps with coefficients
{tuple(MUON['ns_coefficients'])}, update scaled by 0.2·sqrt(max(rows, cols)) to match AdamW's update RMS
(`match_rms_adamw`, as in Moonlight), decoupled weight decay 0.01. Embeddings, LM head, norms, gates, conv
filters and scalars keep S8N's AdamW. Same warmup, cosine decay to 0.1 of peak, clipping at global norm 1,
and the exact episodes in the exact order. Muon runs on the ZeRO-2 CPU-offload partitions: each rank
gathers the momentum update in BF16, orthogonalizes whole matrices on its GPU and updates its own slice,
element for element as `torch.optim.Muon`.""",
    'S8N_MUON_E2': lambda c: f"""S8N_MUON trained for **one more epoch**, exactly as S8N_E2 continues S8N. The run
starts from S8N_MUON's final weights (`pytorch_model.bin` SHA-256 `{c.continuation_weights()['weights_sha256']}`,
the `S8N_MUON/` folder of this repository) with a fresh optimizer state (S8N_MUON's Muon on the hidden-layer
matrices, AdamW elsewhere) and the same schedule as an S8N epoch (116 warmup updates, cosine to 0.1 of peak
LR 5e-6), over the same 30,815 episodes in the seed's second-epoch order (sampler seed 429 + 1): S8N_E2's
order, update for update. With S8N_MUON's epoch it has seen every episode twice (7,704 optimizer updates in
total). The navigation config is S8N_MUON's exactly.""",
}

VERIFICATION = {
    False: """Before training, this source reproduced S8N's logged first updates (loss and gradient norm) and every
first-epoch arm matched S8N's data order; the ZeRO Muon optimizer was checked against `torch.optim.Muon`
under real DeepSpeed and through a fresh-process resume.""",
    True: """Before training, this source bound the completed parent run's final weights (equal to its epoch
checkpoint), ran the first updates from them with a fresh optimizer in S8N_E2's exact data order, checked the
ZeRO Muon optimizer against `torch.optim.Muon` under real DeepSpeed, and resumed the continuation in a fresh
process (weights, loss and optimizer state).""",
}


def readme(campaign, arm, run, identity, resources):
    columns = [(arm, run)]
    if DECLARED[arm].get('continue_from'):
        columns += [(campaign.continuation['arm'], Path(campaign.continuation['run'])),
                    ('S8N_E2', Path(campaign.continuation['order_reference']).parent)]
    else:
        columns.append(('S8N', PARENT_RUN))
    reports = [json.loads((path / 'train/reports/epoch-1.json').read_text()) for _, path in columns]
    rows = '\n'.join(f'| {label} | ' + ' | '.join(f'{r[key]:.5f}' for r in reports) + ' |'
                     for label, key in (('Action-weighted epoch loss', 'loss_action_weighted'),
                                        ('First update loss', 'first_update_loss'),
                                        ('Final update loss', 'last_update_loss')))
    hours = resources['seconds'] / 3600
    training = identity['config']['training']
    continuation = bool(campaign.initialization(arm))
    return f"""# {arm} — S8N follow-up (M5 GDN S sweep), completed epoch

{RECIPE[arm](campaign)}

Memory: S8N's (`episodic_window_gdn_sweep_v1`, score S8N: net state drift
‖S_after − S_before‖_F / max(‖S_before‖_F, ε_l), frame-level) with S8N's own calibrated memory file.

This folder is the arm's serving model; `provenance/` holds the evidence and `SHA256SUMS.json` the published hashes.

## Training

Asserted before launch: the resolved config equals S8N's bound config except
`{', '.join(sorted(DECLARED[arm]['training'])) or 'nothing'}`. Original Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, joint R2R + English RxR15deg (30,815 episodes), 3,852 optimizer
updates, global batch 8 (4 H100 x 1 episode x accumulation 2), seed {training['seed']}, peak LR
{training['backbone_lr']}, Adam 0.9/0.95, weight decay {training['weight_decay']}, clip 1, ZeRO-2 with CPU optimizer offload.

| | {' | '.join(label for label, _ in columns)} |
|---|{'---:|' * len(columns)}
{rows}

Training loss is not navigation success{', and this arm saw its episodes a second time' if continuation else ''}.
Wall time {hours:.2f} h including load and saves; no GPU or host OOM.

## Verification

{VERIFICATION[bool(campaign.continuation)]} The lane controller declared the run complete; the
bound identity re-derives from the sealed source; all 3,852 updates and exposures are covered on every rank
with finite losses; checkpoint-3852 is a complete epoch-boundary recovery checkpoint. Serving tensors are
bitwise identical to it and the real navigation model strictly reloads on CPU with exact serializer and
memory metadata. No completed-model GPU replay is claimed.

## Use

Code: SimpleMemVLN branch `m5_s8n_followup`, source `{SOURCE_COMMIT}`. Load with
`qwen_vl.train.vln_runtime.load_checkpoint('{arm}', base_model_path)` after downloading this folder.
These are navigation-wrapper weights for inference; optimizer/RNG state stays on the cluster.
"""


def collect(campaign, arm, run, identity, checkpoint, record, resources, checks, root, export):
    write(root / 'training_validation.json', checks)
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    supervisor = sorted(run.glob('supervisor-*'))[-1]
    sources = [(run / 'train/reports/epoch-1.json', 'epoch-1.json'),
               (run / 'train/resolved_config.json', 'resolved_config.json'),
               (run / 'train/run_timing.json', 'run_timing.json'),
               (checkpoint / 'trainer_state.json', 'trainer_state.json'),
               (supervisor / 'resources.json', 'completion_resources.json'),
               (root / 'export_validation.json', 'publication_validation.json'),
               (root / 'training_validation.json', 'training_validation.json'),
               (campaign.shared / 'QUALIFIED.json', 'qualification_pass.json'),
               (run / 'run_identity.json', 'run_identity.json'),
               (campaign.arm_memory_path(arm), 'arm_S8N.yaml'),
               (campaign.configs / 'runtime.yaml', 'runtime.yaml'),
               (campaign.policy_path(arm), f'policy_{arm}.yaml')]
    if campaign.initialization(arm):
        sources.append((run / 'train/weights_initialization.json', 'weights_initialization.json'))
    init_parent = campaign.continuation['arm'] if DECLARED[arm].get('continue_from') else None
    if init_parent:
        sources.append((campaign.shared / 'continuation_parent.json', 'continuation_parent.json'))
    for source, name in sources:
        shutil.copy2(source, provenance / name)
    if not (provenance / 'source_and_completion.json').exists():
        write(provenance / 'source_and_completion.json', dict(
            source_commit=SOURCE_COMMIT, source_directory=str(SOURCE), checkpoint=str(checkpoint), arm=arm,
            parent='S8N', init_parent=init_parent, completed_epoch=1, optimizer_updates=EXPECTED_STEPS, global_batch=8,
            recovery_inventory=record, controller_completion=declared_complete(arm),
            export_source_weights_sha256=sha(checkpoint / 'pytorch_model.bin'),
            export_weights_sha256=sha(export / 'pytorch_model.bin'), verified_utc=now()))
    (root / 'README.md').write_text(readme(campaign, arm, run, identity, resources))
    files = {f'{arm}/{p.name}': p for p in export.iterdir() if p.is_file() and p.name in SERVING}
    files.update({f'{arm}/provenance/{p.name}': p for p in provenance.iterdir() if p.is_file()})
    files[f'{arm}/README.md'] = root / 'README.md'
    digests = {name: dict(sha256=sha(path), size=path.stat().st_size) for name, path in files.items()}
    write(root / 'SHA256SUMS.json', digests)
    files[f'{arm}/SHA256SUMS.json'] = root / 'SHA256SUMS.json'
    digests[f'{arm}/SHA256SUMS.json'] = dict(sha256=sha(root / 'SHA256SUMS.json'),
                                             size=(root / 'SHA256SUMS.json').stat().st_size)
    print('HASHED', len(files), flush=True)
    return files, digests


def attributes_change(prior_revision, revision, arm, digests):
    """Return the .gitattributes lines the Hub added; they may only route this arm's new files to LFS."""
    from huggingface_hub import hf_hub_download
    before = Path(hf_hub_download(REPO, '.gitattributes', revision=prior_revision)).read_text()
    after = Path(hf_hub_download(REPO, '.gitattributes', revision=revision)).read_text()
    assert after.startswith(before), '.gitattributes rewritten'
    added = [line for line in after[len(before):].splitlines() if line.strip()]
    for line in added:
        path = line.split()[0]
        assert path in digests and path.startswith(arm + '/') and 'filter=lfs' in line, line
    return added


def publish(arm, files, digests, root):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    assert api.whoami()['name'].lower() == 'anhdao69'
    receipt = root / 'PUBLISHED.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved['digests'] == digests
        revision = saved['revision']
    else:
        info = api.model_info(REPO, files_metadata=True)
        assert not info.private
        assert not any(f.rfilename.startswith(arm + '/') for f in info.siblings), 'destination exists; inspect first'
        write(root / 'prior_repo_files.json', dict(revision=info.sha, files={
            f.rfilename: dict(blob_id=f.blob_id, size=f.size) for f in info.siblings}))
        commit = api.create_commit(
            repo_id=REPO, parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in sorted(files.items())],
            commit_message=f'Publish {arm}: S8N follow-up, completed epoch (step 3852), verified serving weights')
        revision = commit.oid
        write(receipt, dict(repo=REPO, revision=revision, digests=digests))
        print('PUBLISHED', revision, flush=True)
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    prior = json.loads((root / 'prior_repo_files.json').read_text())
    attributes = []
    for name, old in prior['files'].items():
        if name == '.gitattributes' and remote[name].blob_id != old['blob_id']:
            attributes = attributes_change(prior['revision'], revision, arm, digests)
        else:
            assert remote[name].blob_id == old['blob_id'] and remote[name].size == old['size'], name
    assert set(remote) == set(prior['files']) | set(digests), 'unexpected files in the commit'
    verified = []
    for name, expected in sorted(digests.items()):
        f = remote[name]
        assert f.size == expected['size'], name
        if f.lfs:
            assert f.lfs.sha256 == expected['sha256'], name
            method = 'Hub LFS SHA256 and size'
        else:
            assert sha(hf_hub_download(REPO, name, revision=revision)) == expected['sha256'], name
            method = 'Downloaded SHA256 and size'
        verified.append(dict(path=name, **expected, method=method))
    write(root / 'UPLOAD_VERIFIED.json', dict(
        passed=True, repo=REPO, revision=revision, folder=arm, files=verified,
        prior_files_unchanged=len(prior['files']) - int(bool(attributes)), automatic_metadata_change=attributes,
        verified_utc=now()))
    print('UPLOAD_VERIFIED', revision, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = OPS / args.arm
    root.mkdir(exist_ok=True)
    campaign, run, identity, checkpoint, record, resources, checks, navigation = validate_training(args.arm)
    print('COMPLETED_ARM_VERIFIED', args.arm, flush=True)
    export = stage(args.arm, checkpoint, navigation, root)
    files, digests = collect(campaign, args.arm, run, identity, checkpoint, record, resources, checks, root, export)
    if args.check_only:
        print('CHECK_ONLY_PASSED', args.arm, len(files), flush=True)
        return
    publish(args.arm, files, digests, root)


if __name__ == '__main__':
    main()
