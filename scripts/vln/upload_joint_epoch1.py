"""Publish the approved joint FullContext epoch-1 snapshot, never optimizer state."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

from huggingface_hub import HfApi, CommitOperationAdd, get_token, hf_hub_download
from qwen_vl.train.recovery import validate_complete


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    campaign = Path('/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/fullcontext_recovery_20261001')
    checkpoint = campaign/'train/checkpoint-3852'
    publication = campaign/'publication-epoch1'
    publication.mkdir(exist_ok=True)
    complete = validate_complete(checkpoint)
    state = json.loads((checkpoint/'trainer_state.json').read_text())
    assert complete['epoch_boundary'] and state['global_step'] == 3852 and state['epoch'] == 1
    navigation = json.loads((checkpoint/'navigation.json').read_text())
    assert navigation['config']['memory']['mode'] == 'full_context'
    assert navigation['config']['model']['output_mode'] == 'qwen_text'
    token = get_token()
    assert token, 'Missing cached Hub credential'
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'anhdao69'
    repo = 'anhdao69/SimpleMemVLN-R2R-RxR15deg-FullContext-B'
    allowed = {'pytorch_model.bin', 'config.json', 'navigation.json', 'tokenizer.json',
               'tokenizer_config.json', 'processor_config.json', 'preprocessor_config.json',
               'video_preprocessor_config.json', 'chat_template.jinja', 'generation_config.json',
               'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'trainer_state.json'}
    files = {f'epoch-1/{p.name}': p for p in checkpoint.iterdir() if p.is_file() and p.name in allowed}
    for required in ('pytorch_model.bin', 'config.json', 'navigation.json', 'tokenizer.json', 'processor_config.json'):
        assert f'epoch-1/{required}' in files
    files['source.sha256'] = campaign/'source.sha256'
    files['manifest.sha256'] = campaign/'manifest.sha256'
    report = campaign/'train/reports/epoch-1.json'
    if report.is_file():
        files['reports/epoch-1.json'] = report
    card = publication/'README.md'
    card.write_text('''---
base_model: Qwen/Qwen3.5-4B
library_name: transformers
tags:
- vision-language-navigation
- r2r
- rxr
- simplememvln
- intermediate-checkpoint
---
# SimpleMemVLN joint R2R + RxR_15deg FullContext+B

**Not yet evaluated in Habitat.** Training loss is not a navigation success metric.

## Published snapshot
`epoch-1/` contains the completed first epoch, optimizer update 3852.
This is a **mid-schedule snapshot of one two-epoch cosine run**, not an
independently scheduled one-epoch model. The second epoch is not part of this
publication. Training continues separately; uploading does not change training.

## Recipe and provenance
Joint R2R training data and English-guide RxR_15deg trajectories, full episodes.
Initialized from Qwen/Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Source base commit: `6bb2b10a2fe21cb6e25a265afc5a2af39b7de4f6`, with input-staging
and checkpoint-recovery patches. The tested source is identified by
`source.sha256`; the dataset manifest by `manifest.sha256`.
Serializer: `vln_append_only_chat_v1`. Native LLM text-action head (option B),
gold action history; per-action mean token cross-entropy including assistant
terminator, normalized over actions in the distributed accumulation window.
Frozen vision/merger, trainable text backbone and LLM head; BF16, ZeRO-2,
nonreentrant gradient checkpointing with long-sequence activation offload.
Full-context causal attention, no trajectory truncation or TBPTT.
Global batch 8 = four H100 GPUs × one episode/rank × GAS2.
Two epochs, 7704 optimization updates, 232 warmup updates, peak LR 5e-6,
cosine decay to 10% of peak, weight decay 0.01, seed 429.
Observation-before-action alignment is declared but not independently
verified against collector source or Habitat replay.

## Loading and integrity
Use the SimpleMemVLN wrapper loader
`qwen_vl.train.vln_runtime.load_checkpoint(epoch_directory, base_model_path)`
with the pinned base snapshot. These are navigation-wrapper weights, not a
plain AutoModel state dictionary. Navigation metadata records the recipe.
Only model, processor, tokenizer, provenance and selected training metadata
are published. Optimizer/RNG state, dataset images and credentials stay local.
`SHA256SUMS.json` records content hashes. The uploader downloads every file
at its immutable Hub commit and verifies SHA256 before reporting success.
''')
    files['README.md'] = card
    digests = {n: sha256(p) for n, p in files.items()}
    manifest = publication/'SHA256SUMS.json'
    manifest.write_text(json.dumps(digests, indent=2, sort_keys=True)+'\n')
    files['SHA256SUMS.json'] = manifest
    digests['SHA256SUMS.json'] = sha256(manifest)
    print('LOCAL_HASHES_READY', len(files), flush=True)
    published = publication/'PUBLISHED.json'
    if published.exists():
        receipt = json.loads(published.read_text())
        assert receipt['repo'] == repo and receipt['digests'] == digests
        revision = receipt['revision']
    else:
        api.create_repo(repo, repo_type='model', private=False, exist_ok=True)
        info = api.model_info(repo)
        assert not info.private
        assert not any(s.rfilename.startswith('epoch-') for s in info.siblings), 'Refusing to overwrite existing snapshots'
        result = api.create_commit(repo_id=repo, parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n,p in sorted(files.items())],
            commit_message='Publish joint R2R + RxR_15deg FullContext+B epoch 1 (step 3852)')
        revision = result.oid
        published.write_text(json.dumps(dict(repo=repo, revision=revision, digests=digests), indent=2)+'\n')
    print('PUBLISHED', repo, revision, flush=True)
    verified = []
    for name, expected in sorted(digests.items()):
        for attempt in range(3):
            try:
                with tempfile.TemporaryDirectory(prefix='verify-', dir=publication) as cache:
                    path = hf_hub_download(repo, name, revision=revision, token=token,
                                           cache_dir=cache, force_download=True)
                    assert sha256(path) == expected, f'SHA256 mismatch: {name}'
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10)
        verified.append(dict(path=name, sha256=expected))
        print('SHA256_VERIFIED', name, flush=True)
    (publication/'UPLOAD_VERIFIED.json').write_text(json.dumps(dict(
        passed=True, repo=repo, revision=revision, epoch=1, step=3852, files=verified), indent=2)+'\n')
    print('UPLOAD_VERIFIED', repo, revision, len(verified), flush=True)


if __name__ == '__main__':
    main()
