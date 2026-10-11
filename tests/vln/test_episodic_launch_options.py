"""Cluster placement and explicit publication controls for independent H2 jobs."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vln import submit_episodic as submit
from scripts.vln import episodic_job as job


def test_dry_run_preserves_explicit_cluster_resources(tmp_path):
    manifest = dict(source_dir=str(submit.ROOT), source_commit='test', campaign_uuid='test')
    (tmp_path/'campaign_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'source_hashes.json').write_text('{}')
    args=SimpleNamespace(campaign_root=str(tmp_path),run='P1-V0-s429',arm=None,
        allow_phase2=False,allow_existing_lease=False,profiles=['H2'],mem={'H2':'512G'},
        qualify='startup',ablations=None,dry_run=True,partition='main',cpus=30,omp=4)
    result=submit.submit(args)
    command=result[0]['command']
    assert '--partition=main' in command
    assert '--cpus-per-task=30' in command
    assert '--gres=gpu:nvidia_h100_80gb_hbm3:2' in command
    launch=json.loads(next((tmp_path/'runs').rglob('launch_manifest.json')).read_text())
    assert launch['resources']['omp']==4
    assert launch['race_siblings']==[] and launch['submit_ablations']==[]


def test_no_upload_does_not_call_hub_or_require_hub_metadata(tmp_path):
    attempt=SimpleNamespace(manifest={'publish_checkpoints':False})
    with patch.object(job.subprocess,'run') as run:
        assert job.upload(attempt,tmp_path) is None
        run.assert_not_called()
