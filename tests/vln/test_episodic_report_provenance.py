"""Numerical evidence remains bound to the exact tested runtime and immutable attempt."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from scripts.vln import submit_episodic as submit
from scripts.vln import check_episodic as check


def test_selected_runtime_applies_to_all_arm_configs(monkeypatch):
    monkeypatch.setattr(check, 'RUNTIME_PROFILE', 'configs/episodic/profile_h2.yaml')
    for arm in check.ARMS:
        with patch.object(check, 'resolve_config', return_value={}) as resolve:
            check.cfg_for(arm)
            assert resolve.call_args.kwargs['runtime'] == check.RUNTIME_PROFILE


def test_submission_copies_evidence_reference_and_placement(tmp_path):
    ref = {'path': 'immutable.json', 'sha256': 'digest'}
    manifest = dict(source_dir=str(submit.ROOT), source_commit='test', campaign_uuid='test', numerical_report=ref)
    (tmp_path/'campaign_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'source_hashes.json').write_text('{}')
    args = SimpleNamespace(campaign_root=str(tmp_path),run='P1-V0-s429',arm=None,
        allow_phase2=False,allow_existing_lease=False,profiles=['H2'],mem={'H2':'512G'},
        qualify='startup',ablations=None,dry_run=True,partition='main',cpus=30,omp=8)
    submit.submit(args)
    launch = json.loads(next((tmp_path/'runs').rglob('launch_manifest.json')).read_text())
    manifest['numerical_report'] = {'path':'new.json'}
    (tmp_path/'campaign_manifest.json').write_text(json.dumps(manifest))
    assert launch['numerical_report'] == ref
    assert launch['partition'] == 'main'


def test_evidence_versions_are_immutable_and_coexist(tmp_path):
    refs=[]
    for commit in ['old','new']:
        raw=json.dumps(dict(passed=True,source_commit=commit,source_dirty=False,runtime_sha256='a'*64)).encode()
        report=tmp_path/f'{commit}.json';report.write_bytes(raw)
        checked=submit.numerical_report_bytes(report,commit)
        refs.append(submit.store_numerical_report(tmp_path,checked))
    assert refs[0] != refs[1]
    from pathlib import Path
    for ref in refs:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']


@pytest.mark.parametrize('change',[{'source_dirty':True},{'source_commit':'wrong'},{'passed':False}])
def test_report_rejects_unqualified_source(tmp_path,change):
    payload=dict(passed=True,source_commit='wanted',source_dirty=False,runtime_sha256='a'*64)
    payload.update(change)
    report=tmp_path/'report.json';report.write_text(json.dumps(payload))
    with pytest.raises(SystemExit): submit.numerical_report_bytes(report,'wanted')
