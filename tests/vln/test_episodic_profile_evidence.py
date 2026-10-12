"""Races carry evidence for their own runtime profile; gated runs need a recorded reason."""
import json
from types import SimpleNamespace
import pytest
from scripts.vln import submit_episodic as submit


def _campaign(tmp_path, profiles=('H4', 'H2'), legacy=False):
    refs = {}
    for profile in profiles:
        runtime = submit.sha256_file(submit.ROOT / submit.PROFILES[profile]['runtime'])
        raw = json.dumps(dict(passed=True, source_commit='c', source_dirty=False, runtime_sha256=runtime)).encode()
        refs[runtime] = submit.store_numerical_report(tmp_path, raw)
    manifest = dict(source_dir=str(submit.ROOT), source_commit='c', campaign_uuid='u')
    if legacy:
        manifest['numerical_report'] = next(iter(refs.values()))
    else:
        manifest['numerical_reports'] = refs
    (tmp_path / 'campaign_manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'source_hashes.json').write_text('{}')
    return refs


def _args(tmp_path, run='P1-M2c-s429', profiles=('H4', 'H2'), **kw):
    base = dict(campaign_root=str(tmp_path), run=run, arm=None, allow_phase2=False, phase2_reason=None,
                allow_existing_lease=False, profiles=list(profiles), mem={'H4': '768G', 'H2': '512G'},
                qualify='startup', ablations=None, dry_run=True, partition='highgpu',
                cpus=submit.per_profile(int)('H4=32,H2=30'), omp=submit.per_profile(int)('H4=4,H2=8'))
    base.update(kw)
    return SimpleNamespace(**base)


def _launches(tmp_path):
    return {json.loads(p.read_text())['profile']: json.loads(p.read_text())
            for p in (tmp_path / 'runs').rglob('launch_manifest.json')}


def test_race_siblings_each_get_their_own_runtime_evidence(tmp_path):
    refs = _campaign(tmp_path)
    submit.submit(_args(tmp_path))
    launches = _launches(tmp_path)
    for profile, launch in launches.items():
        runtime = submit.sha256_file(submit.ROOT / submit.PROFILES[profile]['runtime'])
        assert launch['numerical_report'] == refs[runtime]
    assert launches['H4']['resources']['cpus'] == 32 and launches['H2']['resources']['omp'] == 8
    assert launches['H4']['resources']['mem'] == '768G'
    assert launches['H4']['race_siblings'] == ['H2']


def test_profile_without_matching_evidence_is_refused(tmp_path):
    _campaign(tmp_path, profiles=('H2',))
    with pytest.raises(SystemExit, match='No numerical evidence'):
        submit.submit(_args(tmp_path))


def test_single_report_campaign_still_resolves_its_own_runtime(tmp_path):
    _campaign(tmp_path, profiles=('H2',), legacy=True)
    submit.submit(_args(tmp_path, profiles=('H2',)))
    with pytest.raises(SystemExit, match='No numerical evidence'):
        submit.submit(_args(tmp_path, profiles=('H4',)))


def test_per_profile_parser():
    assert submit.per_profile(int)('16') == {'H4': 16, 'H2': 16}
    assert submit.per_profile(int)('H4=32') == {'H4': 32}
    with pytest.raises(Exception):
        submit.per_profile(int)('H8=3')


def test_phase2_run_needs_flag_and_reason(tmp_path):
    _campaign(tmp_path)
    with pytest.raises(SystemExit, match='gated'):
        submit.submit(_args(tmp_path, run='P2-M5-s429'))
    with pytest.raises(SystemExit, match='phase2-reason'):
        submit.submit(_args(tmp_path, run='P2-M5-s429', allow_phase2=True))
    submit.submit(_args(tmp_path, run='P2-M5c-s429', allow_phase2=True, phase2_reason='user override'))
    launches = _launches(tmp_path)
    assert {l['arm'] for l in launches.values()} == {'m5c'}
    assert all(l['phase2_reason'] == 'user override' for l in launches.values())


def test_duplicate_runtime_evidence_is_rejected(tmp_path):
    raw = json.dumps(dict(passed=True, source_commit='c', source_dirty=False, runtime_sha256='a' * 64)).encode()
    other = json.dumps(dict(passed=True, source_commit='c', source_dirty=False, runtime_sha256='a' * 64, x=1)).encode()
    with pytest.raises(SystemExit):
        submit.store_numerical_reports(tmp_path, [raw, other])
