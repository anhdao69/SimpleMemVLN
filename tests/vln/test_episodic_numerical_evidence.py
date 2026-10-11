import hashlib
import json
import pytest
from scripts.vln import episodic_job as job


@pytest.mark.parametrize('failure', ['failed','source','runtime','tampered',None])
def test_launch_evidence_rejects_failed_stale_or_tampered_reports(tmp_path,failure):
    assert hasattr(job,'numerical_evidence_problems'), 'launcher must validate numerical evidence'
    runtime=tmp_path/'runtime.yaml';runtime.write_text('runtime: pinned\n')
    report=tmp_path/'numeric.json'
    payload=dict(source_dirty=False,passed=failure!='failed',source_commit='old' if failure=='source' else 'current',
                 runtime_sha256='wrong' if failure=='runtime' else hashlib.sha256(runtime.read_bytes()).hexdigest())
    report.write_text(json.dumps(payload))
    ref=dict(path=str(report),sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    if failure=='tampered': report.write_text(json.dumps(dict(payload,extra='changed')))
    problems=job.numerical_evidence_problems(ref,'current',runtime)
    assert bool(problems)==(failure is not None)


def test_missing_evidence_fails_closed(tmp_path):
    assert hasattr(job,'numerical_evidence_problems'), 'launcher must validate numerical evidence'
    assert job.numerical_evidence_problems(None,'current',tmp_path/'missing')
