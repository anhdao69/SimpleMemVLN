"""The in-job upload must not inherit the job's offline-mode variables."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vln import episodic_job as job
from scripts.vln import upload_episodic

ROOT = job.ROOT


def test_upload_subprocess_env_drops_both_offline_flags(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    checkpoint = tmp_path / 'train' / 'checkpoint-3852'
    checkpoint.mkdir(parents=True)
    a = SimpleNamespace(manifest={'publish_checkpoints': True, 'hf_repo': 'x/y', 'epochs_to_train': 1},
                        launch={'run_id': 'R', 'attempt': 'A'}, dir=tmp_path, status=lambda *a, **k: None)
    with patch.object(job, 'boundary_checkpoint', return_value=checkpoint), \
         patch.object(job.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
        assert job.upload(a, tmp_path / 'train') == 0
    env = run.call_args.kwargs['env']
    assert 'HF_HUB_OFFLINE' not in env and 'TRANSFORMERS_OFFLINE' not in env


def test_uploader_is_online_even_if_only_transformers_offline_is_set():
    code = ("import os; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['HF_HUB_OFFLINE']='1';"
            "from scripts.vln import upload_episodic as u;"
            "[os.environ.pop(n, None) for n in u.ONLINE_BLOCKERS];"
            "import huggingface_hub.constants as c; print(c.HF_HUB_OFFLINE)")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}"))
    assert out.stdout.strip() == 'False', out.stderr
    assert set(upload_episodic.ONLINE_BLOCKERS) == {'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'}
