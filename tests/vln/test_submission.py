"""The Window8 dependency must not accidentally point at the upload job."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def test_training_chain_is_independent_of_uploads(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/vln/submit_campaign.py"
    assert script.is_file(), "Safe campaign submission is missing"
    fake = tmp_path / "sbatch"
    fake.write_text(
        '#!/usr/bin/env python3\nimport json,os,sys\np=os.environ["SBATCH_LOG"]\nrows=open(p).readlines() if os.path.exists(p) else []\nwith open(p,"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(900+len(rows))\n'
    )
    fake.chmod(0o700)
    log = tmp_path / "calls.jsonl"
    (tmp_path / "run").mkdir()
    (tmp_path / "run/preflight_pass.json").write_text(json.dumps({"passed": True}))
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", SBATCH_LOG=str(log))
    result = subprocess.run(
        [
            "python3",
            str(script),
            "--root",
            str(tmp_path),
            "--campaign",
            str(tmp_path / "run"),
            "--verified-preflight",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    assert len(rows) == 4
    assert not any("dependency" in arg for arg in rows[0])
    assert "--dependency=afterok:900" in rows[1]
    assert "--dependency=afterok:900" in rows[2]
    assert "--dependency=afterok:901" in rows[3]
    repeat = subprocess.run(
        [
            "python3",
            str(script),
            "--root",
            str(tmp_path),
            "--campaign",
            str(tmp_path / "run"),
            "--verified-preflight",
        ],
        env=env,
        capture_output=True,
    )
    assert repeat.returncode != 0
    assert len(log.read_text().splitlines()) == 4
