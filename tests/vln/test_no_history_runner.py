"""Exercise the real shell setup without loading models or requesting resources."""
import os
from pathlib import Path
import shutil
import subprocess


def test_smoke_uses_node_local_temp_and_writable_kernel_cache(tmp_path):
    script = Path("scripts/vln/smoke_no_history_interactive.sh").read_text()
    setup = script.split('python - "$campaign" "$manifest"', 1)[0]
    shared_temp = tmp_path / "inherited_nfs_tmp"
    shared_temp.mkdir()
    env = dict(os.environ, SLURM_JOB_ID="testjob", TMPDIR=str(shared_temp))
    result = subprocess.run(
        ["bash", "-c", setup + '\nprintf "%s\\n%s\\n" "$TMPDIR" "${PYTORCH_KERNEL_CACHE_PATH:-unset}"',
         "runner", str(tmp_path / "run"), "testjob"],
        env=env, capture_output=True, text=True, check=True,
    )
    scratch, kernel = map(Path, result.stdout.strip().splitlines())
    try:
        assert scratch != shared_temp
        assert scratch.parent.resolve() == Path("/tmp").resolve()
        assert kernel.is_dir() and os.access(kernel, os.W_OK)
        assert kernel.is_relative_to(scratch)
    finally:
        # Remove only the unique scratch created by this test, never inherited paths.
        if scratch != shared_temp and scratch.name.startswith("smv-nohistory-testjob-"):
            shutil.rmtree(scratch)
