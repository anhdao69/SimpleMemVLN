"""Detached CPU-only watcher: publish each S8N follow-up arm to M5_S once its lane controller declares it complete.

One arm at a time (each Hub commit is made on an optimistic parent revision), up to three publisher attempts
per arm. State in ``status.json``; create ``STOP`` in the operations directory to end after the current step.
Exits once every arm is uploaded or failed, or every campaign lane has ended with nothing left to publish
(a lane that has not started yet has not ended).
"""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
from publish_followup import ARMS, campaign_data, declared_complete  # noqa: E402

LANES = Path(campaign_data['root']) / 'lanes'
RUNS = Path(campaign_data['root']) / 'runs'
POLL, RETRY = 60, 300


def lane_ended(lane):
    if not (lane / 'history.jsonl').exists():
        return False
    rows = [json.loads(line) for line in (lane / 'history.jsonl').read_text().splitlines()]
    return rows[-1]['status'] in ('lane-complete', 'stopped-by-request')


def progress(arm):
    path = RUNS / f'{arm}_seed429_e1/train/profile_rank0.jsonl'
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 4096))
        lines = f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)['update']
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def main():
    lock = (OPS / 'watcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), arms={})

    def update(arm, status, **details):
        state['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if arm:
            state['arms'][arm] = dict(status=status, **details)
        else:
            state['watcher'] = status
        temporary = OPS / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(OPS / 'status.json')
        print(state['utc'], arm or 'watcher', status, details, flush=True)

    attempts = {arm: 0 for arm in ARMS}
    update(None, 'running')
    while True:
        if (OPS / 'STOP').exists():
            update(None, 'stopped-by-request')
            return
        for arm in ARMS:
            receipt = OPS / arm / 'UPLOAD_VERIFIED.json'
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                assert saved['passed'] and saved['folder'] == arm
                if state['arms'].get(arm, {}).get('status') != 'uploaded_verified':
                    update(arm, 'uploaded_verified', revision=saved['revision'])
                continue
            if attempts[arm] >= 3:
                continue
            if not declared_complete(arm):
                update(arm, 'waiting_for_training', update=progress(arm))
                continue
            attempts[arm] += 1
            update(arm, 'validating_and_uploading', attempt=attempts[arm])
            with (OPS / f'{arm}-publication.log').open('a') as log:
                result = subprocess.run([sys.executable, str(OPS / 'publish_followup.py'), arm], stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, cwd=OPS)
            if result.returncode == 0 and receipt.exists():
                update(arm, 'uploaded_verified', revision=json.loads(receipt.read_text())['revision'])
            elif attempts[arm] >= 3:
                update(arm, 'failed', exit_code=result.returncode, log=f'{arm}-publication.log')
            else:
                update(arm, 'publisher_retry_pending', attempt=attempts[arm], exit_code=result.returncode)
                time.sleep(RETRY)
            break                                   # re-check controls and all arms after every publisher run
        else:
            settled = all((OPS / a / 'UPLOAD_VERIFIED.json').exists() or attempts[a] >= 3 for a in ARMS)
            ended = all(lane_ended(LANES / lane) for lane in campaign_data['lanes'])
            if settled or (ended and not any(declared_complete(a) and attempts[a] < 3 and not
                                             (OPS / a / 'UPLOAD_VERIFIED.json').exists() for a in ARMS)):
                update(None, 'finished')
                return
            time.sleep(POLL)


if __name__ == '__main__':
    main()
