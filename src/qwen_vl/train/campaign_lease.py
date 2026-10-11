"""Single-writer ownership for one logical training run (shared filesystem).

* ``claim`` uses one atomic exclusive create (``O_CREAT|O_EXCL``) of
  ``leases/<run>/LEASE.json``; two processes can never both succeed.
* A lease is stale only with scheduler evidence: the owner's Slurm job is no
  longer PENDING/RUNNING, its heartbeat is old, and it never reached
  TRAINING_VERIFIED.  A stale lease is archived by one atomic rename (only one
  contender can rename it) before a new exclusive claim; nothing is deleted.
* Cancellation is allowed only for an exact job ID recorded in this
  campaign's submission ledger whose live scheduler record still matches the
  recorded user, job name and campaign comment.  There is no user-wide,
  wildcard or name-based cancellation path.
"""
import json
import os
from pathlib import Path
import subprocess
import time

STATES = ('SUBMITTED', 'ALLOCATED', 'PREFLIGHT_PASSED', 'CLAIMED', 'TRAINING_VERIFIED',
          'ACTIVE', 'COMPLETE', 'FAILED', 'DUPLICATE_STANDBY')
VERIFIED = ('TRAINING_VERIFIED', 'ACTIVE', 'COMPLETE')
LIVE_SLURM = ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'REQUEUED', 'RESIZING', 'SUSPENDED')


def now():
    return time.time()


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record, unix_time=now())) + '\n'
    with open(path, 'a') as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def slurm_state(job_id, runner=subprocess.run):
    """Live scheduler state (squeue first, then sacct); None if unknown."""
    if not job_id:
        return None
    result = runner(['squeue', '-h', '-j', str(job_id), '-o', '%T'], capture_output=True, text=True)
    state = result.stdout.strip().split('\n')[0].strip() if result.returncode == 0 else ''
    if state:
        return state
    result = runner(['sacct', '-n', '-X', '-j', str(job_id), '-o', 'State'], capture_output=True, text=True)
    state = result.stdout.strip().split('\n')[0].strip() if result.returncode == 0 else ''
    return state.split()[0] if state else None


class Lease:
    def __init__(self, campaign_root, run_id, runner=subprocess.run):
        self.root = Path(campaign_root)
        self.run_id = run_id
        self.dir = self.root / 'leases' / run_id
        self.path = self.dir / 'LEASE.json'
        self.runner = runner

    def read(self):
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return dict(state='CLAIMING', partial=True)   # exists => held

    def claim(self, attempt, job_id, profile, host=None):
        self.dir.mkdir(parents=True, exist_ok=True)
        record = dict(run_id=self.run_id, attempt=attempt, job_id=str(job_id), profile=profile,
                      host=host or os.uname().nodename, pid=os.getpid(), claimed=now(),
                      heartbeat=now(), state='CLAIMED')
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False, self.read()
        with os.fdopen(fd, 'w') as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        append_jsonl(self.dir / 'events.jsonl', dict(event='claimed', **record))
        return True, record

    def _owned(self, attempt):
        current = self.read()
        if not current or current.get('attempt') != attempt:
            raise RuntimeError(f'Lease for {self.run_id} is not owned by {attempt}: {current}')
        return current

    def update(self, attempt, **fields):
        current = self._owned(attempt)
        if 'state' in fields and fields['state'] not in STATES:
            raise ValueError(fields['state'])
        current.update(fields, heartbeat=now())
        atomic_write_json(self.path, current)
        if 'state' in fields:
            append_jsonl(self.dir / 'events.jsonl', dict(event='state', attempt=attempt, **fields))
        return current

    def stale_evidence(self, max_age=900):
        current = self.read()
        if current is None:
            return None
        state = slurm_state(current.get('job_id'), self.runner)
        evidence = dict(lease=current, slurm_state=state, heartbeat_age=now() - current.get('heartbeat', 0))
        stale = (state is not None and state.split('+')[0] not in LIVE_SLURM
                 and evidence['heartbeat_age'] > max_age and current.get('state') not in VERIFIED)
        evidence['stale'] = bool(stale)
        return evidence

    def takeover_if_stale(self, attempt, job_id, profile, max_age=900):
        evidence = self.stale_evidence(max_age)
        if not evidence or not evidence['stale']:
            return False, evidence
        archive = self.dir / 'history' / f'LEASE.{int(now())}.{evidence["lease"].get("attempt")}.json'
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.path, archive)          # atomic; only one contender succeeds
        except FileNotFoundError:
            return False, evidence
        append_jsonl(self.dir / 'events.jsonl', dict(event='stale_archived', by=attempt, evidence=evidence))
        ok, record = self.claim(attempt, job_id, profile)
        return ok, dict(evidence=evidence, claim=record)


def ledger_jobs(campaign_root, run_id):
    rows = []
    path = Path(campaign_root) / 'submission_ledger.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') == 'submitted' and row.get('run_id') == run_id:
                rows.append(row)
    return rows


def scheduler_identity(job_id, runner=subprocess.run):
    out = runner(['scontrol', 'show', 'job', str(job_id), '-o'], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    fields = {}
    for token in out.stdout.split():
        if '=' in token:
            key, value = token.split('=', 1)
            fields.setdefault(key, value)
    return fields


def cancel_recorded_sibling(campaign_root, run_id, winner_attempt, runner=subprocess.run,
                            user=None):
    """Cancel only the ledger-recorded sibling(s) of the winner for this run."""
    user = user or os.environ.get('USER')
    actions = []
    for row in ledger_jobs(campaign_root, run_id):
        if row['attempt'] == winner_attempt:
            continue
        job = str(row['job_id'])
        identity = scheduler_identity(job, runner)
        state = slurm_state(job, runner)
        record = dict(run_id=run_id, sibling_attempt=row['attempt'], job_id=job, state=state)
        if not identity or (state or '').split('+')[0] not in LIVE_SLURM:
            record['action'] = 'none_not_live'
        elif (identity.get('JobName') != row['job_name'] or not identity.get('UserId', '').startswith(f'{user}(')
              or identity.get('Comment') != row['comment']):
            record['action'] = 'refused_identity_mismatch'
            record['identity'] = {k: identity.get(k) for k in ('JobName', 'UserId', 'Comment')}
        else:
            result = runner(['scancel', job], capture_output=True, text=True)
            record['action'] = 'cancelled' if result.returncode == 0 else f'scancel_failed:{result.stderr.strip()}'
        append_jsonl(Path(campaign_root) / 'submission_ledger.jsonl', dict(event='sibling', **record))
        actions.append(record)
    return actions
