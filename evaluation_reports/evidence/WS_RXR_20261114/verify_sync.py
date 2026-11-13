#!/usr/bin/env python3
"""Recheck transferred compact metrics, certificates, report cells and archives."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
verification = read(HERE / 'SYNC_VERIFICATION.json')
manifest = read(HERE / 'SYNC_MANIFEST.json')
rb = read(REPORTS / 'evidence/S8N_R_B_ABLATION_SR_20261113.json')
stage = read(REPORTS / 'evidence/S8N_STAGE2_REMAINING_RXR_SR_20261113.json')
expected = {r['model']: r['rxr15'] for r in rb['results']}
expected[stage['model']] = stage
assert len(expected) == len(verification['checks']) == 9
total = 0
for item in verification['checks']:
    folder = HERE / item['model']
    cert_path = folder / 'FINAL_VERIFIED.json'
    metrics_path = folder / 'episode_metrics.jsonl'
    cert = read(cert_path)
    assert sha(cert_path) == item['certificate_sha256'] == expected[item['model']]['source_file_sha256']
    assert sha(metrics_path) == item['episode_metrics_sha256']
    assert cert['journal_sha256'] == item['journal_sha256']
    ids, successes, failures = set(), 0, 0
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        eid = str(row['episode_id'])
        assert eid not in ids
        ids.add(eid)
        successes += int(not row.get('failed', False) and row['metrics'].get('success', 0) == 1)
        failures += int(bool(row.get('failed', False)))
    assert len(ids) == item['episodes'] == expected[item['model']]['episodes'] == 3669
    assert successes == item['successes'] == expected[item['model']]['successes']
    assert failures == item['failures']
    assert abs(successes / len(ids) * 100 - expected[item['model']]['sr_percent']) < 1e-9
    total += len(ids)

old_root = REPORTS / 'archive/WS_RXR_20261114/prior_main'
old_rb = read(old_root / 'evidence/S8N_R_B_ABLATION_SR_20261113.json')
rows = list(csv.DictReader((REPORTS / 'evidence/S8N_R_B_ABLATION_SR_20261113.csv').open()))
assert len(rows) == len(rb['results']) == 8
for before, after, row in zip(old_rb['results'], rb['results'], rows):
    assert before['model'] == after['model'] == row['Model']
    assert before['r2r'] == after['r2r']
    r2r, rxr = (f"{after[k]['sr_percent']:.2f}" for k in ('r2r', 'rxr15'))
    assert row['SR R2R (%)'] == r2r and row['SR RxR (%)'] == rxr
    for name in ('ALL_RESULTS.md', 'S8N_R_B_ABLATION.md'):
        assert f"| {after['model']} | {r2r} | {rxr} |" in (REPORTS / name).read_text()
stage_rows = list(csv.DictReader((REPORTS / 'evidence/S8N_STAGE2_REMAINING_RXR_SR_20261113.csv').open()))
assert stage_rows == [{'Model': stage['model'], 'SR RxR (%)': f"{stage['sr_percent']:.2f}"}]
marker = '## Historical R2R index through 2026-11-01'
assert (old_root / 'ALL_RESULTS.md').read_text().split(marker, 1)[1] == (REPORTS / 'ALL_RESULTS.md').read_text().split(marker, 1)[1]
for name, digest in manifest['published_report_sha256'].items():
    assert sha(REPORTS / name) == digest
audit = read(REPORTS / 'evidence/RXR_RERUN_RESULTS_FINAL_AUDIT_20261114.json')
for name, digest in audit['published_file_sha256'].items():
    path = REPORTS / name if name.startswith('evidence/') else REPORTS / 'archive/WS_RXR_20261114/server_original' / name
    assert sha(path) == digest
assert total == 33021
print(f'PASS: 9 journals, {total} compact episode records, certificate and transfer hashes, CSV/Markdown cells, unchanged R2R evidence and historical index.')
print('Full server journal hashes were freshly checked during export; this offline verifier does not repeat simulator or media audits.')
