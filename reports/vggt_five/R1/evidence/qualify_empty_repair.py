"""One-campaign R1 repair audit, with exact revision binding (not a general bypass).

Run inside allocation4925 from source-bcb14ea. Reuse supported-path experiments
only after real-FLA output/gradient equality and all-rank positive-support proof.
Empty-support and serving gates execute again. All original evidence is retained.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.vln.queue_vggt_r1 import check_baseline, check_qualification, source_files, write_seal
from qwen_vl.train.recovery import validate_complete
import yaml

B = Path('/mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015')
SOURCE = B/'source-bcb14ea'
OLD_SOURCE = B/'source-dca86b6'
PRIOR = B/'attempt2/qualification'
ROOT = B/'attempt3/qualification'
MODULE = 'src/qwen_vl/models/vggt_r1_forward.py'
OLD_SHA = '376627007604bad544399d8f94a4aec0b7294669420097ba83955001c400eaa5'
NEW_SHA = '853af8ddf838bed3483e9a81cc92a1da3d89a5021ccab9e177eeb72c99510471'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def option(command, key):
    assert command.count(key) == 1, (key, command)
    return command[command.index(key)+1]


def replay_proof(name, fixture):
    folder = B/name
    buckets = [rows(folder/f'buckets_rank{rank}.jsonl') for rank in range(4)]
    assert buckets[0] and all(b == buckets[0] for b in buckets), name
    supports = []
    for rank in range(4):
        identity = read(folder/f'identity_rank{rank}.json')
        assert identity['completed'] is True and identity['module_path'] == str(SOURCE/MODULE)
        assert identity['module_sha256'] == NEW_SHA and identity['source_files'] == source_files(SOURCE)
        assert identity['fixture'] == str(B/fixture) and identity['fixture_sha256'] == digest(B/fixture)
        assert identity['slurm_job'] == '4925' and identity['node'] == 'worker-2'
        argv = identity['argv']
        for key, value in {'--deepspeed':'configs/deepspeed_zero2_cpu_optimizer.json',
                           '--runtime_profile':'configs/vggt/r1_omega_h4.yaml',
                           '--policy_config':'configs/vggt/r1_joint_one_epoch.yaml',
                           '--memory_config':'configs/vln_memory_window8.yaml',
                           '--max-optimizer-updates':'2'}.items():
            assert option(argv,key) == value
        assert '--profile-only' in argv
        assert not any('gradient_accumulation' in arg or 'batch_size' in arg for arg in argv)
        profiles = rows(folder/'train'/f'profile_rank{rank}.jsonl')
        assert [r['update'] for r in profiles] == [1, 2], (name, rank, profiles)
        exposures = rows(folder/'train'/f'exposures_rank{rank}.jsonl')
        assert len(exposures) == 4, (name, rank)
        supports.extend(r['aux_pair_count'] for r in exposures)
    assert 0 in supports and max(supports) > 0
    return dict(name=name, identical_collective_sequences=True,
                buckets_per_rank=len(buckets[0]), updates=2,
                fixture_sha256=digest(B/fixture), log_sha256=digest(B/(name+'.log')),
                bucket_sha256=[digest(folder/f'buckets_rank{rank}.jsonl') for rank in range(4)],
                execution_identity_sha256=[digest(folder/f'identity_rank{rank}.json') for rank in range(4)])


def main():
    assert Path.cwd().resolve() == SOURCE
    expected = source_files(SOURCE)
    prior = check_qualification(OLD_SOURCE, PRIOR/'qualification_pass.json')
    old = prior['source_files']
    changed = {k for k in old.keys() | expected.keys() if old.get(k) != expected.get(k)}
    assert changed == {MODULE} and old[MODULE] == OLD_SHA and expected[MODULE] == NEW_SHA
    assert digest(OLD_SOURCE/MODULE) == OLD_SHA and digest(SOURCE/MODULE) == NEW_SHA
    parity = read(B/'repair_supported_parity.json')
    assert parity['passed'] and parity['old_source'] == str(OLD_SOURCE)
    assert parity['old_module_sha256'] == OLD_SHA and parity['new_module_sha256'] == NEW_SHA
    assert parity['new_module_path'] == str(SOURCE/MODULE)
    assert [r['pairs'] for r in parity['cases']] == [2, 0]
    assert all(r['outputs_exact'] and r['loss_exact'] and r['all_gradients_exact'] for r in parity['cases'])
    repair = dict(old_module_sha256=OLD_SHA, new_module_sha256=NEW_SHA,
                  harness_sha256=digest(__file__), parity_sha256=digest(B/'repair_supported_parity.json'),
                  parity_script_sha256=digest(B/'repair_supported_parity.py'),
                  prior_seal_sha256=digest(PRIOR/'qualification_pass.json'),
                  replay_script_sha256=digest(B/'repro_empty_verified.py'),
                  replays=[replay_proof('empty-repro-fixed-verified', 'stall-update19-fixture.jsonl'),
                           replay_proof('empty-repro-reversed-verified', 'stall-update19-reversed.jsonl')])
    # Require reversed accumulation slots to contain exactly the same episodes.
    original = rows(B/'stall-update19-fixture.jsonl')
    assert rows(B/'stall-update19-reversed.jsonl') == original[4:] + original[:4]
    # Original source must reproduce the actual collective failure.
    original_log = (B/'empty-repro-original2.log').read_text()
    assert '48501376' in original_log and '49813504' in original_log
    repair['original_failure_log_sha256'] = digest(B/'empty-repro-original2.log')
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'qualification_pass.json').exists()
    (ROOT/'source_start.json').write_text(json.dumps(expected, indent=2))
    cfg = yaml.safe_load((SOURCE/'configs/vggt/r1_omega_h4.yaml').read_text())
    check_baseline(cfg)
    records = []
    for previous in prior['records']:
        assert source_files(SOURCE) == expected
        name, command = previous['name'], list(previous['command'])
        # All other gates execute under the repaired source, including export,
        # recovery comparison, both full suites, and mixed empty-support batches.
        carry = 'scripts.vln.profile_vggt_g1' in command and name != 'mixed-empty'
        if carry:
            origin = Path(option(command, '--root'))
            folder = origin/'experiments'/name
            fixture = Path(option(command, '--fixture'))
            manifest = read(folder/'source_manifest.json')
            assert manifest == read(origin/'source_start.json')
            if origin == PRIOR:
                assert manifest == old
            else:
                # Earlier carried experiments permit only the two already
                # reviewed audit-script differences; no training differences.
                assert origin == B/'qualification'
                differences = {k for k in old.keys() | manifest.keys() if old.get(k) != manifest.get(k)}
                assert differences <= {'scripts/vln/check_vggt_export.py', 'scripts/vln/qualify_vggt_r1.py'}
                carry_record = previous['carried_evidence']
                assert carry_record['source_manifest_sha256'] == digest(origin/'source_start.json')
                assert carry_record['log_sha256'] == digest(origin/(name+'.log'))
            result = read(folder/'result.json')
            assert result['passed'] and result['fixture_sha256'] == digest(fixture)
            assert result['check']['peak_allocated_gib'] <= 72
            exposure_hashes, counts = [], []
            for rank in range(4):
                file = folder/'train'/f'exposures_rank{rank}.jsonl'
                exposure = rows(file)
                updates = int(option(command, '--updates'))
                first = int(option(command, '--first-update')) if '--first-update' in command else 1
                assert len(exposure) == 2*(updates-first+1), (name, rank, len(exposure))
                assert all(r['aux_pair_count'] > 0 for r in exposure), (name, rank)
                exposure_hashes.append(digest(file)); counts.append(len(exposure))
            inventories = []
            if '--save' in command:
                checkpoint = folder/'train'/('checkpoint-'+option(command, '--updates'))
                saved = validate_complete(checkpoint)
                assert saved['world_size'] == 4 and saved['step'] == int(option(command, '--updates'))
                inventories.append(dict(path=str(checkpoint), marker_sha256=digest(checkpoint/'RECOVERY_COMPLETE.json')))
            records.append(dict(previous, repair_carry=dict(
                reason='all observed micros have support; exact old/new real-FLA outputs and all gradients',
                original_root=str(origin), exposure_sha256=exposure_hashes, micro_counts=counts,
                source_manifest_sha256=digest(folder/'source_manifest.json'),
                result_sha256=digest(folder/'result.json'), fixture_sha256=digest(fixture),
                full_checkpoint_inventories=inventories)))
            print('CARRIED_POSITIVE_SUPPORT', name, counts, flush=True)
        else:
            # Relocate outputs only, retaining saved recovery inputs.
            for flag in ('--out', '--output', '--root'):
                if flag in command:
                    index = command.index(flag)+1
                    if flag == '--root': command[index] = str(ROOT)
                    else: command[index] = str(ROOT/Path(command[index]).name)
            print('START', name, flush=True)
            with (ROOT/(name+'.log')).open('x') as log:
                completed = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
            record = dict(name=name, command=command, returncode=completed.returncode)
            records.append(record)
            (ROOT/'qualification.json').write_text(json.dumps(records, indent=2))
            assert completed.returncode == 0, name
            if name == 'mixed-empty':
                folder = ROOT/'experiments'/name
                result = read(folder/'result.json')
                assert result['passed'] and result['fixture_sha256'] == digest(option(command,'--fixture'))
                assert read(folder/'source_manifest.json') == expected
                supports = [r['aux_pair_count'] for rank in range(4) for r in rows(folder/'train'/f'exposures_rank{rank}.jsonl')]
                assert 0 in supports and max(supports) > 0
            print('DONE', name, flush=True)
        (ROOT/'qualification.json').write_text(json.dumps(records, indent=2))
    (ROOT/'repair_proof.json').write_text(json.dumps(repair, indent=2))
    for record in records:
        record['repair_proof_sha256'] = digest(ROOT/'repair_proof.json')
    (ROOT/'qualification.json').write_text(json.dumps(records, indent=2))
    write_seal(SOURCE, ROOT/'qualification_pass.json', records, expected)
    print('R1_REPAIR_QUALIFICATION_PASS', flush=True)


if __name__ == '__main__':
    main()
