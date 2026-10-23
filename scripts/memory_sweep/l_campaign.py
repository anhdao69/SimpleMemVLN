"""Phase-B learned-ranker campaign (L0-L3): rankers, qualification and matched training on two lanes.

Every L arm reuses the completed M5_E1 recipe exactly (fresh original Qwen3.5-4B,
joint R2R + English RxR15deg, seed 429 and its data order, 3,852 updates, LR
5e-6, global batch 8, full BPTT, canonical text actions, frozen vision), like
the S arms; only the memory section differs: the frozen learned ranker is
embedded in it with its canonical SHA-256.  The ranker's inputs are computed
under no_grad and its weights/normalizers never change during training
(plan 11.7); navigation gradients reach the backbone through the retained K/V.

Lane B prepares and qualifies once for everyone:

* ``labels``: every selected episode has a complete label file
  (``gdn_utility_near_tail_v2``, shadow fixed-query LOO on frozen M5_E1);
* ``dataset``: columnar ranker rows, fit/dev/holdout by scene;
* ``rankers``: L0-L3 and the controls (L0_raw, L0_S0res, H_linear,
  H_mlp_matched) fitted on identical rows/folds/target/tuning budget, one
  process per GPU; fixed S scores evaluated on the same rows;
* ``arm-configs``: generated memory sections with the embedded artifacts;
* ``drift``: frozen-ranker score drift base vs M5_E1 on identical histories;
* ``cpu-suite``, ``provenance``, ``serving-replay`` (L arms on M5_E1 with the
  sweep-S0 == legacy-S0 harness parity), ``recovery`` (fresh-process resume of L3).

Lane A waits for its predecessor (the S campaign's lane A) and the seal.  Per
arm: longest-RxR smoke, then production with identity-bound automatic resume,
completion validation and pruning of rolling checkpoints, exactly as for S.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.memory_sweep import s_campaign as base
from scripts.memory_sweep.s_campaign import WORLD, expected_config, ready, sha, summarize_replay, write

VERSION = 'm5_gdn_l_sweep_campaign_v1'
ARMS = ('L0', 'L1', 'L2', 'L3')
CONTROLS = ('L0_raw', 'L0_S0res', 'H_linear', 'H_mlp_matched')
FIT_SHARDS = (('L3', 'L0'), ('L2', 'L0_raw', 'L0_S0res'), ('L1', 'H_linear'), ('H_mlp_matched',))
PREPARATION_GATES = ('labels', 'dataset', 'rankers', 'arm-configs', 'drift')
QUALIFICATION_GATES = ('cpu-suite', 'provenance', 'serving-replay', 'recovery')
CPU_SUITE = ('tests/memory_sweep', 'tests/vln/test_episodic_contract.py', 'tests/vln/test_episodic_selector.py',
             'tests/vln/test_episodic_attention.py', 'tests/vln/test_episodic_forward.py',
             'tests/vln/test_committed_write_reference.py')


def merge_fit_reports(reports):
    """One ledger from the per-GPU shards; shard 0 alone evaluates the fixed scores."""
    merged = None
    for report in reports:
        if merged is None:
            merged = copy.deepcopy(report)
            continue
        for key in ('target', 'seed', 'lrs', 'weight_decay', 'batch', 'epochs', 'clip', 'patience',
                    'budget_rows', 'dataset', 'source_commit', 'h_projection'):
            if report.get(key) != merged.get(key):
                raise ValueError(f'Fit shards disagree on {key}')
        if set(report['arms']) & set(merged['arms']):
            raise ValueError('An arm was fitted twice')
        merged['arms'].update(report['arms'])
        if report['fixed']:
            if merged['fixed']:
                raise ValueError('Fixed scores evaluated twice')
            merged['fixed'] = report['fixed']
    return merged


class Campaign(base.Campaign):
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['version'] != VERSION:
            raise ValueError('Unknown campaign version')
        self.root = Path(self.data['root'])
        self.source = Path(self.data['source'])
        self.base = Path(self.data['base'])
        self.shared = self.root / 'shared'
        self.configs = self.root / 'configs'
        self.labels = self.root / 'labels'
        self.dataset = self.root / 'ranker_data'
        self.rankers = self.root / 'rankers'

    def arm_memory_path(self, arm, role='training'):
        if role != 'training' and arm not in ARMS:          # S0 screening config for the replay parity
            return self.source / f'configs/memory_sweep/arm_{arm}.yaml'
        return self.configs / f'arm_{arm}.yaml'

    def artifact(self, arm):
        path = self.rankers / f'{arm}_artifact.json'
        artifact = json.loads(path.read_text())
        from qwen_vl.research.gdn_ranker import canonical_sha256
        ledger = json.loads((self.rankers / 'fit_report.json').read_text())
        digest = canonical_sha256(artifact)
        if ledger['arms'][arm]['artifact_sha256'] != digest:
            raise ValueError(f'{arm}: artifact differs from the fit ledger')
        return artifact, digest

    def run_identity(self, arm):
        import yaml
        from qwen_vl.contracts import validate_config
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        reference = json.loads(Path(self.data['m5_e1_reference_config']).read_text())
        memory = yaml.safe_load(self.arm_memory_path(arm).read_text())['memory']
        cfg = self.resolve(arm)
        if cfg != expected_config(reference, memory):
            raise ValueError(f'{arm}: resolved config differs from the M5_E1 recipe beyond memory/loader')
        validate_config(copy.deepcopy(cfg), WORLD)
        spec = parse_episodic_spec(cfg)
        if spec.score != arm or spec.version != 'episodic_window_gdn_sweep_v1' or not spec.is_learned:
            raise ValueError(f'{arm}: config declares {spec.score}')
        _, digest = self.artifact(arm)
        if spec.ranker_sha256 != digest:
            raise ValueError(f'{arm}: embedded ranker differs from the fitted artifact')
        seal = self.shared / 'QUALIFIED.json'
        return dict(campaign=VERSION, arm=arm, config=cfg, source_commit=self.source_identity(),
                    manifest_sha256=self.data['manifest_sha256'], model=self.data['model'],
                    memory_sha256=sha(self.arm_memory_path(arm)), runtime_sha256=sha(self.configs / 'runtime.yaml'),
                    ranker_sha256=digest, fit_report_sha256=sha(self.rankers / 'fit_report.json'),
                    qualification_sha256=sha(seal) if seal.exists() else None)


class Lane(base.Lane):
    def wait_predecessor(self):
        """A predecessor is a controller state reaching its declared final status, or just a Slurm step ending."""
        predecessor = self.spec.get('predecessor')
        if predecessor:
            final = predecessor.get('final_status', 'complete')
            while True:
                status = None
                if predecessor.get('state'):
                    try:
                        status = json.loads(Path(predecessor['state']).read_text()).get('status')
                    except (OSError, ValueError):
                        status = 'unreadable'
                alive = predecessor['step'] in self.active_steps()
                if self.control('OVERRIDE_PREDECESSOR'):
                    self.record('predecessor-overridden', predecessor_status=status)
                    break
                if not alive and (not predecessor.get('state') or status == final):
                    break
                self.record('waiting-predecessor', predecessor_status=status, predecessor_alive=alive)
                time.sleep(120)
        while not (ready(self.active_steps(), self.own, self.job) and not self.gpu_busy()):
            self.record('waiting-for-idle-allocation', steps=self.active_steps())
            time.sleep(60)

    def wait_qualification(self):
        seal = self.campaign.shared / 'QUALIFIED.json'
        while not seal.exists():
            self.record('waiting-shared-qualification')
            time.sleep(120)
        data = json.loads(seal.read_text())
        if not data.get('passed') or data.get('source_commit') != self.campaign.source_identity():
            raise RuntimeError('Shared qualification failed or belongs to another source')
        return data

    def parallel(self, name, commands):
        """One process per GPU; every process must succeed."""
        processes = []
        for index, command in enumerate(commands):
            log = (self.dir / f'{name}-gpu{index}-{time.time_ns()}.log').open('x')
            processes.append((subprocess.Popen(command, cwd=self.campaign.source, stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index),
                                                        OMP_NUM_THREADS='12')), log))
        codes = [p.wait() for p, _ in processes]
        for _, log in processes:
            log.close()
        if any(codes):
            raise RuntimeError(f'{name} failed with exits {codes}')

    def qualify(self):
        c, shared = self.campaign, self.campaign.shared
        seal = shared / 'QUALIFIED.json'
        if seal.exists():
            return self.wait_qualification()
        shared.mkdir(parents=True, exist_ok=True)
        gates = json.loads((shared / 'gates.json').read_text()) if (shared / 'gates.json').exists() else {}

        def gate(name, fn):
            if gates.get(name, {}).get('passed'):
                return
            self.record('qualifying', gate=name)
            result = fn()
            gates[name] = dict(passed=True, result=result, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write(shared / 'gates.json', gates)
        source = c.source_identity()
        py = sys.executable

        def labels():
            out = shared / 'labels_verified.json'
            self.run_logged('labels', [py, '-m', 'scripts.memory_sweep.l_campaign', '--campaign', str(c.path),
                                       '--verify-labels', str(out)])
            return json.loads(out.read_text())
        gate('labels', labels)

        def dataset():
            if c.dataset.exists():
                raise RuntimeError(f'Preserve interrupted dataset {c.dataset} before retry')
            self.run_logged('dataset', [py, '-c', 'import sys; from qwen_vl.research.gdn_ranker_dataset import build; '
                                        'build(sys.argv[1], sys.argv[2], subset="full")', str(c.labels), str(c.dataset)])
            return dict(manifest_sha256=sha(c.dataset / 'manifest.json'), rows_sha256=sha(c.dataset / 'rows.npz'))
        gate('dataset', dataset)

        def rankers():
            shards = c.rankers / 'shards'
            if shards.exists():
                raise RuntimeError(f'Preserve interrupted ranker fit {shards} before retry')
            commands = []
            for index, arms in enumerate(FIT_SHARDS):
                command = [py, '-m', 'scripts.memory_sweep.train_ranker', '--dataset', str(c.dataset),
                           '--out', str(shards / f'shard{index}'), '--source-commit', source, '--device', 'cuda',
                           '--arms', *arms]
                commands.append(command + ([] if index == 0 else ['--no-fixed']))
            self.parallel('rankers', commands)
            reports = [json.loads((shards / f'shard{i}' / 'fit_report.json').read_text())
                       for i in range(len(FIT_SHARDS))]
            merged = merge_fit_reports(reports)
            if set(merged['arms']) != set(ARMS) | set(CONTROLS) or not merged['fixed']:
                raise RuntimeError('Ranker ledger is incomplete')
            for index, arms in enumerate(FIT_SHARDS):
                for arm in arms:
                    name = f'{arm}_artifact.json' if arm in ARMS else f'{arm}_control.pt'
                    shutil.copy2(shards / f'shard{index}' / name, c.rankers / name)
            write(c.rankers / 'fit_report.json', merged)
            return dict(fit_report_sha256=sha(c.rankers / 'fit_report.json'),
                        artifacts={arm: merged['arms'][arm]['artifact_sha256'] for arm in ARMS},
                        holdout_capture={arm: merged['arms'][arm]['holdout']['capture'] for arm in (*ARMS, *CONTROLS)},
                        holdout_capture_fixed={k: v['ranker_holdout']['capture'] for k, v in merged['fixed'].items()})
        gate('rankers', rankers)

        def arm_configs():
            import yaml
            from qwen_vl.research.gdn_sweep_contract import arm_memory_config
            for arm in ARMS:
                artifact, digest = c.artifact(arm)
                memory = arm_memory_config(arm, artifacts=dict(ranker=artifact, ranker_sha256=digest))
                path = c.arm_memory_path(arm)
                if path.exists():
                    if yaml.safe_load(path.read_text()) != memory:
                        raise RuntimeError(f'{path} exists with different content')
                else:
                    path.write_text(yaml.safe_dump(memory, sort_keys=False))
            return {arm: dict(memory_sha256=sha(c.arm_memory_path(arm)),
                              kv_guard=c.run_identity(arm)['config']['memory']['kv_guard_tokens']) for arm in ARMS}
        gate('arm-configs', arm_configs)

        def drift():
            out = c.rankers / 'drift_base_vs_m5_e1.json'
            self.run_logged('drift', [py, '-m', 'scripts.memory_sweep.ranker_drift', '--checkpoints', c.data['m5_e1'],
                                      'base', '--artifacts', *[str(c.rankers / f'{a}_artifact.json') for a in ARMS],
                                      '--selection', str(c.labels / 'selection.json'), '--manifest', c.data['manifest'],
                                      '--model-path', c.data['model'], '--runtime-profile', str(c.configs / 'runtime.yaml'),
                                      '--out', str(out)], env_extra=dict(CUDA_VISIBLE_DEVICES='0'))
            report = json.loads(out.read_text())['comparisons']['base']
            return {arm: dict(spearman=sum(v['spearman_mean'] for v in layers.values()) / len(layers),
                              bank_overlap=sum(v['bank_overlap_mean'] for v in layers.values()) / len(layers))
                    for arm, layers in report.items()}
        gate('drift', drift)

        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                                                 *CPU_SUITE]) or 'pass')

        def provenance():
            self.run_logged('provenance', [py, '-m', 'scripts.memory_sweep.audit_manifest', '--campaign', str(c.path),
                                           '--out', str(shared / 'provenance.json')])
            return json.loads((shared / 'provenance.json').read_text())['checks']
        gate('provenance', provenance)

        def serving_replay():
            arms = c.data['screening_arms']
            shards = [arms[i::WORLD] for i in range(WORLD)]
            commands = []
            for index, shard in enumerate(shards):
                command = [py, '-m', 'scripts.memory_sweep.replay_check', '--checkpoint', c.data['m5_e1'],
                           '--model-path', c.data['model'], '--manifest', c.data['manifest'],
                           '--out', str(shared / 'replay' / f'shard{index}.json')]
                for arm in shard:
                    command += ['--arm', arm, str(c.arm_memory_path(arm, 'screening'))]
                commands.append(command)
            processes = []
            for index, command in enumerate(commands):
                log = (self.dir / f'replay-shard{index}-{time.time_ns()}.log').open('x')
                processes.append((subprocess.Popen(command, cwd=c.source, stdout=log, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index))), log))
            codes = [p.wait() for p, _ in processes]
            for _, log in processes:
                log.close()
            records = []
            for index in range(WORLD):
                path = shared / 'replay' / f'shard{index}.json'
                if not path.exists():           # a crash, not a tolerance result
                    raise RuntimeError(f'Serving replay shard {index} crashed (exit {codes[index]})')
                records += json.loads(path.read_text())['replay']
            summary = summarize_replay(records)
            for arm in ARMS:
                for key, record in summary['records'].items():
                    if key.startswith(arm + '/') and record['max_resident_kv'] > record['kv_guard']:
                        raise RuntimeError(f'{key} exceeded its KV guard')
            return summary
        gate('serving-replay', serving_replay)

        def recovery():
            arm, root = c.data['recovery_arm'], shared / 'recovery'
            fixture = Path(c.data['fixtures']) / 'mixed_tail.jsonl'
            diagnostic = self.diagnostic_runtime()
            for name, extra in (('reference', dict(save_every=2)),
                                ('resumed', dict(save_every=2, resume=root / 'reference/train/checkpoint-2', first_save=3))):
                out = root / name
                if out.exists():
                    raise RuntimeError(f'Preserve interrupted recovery evidence {out} before retry')
                self.run_logged(f'recovery-{name}', base.train_command(c, arm, out, manifest=fixture, updates=3,
                                                                       runtime=diagnostic, host_guard=self.host_guard,
                                                                       **extra))
                self.check_profile(out / 'train', 3, 1 if name == 'reference' else 3)
                cfg = json.loads((out / 'train/resolved_config.json').read_text())
                if cfg != c.resolve(arm, diagnostic):
                    raise RuntimeError('Recovery diagnostic config differs')
            self.run_logged('recovery-compare', [py, '-m', 'scripts.memory_sweep.compare_recovery',
                                                 '--root', str(root), '--out', str(root / 'comparison.json')])
            return json.loads((root / 'comparison.json').read_text())
        gate('recovery', recovery)
        expected = set(PREPARATION_GATES) | set(QUALIFICATION_GATES)
        if set(gates) != expected or not all(g['passed'] for g in gates.values()):
            raise RuntimeError('Incomplete shared qualification')
        write(seal, dict(passed=True, source_commit=source, gates=gates,
                         fit_report_sha256=gates['rankers']['result']['fit_report_sha256'],
                         artifacts=gates['rankers']['result']['artifacts']))
        self.record('qualified')
        return json.loads(seal.read_text())


def verify_labels(campaign, out):
    """Every selected episode has a complete label file from the declared label source."""
    import numpy as np
    from scripts.memory_sweep.build_labels import episode_file
    labels = campaign.labels
    selection_text = (labels / 'selection.json').read_text()
    source = json.loads((labels / 'LABEL_SOURCE.json').read_text())
    import hashlib
    if hashlib.sha256(selection_text.encode()).hexdigest() != source['selection_sha256']:
        raise ValueError('Selection differs from the label source record')
    if source['m5_e1_weights_sha256'] != campaign.data['m5_e1_weights_sha256']:
        raise ValueError('Labels were not collected on M5_E1')
    selection = json.loads(selection_text)
    totals = dict(episodes=0, tokens=0, gpu_seconds=0.0, labels=0, stable_fallbacks=0, dense_episodes=0)
    recon_max, detection_min = 0.0, 1.0
    for e in selection['episodes']:
        with np.load(episode_file(labels, e['episode_uid'])) as z:
            meta = json.loads(str(z['meta']))
            if meta['episode'] != e or meta['m5_e1'] != campaign.data['m5_e1']:
                raise ValueError(f'{e["episode_uid"]}: label file does not match the selection or checkpoint')
            coverage = meta['coverage']
            if coverage['missing'] or coverage['finite'] != coverage['expected'] or coverage.get('dense_missing'):
                raise ValueError(f'{e["episode_uid"]}: incomplete labels {coverage}')
            totals['labels'] += coverage['finite']
        totals['episodes'] += 1
        totals['tokens'] += meta['tokens']
        totals['gpu_seconds'] += meta['seconds']
        totals['stable_fallbacks'] += meta['stable_fallbacks']
        totals['dense_episodes'] += int(meta['dense'])
        for record in meta['reconstruction'].values():
            recon_max = max(recon_max, record['max_rel_error'])
            if record['bank_ablation_detected'] is not None:
                detection_min = min(detection_min, record['bank_ablation_detected'])
    report = dict(totals, source=source, reconstruction_max_rel_error=recon_max,
                  bank_ablation_detected_min=detection_min)
    if totals['episodes'] != len(selection['episodes']) or recon_max > 0.02 or detection_min < 0.9:
        raise ValueError(f'Label verification failed: {report}')
    write(out, report)
    print('LABELS_VERIFIED', json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--lane')
    p.add_argument('--preflight', action='store_true', help='Check identities of every arm without GPU work')
    p.add_argument('--verify-labels', metavar='OUT')
    args = p.parse_args()
    campaign = Campaign(args.campaign)
    if args.verify_labels:
        return verify_labels(campaign, args.verify_labels)
    if args.preflight:
        report = {}
        for arm in ARMS:
            if not campaign.arm_memory_path(arm).exists():
                report[arm] = 'pending-ranker-fit'
                continue
            identity = campaign.run_identity(arm)
            report[arm] = dict(memory_sha256=identity['memory_sha256'], ranker_sha256=identity['ranker_sha256'],
                               kv_guard=identity['config']['memory']['kv_guard_tokens'])
        print(json.dumps(report, indent=2))
        print('PREFLIGHT_PASS', flush=True)
        return
    Lane(campaign, args.lane).main()


if __name__ == '__main__':
    main()
