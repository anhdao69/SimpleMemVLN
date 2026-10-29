"""scripts/rl/analyze_event_forks.py on a synthetic grpo_fork run whose fork informativeness rises with drift."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'rl' / 'analyze_event_forks.py'


def test_informativeness_rises_with_event_score(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / 'rollouts').mkdir()
    lines = []
    for i in range(400):
        n = 30
        drift = [[float(math.exp(rng.normal())), float(math.exp(rng.normal()))] for _ in range(n)]
        t = int(rng.integers(1, n))
        score = float(np.mean(np.log(drift[t])))
        informative = rng.random() < 1 / (1 + math.exp(-2 * score))
        main = 15.0 if rng.random() < 0.5 else 0.0
        branch = (15.0 - main) if informative else main
        lines.append(json.dumps(dict(episode_id=str(i % 100), uid=f'u{i}', N_star=20, num_steps=n, reward=main,
                                     step_drift=drift, step_entropy=[float(rng.random()) for _ in range(n)],
                                     fork_summary=[dict(fork_t=t, reward=branch)])))
    (tmp_path / 'rollouts' / 'step_0003_rank0.jsonl').write_text('\n'.join(lines) + '\n')
    out = tmp_path / 'report.json'
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), '--out', str(out)], check=True,
                   capture_output=True)
    report = json.loads(out.read_text())
    assert report['forks'] == 400 and report['episodes'] == 100
    p = [b['p_informative'] for b in report['by_event_quintile']]
    assert p[-1] > p[0] + 0.3 and sum(b['forks'] for b in report['by_event_quintile']) == 400
    assert report['spearman']['event_informative'] > 0.2
    names, z = report['logistic']['features'], report['logistic']['z']
    assert z[names.index('event')] > 4 and abs(z[names.index('entropy')]) < 3


def test_position_normalised_score_recovers_surprise_under_a_position_trend(tmp_path):
    """Event scores fall with t as in S8N; informativeness depends only on the surprise beyond that trend."""
    rng = np.random.default_rng(1)
    (tmp_path / 'rollouts').mkdir()
    lines = []
    for i in range(600):
        n = 60
        surprise = rng.normal(size=n)
        drift = [[float(math.exp(-0.5 * math.log1p(t) + 0.3 * surprise[t]))] * 2 for t in range(n)]
        t = int(rng.integers(1, n))
        informative = rng.random() < 1 / (1 + math.exp(-2 * surprise[t]))
        main = 15.0 if rng.random() < 0.5 else 0.0
        lines.append(json.dumps(dict(episode_id=str(i % 150), uid=f'u{i}', N_star=40, num_steps=n, reward=main,
                                     step_drift=drift, step_entropy=[float(rng.random()) for _ in range(n)],
                                     fork_summary=[dict(fork_t=t, reward=(15.0 - main) if informative else main)])))
    (tmp_path / 'rollouts' / 'step_0001_rank0.jsonl').write_text('\n'.join(lines) + '\n')
    out = tmp_path / 'report.json'
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), '--out', str(out)], check=True,
                   capture_output=True)
    report = json.loads(out.read_text())
    assert abs(report['spearman']['event_pos_t']) < 0.15 < abs(report['spearman']['event_t'])
    assert report['spearman']['event_pos_informative'] > report['spearman']['event_informative'] > 0
    names, z = report['logistic_position']['features'], report['logistic_position']['z']
    assert z[names.index('event_pos')] > 5 and abs(z[names.index('log1p_t')]) < 3
    p = [b['p_informative'] for b in report['by_event_pos_quintile']]
    assert p[-1] > p[0] + 0.4
