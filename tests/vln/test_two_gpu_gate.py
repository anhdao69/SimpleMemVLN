import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'train/slurm/verify_two_gpu_profile.py'


@pytest.mark.parametrize('fault', [None, 'missing_rank', 'missing_update', 'high_peak', 'nan_peak', 'bad_exposures', 'nan_loss', 'nan_rank_loss', 'wrong_mode'])
def test_two_gpu_gate_fails_closed(tmp_path, fault):
    # Removing a required rank/update/finite-value check must fail this test.
    uid = 'longest-episode'
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps({'episode_uid': uid, 'steps': [0]*627})+'\n' for _ in range(8)))
    for rank in range(2):
        rows = [dict(update=i, seconds=100., peak_reserved_gib=70.) for i in (1, 2, 3)]
        if fault == 'missing_rank' and rank == 1:
            continue
        if fault == 'missing_update' and rank == 1:
            rows.pop()
        if fault in ('high_peak', 'nan_peak') and rank == 1:
            rows[-1]['peak_reserved_gib'] = 79. if fault == 'high_peak' else float('nan')
        (tmp_path / f'profile_rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        exposures = [dict(update=i, episode_uid=uid, loss_sum=.3) for i in (0, 1, 2) for _ in range(4)]
        if fault == 'nan_rank_loss' and rank == 1:
            exposures[-1]['loss_sum'] = float('nan')
        if fault == 'bad_exposures':
            exposures.pop()
        (tmp_path / f'exposures_rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exposures))
    (tmp_path / 'resolved_config.json').write_text(json.dumps(dict(model=dict(output_mode='candidate_logits' if fault == 'wrong_mode' else 'qwen_text'), memory=dict(mode='window8'), training=dict(gradient_accumulation_steps=4, nominal_episodes_per_update=8))))
    (tmp_path / 'trainer_state.json').write_text(json.dumps(dict(global_step=3, log_history=[dict(step=i, loss=float('nan') if fault == 'nan_loss' else .3, grad_norm=1.) for i in (1, 2, 3)])))
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), str(manifest)], capture_output=True, text=True)
    if fault is None:
        assert result.returncode == 0, result.stderr
        gate = json.loads((tmp_path / 'PASS.json').read_text())
        assert gate['passed'] and gate['peak_reserved_gib'] == 70.
    else:
        assert result.returncode != 0
        assert not (tmp_path / 'PASS.json').exists()
