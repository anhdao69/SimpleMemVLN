"""The simulator subprocess renders on the trainer rank's own device under a scheduler mask."""
import json
import sys

import pytest

from qwen_vl.rl.env_client import EnvClient

FAKE_SERVER = '''
import json, os, sys
print(json.dumps(dict(status='ready', cvd=os.environ.get('CUDA_VISIBLE_DEVICES'), argv=sys.argv[1:])), flush=True)
sys.stdin.readline()
'''


def start(tmp_path, gpu):
    python = tmp_path / 'python'
    python.write_text(f'#!{sys.executable}\nimport runpy, sys\nsys.argv = sys.argv[3:]\n'
                      f'exec({FAKE_SERVER!r})\n')
    python.chmod(0o755)
    client = EnvClient(str(python), 'cfg.yaml', 'data', 'scenes', gpu, 'ids.json')
    info = client.info
    client.close()
    return info['cvd'], info['argv'][info['argv'].index('--gpu') + 1]


@pytest.mark.parametrize('mask,gpu,expected', [('3,5', 1, ('5', '0')), ('3,5', 0, ('3', '0')),
                                               ('0,1', 1, ('1', '0'))])
def test_masked_rank_device_is_the_simulator_ordinal_zero(tmp_path, monkeypatch, mask, gpu, expected):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    assert start(tmp_path, gpu) == expected


def test_unmasked_host_keeps_the_physical_ordinal(tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert start(tmp_path, 1) == (None, '1')
