import os
from pathlib import Path

import pytest

MODEL_PATH = Path(os.environ.get('MODEL_PATH', '/data/anh/checkpoints/simplemem/qwen-base'))
CHECKPOINT = Path(os.environ.get('CHECKPOINT',
                                 '/data/anh/checkpoints/simplemem/joint-window8-text-b/epoch-2'))


@pytest.fixture(scope='session')
def serializer():
    import json
    import transformers
    from qwen_vl.data.episode_serializer import EpisodeSerializer

    if not MODEL_PATH.exists():
        pytest.skip('Pinned Qwen base processor unavailable')
    config = json.loads((CHECKPOINT / 'navigation.json').read_text())['config']
    processor = transformers.AutoProcessor.from_pretrained(MODEL_PATH)
    return EpisodeSerializer(processor, config)
