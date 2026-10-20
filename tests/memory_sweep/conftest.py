import sys
from pathlib import Path

# Reuse the established CPU FlashAttention fakes and tiny-model fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vln'))
