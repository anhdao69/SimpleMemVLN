import torch

from qwen_vl.rl.sampling import tempered_token


def test_tempered_sampling_frequencies_and_logp():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    temperature = 1.2
    expected = torch.softmax(logits / temperature, -1)
    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(4)
    for _ in range(20000):
        token, logp, entropy = tempered_token(logits, temperature, generator)
        counts[token] += 1
        assert abs(logp - float(expected[token].log())) < 1e-6
    assert torch.allclose(counts / counts.sum(), expected, atol=0.01)
    assert abs(entropy - float(-(expected * expected.log()).sum())) < 1e-6


def test_sampling_is_reproducible_for_a_seed():
    logits = torch.randn(1000)
    draws = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(7)
        draws.append([tempered_token(logits, 1.2, generator)[0] for _ in range(50)])
    assert draws[0] == draws[1]
