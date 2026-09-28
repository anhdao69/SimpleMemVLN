import torch
from qwen_vl.data.episode_dataset import EpisodeLengthSampler
from qwen_vl.train.trainer import QwenSFTTrainer


def test_sampler_never_pads_pools_or_drops_tail():
    lengths = [i % 37 + 1 for i in range(131)]
    sampler = EpisodeLengthSampler(lengths, group_size=64, pool_size=64, world_size=4)
    first = list(sampler)
    assert len(first) == 131 and sorted(first) == list(range(131))
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_actual_accelerate_tail_for_full_r2r_batch64():
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler
    from collections import Counter

    sampler = EpisodeLengthSampler([i % 183 + 1 for i in range(10819)])
    batches = BatchSampler(sampler, batch_size=1, drop_last=False)
    ranks = [
        list(
            BatchSamplerShard(
                batches,
                num_processes=4,
                process_index=rank,
                split_batches=False,
                even_batches=True,
            )
        )
        for rank in range(4)
    ]
    assert [len(rank) for rank in ranks] == [2705] * 4
    exposed = [batch[0] for rank in ranks for batch in rank]
    counts = Counter(exposed)
    assert len(exposed) == 10820 and len(counts) == 10819
    assert sorted(counts.values()).count(2) == 1
    assert len(ranks[0]) % 16 == 1  # Last optimizer update: four exposures, not 64.


def test_trainer_counts_actions_and_requires_global_denominator():
    from types import SimpleNamespace

    trainer = object.__new__(QwenSFTTrainer)
    trainer.accelerator = SimpleNamespace(
        num_processes=2, gather=lambda n: torch.stack([n, n + 1])
    )
    count = trainer._get_num_items_in_batch(
        [{"num_actions": 2}, {"num_actions": 5}], torch.device("cpu")
    )
    assert count == 15

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            return {"loss_sum": torch.tensor(6.0, requires_grad=True)}

    model = Model()
    loss = trainer.compute_loss(model, {"num_actions": 2}, num_items_in_batch=count)
    torch.testing.assert_close(loss, torch.tensor(0.8))
    import pytest

    with pytest.raises(ValueError):
        trainer.compute_loss(model, {"num_actions": 2})


def test_model_cast_preserves_nonpersistent_rotary_precision():
    from qwen_vl.models.nav_model import SimpleMemVLNForNavigation

    model = SimpleMemVLNForNavigation.__new__(SimpleMemVLNForNavigation)
    torch.nn.Module.__init__(model)
    model.rope = torch.nn.Module()
    freq = torch.tensor([1.0, 0.123456789, 0.000123456789], dtype=torch.float32)
    model.rope.register_buffer("inv_freq", freq.clone(), persistent=False)
    model.rope.register_buffer("original_inv_freq", freq.clone(), persistent=False)
    model.weight = torch.nn.Parameter(torch.ones(3))
    model.bfloat16()
    assert model.weight.dtype == torch.bfloat16
    assert model.rope.inv_freq.dtype == torch.float32
    torch.testing.assert_close(model.rope.inv_freq, freq, atol=0, rtol=0)
    torch.testing.assert_close(model.rope.original_inv_freq, freq, atol=0, rtol=0)
