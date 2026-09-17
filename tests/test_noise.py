from __future__ import annotations

import pytest
import torch

from stochastic_bridge import SUPPORTED_CORRUPTIONS, CorruptionMixture, VPNoiseSchedule
from stochastic_bridge.data import build_bridge_batch
from stochastic_bridge.prepared import (
    PreparedBridgeDataset,
    PreparedShardWriter,
    ShardBatchSampler,
)


@pytest.mark.parametrize("name", SUPPORTED_CORRUPTIONS)
def test_every_corruption_is_finite_and_preserves_shape(name: str) -> None:
    torch.manual_seed(101)
    schedule = VPNoiseSchedule(steps=100)
    mixture = CorruptionMixture(schedule, {name: 1.0})
    clean = torch.rand(2, 3, 32, 32) * 2.0 - 1.0

    level_zero = mixture.corrupt(clean, torch.zeros(2, dtype=torch.long), (name, name))
    severe = mixture.corrupt(
        clean, torch.full((2,), 100, dtype=torch.long), (name, name)
    )

    torch.testing.assert_close(level_zero, clean)
    assert severe.shape == clean.shape
    assert torch.isfinite(severe).all()
    assert severe.min() >= -1.0
    assert severe.max() <= 1.0


def test_mixed_corruption_builds_answer_cloud() -> None:
    torch.manual_seed(103)
    schedule = VPNoiseSchedule(steps=100)
    mixture = CorruptionMixture(
        schedule,
        {
            "white_gaussian": 1.0,
            "edge_gaussian": 1.0,
            "poisson": 1.0,
            "mask": 1.0,
        },
    )
    clean = torch.rand(4, 3, 32, 32) * 2.0 - 1.0
    batch = build_bridge_batch(
        clean,
        schedule,
        target_samples=3,
        corruption_mixture=mixture,
    )

    assert batch.current.shape == clean.shape
    assert batch.goal.shape == clean.shape
    assert batch.target_cloud.shape == (4, 3, 3, 32, 32)
    assert batch.corruption_types is not None
    assert len(batch.corruption_types) == clean.shape[0]
    assert torch.isfinite(batch.target_cloud).all()


def test_weights_are_normalized() -> None:
    schedule = VPNoiseSchedule(steps=100)
    mixture = CorruptionMixture(
        schedule, {"white_gaussian": 2.0, "local_gaussian": 3.0}
    )
    torch.testing.assert_close(mixture.probabilities.sum(), torch.tensor(1.0))


def test_unknown_corruption_is_rejected() -> None:
    schedule = VPNoiseSchedule(steps=100)
    with pytest.raises(ValueError, match="unknown corruption"):
        CorruptionMixture(schedule, {"not_a_real_noise": 1.0})


def test_prepared_dataset_round_trip(tmp_path) -> None:
    torch.manual_seed(107)
    schedule = VPNoiseSchedule(steps=100)
    mixture = CorruptionMixture(schedule, {"edge_gaussian": 1.0})
    clean = torch.rand(3, 3, 16, 16) * 2.0 - 1.0
    batch = build_bridge_batch(
        clean, schedule, target_samples=2, corruption_mixture=mixture
    )
    writer = PreparedShardWriter(
        tmp_path / "prepared",
        shard_size=2,
        metadata={"target_samples": 2, "image_size": 16},
        corruption_names=SUPPORTED_CORRUPTIONS,
        save_target_noise=False,
    )
    writer.add_batch(batch)
    writer.finalize()

    dataset = PreparedBridgeDataset(tmp_path / "prepared", cache_shards=1)
    assert len(dataset) == 3
    assert len(dataset.manifest["shards"]) == 2
    assert not dataset.has_target_noise
    record = dataset[1]
    assert record["target_cloud"].shape == (2, 3, 16, 16)
    torch.testing.assert_close(record["current"], batch.current[1].cpu(), atol=1 / 127.5, rtol=0)

    sampler = ShardBatchSampler(dataset, batch_size=2, seed=5)
    first_epoch = list(sampler)
    assert sorted(index for indices in first_epoch for index in indices) == [0, 1, 2]
    assert all(
        not (0 in indices and 2 in indices) for indices in first_epoch
    ), "a batch must not cross a shard boundary"
    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert sorted(index for indices in second_epoch for index in indices) == [0, 1, 2]
