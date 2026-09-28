from __future__ import annotations

import json

import pytest
import torch

from stochastic_bridge import SUPPORTED_CORRUPTIONS, CorruptionMixture, VPNoiseSchedule
from stochastic_bridge.data import build_bridge_batch
from stochastic_bridge.prepared import (
    PreparedBridgeDataset,
    PreparedShardWriter,
    ShardBatchSampler,
    materialize_target_noise,
)
from stochastic_bridge.stateless import stateless_normal


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

    rank_zero = ShardBatchSampler(
        dataset, batch_size=2, seed=5, num_replicas=2, rank=0
    )
    rank_one = ShardBatchSampler(
        dataset, batch_size=2, seed=5, num_replicas=2, rank=1
    )
    zero_batches = list(rank_zero)
    one_batches = list(rank_one)
    assert len(zero_batches) == len(one_batches) == 1
    assert set(zero_batches[0]).isdisjoint(one_batches[0])
    assert sorted(zero_batches[0] + one_batches[0]) == [0, 1, 2]


def test_stateless_normal_is_reproducible_and_seed_specific() -> None:
    seeds = torch.tensor([7, 11, 7], dtype=torch.int64)
    first = stateless_normal(seeds, (4, 3, 16, 16))
    second = stateless_normal(seeds, (4, 3, 16, 16))

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[0], first[2], rtol=0, atol=0)
    assert not torch.equal(first[0], first[1])
    assert abs(first.mean().item()) < 0.05
    assert 0.90 < first.std().item() < 1.10


def test_prepared_paired_noise_is_stored_as_seed(tmp_path) -> None:
    torch.manual_seed(109)
    schedule = VPNoiseSchedule(steps=100)
    clean = torch.rand(3, 3, 16, 16) * 2.0 - 1.0
    batch = build_bridge_batch(clean, schedule, target_samples=2)
    writer = PreparedShardWriter(
        tmp_path / "prepared",
        shard_size=3,
        metadata={"target_samples": 2, "image_size": 16},
        corruption_names=SUPPORTED_CORRUPTIONS,
        save_target_noise=True,
    )
    writer.add_batch(batch)
    manifest_path = writer.finalize()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
    assert manifest["target_noise_storage"] == "seed"
    payload = torch.load(
        tmp_path / "prepared" / manifest["shards"][0]["file"],
        map_location="cpu",
        weights_only=True,
    )
    assert "target_noise_seed" in payload
    assert "target_noise" not in payload

    dataset = PreparedBridgeDataset(tmp_path / "prepared")
    records = [dataset[index] for index in range(len(dataset))]
    raw = {
        "target_cloud": torch.stack([record["target_cloud"] for record in records]),
        "target_noise_seed": torch.stack(
            [record["target_noise_seed"] for record in records]
        ),
    }
    reconstructed = materialize_target_noise(raw, "cpu")
    assert reconstructed is not None
    torch.testing.assert_close(reconstructed, batch.target_noise, rtol=0, atol=0)


def test_format_v1_tensor_noise_remains_readable(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    image = torch.zeros(1, 3, 4, 4, dtype=torch.uint8)
    cloud = torch.zeros(1, 2, 3, 4, 4, dtype=torch.uint8)
    noise = torch.randn(1, 2, 3, 4, 4).half()
    torch.save(
        {
            "clean": image,
            "current": image,
            "goal": image,
            "target_cloud": cloud,
            "current_level": torch.zeros(1, dtype=torch.int16),
            "goal_level": torch.zeros(1, dtype=torch.int16),
            "answer_level": torch.zeros(1, dtype=torch.int16),
            "corruption_type_id": torch.full((1,), -1, dtype=torch.int16),
            "target_noise": noise,
        },
        root / "shard-000000.pt",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "length": 1,
                "has_target_noise": True,
                "shards": [{"file": "shard-000000.pt", "count": 1}],
            }
        ),
        encoding="utf-8",
    )

    dataset = PreparedBridgeDataset(root)
    record = dataset[0]
    torch.testing.assert_close(record["target_noise"], noise[0].float())


def test_compact_clean_repeat_target_storage(tmp_path) -> None:
    root = tmp_path / "compact"
    root.mkdir()
    clean = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
    torch.save(
        {
            "clean": clean,
            "current": clean,
            "goal": clean,
            "current_level": torch.ones(2, dtype=torch.int16),
            "goal_level": torch.zeros(2, dtype=torch.int16),
            "answer_level": torch.zeros(2, dtype=torch.int16),
            "corruption_type_id": torch.full((2,), -1, dtype=torch.int16),
            "target_noise_seed": torch.tensor([7, 11], dtype=torch.int64),
        },
        root / "shard-000000.pt",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "length": 2,
                "has_target_noise": True,
                "target_noise_storage": "seed",
                "target_cloud_storage": "clean_repeat",
                "target_samples": 4,
                "shards": [{"file": "shard-000000.pt", "count": 2}],
            }
        ),
        encoding="utf-8",
    )

    dataset = PreparedBridgeDataset(root)
    record = dataset[1]
    assert record["target_cloud"].shape == (4, 3, 4, 4)
    for sample in record["target_cloud"]:
        torch.testing.assert_close(sample, record["clean"])
