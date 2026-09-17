from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .data import BridgeBatch


FORMAT_VERSION = 1
IMAGE_KEYS = ("clean", "current", "goal", "target_cloud")
LEVEL_KEYS = ("current_level", "goal_level", "answer_level")


def encode_image_tensor(value: Tensor) -> Tensor:
    """Quantize normalized [-1, 1] images to compact uint8 storage."""
    return ((value.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte()


def decode_image_tensor(value: Tensor) -> Tensor:
    return value.float().div(127.5).sub(1.0)


class PreparedShardWriter:
    """Write fixed bridge records as compact torch tensor shards plus a manifest."""

    def __init__(
        self,
        root: str | Path,
        shard_size: int,
        metadata: dict[str, Any],
        corruption_names: tuple[str, ...],
        save_target_noise: bool,
    ) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"prepared dataset directory is not empty: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.metadata = metadata
        self.corruption_names = corruption_names
        self.type_to_id = {name: index for index, name in enumerate(corruption_names)}
        self.save_target_noise = save_target_noise
        self._records: list[dict[str, Tensor]] = []
        self._shards: list[dict[str, Any]] = []
        self._total = 0

    def add_batch(self, batch: BridgeBatch) -> None:
        for index in range(batch.clean.shape[0]):
            record = {
                "clean": encode_image_tensor(batch.clean[index]),
                "current": encode_image_tensor(batch.current[index]),
                "goal": encode_image_tensor(batch.goal[index]),
                "target_cloud": encode_image_tensor(batch.target_cloud[index]),
                "current_level": batch.current_level[index].detach().cpu().to(torch.int16),
                "goal_level": batch.goal_level[index].detach().cpu().to(torch.int16),
                "answer_level": batch.answer_level[index].detach().cpu().to(torch.int16),
                "corruption_type_id": torch.tensor(
                    -1
                    if batch.corruption_types is None
                    else self.type_to_id[batch.corruption_types[index]],
                    dtype=torch.int16,
                ),
            }
            if self.save_target_noise:
                if batch.target_noise is None:
                    raise ValueError("target noise was requested but is unavailable")
                record["target_noise"] = (
                    batch.target_noise[index].detach().cpu().to(torch.float16)
                )
            self._records.append(record)
            if len(self._records) == self.shard_size:
                self._flush()

    def finalize(self) -> Path:
        self._flush()
        if self._total == 0:
            raise RuntimeError("cannot finalize an empty prepared dataset")
        manifest = {
            "format_version": FORMAT_VERSION,
            "storage": "uint8_-1_1",
            "length": self._total,
            "has_target_noise": self.save_target_noise,
            "corruption_names": list(self.corruption_names),
            "shards": self._shards,
            **self.metadata,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def _flush(self) -> None:
        if not self._records:
            return
        keys = self._records[0].keys()
        payload = {key: torch.stack([record[key] for record in self._records]) for key in keys}
        name = f"shard-{len(self._shards):06d}.pt"
        torch.save(payload, self.root / name)
        count = len(self._records)
        self._shards.append({"file": name, "count": count})
        self._total += count
        self._records.clear()


class PreparedBridgeDataset(Dataset[dict[str, Tensor]]):
    """Read a generated bridge dataset without resampling its corruptions."""

    def __init__(self, root: str | Path, cache_shards: int = 2) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"prepared dataset manifest does not exist: {manifest_path}; "
                "run prepare_dataset.py first"
            )
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("unsupported prepared dataset format version")
        self.shards = self.manifest["shards"]
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.ends.append(total)
        if total != int(self.manifest["length"]):
            raise RuntimeError("prepared dataset manifest has inconsistent lengths")
        self.cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[int, dict[str, Tensor]] = OrderedDict()

    @property
    def has_target_noise(self) -> bool:
        return bool(self.manifest.get("has_target_noise", False))

    def __len__(self) -> int:
        return int(self.manifest["length"])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        payload = self._load_shard(shard_index)
        offset = index - start
        record = {
            key: decode_image_tensor(payload[key][offset]) for key in IMAGE_KEYS
        }
        record.update(
            {key: payload[key][offset].long() for key in LEVEL_KEYS}
        )
        record["corruption_type_id"] = payload["corruption_type_id"][offset].long()
        if "target_noise" in payload:
            record["target_noise"] = payload["target_noise"][offset].float()
        return record

    def _load_shard(self, index: int) -> dict[str, Tensor]:
        if index in self._cache:
            payload = self._cache.pop(index)
            self._cache[index] = payload
            return payload
        path = self.root / self.shards[index]["file"]
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self._cache[index] = payload
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return payload


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle records while keeping each batch inside one memory-mapped shard.

    This prevents random record-level sampling from repeatedly opening unrelated
    shard files. Shard order and record order both change on every epoch.
    """

    def __init__(
        self,
        dataset: PreparedBridgeDataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator)
        starts = [0, *self.dataset.ends[:-1]]
        for shard_index_tensor in shard_order:
            shard_index = int(shard_index_tensor)
            count = int(self.dataset.shards[shard_index]["count"])
            local_order = torch.randperm(count, generator=generator)
            start = starts[shard_index]
            for offset in range(0, count, self.batch_size):
                selection = local_order[offset : offset + self.batch_size]
                if self.drop_last and len(selection) < self.batch_size:
                    continue
                yield (selection + start).tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                int(shard["count"]) // self.batch_size for shard in self.dataset.shards
            )
        return sum(
            (int(shard["count"]) + self.batch_size - 1) // self.batch_size
            for shard in self.dataset.shards
        )
