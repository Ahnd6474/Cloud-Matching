from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from stochastic_bridge.config import load_config
from stochastic_bridge.corruptions import GoalDetailCorruptor
from stochastic_bridge.data import build_bridge_batch
from stochastic_bridge.datasets import ImageDirectoryDataset, SyntheticImageDataset
from stochastic_bridge.noise import CorruptionMixture
from stochastic_bridge.prepared import PreparedBridgeDataset
from stochastic_bridge.schedule import VPNoiseSchedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview on-the-fly bridge training data")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data")
    parser.add_argument("--prepared")
    parser.add_argument("--output", default="data-preview.png")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data:
        config.data.root = args.data
    prepared_root = (
        args.prepared
        or (config.data.prepared_root if not args.data and not args.synthetic else "")
    )
    if prepared_root:
        dataset = PreparedBridgeDataset(
            prepared_root, cache_shards=config.data.prepared_cache_shards
        )
        fixed = next(
            iter(DataLoader(dataset, batch_size=min(4, config.data.batch_size)))
        )
        rows = torch.cat(
            [
                fixed["clean"],
                fixed["current"],
                fixed["goal"],
                fixed["target_cloud"][:, 0],
            ],
            dim=0,
        )
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_image((rows.clamp(-1, 1) + 1) / 2, destination, nrow=fixed["clean"].shape[0])
        print("fixed rows: clean | current | goal | target answer")
        print(destination)
        return
    dataset = (
        SyntheticImageDataset(16, config.data.image_size)
        if args.synthetic
        else ImageDirectoryDataset(
            config.data.root,
            config.data.image_size,
            random_crop=config.data.random_crop,
            crop_mode=config.data.crop_mode,
            horizontal_flip=config.data.horizontal_flip,
        )
    )
    clean = next(iter(DataLoader(dataset, batch_size=min(4, config.data.batch_size))))
    schedule = VPNoiseSchedule(
        config.schedule.steps, config.schedule.beta_start, config.schedule.beta_end
    )
    goal_config = config.goal_corruption.__dict__.copy()
    enabled = goal_config.pop("enabled")
    corruptor = GoalDetailCorruptor(**goal_config) if enabled else None
    mixture_config = config.corruption.__dict__.copy()
    mixture_enabled = mixture_config.pop("enabled")
    mixture = (
        CorruptionMixture(schedule=schedule, **mixture_config)
        if mixture_enabled
        else None
    )
    endpoint_config = config.endpoint_corruption.__dict__.copy()
    endpoint_enabled = endpoint_config.pop("enabled")
    endpoint_probability = endpoint_config.pop("probability")
    endpoint_mixture = (
        CorruptionMixture(schedule=schedule, **endpoint_config)
        if endpoint_enabled and endpoint_probability > 0.0
        else None
    )
    batch = build_bridge_batch(
        clean,
        schedule,
        target_samples=config.loss.samples,
        answer_jump=config.schedule.answer_jump,
        goal_corruptor=corruptor,
        corruption_mixture=mixture,
        clean_answer_probability=config.schedule.clean_answer_probability,
        endpoint_corruption_mixture=endpoint_mixture,
        endpoint_corruption_probability=endpoint_probability,
    )
    rows = torch.cat(
        [clean, batch.current, batch.goal, batch.target_cloud[:, 0]], dim=0
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image((rows.clamp(-1, 1) + 1) / 2, destination, nrow=clean.shape[0])
    print("rows: clean | current | goal | target answer")
    print(destination)


if __name__ == "__main__":
    main()
