from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stochastic_bridge.config import load_config
from stochastic_bridge.corruptions import GoalDetailCorruptor
from stochastic_bridge.data import build_bridge_batch
from stochastic_bridge.datasets import ImageDirectoryDataset, SyntheticImageDataset
from stochastic_bridge.noise import CorruptionMixture, SUPPORTED_CORRUPTIONS
from stochastic_bridge.prepared import PreparedShardWriter
from stochastic_bridge.schedule import VPNoiseSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed sharded bridge dataset before training"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data", help="Override source image directory")
    parser.add_argument("--output", help="Override data.prepared_root")
    parser.add_argument("--variants", type=int, help="Records per source image")
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config.data.root = args.data
        config.data.synthetic = False
    if args.synthetic:
        config.data.synthetic = True
    output = Path(args.output or config.data.prepared_root).expanduser().resolve()
    variants = args.variants or config.prepare.variants_per_image
    if variants < 1:
        raise ValueError("variants must be positive")

    seed = config.prepare.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    source = (
        SyntheticImageDataset(config.data.synthetic_length, config.data.image_size)
        if config.data.synthetic
        else ImageDirectoryDataset(
            config.data.root,
            config.data.image_size,
            random_crop=config.data.random_crop,
            horizontal_flip=config.data.horizontal_flip,
        )
    )
    loader = DataLoader(
        source,
        batch_size=config.prepare.batch_size,
        shuffle=False,
        num_workers=config.prepare.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.prepare.workers > 0,
    )
    schedule = VPNoiseSchedule(
        config.schedule.steps,
        config.schedule.beta_start,
        config.schedule.beta_end,
    ).to(device)
    corruption = config.corruption
    mixture = (
        CorruptionMixture(
            schedule,
            corruption.weights,
            corruption.spatial_floor,
            corruption.student_t_df,
        ).to(device)
        if corruption.enabled
        else None
    )
    goal_values = config.goal_corruption.__dict__.copy()
    goal_enabled = goal_values.pop("enabled")
    goal_corruptor = (
        GoalDetailCorruptor(**goal_values).to(device) if goal_enabled else None
    )
    if mixture is not None and config.loss.name.lower() == "paired":
        raise ValueError("paired loss cannot generate a mixed-corruption dataset")

    writer = PreparedShardWriter(
        root=output,
        shard_size=config.prepare.shard_size,
        corruption_names=SUPPORTED_CORRUPTIONS,
        save_target_noise=mixture is None,
        metadata={
            "source_images": len(source),
            "variants_per_image": variants,
            "image_size": config.data.image_size,
            "channels": config.model.in_channels,
            "target_samples": config.loss.samples,
            "analytic_vp": mixture is None,
            "config": config.to_dict(),
        },
    )
    progress = tqdm(total=len(loader) * variants, desc="preparing fixed dataset")
    with torch.inference_mode():
        for _ in range(variants):
            for clean in loader:
                clean = clean.to(device, non_blocking=True)
                batch = build_bridge_batch(
                    clean,
                    schedule,
                    target_samples=config.loss.samples,
                    answer_jump=config.schedule.answer_jump,
                    goal_corruptor=goal_corruptor,
                    corruption_mixture=mixture,
                )
                writer.add_batch(batch)
                progress.update(1)
    progress.close()
    manifest = writer.finalize()
    print(f"records: {len(source) * variants}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
