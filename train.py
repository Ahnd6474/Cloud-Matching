from __future__ import annotations

import argparse

from stochastic_bridge.config import load_config
from stochastic_bridge.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stochastic image bridge")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data", help="Override data.root")
    parser.add_argument("--prepared", help="Override data.prepared_root")
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--resume", help="Override train.resume")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the synthetic dataset for a short environment check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config.data.root = args.data
        config.data.synthetic = False
    if args.prepared:
        config.data.prepared_root = args.prepared
    if args.resume:
        config.train.resume = args.resume
    if args.smoke:
        # Smoke mode intentionally exercises the on-the-fly fallback so it
        # does not require a prepared dataset to exist.
        config.data.prepared_root = ""
        config.data.synthetic = True
        config.data.synthetic_length = 8
        config.data.image_size = 32
        config.data.batch_size = 2
        config.data.workers = 0
        config.model.base_channels = 8
        config.model.heads = 4
        config.model.attention_depth = 1
        config.model.fullres_dim = 32
        config.model.fullres_depth = 2
        config.model.fullres_cross_depth = 1
        config.model.fullres_window_size = 4
        config.model.fullres_gradient_checkpointing = False
        config.loss.samples = 2
        config.train.epochs = 1
        config.train.log_every = 1
        config.train.preview_every = 2
        config.train.save_every = 100
        config.train.output_dir = "runs/smoke"
    Trainer(config, device=args.device).train()


if __name__ == "__main__":
    main()
