from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample next states from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", default="samples.png")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device")
    return parser.parse_args()


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda value: value * 2.0 - 1.0),
        ]
    )
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor[None].to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    current = load_image(args.current, config.data.image_size, device)
    goal = load_image(args.goal, config.data.image_size, device)
    with torch.inference_mode():
        predictions = model(current, goal, samples=args.samples)[0]
    grid = torch.cat([current.cpu(), goal.cpu(), predictions.cpu()], dim=0)
    grid = (grid.clamp(-1.0, 1.0) + 1.0) / 2.0
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, destination, nrow=min(args.samples + 2, 10))
    print(destination)


if __name__ == "__main__":
    main()

