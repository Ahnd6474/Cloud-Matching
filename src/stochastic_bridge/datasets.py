from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_image(value: Tensor) -> Tensor:
    """Map torchvision's [0, 1] image tensor to the model's [-1, 1] range."""
    return value * 2.0 - 1.0


class ImageDirectoryDataset(Dataset[Tensor]):
    """Load every image below a directory; class subfolders are optional."""

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        random_crop: bool = True,
        crop_mode: str = "resized",
        horizontal_flip: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"image directory does not exist: {self.root}")
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"no supported images found below {self.root}")

        normalized_crop_mode = crop_mode.strip().lower()
        if normalized_crop_mode == "native":
            crop = (
                transforms.RandomCrop(image_size, pad_if_needed=True)
                if random_crop
                else transforms.CenterCrop(image_size)
            )
        elif normalized_crop_mode == "resized":
            crop = (
                transforms.RandomResizedCrop(
                    image_size, scale=(0.65, 1.0), antialias=True
                )
                if random_crop
                else transforms.Compose(
                    [
                        transforms.Resize(image_size, antialias=True),
                        transforms.CenterCrop(image_size),
                    ]
                )
            )
        else:
            raise ValueError("crop_mode must be 'native' or 'resized'")
        augmentation: list[object] = [crop]
        if horizontal_flip:
            augmentation.append(transforms.RandomHorizontalFlip())
        augmentation.extend([transforms.ToTensor(), transforms.Lambda(normalize_image)])
        self.transform = transforms.Compose(augmentation)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        return self.sample_variants(index, 1)[0]

    def sample_variants(self, index: int, count: int) -> Tensor:
        """Decode one source image once and draw ``count`` independent crops."""
        if count < 1:
            raise ValueError("variant count must be positive")
        path = self.paths[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return torch.stack([self.transform(rgb) for _ in range(count)])


class SyntheticImageDataset(Dataset[Tensor]):
    """Deterministic structured images for smoke tests without data files."""

    def __init__(self, length: int = 128, image_size: int = 32) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        coordinates = torch.linspace(-1.0, 1.0, self.image_size)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        radius = 0.20 + 0.04 * (index % 8)
        circle = ((x**2 + y**2) < radius).float() * 2.0 - 1.0
        stripe = torch.sin((2 + index % 5) * torch.pi * x).clamp(-1.0, 1.0)
        diagonal = torch.tanh(3.0 * (x + y - 0.1 * (index % 5)))
        return torch.stack([circle, stripe, diagonal])
