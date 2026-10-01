from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json
import random

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tvf
from torch.nn import functional as F


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_image(value: Tensor) -> Tensor:
    """Map torchvision's [0, 1] image tensor to the model's [-1, 1] range."""
    return value * 2.0 - 1.0


class ImageDirectoryDataset(Dataset[Tensor]):
    """Load every image below one or more directories.

    Multiple roots are useful for DF2K-style training: DIV2K and Flickr2K can
    remain in their original layouts without duplicating several gigabytes of
    HR images into a merged directory.
    """

    def __init__(
        self,
        root: str | Path | Sequence[str | Path],
        image_size: int,
        random_crop: bool = True,
        crop_mode: str = "resized",
        horizontal_flip: bool = True,
    ) -> None:
        raw_roots = [root] if isinstance(root, (str, Path)) else list(root)
        if not raw_roots:
            raise ValueError("at least one image directory is required")
        self.roots = [Path(value).expanduser().resolve() for value in raw_roots]
        for resolved_root in self.roots:
            if not resolved_root.is_dir():
                raise FileNotFoundError(
                    f"image directory does not exist: {resolved_root}"
                )
        # Keep ``root`` for callers that used it for display/debugging.
        self.root = self.roots[0]
        self.paths = sorted(
            path
            for resolved_root in self.roots
            for path in resolved_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(
                "no supported images found below: "
                + ", ".join(str(value) for value in self.roots)
            )

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


class CocoCaptionDataset(Dataset[dict[str, Tensor | str | int]]):
    """COCO images paired with one of their human-written captions.

    Images are fitted inside a square instead of aggressively cropped because
    COCO captions describe the whole scene. Padding is neutral gray in the
    normalized model range and source files always remain at native resolution.
    """

    def __init__(
        self,
        image_root: str | Path,
        captions_file: str | Path,
        image_size: int,
        *,
        random_caption: bool = True,
        horizontal_flip: bool = False,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        self.captions_file = Path(captions_file).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"COCO image directory does not exist: {self.image_root}")
        if not self.captions_file.is_file():
            raise FileNotFoundError(f"COCO captions file does not exist: {self.captions_file}")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.image_size = int(image_size)
        self.random_caption = bool(random_caption)
        self.horizontal_flip = bool(horizontal_flip)

        raw = json.loads(self.captions_file.read_text(encoding="utf-8"))
        captions_by_image: dict[int, list[str]] = {}
        for annotation in raw.get("annotations", []):
            image_id = int(annotation["image_id"])
            caption = " ".join(str(annotation["caption"]).split())
            if caption:
                captions_by_image.setdefault(image_id, []).append(caption)

        samples: list[tuple[Path, int, tuple[str, ...]]] = []
        for image in raw.get("images", []):
            image_id = int(image["id"])
            captions = captions_by_image.get(image_id)
            if not captions:
                continue
            path = self.image_root / str(image["file_name"])
            if not path.is_file():
                raise FileNotFoundError(f"COCO image listed by annotations is missing: {path}")
            samples.append((path, image_id, tuple(captions)))
        self.samples = sorted(samples, key=lambda item: item[1])
        if not self.samples:
            raise RuntimeError("COCO annotations contained no usable image-caption pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        path, image_id, captions = self.samples[index]
        caption = random.choice(captions) if self.random_caption else captions[0]
        with Image.open(path) as image:
            clean = self._fit_square(image.convert("RGB"))
        if self.horizontal_flip and random.random() < 0.5:
            clean = clean.flip(-1)
        return {"clean": clean, "caption": caption, "image_id": image_id}

    def _fit_square(self, image: Image.Image) -> Tensor:
        tensor = tvf.pil_to_tensor(image).float().div_(255.0)
        height, width = tensor.shape[-2:]
        scale = self.image_size / max(height, width)
        resized_height = max(1, min(self.image_size, round(height * scale)))
        resized_width = max(1, min(self.image_size, round(width * scale)))
        tensor = tvf.resize(
            tensor,
            [resized_height, resized_width],
            antialias=True,
        )
        pad_height = self.image_size - resized_height
        pad_width = self.image_size - resized_width
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        # 0.5 maps to zero after normalization and therefore does not inject a
        # strong black/white border into the denoising target.
        tensor = F.pad(tensor, (left, right, top, bottom), value=0.5)
        return normalize_image(tensor)
