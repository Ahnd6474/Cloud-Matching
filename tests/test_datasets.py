from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from stochastic_bridge.datasets import ImageDirectoryDataset


def test_native_crop_preserves_source_pixels(tmp_path) -> None:
    yy, xx = np.indices((256, 256))
    checkerboard = ((xx + yy) % 2 * 255).astype(np.uint8)
    rgb = np.repeat(checkerboard[..., None], 3, axis=-1)
    Image.fromarray(rgb).save(tmp_path / "checkerboard.png")

    dataset = ImageDirectoryDataset(
        tmp_path,
        image_size=64,
        random_crop=True,
        crop_mode="native",
        horizontal_flip=False,
    )
    image = dataset[0]

    assert image.shape == (3, 64, 64)
    assert set(image.unique().tolist()) == {-1.0, 1.0}


def test_unknown_crop_mode_is_rejected(tmp_path) -> None:
    Image.new("RGB", (16, 16)).save(tmp_path / "image.png")
    with pytest.raises(ValueError, match="crop_mode"):
        ImageDirectoryDataset(tmp_path, image_size=8, crop_mode="unknown")


def test_image_directory_dataset_combines_multiple_roots(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    Image.new("RGB", (12, 12), color=(255, 0, 0)).save(first / "a.png")
    Image.new("RGB", (12, 12), color=(0, 255, 0)).save(second / "b.png")

    dataset = ImageDirectoryDataset(
        [first, second], image_size=8, random_crop=False, horizontal_flip=False
    )

    assert len(dataset) == 2
    assert dataset.roots == [first.resolve(), second.resolve()]
