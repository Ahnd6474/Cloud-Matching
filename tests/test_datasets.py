from __future__ import annotations

import json
import numpy as np
import pytest
from PIL import Image

from stochastic_bridge.datasets import CocoCaptionDataset, ImageDirectoryDataset
from stochastic_bridge.text import CocoWordTokenizer


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


def test_coco_caption_dataset_preserves_whole_scene_and_caption(tmp_path) -> None:
    image_root = tmp_path / "train2017"
    image_root.mkdir()
    Image.new("RGB", (20, 10), color=(255, 0, 0)).save(
        image_root / "000000000001.jpg"
    )
    annotation_path = tmp_path / "captions.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg"}],
                "annotations": [
                    {"image_id": 1, "caption": "A red rectangle."},
                    {"image_id": 1, "caption": "A wide red image."},
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = CocoCaptionDataset(
        image_root,
        annotation_path,
        16,
        random_caption=False,
    )
    sample = dataset[0]
    assert sample["clean"].shape == (3, 16, 16)
    assert sample["caption"] == "A red rectangle."
    # The 2:1 image is fitted to 16x8 with neutral padding, not cropped.
    assert sample["clean"][:, :4].abs().max().item() == 0.0


def test_coco_tokenizer_build_save_and_batch_encode(tmp_path) -> None:
    annotation_path = tmp_path / "captions.json"
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"caption": "A small red bird."},
                    {"caption": "A small blue bird."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = CocoWordTokenizer.build_from_coco(
        annotation_path,
        vocab_size=12,
        min_frequency=1,
        max_length=8,
    )
    saved = tokenizer.save(tmp_path / "vocab.json")
    loaded = CocoWordTokenizer.load(saved)
    ids, mask = loaded.batch_encode(["A red bird.", "unknown creature"])
    assert ids.shape == mask.shape == (2, 8)
    assert ids[0, 0].item() == 1
    assert mask.sum(dim=1).tolist() == [6, 4]
    assert loaded.token_to_id["<unk>"] in ids[1].tolist()
