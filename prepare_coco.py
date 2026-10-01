from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

from stochastic_bridge.text import CocoWordTokenizer


FILES = {
    "train2017.zip": (
        "http://images.cocodataset.org/zips/train2017.zip",
        19_336_861_798,
    ),
    "val2017.zip": (
        "http://images.cocodataset.org/zips/val2017.zip",
        815_585_330,
    ),
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        252_907_541,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, extract, validate, and tokenize COCO Captions 2017"
    )
    parser.add_argument("--root", default="data/coco2017")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=48)
    return parser.parse_args()


def download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing == expected_size:
        print(f"already downloaded: {destination}")
        return
    if existing > expected_size:
        raise RuntimeError(f"download is larger than expected: {destination}")
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"downloading {url} -> {destination} (resume={existing:,})")
    with urllib.request.urlopen(request) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        if existing and not resumed:
            print("server did not honor Range; restarting this archive")
        with destination.open("ab" if resumed else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = destination.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"incomplete download for {destination.name}: {actual:,}/{expected_size:,}"
        )


def safe_extract(archive: Path, root: Path) -> None:
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        print(f"extracting {archive.name}")
        bundle.extractall(root)


def validate_layout(root: Path) -> None:
    required = (
        root / "train2017",
        root / "val2017",
        root / "annotations" / "captions_train2017.json",
        root / "annotations" / "captions_val2017.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("COCO preparation is incomplete: " + ", ".join(missing))
    train_count = sum(1 for _ in (root / "train2017").glob("*.jpg"))
    val_count = sum(1 for _ in (root / "val2017").glob("*.jpg"))
    if train_count != 118_287 or val_count != 5_000:
        raise RuntimeError(
            f"unexpected COCO image counts: train={train_count}, val={val_count}"
        )
    print(f"validated COCO: train={train_count:,}, val={val_count:,}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        for filename, (url, expected_size) in FILES.items():
            download_with_resume(url, downloads / filename, expected_size)
    if not args.skip_extract:
        for filename in FILES:
            safe_extract(downloads / filename, root)
    validate_layout(root)
    tokenizer = CocoWordTokenizer.build_from_coco(
        root / "annotations" / "captions_train2017.json",
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_length=args.max_length,
    )
    vocab_path = tokenizer.save(root / "coco_vocab.json")
    print(f"vocabulary: {len(tokenizer):,} tokens -> {vocab_path}")


if __name__ == "__main__":
    main()
