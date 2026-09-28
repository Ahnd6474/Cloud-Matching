from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from stochastic_bridge.prepared import (
    COMPACT_CLEAN_FORMAT_VERSION,
    PreparedBridgeDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move a prepared bridge dataset and convert every target to the "
            "clean endpoint using compact clean-repeat target storage."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def convert_split(root: Path, split: str) -> None:
    split_root = root / split
    manifest_path = split_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in tqdm(manifest["shards"], desc=f"compact clean {split}"):
        path = split_root / shard["file"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload.pop("target_cloud", None)
        payload["answer_level"] = torch.zeros_like(payload["answer_level"])
        temporary = path.with_name(path.name + ".clean100.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    manifest["format_version"] = COMPACT_CLEAN_FORMAT_VERSION
    manifest["target_cloud_storage"] = "clean_repeat"
    manifest["clean_answer_probability"] = 1.0
    config = manifest.get("config")
    if isinstance(config, dict):
        schedule = config.setdefault("schedule", {})
        schedule["clean_answer_probability"] = 1.0
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)


def validate(root: Path) -> None:
    expected_lengths = {"train": 27_600, "valid": 200}
    for split, expected_length in expected_lengths.items():
        dataset = PreparedBridgeDataset(root / split, cache_shards=1)
        if len(dataset) != expected_length:
            raise RuntimeError(
                f"{split} length mismatch: {len(dataset)} != {expected_length}"
            )
        manifest = dataset.manifest
        if manifest.get("format_version") != COMPACT_CLEAN_FORMAT_VERSION:
            raise RuntimeError(f"{split} was not converted to format v3")
        if manifest.get("target_cloud_storage") != "clean_repeat":
            raise RuntimeError(f"{split} target storage is not clean_repeat")
        indices = {0, len(dataset) // 2, len(dataset) - 1}
        for index in sorted(indices):
            record = dataset[index]
            expected = record["clean"].unsqueeze(0).expand_as(
                record["target_cloud"]
            )
            torch.testing.assert_close(
                record["target_cloud"], expected, rtol=0, atol=0
            )
            if int(record["answer_level"]) != 0:
                raise RuntimeError(f"{split}[{index}] is not a clean endpoint")
        print(
            json.dumps(
                {
                    "split": split,
                    "records": len(dataset),
                    "format_version": manifest["format_version"],
                    "target_cloud_storage": manifest["target_cloud_storage"],
                }
            )
        )


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("source and output must differ")
    if source.exists():
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        source.rename(output)
    elif not output.exists():
        raise FileNotFoundError(f"source does not exist: {source}")

    for split in ("train", "valid"):
        convert_split(output, split)
    validate(output)
    (output / ".complete").write_text("validated clean100\n", encoding="utf-8")


if __name__ == "__main__":
    main()
