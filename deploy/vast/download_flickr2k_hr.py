from __future__ import annotations

import os
import urllib.request
from pathlib import Path, PurePosixPath

from stream_unzip import stream_unzip


URL = "https://www.kaggle.com/api/v1/datasets/download/daehoyang/flickr2k"
OUTPUT = Path("/workspace/data/flickr2k/Flickr2K/Flickr2K_HR")
CHUNK_SIZE = 4 * 1024 * 1024
REPORT_BYTES = 256 * 1024 * 1024


def zipped_chunks():
    request = urllib.request.Request(URL, headers={"User-Agent": "cloud-matching/1"})
    downloaded = 0
    next_report = REPORT_BYTES
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        print(f"Kaggle Flickr2K stream: {total / 2**30:.2f} GiB", flush=True)
        while chunk := response.read(CHUNK_SIZE):
            downloaded += len(chunk)
            if downloaded >= next_report:
                ratio = 100.0 * downloaded / total if total else 0.0
                print(
                    f"downloaded {downloaded / 2**30:.2f} GiB ({ratio:.1f}%)",
                    flush=True,
                )
                next_report += REPORT_BYTES
            yield chunk


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for raw_name, expected_size, chunks in stream_unzip(zipped_chunks()):
        archive_path = PurePosixPath(raw_name.decode("utf-8"))
        # The original SNU tar uses Flickr2K/Flickr2K_HR/*.png, while the
        # compact Kaggle mirror flattens the same HR images to Flickr2K/*.png.
        is_hr_png = (
            archive_path.suffix.lower() == ".png"
            and archive_path.name not in {"", ".", ".."}
            and (
                "Flickr2K_HR" in archive_path.parts
                or archive_path.parent.name == "Flickr2K"
            )
        )
        if not is_hr_png:
            for _ in chunks:
                pass
            continue

        destination = OUTPUT / archive_path.name
        temporary = destination.with_suffix(destination.suffix + ".part")
        written = 0
        with temporary.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                written += len(chunk)
        if expected_size is not None and written != expected_size:
            raise RuntimeError(
                f"size mismatch for {archive_path}: {written} != {expected_size}"
            )
        os.replace(temporary, destination)
        extracted += 1
        if extracted % 50 == 0:
            print(f"extracted {extracted}/2650 HR images", flush=True)

    if extracted != 2650:
        raise RuntimeError(f"expected 2650 Flickr2K HR images, extracted {extracted}")
    print(f"Flickr2K ready: {extracted} HR images below {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
