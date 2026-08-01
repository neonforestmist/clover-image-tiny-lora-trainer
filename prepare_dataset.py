#!/usr/bin/env python3
"""Validate a local image/caption dataset before training.

A Clover style dataset is a Diffusers *imagefolder*: a directory containing an
`images/` folder and a `metadata.jsonl` file. Each line of `metadata.jsonl`
pairs one image with its caption:

    {"file_name": "images/0001.png", "text": "Monet Style, a quiet lily pond"}

This script checks that every referenced image exists, opens cleanly, and is a
reasonable size, and that every caption is a non-empty string. It prints a
short report and exits non-zero if anything is wrong, so you never launch a
2-hour training run against a broken dataset.

    python prepare_dataset.py data/example-monet
    python prepare_dataset.py data/example-monet --trigger "Monet Style"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - guidance only
    sys.exit("Pillow is required: pip install pillow")

MIN_SIDE = 256


def validate(root: Path, trigger: str | None) -> int:
    metadata = root / "metadata.jsonl"
    if not metadata.exists():
        print(f"FAIL: {metadata} not found")
        return 1

    rows: list[dict] = []
    for number, line in enumerate(metadata.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"FAIL: {metadata}:{number} is not valid JSON: {exc}")
            return 1

    if not rows:
        print(f"FAIL: {metadata} has no rows")
        return 1

    problems = 0
    without_trigger = 0
    small = 0
    sizes: set[tuple[int, int]] = set()

    for number, row in enumerate(rows, start=1):
        file_name = row.get("file_name")
        text = row.get("text")

        if not isinstance(file_name, str) or not file_name:
            print(f"FAIL: row {number} has no 'file_name'")
            problems += 1
            continue
        if not isinstance(text, str) or not text.strip():
            print(f"FAIL: row {number} ({file_name}) has an empty 'text' caption")
            problems += 1
            continue

        image_path = root / file_name
        if not image_path.exists():
            print(f"FAIL: row {number} references missing image {image_path}")
            problems += 1
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:  # noqa: BLE001 - report any decode failure
            print(f"FAIL: row {number} image {image_path} will not open: {exc}")
            problems += 1
            continue

        sizes.add((width, height))
        if min(width, height) < MIN_SIDE:
            small += 1
        if trigger and not text.lstrip().lower().startswith(trigger.lower()):
            without_trigger += 1

    print(f"Dataset : {root}")
    print(f"Pairs   : {len(rows)}")
    print(f"Sizes   : {sorted(sizes)}")
    if small:
        print(f"WARN    : {small} image(s) smaller than {MIN_SIDE}px on a side")
    if trigger:
        if without_trigger:
            print(
                f"WARN    : {without_trigger} caption(s) do not start with the "
                f"trigger {trigger!r}"
            )
        else:
            print(f"OK      : every caption starts with the trigger {trigger!r}")

    if problems:
        print(f"\n{problems} problem(s) found — fix these before training.")
        return 1
    print("\nAll pairs valid. Ready to train.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset directory (imagefolder)")
    parser.add_argument(
        "--trigger",
        help="Warn about captions that do not start with this trigger phrase",
    )
    args = parser.parse_args()
    raise SystemExit(validate(args.root.resolve(), args.trigger))


if __name__ == "__main__":
    main()
