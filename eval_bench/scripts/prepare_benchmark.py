#!/usr/bin/env python
"""Prepare benchmark helper utilities.

This script intentionally does not download proprietary/large datasets. It can
materialize a tiny synthetic dataset for smoke tests and validate configured
datasets after users place official files on disk.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from eval_bench.io.images import ImageData, write_png_rgb
from eval_bench.io.splits import write_split_file
from eval_bench.io.transforms import write_json
from eval_bench.utils.config import load_yaml
from eval_bench.utils.logging import log
from eval_bench.utils.registry import build_dataset


def _camera(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _gradient(width: int, height: int, offset: int) -> ImageData:
    pixels = []
    for y in range(height):
        for x in range(width):
            pixels.append(((x * 7 + offset) % 256, (y * 9 + offset) % 256, (x * 3 + y * 5 + offset) % 256))
    return ImageData(width, height, pixels)


def make_tiny_dataset(out: Path) -> None:
    """Create a 3-frame transforms.json dataset for pipeline smoke tests."""
    scene_dir = out / "scene_000"
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_id in range(3):
        image_name = f"{frame_id:03d}.png"
        write_png_rgb(image_dir / image_name, _gradient(32, 24, frame_id * 40))
        frames.append(
            {
                "id": frame_id,
                "file_path": f"images/{image_name}",
                "transform_matrix": _camera((frame_id - 1) * 0.1),
                "fl_x": 35.0,
                "fl_y": 35.0,
                "cx": 16.0,
                "cy": 12.0,
                "w": 32,
                "h": 24,
            }
        )
    write_json(
        scene_dir / "transforms.json",
        {
            "camera_angle_x": math.radians(50.0),
            "camera_convention": "opengl_c2w",
            "frames": frames,
        },
    )
    write_split_file(scene_dir / "train_test_split_1.json", [0], [1, 2])
    log(f"Created tiny dataset at {scene_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-tiny", action="store_true", help="Create a tiny synthetic dataset.")
    parser.add_argument("--out", default="outputs/tiny_dataset", help="Output directory for --make-tiny.")
    parser.add_argument("--dataset-config", help="Optionally validate a dataset config after preparing files.")
    args = parser.parse_args()

    if args.make_tiny:
        make_tiny_dataset(Path(args.out))

    if args.dataset_config:
        dataset = build_dataset(load_yaml(args.dataset_config))
        issues = dataset.validate()
        errors = [issue for issue in issues if issue["level"] == "error"]
        for issue in issues:
            log(f"{issue['level'].upper()}: {issue['scene_id']}: {issue['message']}")
        if errors:
            raise SystemExit(1)
        log("Dataset validation passed.")

    if not args.make_tiny and not args.dataset_config:
        parser.error("Provide --make-tiny and/or --dataset-config.")


if __name__ == "__main__":
    main()
