"""Compatibility adapter for Stability-AI Stable Virtual Camera benchmark data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from eval_bench.datasets.base import BaseDatasetAdapter, SceneSample
from eval_bench.io.splits import read_split_file
from eval_bench.io.transforms import load_transforms_scene


class SevaBenchmarkDataset(BaseDatasetAdapter):
    """Read Stable Virtual Camera benchmark-style scene folders.

    Expected scene layout:

    scene_id/
      images/
      transforms.json
      train_test_split_N.json

    The adapter honors the official benchmark convention that frame ordering is
    sorted by image path when no explicit frame id is provided, and it keeps
    OpenGL camera-to-world matrices untouched.
    """

    def _scene_dirs(self) -> list[Path]:
        transforms_file = self.config.get("transforms_file", "transforms.json")
        if (self.root / transforms_file).exists():
            return [self.root]
        scene_ids = self.config.get("scene_ids")
        if scene_ids:
            return [self.root / str(scene_id) for scene_id in scene_ids]
        scene_glob = self.config.get("scene_glob", "*")
        return sorted(
            [p for p in self.root.glob(str(scene_glob)) if p.is_dir() and (p / transforms_file).exists()]
        )

    def iter_samples(self) -> Iterable[SceneSample]:
        scene_dirs = self._scene_dirs()
        if not scene_dirs:
            raise FileNotFoundError(
                f"No benchmark scenes found under {self.root}. Expected scene/images, "
                "scene/transforms.json, and scene/train_test_split_N.json."
            )
        split_file = self.config.get("split_file")
        if not split_file:
            num_inputs = self.config.get("num_inputs")
            if not num_inputs:
                raise ValueError("seva_benchmark config needs split_file or num_inputs.")
            split_file = f"train_test_split_{num_inputs}.json"
        transforms_file = self.config.get("transforms_file", "transforms.json")
        image_dir = self.config.get("image_dir", "images")
        sort_images = bool(self.config.get("sort_images", True))

        for scene_dir in scene_dirs:
            meta, frames = load_transforms_scene(
                scene_dir,
                transforms_file=transforms_file,
                image_dir=image_dir,
                sort_images=sort_images,
            )
            split_path = Path(split_file).expanduser()
            if not split_path.is_absolute():
                split_path = scene_dir / split_path
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Missing split file: {split_path}. Download/copy the official benchmark "
                    "split files into each scene folder or set dataset.split_file to an absolute path."
                )
            split = read_split_file(split_path)
            frame_map = {frame.frame_id: frame for frame in frames}
            yield SceneSample(
                dataset_name=self.name,
                split_name=Path(str(split_file)).stem,
                num_inputs=self.num_inputs or len(split["input_ids"]),
                scene_id=str(scene_dir.name),
                scene_dir=scene_dir,
                frames=frame_map,
                input_ids=split["input_ids"],
                target_ids=split["target_ids"],
                metadata={
                    "dataset_type": "seva_benchmark",
                    "camera_convention": "opengl_c2w",
                    "image_sorting": "path_sort" if sort_images else "transforms_order",
                    "transforms_meta": meta,
                    "split": split.get("metadata", {}),
                },
            )
