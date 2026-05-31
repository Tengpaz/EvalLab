"""Adapter for user-owned NeRF/transforms.json datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from eval_bench.datasets.base import BaseDatasetAdapter, SceneSample
from eval_bench.io.splits import generate_split, read_split_file
from eval_bench.io.transforms import load_transforms_scene


class GenericTransformsDataset(BaseDatasetAdapter):
    """Load one or more scenes with NeRF-style transforms.json files."""

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
        """Yield one sample per scene for the configured split."""
        scene_dirs = self._scene_dirs()
        if not scene_dirs:
            raise FileNotFoundError(
                f"No scenes found under {self.root}. Expected root/transforms.json or "
                "subdirectories containing transforms.json. Set scene_ids or scene_glob if needed."
            )
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
            frame_map = {frame.frame_id: frame for frame in frames}
            frame_ids = [frame.frame_id for frame in frames]
            split_file = self.config.get("split_file")
            if split_file:
                split_path = Path(split_file).expanduser()
                if not split_path.is_absolute():
                    split_path = scene_dir / split_path
                split = read_split_file(split_path)
            else:
                split = generate_split(frame_ids, self.config.get("auto_split", {}))
            yield SceneSample(
                dataset_name=self.name,
                split_name=self.split_name,
                num_inputs=self.num_inputs or len(split["input_ids"]),
                scene_id=str(self.config.get("scene_id") or scene_dir.name),
                scene_dir=scene_dir,
                frames=frame_map,
                input_ids=split["input_ids"],
                target_ids=split["target_ids"],
                metadata={
                    "dataset_type": "generic_transforms",
                    "transforms_meta": meta,
                    "split": split.get("metadata", {}),
                },
            )
