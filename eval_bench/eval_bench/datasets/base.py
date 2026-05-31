"""Dataset adapter base classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from eval_bench.io.images import read_image
from eval_bench.io.transforms import FrameRecord


@dataclass
class SceneSample:
    """One scene and split to evaluate."""

    dataset_name: str
    split_name: str
    num_inputs: int | None
    scene_id: str
    scene_dir: Path
    frames: dict[int, FrameRecord]
    input_ids: list[int]
    target_ids: list[int]
    metadata: dict[str, Any]

    def input_frames(self) -> list[FrameRecord]:
        return [self.frames[i] for i in self.input_ids]

    def target_frames(self) -> list[FrameRecord]:
        return [self.frames[i] for i in self.target_ids]


class BaseDatasetAdapter:
    """Base class for scene/split dataset adapters."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = str(config.get("name") or config.get("type") or "dataset")
        self.root = Path(config.get("root", ".")).expanduser()
        self.split_name = str(
            config.get("split_name")
            or Path(str(config.get("split_file", "auto_split"))).stem
            or "split"
        )
        self.num_inputs = config.get("num_inputs")

    def iter_samples(self) -> Iterable[SceneSample]:
        """Yield SceneSample objects."""
        raise NotImplementedError

    def validate(self, check_images: bool = True) -> list[dict[str, Any]]:
        """Validate scenes, camera records, image presence, and split ids."""
        records: list[dict[str, Any]] = []
        for sample in self.iter_samples():
            records.extend(self.validate_sample(sample, check_images=check_images))
        return records

    def validate_sample(self, sample: SceneSample, check_images: bool = True) -> list[dict[str, Any]]:
        """Validate one sample and return issue records."""
        issues: list[dict[str, Any]] = []
        frame_ids = set(sample.frames)
        for label, ids in (("input", sample.input_ids), ("target", sample.target_ids)):
            for frame_id in ids:
                if frame_id not in frame_ids:
                    issues.append(
                        {
                            "level": "error",
                            "scene_id": sample.scene_id,
                            "message": f"{label} id {frame_id} is outside available frame ids.",
                            "fix": "Check split ids against transforms.json frame order/ids.",
                        }
                    )
        overlap = sorted(set(sample.input_ids).intersection(sample.target_ids))
        if overlap:
            issues.append(
                {
                    "level": "error",
                    "scene_id": sample.scene_id,
                    "message": f"input_ids and target_ids overlap: {overlap}",
                    "fix": "Remove conditioning views from target_ids; metrics only evaluate targets.",
                }
            )
        if not sample.input_ids:
            issues.append(
                {
                    "level": "error",
                    "scene_id": sample.scene_id,
                    "message": "split has no input_ids.",
                    "fix": "Use num_inputs/split_file or auto_split to define conditioning views.",
                }
            )
        if not sample.target_ids:
            issues.append(
                {
                    "level": "error",
                    "scene_id": sample.scene_id,
                    "message": "split has no target_ids.",
                    "fix": "Define held-out target views in the split.",
                }
            )
        sizes: set[tuple[int, int]] = set()
        for frame_id, frame in sample.frames.items():
            matrix = frame.camera.get("transform_matrix")
            if not isinstance(matrix, list) or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                issues.append(
                    {
                        "level": "error",
                        "scene_id": sample.scene_id,
                        "frame_id": frame_id,
                        "message": "camera transform_matrix is not 4x4.",
                        "fix": "Use OpenGL camera-to-world 4x4 matrices in transforms.json.",
                    }
                )
            if check_images:
                if not frame.image_path.exists():
                    issues.append(
                        {
                            "level": "error",
                            "scene_id": sample.scene_id,
                            "frame_id": frame_id,
                            "message": f"missing image: {frame.image_path}",
                            "fix": "Verify transforms.json file_path values and image_dir.",
                        }
                    )
                    continue
                try:
                    img = read_image(frame.image_path)
                    sizes.add((img.width, img.height))
                except Exception as exc:
                    issues.append(
                        {
                            "level": "error",
                            "scene_id": sample.scene_id,
                            "frame_id": frame_id,
                            "message": f"could not read image {frame.image_path}: {exc}",
                            "fix": "Install Pillow for JPEG/non-standard PNGs, or convert images to 8-bit PNG.",
                        }
                    )
        if len(sizes) > 1:
            issues.append(
                {
                    "level": "warning",
                    "scene_id": sample.scene_id,
                    "message": f"image sizes are not consistent: {sorted(sizes)}",
                    "fix": "Set image_preprocess or metric_postprocess resize/crop policy.",
                }
            )
        return issues
