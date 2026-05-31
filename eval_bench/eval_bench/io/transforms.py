"""Transforms.json parsing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_bench.io.cameras import camera_record


@dataclass(frozen=True)
class FrameRecord:
    """One image and camera from a scene."""

    frame_id: int
    image_path: Path
    camera: dict[str, Any]
    metadata: dict[str, Any]


def read_json(path: str | Path) -> Any:
    """Read JSON with a helpful filename on parse errors."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: str | Path, data: Any) -> None:
    """Write compact, readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_image_path(scene_dir: Path, file_path: str, image_dir: str = "images") -> Path:
    """Resolve a transforms.json frame file_path to an image path."""
    raw = Path(file_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                scene_dir / raw,
                scene_dir / image_dir / raw,
            ]
        )
        if raw.suffix == "":
            for suffix in (".png", ".jpg", ".jpeg"):
                candidates.append(scene_dir / f"{raw}{suffix}")
                candidates.append(scene_dir / image_dir / f"{raw}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_transforms_scene(
    scene_dir: str | Path,
    transforms_file: str = "transforms.json",
    image_dir: str = "images",
    sort_images: bool = True,
) -> tuple[dict[str, Any], list[FrameRecord]]:
    """Load a NeRF-style transforms scene.

    The returned frames use stable integer frame ids. If frames contain an
    explicit ``id`` or ``frame_id`` it is honored; otherwise ids come from the
    sorted frame order.
    """
    scene_dir = Path(scene_dir)
    path = scene_dir / transforms_file
    if not path.exists():
        raise FileNotFoundError(
            f"Missing transforms file: {path}. Place transforms.json in each scene "
            "directory or set transforms_file in the dataset config."
        )
    meta = read_json(path)
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty 'frames' list.")

    global_meta = {k: v for k, v in meta.items() if k != "frames"}
    ordered = list(frames)
    if sort_images:
        ordered.sort(key=lambda f: str(f.get("file_path", f.get("image_path", ""))))

    records: list[FrameRecord] = []
    for index, frame in enumerate(ordered):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {index} in {path} is not an object.")
        file_path = frame.get("file_path") or frame.get("image_path")
        if not file_path:
            raise ValueError(f"Frame {index} in {path} is missing file_path/image_path.")
        frame_id = int(frame.get("id", frame.get("frame_id", index)))
        image_path = resolve_image_path(scene_dir, str(file_path), image_dir=image_dir)
        records.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                camera=camera_record(frame, global_meta),
                metadata={k: v for k, v in frame.items() if k not in {"transform_matrix", "c2w"}},
            )
        )
    return global_meta, records
