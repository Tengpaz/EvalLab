"""Camera parsing helpers for NeRF-style transforms files."""

from __future__ import annotations

import math
from typing import Any


def _as_matrix(value: Any, rows: int, cols: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must be a {rows}x{cols} list.")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name} must be a {rows}x{cols} list.")
        matrix.append([float(x) for x in row])
    return matrix


def normalize_transform_matrix(frame: dict[str, Any]) -> list[list[float]]:
    """Read a 4x4 camera-to-world matrix from common transforms.json keys."""
    for key in ("transform_matrix", "c2w", "camera_to_world"):
        if key in frame:
            return _as_matrix(frame[key], 4, 4, key)
    raise ValueError(
        "Frame is missing a 4x4 camera-to-world matrix. Expected one of: "
        "transform_matrix, c2w, camera_to_world."
    )


def intrinsics_from_frame(frame: dict[str, Any], global_meta: dict[str, Any]) -> dict[str, float | int | None]:
    """Return camera intrinsics from frame-level or global NeRF metadata."""
    meta = dict(global_meta)
    meta.update({k: v for k, v in frame.items() if v is not None})
    width = meta.get("w") or meta.get("width")
    height = meta.get("h") or meta.get("height")
    fl_x = meta.get("fl_x") or meta.get("fx")
    fl_y = meta.get("fl_y") or meta.get("fy")
    cx = meta.get("cx")
    cy = meta.get("cy")

    camera_angle_x = meta.get("camera_angle_x")
    camera_angle_y = meta.get("camera_angle_y")
    if fl_x is None and camera_angle_x is not None and width:
        fl_x = 0.5 * float(width) / math.tan(0.5 * float(camera_angle_x))
    if fl_y is None and camera_angle_y is not None and height:
        fl_y = 0.5 * float(height) / math.tan(0.5 * float(camera_angle_y))
    if fl_y is None and fl_x is not None:
        fl_y = fl_x
    if fl_x is None and fl_y is not None:
        fl_x = fl_y
    if cx is None and width:
        cx = float(width) / 2.0
    if cy is None and height:
        cy = float(height) / 2.0

    return {
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "fl_x": float(fl_x) if fl_x is not None else None,
        "fl_y": float(fl_y) if fl_y is not None else None,
        "cx": float(cx) if cx is not None else None,
        "cy": float(cy) if cy is not None else None,
        "camera_angle_x": float(camera_angle_x) if camera_angle_x is not None else None,
        "camera_angle_y": float(camera_angle_y) if camera_angle_y is not None else None,
    }


def camera_record(frame: dict[str, Any], global_meta: dict[str, Any]) -> dict[str, Any]:
    """Build a serializable camera record for model adapters."""
    return {
        "transform_matrix": normalize_transform_matrix(frame),
        "intrinsics": intrinsics_from_frame(frame, global_meta),
        "convention": global_meta.get("camera_convention", "opengl_c2w"),
    }
