"""Model adapter base classes and shared helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval_bench.io.images import ImageData, save_image


class BaseModelAdapter:
    """Base model adapter interface."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = str(config.get("name") or config.get("adapter") or config.get("type") or "model")

    def setup(self) -> None:
        """Initialize model resources."""

    def predict(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """Predict target views for one scene split."""
        raise NotImplementedError


def expected_prediction_path(output_dir: str | Path, target_id: int | str, pattern: str = "{target_id}.png") -> Path:
    """Return the canonical output path for a target id."""
    return Path(output_dir) / pattern.format(target_id=target_id)


def persist_predictions(
    prediction: dict[str, Any] | None,
    target_ids: list[int],
    output_dir: str | Path,
    output_pattern: str = "{target_id}.png",
) -> dict[int, Path]:
    """Normalize model predictions to files and return target_id->path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = prediction or {}
    result: dict[int, Path] = {}
    missing: list[int] = []
    for target_id in target_ids:
        value = prediction.get(target_id, prediction.get(str(target_id)))
        out_path = expected_prediction_path(output_dir, target_id, output_pattern)
        if value is None and out_path.exists():
            result[target_id] = out_path
            continue
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                missing.append(target_id)
                continue
            if path.resolve() != out_path.resolve():
                image = path.read_bytes()
                out_path.write_bytes(image)
            result[target_id] = out_path
        elif isinstance(value, ImageData) or hasattr(value, "save") or hasattr(value, "shape"):
            save_image(out_path, value)
            result[target_id] = out_path
        else:
            missing.append(target_id)
    if missing:
        raise FileNotFoundError(
            "Model did not produce predictions for target ids "
            f"{missing}. Return a dict target_id->image/path or write {output_pattern} files to output_dir."
        )
    return result
