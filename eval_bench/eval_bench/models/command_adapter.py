"""Black-box command-line model adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from eval_bench.models.base import BaseModelAdapter, expected_prediction_path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


class CommandAdapter(BaseModelAdapter):
    """Adapter that calls a configured external inference command."""

    def setup(self) -> None:
        if "command_template" not in self.config:
            raise ValueError("command model config requires command_template.")

    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(batch["output_dir"])
        manifest_dir = output_dir / "_command_inputs"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        input_images_json = manifest_dir / "input_images.json"
        input_cameras_json = manifest_dir / "input_cameras.json"
        target_cameras_json = manifest_dir / "target_cameras.json"
        metadata_json = manifest_dir / "metadata.json"

        _write_json(input_images_json, [str(p) for p in batch["input_image_paths"]])
        _write_json(input_cameras_json, batch["input_cameras"])
        _write_json(target_cameras_json, batch["target_cameras"])
        _write_json(
            metadata_json,
            {
                "scene_id": batch["scene_id"],
                "target_ids": batch["target_ids"],
                "dataset_name": batch["dataset_name"],
                "split_name": batch["split_name"],
            },
        )
        values = {
            "input_images_json": input_images_json,
            "input_cameras_json": input_cameras_json,
            "target_cameras_json": target_cameras_json,
            "metadata_json": metadata_json,
            "weights": self.config.get("weights", ""),
            "config": self.config.get("config", ""),
            "output_dir": output_dir,
            "scene_id": batch["scene_id"],
        }
        command = str(self.config["command_template"]).format(**values)
        env = os.environ.copy()
        for key, value in (self.config.get("env") or {}).items():
            env[str(key)] = str(value)
        proc = subprocess.run(command, shell=True, cwd=self.config.get("cwd"), env=env, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command model failed with exit code {proc.returncode}: {command}")

        pattern = self.config.get("output_pattern", "{target_id}.png")
        missing = [
            target_id
            for target_id in batch["target_ids"]
            if not expected_prediction_path(output_dir, target_id, pattern).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Command completed but missing output images for target ids "
                f"{missing}. Expected files like {output_dir / pattern.format(target_id=missing[0])}."
            )
        return {target_id: expected_prediction_path(output_dir, target_id, pattern) for target_id in batch["target_ids"]}
