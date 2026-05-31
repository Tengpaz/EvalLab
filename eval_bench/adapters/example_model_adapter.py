"""Dependency-free mock model adapter for smoke tests.

The adapter either copies the first conditioning view or writes a gray image.
It demonstrates the minimal Python adapter contract expected by PythonAdapter.
"""

from __future__ import annotations

from pathlib import Path

from eval_bench.io.images import gray_image, read_image


class ExampleModelAdapter:
    """Simple mock adapter implementing setup and predict."""

    def setup(self, model_config):
        self.mode = (model_config.get("extra_args") or {}).get("mode", "copy_first_input")
        self.gray_value = int((model_config.get("extra_args") or {}).get("gray_value", 128))

    def predict(self, batch):
        predictions = {}
        if self.mode == "gray":
            first = read_image(batch["input_image_paths"][0])
            for target_id in batch["target_ids"]:
                predictions[target_id] = gray_image(first.width, first.height, self.gray_value)
            return predictions

        first_input = Path(batch["input_image_paths"][0])
        for target_id in batch["target_ids"]:
            predictions[target_id] = first_input
        return predictions
