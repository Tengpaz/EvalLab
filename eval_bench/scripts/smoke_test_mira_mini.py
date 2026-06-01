#!/usr/bin/env python
"""Smoke test for MiraMiniAdapter.

Verifies that the adapter can:
  1. Load the checkpoint and build the model.
  2. Run a forward pass on a synthetic 8-view batch (3 inputs + 5 targets).

Usage (from eval_bench/):
    python scripts/smoke_test_mira_mini.py --model-config configs/models/mira_mini_prope.yaml
    python scripts/smoke_test_mira_mini.py --model-config configs/models/mira_mini_prope_window40.yaml
    python scripts/smoke_test_mira_mini.py --all
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401 — adds eval_bench to sys.path

from eval_bench.utils.config import load_yaml
from eval_bench.utils.logging import log


# ------------------------------------------------------------------ #
# Synthetic batch helpers                                              #
# ------------------------------------------------------------------ #

def _make_camera(tx: float) -> dict:
    """Simple camera at position (tx, 0, 1) looking along -Z (OpenGL c2w)."""
    return {
        "transform_matrix": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "intrinsics": {
            "width": 256, "height": 256,
            "fl_x": 256.0, "fl_y": 256.0,
            "cx": 128.0, "cy": 128.0,
        },
        "convention": "opengl_c2w",
    }


def _make_batch(num_inputs: int = 3, num_targets: int = 5) -> dict:
    """Create a synthetic eval_bench batch with num_inputs+num_targets views."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    import numpy as np

    total = num_inputs + num_targets
    rng = np.random.default_rng(42)

    input_images = [
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        for _ in range(num_inputs)
    ]
    input_cameras  = [_make_camera(tx=(i - num_inputs // 2) * 0.1) for i in range(num_inputs)]
    target_cameras = [_make_camera(tx=(i + 1) * 0.15) for i in range(num_targets)]
    target_ids     = list(range(num_inputs, total))

    return {
        "scene_id":           "smoke_scene",
        "dataset_name":       "smoke",
        "split_name":         "test",
        "num_inputs":         num_inputs,
        "input_ids":          list(range(num_inputs)),
        "target_ids":         target_ids,
        "input_images":       input_images,
        "input_image_paths":  [Path(f"/tmp/smoke_input_{i}.png") for i in range(num_inputs)],
        "input_cameras":      input_cameras,
        "target_cameras":     target_cameras,
        "target_image_paths": [Path(f"/tmp/smoke_target_{i}.png") for i in target_ids],
        "output_dir":         Path("/tmp/smoke_output"),
        "metadata":           {},
    }


# ------------------------------------------------------------------ #
# Test runner                                                          #
# ------------------------------------------------------------------ #

def run_smoke(config_path: str) -> bool:
    config_path = Path(config_path)
    log(f"=== Smoke test: {config_path.name} ===")

    # 1. Load YAML config
    model_config = load_yaml(str(config_path))
    # Resolve adapter path relative to config file
    if "adapter" in model_config:
        raw = model_config["adapter"]
        if ":" in raw:
            rel_path, cls = raw.split(":", 1)
            abs_path = (config_path.parent.parent.parent / rel_path).resolve()
            model_config["adapter"] = f"{abs_path}:{cls}"

    # 2. Load adapter class
    from eval_bench.models.python_adapter import load_symbol
    cls = load_symbol(model_config["adapter"])
    adapter = cls()
    log("  [1/3] Adapter class imported")

    # 3. setup() — loads model + checkpoint
    t0 = time.time()
    adapter.setup(model_config)
    elapsed = time.time() - t0
    log(f"  [2/3] setup() OK  ({elapsed:.1f}s)")

    # Count parameters
    num_params = sum(p.numel() for p in adapter.model.parameters()) / 1e6
    log(f"        model params: {num_params:.1f}M")

    # 4. predict() — one synthetic forward pass
    num_views = model_config.get("extra_args", {}).get("model_params", {}).get("num_views", 8)
    num_inputs = 3
    num_targets = num_views - num_inputs
    batch = _make_batch(num_inputs=num_inputs, num_targets=num_targets)

    t0 = time.time()
    predictions = adapter.predict(batch)
    elapsed = time.time() - t0
    log(f"  [3/3] predict() OK ({elapsed:.1f}s)  →  {len(predictions)} target image(s)")

    for tid, img in predictions.items():
        log(f"        target_id={tid}  size={img.size}  mode={img.mode}")

    log(f"  PASS: {config_path.name}\n")
    return True


def main() -> None:
    MODEL_CONFIGS = [
        "configs/models/mira_mini.yaml",
        "configs/models/mira_mini_window40.yaml",
        "configs/models/mira_mini_prope.yaml",
        "configs/models/mira_mini_prope_window40.yaml",
        "configs/models/mira_mini_prope_window40_lr2e-5.yaml",
        "configs/models/mira_mini_prope_register.yaml",
    ]

    parser = argparse.ArgumentParser(description="Smoke test for MiraMiniAdapter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-config", help="Path to a single model YAML config")
    group.add_argument("--all", action="store_true", help="Run all model configs")
    args = parser.parse_args()

    configs = MODEL_CONFIGS if args.all else [args.model_config]

    failures = []
    for cfg in configs:
        try:
            run_smoke(cfg)
        except Exception as exc:
            log(f"  FAIL: {cfg}: {exc}\n")
            failures.append((cfg, exc))

    if failures:
        log(f"{len(failures)}/{len(configs)} config(s) FAILED:")
        for cfg, exc in failures:
            log(f"  {cfg}: {exc}")
        sys.exit(1)
    else:
        log(f"All {len(configs)} config(s) passed.")


if __name__ == "__main__":
    main()
