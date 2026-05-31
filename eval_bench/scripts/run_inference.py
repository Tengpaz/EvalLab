#!/usr/bin/env python
"""Run model inference for a configured virtual camera benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from eval_bench.io.images import apply_preprocess, read_image, save_image
from eval_bench.models.base import persist_predictions
from eval_bench.utils.config import dump_yaml, load_yaml, resolve_config_refs
from eval_bench.utils.logging import log, now_iso, write_jsonl
from eval_bench.utils.paths import ensure_dir, environment_info, write_json
from eval_bench.utils.registry import build_dataset, build_model


def _abspath(path_value: str | Path, base_dir: Path) -> str:
    path = Path(path_value).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _normalize_run_paths(config: dict, base_dir: Path) -> dict:
    dataset = config.get("dataset", {})
    if dataset.get("root"):
        dataset["root"] = _abspath(dataset["root"], base_dir)
    model = config.get("model", {})
    if model.get("adapter") and ":" in str(model["adapter"]):
        path_text, symbol = str(model["adapter"]).split(":", 1)
        model["adapter"] = f"{_abspath(path_text, base_dir)}:{symbol}"
    for key in ("weights", "config"):
        if model.get(key) and str(model[key]).startswith((".", "/", "~")):
            model[key] = _abspath(model[key], base_dir)
    return config


def _all_predictions_exist(output_dir: Path, target_ids: list[int]) -> bool:
    return all((output_dir / f"{target_id}.png").exists() for target_id in target_ids)


def _build_batch(sample, output_dir: Path, image_preprocess: dict | None, save_inputs_for_debug: bool) -> dict:
    input_frames = sample.input_frames()
    target_frames = sample.target_frames()
    input_images = []
    input_paths = []
    debug_dir = output_dir / "_debug_inputs"
    for frame in input_frames:
        img = apply_preprocess(read_image(frame.image_path), image_preprocess)
        input_images.append(img)
        input_paths.append(frame.image_path)
        if save_inputs_for_debug:
            save_image(debug_dir / f"{frame.frame_id}.png", img)
    return {
        "scene_id": sample.scene_id,
        "dataset_name": sample.dataset_name,
        "split_name": sample.split_name,
        "num_inputs": sample.num_inputs,
        "input_ids": sample.input_ids,
        "target_ids": sample.target_ids,
        "input_images": input_images,
        "input_image_paths": input_paths,
        "input_cameras": [frame.camera for frame in input_frames],
        "target_cameras": [frame.camera for frame in target_frames],
        "target_image_paths": [frame.image_path for frame in target_frames],
        "output_dir": output_dir,
        "metadata": sample.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()

    raw = load_yaml(args.run_config)
    config = resolve_config_refs(raw)
    config = _normalize_run_paths(config, Path.cwd())
    run_name = config.get("run_name") or Path(args.run_config).stem
    output_root = Path(config.get("output_root", "outputs"))
    run_dir = ensure_dir(output_root / run_name)
    metadata_dir = ensure_dir(run_dir / "metadata")
    dump_yaml(config, metadata_dir / "resolved_config.yaml")
    write_json(metadata_dir / "environment.json", environment_info(sys.argv, cwd=Path.cwd()))

    dataset = build_dataset(config["dataset"])
    model = build_model(config["model"])
    model.setup()

    inference_cfg = config.get("inference", {})
    overwrite = bool(inference_cfg.get("overwrite", False))
    resume = bool(inference_cfg.get("resume", True))
    continue_on_error = bool(inference_cfg.get("continue_on_error", True))
    save_inputs_for_debug = bool(inference_cfg.get("save_inputs_for_debug", False))
    image_preprocess = config["dataset"].get("image_preprocess", {})

    status_path = metadata_dir / "per_scene_status.jsonl"
    failure_path = metadata_dir / "failures.jsonl"
    if overwrite:
        for path in (status_path, failure_path):
            if path.exists():
                path.unlink()

    failures = 0
    total = 0
    for sample in dataset.iter_samples():
        total += 1
        pred_dir = ensure_dir(run_dir / "predictions" / sample.dataset_name / sample.split_name / sample.scene_id)
        if resume and not overwrite and _all_predictions_exist(pred_dir, sample.target_ids):
            write_jsonl(
                status_path,
                {
                    "time": now_iso(),
                    "scene_id": sample.scene_id,
                    "dataset": sample.dataset_name,
                    "split": sample.split_name,
                    "status": "skipped_existing",
                    "num_targets": len(sample.target_ids),
                },
            )
            log(f"SKIP {sample.dataset_name}/{sample.split_name}/{sample.scene_id}")
            continue
        try:
            batch = _build_batch(sample, pred_dir, image_preprocess, save_inputs_for_debug)
            prediction = model.predict(batch)
            persist_predictions(
                prediction,
                sample.target_ids,
                pred_dir,
            )
            write_jsonl(
                status_path,
                {
                    "time": now_iso(),
                    "scene_id": sample.scene_id,
                    "dataset": sample.dataset_name,
                    "split": sample.split_name,
                    "status": "ok",
                    "num_targets": len(sample.target_ids),
                },
            )
            log(f"OK {sample.dataset_name}/{sample.split_name}/{sample.scene_id}")
        except Exception as exc:
            failures += 1
            record = {
                "time": now_iso(),
                "scene_id": sample.scene_id,
                "dataset": sample.dataset_name,
                "split": sample.split_name,
                "status": "failed",
                "error": str(exc),
            }
            write_jsonl(status_path, record)
            write_jsonl(failure_path, record)
            log(f"FAIL {sample.dataset_name}/{sample.split_name}/{sample.scene_id}: {exc}")
            if not continue_on_error:
                raise

    if failures:
        log(f"Inference finished with {failures}/{total} failed scene(s). See {failure_path}.")
        raise SystemExit(1)
    log(f"Inference complete: {total} scene(s), run_dir={run_dir}")


if __name__ == "__main__":
    main()
