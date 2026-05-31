"""Metric computation over run directories."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_bench.io.images import align_for_metric, read_image
from eval_bench.metrics.lpips_metric import LPIPSMetric
from eval_bench.metrics.psnr import compute_psnr
from eval_bench.metrics.ssim import compute_ssim
from eval_bench.utils.config import load_yaml
from eval_bench.utils.registry import build_dataset


def _mean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return None
    if any(math.isinf(v) for v in finite):
        return math.inf
    return sum(finite) / len(finite)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_run_metrics(run_dir: str | Path) -> dict[str, Path]:
    """Compute configured metrics for one inference run directory."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "metadata" / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {cfg_path}. Run inference first.")
    config = load_yaml(cfg_path)
    dataset = build_dataset(config["dataset"])
    metrics_cfg = config.get("metrics", {})
    enabled = metrics_cfg.get("enabled", ["psnr", "ssim"])
    postprocess = metrics_cfg.get("metric_postprocess") or config["dataset"].get("metric_postprocess") or {}
    skip_unavailable = bool(metrics_cfg.get("skip_unavailable", True))
    lpips_metric = None
    if "lpips" in enabled:
        try:
            lpips_metric = LPIPSMetric(metrics_cfg.get("lpips_net", "alex"), metrics_cfg.get("lpips_device", config.get("model", {}).get("device", "cuda:0")))
        except ImportError:
            if not skip_unavailable:
                raise
            lpips_metric = None

    per_image_path = run_dir / "metrics" / "per_image_metrics.jsonl"
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    if per_image_path.exists():
        per_image_path.unlink()
    per_rows: list[dict[str, Any]] = []
    scene_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in dataset.iter_samples():
        pred_dir = run_dir / "predictions" / sample.dataset_name / sample.split_name / sample.scene_id
        for frame in sample.target_frames():
            pred_path = pred_dir / f"{frame.frame_id}.png"
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"Missing prediction {pred_path}. Run inference or disable resume skipping issues."
                )
            pred_img = read_image(pred_path)
            gt_img = read_image(frame.image_path)
            pred_aligned, gt_aligned = align_for_metric(pred_img, gt_img, postprocess)
            row: dict[str, Any] = {
                "dataset": sample.dataset_name,
                "split": sample.split_name,
                "num_inputs": sample.num_inputs,
                "scene_id": sample.scene_id,
                "target_id": frame.frame_id,
                "prediction": str(pred_path),
                "ground_truth": str(frame.image_path),
            }
            if "psnr" in enabled:
                row["psnr"] = compute_psnr(pred_aligned, gt_aligned)
            if "ssim" in enabled:
                row["ssim"] = compute_ssim(pred_aligned, gt_aligned)
            if "lpips" in enabled:
                row["lpips"] = lpips_metric(pred_aligned, gt_aligned) if lpips_metric else None
            with per_image_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            per_rows.append(row)
            key = (sample.dataset_name, sample.split_name, sample.scene_id)
            for metric in ("psnr", "ssim", "lpips"):
                if metric in row and row[metric] is not None:
                    scene_values[key][metric].append(float(row[metric]))

    scene_rows: list[dict[str, Any]] = []
    for (dataset_name, split_name, scene_id), values in sorted(scene_values.items()):
        row = {
            "dataset": dataset_name,
            "split": split_name,
            "num_inputs": next((r["num_inputs"] for r in per_rows if r["scene_id"] == scene_id), None),
            "scene_id": scene_id,
            "num_targets": sum(1 for r in per_rows if r["scene_id"] == scene_id and r["dataset"] == dataset_name and r["split"] == split_name),
        }
        for metric in ("psnr", "ssim", "lpips"):
            if values.get(metric):
                row[metric] = _mean(values[metric])
        scene_rows.append(row)

    summary_groups: dict[tuple[str, str, Any], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scene_rows:
        key = (row["dataset"], row["split"], row.get("num_inputs"))
        for metric in ("psnr", "ssim", "lpips"):
            if metric in row and row[metric] is not None:
                summary_groups[key][metric].append(float(row[metric]))
    summary_rows: list[dict[str, Any]] = []
    model_name = config.get("model", {}).get("name") or config.get("model", {}).get("type")
    for (dataset_name, split_name, num_inputs), values in sorted(summary_groups.items()):
        row = {
            "run_name": config.get("run_name", run_dir.name),
            "model": model_name,
            "dataset": dataset_name,
            "split": split_name,
            "num_inputs": num_inputs,
            "num_scenes": sum(1 for r in scene_rows if r["dataset"] == dataset_name and r["split"] == split_name),
        }
        for metric in ("psnr", "ssim", "lpips"):
            if values.get(metric):
                row[metric] = _mean(values[metric])
        summary_rows.append(row)

    scene_csv = run_dir / "metrics" / "per_scene_metrics.csv"
    summary_csv = run_dir / "metrics" / "summary.csv"
    scene_fields = ["dataset", "split", "num_inputs", "scene_id", "num_targets", "psnr", "ssim", "lpips"]
    summary_fields = ["run_name", "model", "dataset", "split", "num_inputs", "num_scenes", "psnr", "ssim", "lpips"]
    _write_csv(scene_csv, scene_rows, scene_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)
    return {
        "per_image": per_image_path,
        "per_scene": scene_csv,
        "summary": summary_csv,
    }
