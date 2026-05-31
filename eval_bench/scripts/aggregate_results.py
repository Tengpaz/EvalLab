#!/usr/bin/env python
"""Aggregate one or more run summary files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from eval_bench.utils.logging import log


def _float_or_none(value: str):
    if value in ("", "None", "null"):
        return None
    return float(value)


def _mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    if any(math.isinf(v) for v in values):
        return math.inf
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    for run in args.runs:
        summary = Path(run) / "metrics" / "summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"Missing summary: {summary}. Run compute_metrics first.")
        with summary.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("model"), row.get("dataset"), row.get("split"), row.get("num_inputs"))
                counts[key] += int(row.get("num_scenes") or 0)
                for metric in ("psnr", "ssim", "lpips"):
                    if metric in row:
                        groups[key][metric].append(_float_or_none(row[metric]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["model", "dataset", "split", "num_inputs", "num_scenes", "psnr", "ssim", "lpips"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in sorted(groups.items()):
            model, dataset, split, num_inputs = key
            writer.writerow(
                {
                    "model": model,
                    "dataset": dataset,
                    "split": split,
                    "num_inputs": num_inputs,
                    "num_scenes": counts[key],
                    "psnr": _mean(values["psnr"]),
                    "ssim": _mean(values["ssim"]),
                    "lpips": _mean(values["lpips"]),
                }
            )
    log(f"Wrote aggregate: {out}")


if __name__ == "__main__":
    main()
