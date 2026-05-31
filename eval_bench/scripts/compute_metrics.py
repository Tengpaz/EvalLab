#!/usr/bin/env python
"""Compute PSNR/SSIM/LPIPS for an inference run."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from eval_bench.metrics.runner import compute_run_metrics
from eval_bench.utils.logging import log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    outputs = compute_run_metrics(args.run_dir)
    for name, path in outputs.items():
        log(f"{name}: {path}")


if __name__ == "__main__":
    main()
