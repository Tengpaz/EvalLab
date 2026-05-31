"""PSNR metric."""

from __future__ import annotations

from eval_bench.io.images import ImageData, mse_rgb, psnr_from_mse


def compute_psnr(pred: ImageData, gt: ImageData) -> float:
    """Compute RGB PSNR in dB for uint8 images."""
    return psnr_from_mse(mse_rgb(pred, gt))
