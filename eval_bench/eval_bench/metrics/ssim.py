"""Dependency-free SSIM approximation.

This implementation computes global RGB SSIM rather than the common windowed
variant from skimage. It is useful for smoke tests and lightweight runs. For
paper numbers, install skimage or adapt this module to your benchmark standard.
"""

from __future__ import annotations

from eval_bench.io.images import ImageData


def _channel_values(image: ImageData, channel: int) -> list[float]:
    return [float(pixel[channel]) for pixel in image.pixels]


def _ssim_1d(a: list[float], b: list[float]) -> float:
    n = max(1, len(a))
    mu_a = sum(a) / n
    mu_b = sum(b) / n
    var_a = sum((x - mu_a) ** 2 for x in a) / n
    var_b = sum((x - mu_b) ** 2 for x in b) / n
    cov = sum((x - mu_a) * (y - mu_b) for x, y in zip(a, b)) / n
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )


def compute_ssim(pred: ImageData, gt: ImageData) -> float:
    """Compute mean global SSIM across RGB channels."""
    if (pred.width, pred.height) != (gt.width, gt.height):
        raise ValueError("Images must have matching dimensions for SSIM.")
    return sum(_ssim_1d(_channel_values(pred, ch), _channel_values(gt, ch)) for ch in range(3)) / 3.0
