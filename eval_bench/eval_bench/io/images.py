"""Image loading, saving, preprocessing, and simple PNG support.

Pillow is used when available. The stdlib PNG path keeps the smoke test
dependency-free in minimal Python environments.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class ImageData:
    """RGB image stored as uint8 triples in row-major order."""

    width: int
    height: int
    pixels: list[tuple[int, int, int]]

    def copy(self) -> "ImageData":
        return ImageData(self.width, self.height, list(self.pixels))


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)


def write_png_rgb(path: str | Path, image: ImageData) -> None:
    """Write an RGB PNG using only the standard library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    idx = 0
    for _ in range(image.height):
        raw.append(0)
        for _ in range(image.width):
            r, g, b = image.pixels[idx]
            raw.extend((r & 255, g & 255, b & 255))
            idx += 1
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    data = PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw))) + _chunk(b"IEND", b"")
    path.write_bytes(data)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgb(path: str | Path) -> ImageData:
    """Read common 8-bit PNG files as RGB.

    Supported PNG color types: grayscale, RGB, and RGBA.
    """
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG file and Pillow is not installed.")
    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        chunk_type = data[offset : offset + 4]
        offset += 4
        payload = data[offset : offset + length]
        offset += length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError(
            f"Unsupported PNG in {path}. Need 8-bit grayscale/RGB/RGBA or install Pillow."
        )

    channels = {0: 1, 2: 3, 6: 4}[color_type]
    bpp = channels
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    rows: list[bytearray] = []
    pos = 0
    prev = bytearray(stride)
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos : pos + stride])
        pos += stride
        for i in range(stride):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 1:
                row[i] = (row[i] + left) & 255
            elif filter_type == 2:
                row[i] = (row[i] + up) & 255
            elif filter_type == 3:
                row[i] = (row[i] + ((left + up) // 2)) & 255
            elif filter_type == 4:
                row[i] = (row[i] + _paeth(left, up, up_left)) & 255
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type} in {path}.")
        rows.append(row)
        prev = row

    pixels: list[tuple[int, int, int]] = []
    for row in rows:
        for x in range(width):
            base = x * channels
            if channels == 1:
                v = row[base]
                pixels.append((v, v, v))
            else:
                pixels.append((row[base], row[base + 1], row[base + 2]))
    return ImageData(width, height, pixels)


def read_image(path: str | Path) -> ImageData:
    """Read an image as RGB ImageData."""
    path = Path(path)
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            pixels = list(rgb.getdata())
            return ImageData(rgb.width, rgb.height, [(int(r), int(g), int(b)) for r, g, b in pixels])
    except ModuleNotFoundError:
        if path.suffix.lower() != ".png":
            raise ValueError(f"Cannot read {path}: install Pillow for non-PNG image support.")
        return read_png_rgb(path)


def save_image(path: str | Path, image: ImageData | object) -> None:
    """Save ImageData, a Pillow image, or a numpy-like HxWxC array to PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, ImageData):
        write_png_rgb(path, image)
        return
    if hasattr(image, "save"):
        image.save(path)
        return
    if hasattr(image, "shape") and hasattr(image, "tolist"):
        data = image.tolist()
        h = len(data)
        w = len(data[0]) if h else 0
        pixels: list[tuple[int, int, int]] = []
        for row in data:
            for value in row:
                rgb = list(value[:3])
                if rgb and max(rgb) <= 1.0:
                    rgb = [round(float(v) * 255.0) for v in rgb]
                pixels.append(tuple(max(0, min(255, int(v))) for v in rgb))  # type: ignore[arg-type]
        write_png_rgb(path, ImageData(w, h, pixels))
        return
    raise TypeError(
        f"Unsupported image result type {type(image).__name__}. Return an image path, "
        "ImageData, Pillow image, or numpy-like HxWxC array."
    )


def center_crop(image: ImageData, size: int | None = None) -> ImageData:
    """Center-crop to a square and optionally resize to ``size``."""
    crop_size = min(image.width, image.height)
    x0 = (image.width - crop_size) // 2
    y0 = (image.height - crop_size) // 2
    pixels = []
    for y in range(y0, y0 + crop_size):
        start = y * image.width + x0
        pixels.extend(image.pixels[start : start + crop_size])
    cropped = ImageData(crop_size, crop_size, pixels)
    if size and size != crop_size:
        return resize_nearest(cropped, size, size)
    return cropped


def resize_nearest(image: ImageData, width: int, height: int) -> ImageData:
    """Resize an image with nearest-neighbor sampling."""
    if width <= 0 or height <= 0:
        raise ValueError("Resize dimensions must be positive.")
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        src_y = min(image.height - 1, int(y * image.height / height))
        for x in range(width):
            src_x = min(image.width - 1, int(x * image.width / width))
            pixels.append(image.pixels[src_y * image.width + src_x])
    return ImageData(width, height, pixels)


def resize_short(image: ImageData, short_side: int) -> ImageData:
    """Resize so the shorter side matches ``short_side``."""
    if image.width <= image.height:
        width = short_side
        height = max(1, round(image.height * short_side / image.width))
    else:
        height = short_side
        width = max(1, round(image.width * short_side / image.height))
    return resize_nearest(image, width, height)


def apply_preprocess(image: ImageData, spec: dict | None) -> ImageData:
    """Apply a configured image preprocess."""
    if not spec:
        return image
    mode = spec.get("mode", "none")
    result = image
    if mode == "center_crop":
        result = center_crop(result, spec.get("size"))
    elif mode == "resize":
        size = spec.get("size")
        if isinstance(size, int):
            result = resize_nearest(result, size, size)
        elif isinstance(size, Sequence) and len(size) == 2:
            result = resize_nearest(result, int(size[0]), int(size[1]))
        else:
            raise ValueError("resize preprocess requires size: int or [width, height].")
    elif mode in ("none", None):
        result = image
    else:
        raise ValueError(f"Unknown image preprocess mode {mode!r}.")
    return result


def align_for_metric(pred: ImageData, gt: ImageData, spec: dict | None) -> tuple[ImageData, ImageData]:
    """Align prediction and GT dimensions according to metric config."""
    spec = spec or {}
    pred2, gt2 = pred, gt
    resize_exact = spec.get("resize_exact")
    resize_short_value = spec.get("resize_short")
    if resize_exact:
        if isinstance(resize_exact, int):
            pred2 = resize_nearest(pred2, resize_exact, resize_exact)
            gt2 = resize_nearest(gt2, resize_exact, resize_exact)
        else:
            pred2 = resize_nearest(pred2, int(resize_exact[0]), int(resize_exact[1]))
            gt2 = resize_nearest(gt2, int(resize_exact[0]), int(resize_exact[1]))
    if resize_short_value:
        pred2 = resize_short(pred2, int(resize_short_value))
        gt2 = resize_short(gt2, int(resize_short_value))
    if spec.get("center_crop"):
        size = spec.get("crop_size")
        pred2 = center_crop(pred2, size)
        gt2 = center_crop(gt2, size)
    if (pred2.width, pred2.height) != (gt2.width, gt2.height):
        policy = spec.get("mismatch", "error")
        if policy == "resize_pred_to_gt":
            pred2 = resize_nearest(pred2, gt2.width, gt2.height)
        elif policy == "resize_gt_to_pred":
            gt2 = resize_nearest(gt2, pred2.width, pred2.height)
        elif policy == "center_crop_common":
            common = min(pred2.width, pred2.height, gt2.width, gt2.height)
            pred2 = center_crop(pred2, common)
            gt2 = center_crop(gt2, common)
        else:
            raise ValueError(
                "Prediction and GT dimensions differ "
                f"({pred2.width}x{pred2.height} vs {gt2.width}x{gt2.height}). "
                "Set metrics.metric_postprocess.mismatch to resize_pred_to_gt, "
                "resize_gt_to_pred, or center_crop_common."
            )
    return pred2, gt2


def gray_image(width: int, height: int, value: int = 128) -> ImageData:
    """Create a solid gray RGB image."""
    return ImageData(width, height, [(value, value, value)] * (width * height))


def mse_rgb(a: ImageData, b: ImageData) -> float:
    """Mean squared RGB error in uint8 space."""
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError("Images must have the same dimensions for MSE.")
    total = 0.0
    count = a.width * a.height * 3
    for pa, pb in zip(a.pixels, b.pixels):
        total += (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
    return total / max(1, count)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(1, len(vals))


def psnr_from_mse(mse: float, max_value: float = 255.0) -> float:
    """Convert MSE to PSNR."""
    if mse <= 0:
        return math.inf
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)
