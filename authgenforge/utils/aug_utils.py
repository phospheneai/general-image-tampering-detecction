from __future__ import annotations

import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


def recursively_read(
    root: str | Path,
    must_contain: str = "",
    exts: set[str] | None = None,
) -> list[str]:
    """
    Recursively find image files under a directory.
    """
    root = Path(root)
    exts = IMAGE_EXTS if exts is None else exts

    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in exts:
            continue

        if must_contain and must_contain not in str(path):
            continue

        files.append(str(path))

    return sorted(files)


def jpeg_compress(
    image: Image.Image,
    quality: int = 75,
) -> Image.Image:
    """
    Apply JPEG compression and return the decoded image.
    """
    buffer = io.BytesIO()

    image = image.convert("RGB")
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


RESAMPLE_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resize_pil(
    image: Image.Image,
    scale: float,
    method: str = "bicubic",
) -> Image.Image:
    """
    Resize a PIL image by a scale factor.
    """
    width, height = image.size

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resample = RESAMPLE_METHODS.get(
        method,
        Image.Resampling.BICUBIC,
    )

    return image.resize(
        (new_width, new_height),
        resample=resample,
    )


def make_resized_versions(
    image: Image.Image,
    down_scales: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    up_scales: tuple[float, ...] = (1.25, 1.5, 2.0),
    max_pixels: int = 16_000_000,
) -> list[Image.Image]:
    """
    Create resized versions of an image while respecting
    a maximum pixel count.
    """
    versions = []

    for scale in (*down_scales, *up_scales):
        width, height = image.size

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        if new_width * new_height > max_pixels:
            continue

        versions.append(
            resize_pil(
                image,
                scale,
            )
        )

    return versions


def snapshot_rng_state():
    """
    Save Python, NumPy and PyTorch RNG states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state):
    """
    Restore Python, NumPy and PyTorch RNG states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def apply_same_transform(transform, images):
    """
    Apply the same stochastic transform to multiple images.

    Useful when image and mask must undergo identical
    geometric transformations.
    """
    state = snapshot_rng_state()

    transformed = []

    for image in images:
        restore_rng_state(state)
        transformed.append(transform(image))

    return transformed