from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".ogv", ".mov", ".avi", ".mkv"}


def ensure_bgr(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        if array.ndim == 3 and array.shape[2] == 3:
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    if isinstance(image, (str, Path)):
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Could not read image: {image}")
        return frame

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def discover_examples(directory: str | Path = "assets/images") -> list[str]:
    asset_dir = Path(directory)
    if not asset_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(asset_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def discover_video_examples(directory: str | Path = "assets/videos") -> list[str]:
    asset_dir = Path(directory)
    if not asset_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(asset_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
