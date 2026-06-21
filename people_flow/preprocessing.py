from __future__ import annotations

import cv2
import numpy as np

from .config import MIN_BLOB, PROC_WIDTH


def resize_for_processing(frame: np.ndarray, target_width: int = PROC_WIDTH) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame.copy()
    scale = target_width / frame.shape[1]
    target_height = int(round(frame.shape[0] * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def enhance(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def clean_mask(mask: np.ndarray, min_area: int = MIN_BLOB) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    output = np.zeros_like(mask)
    for component_index in range(1, component_count):
        if stats[component_index, cv2.CC_STAT_AREA] >= min_area:
            output[labels == component_index] = 255
    return output
