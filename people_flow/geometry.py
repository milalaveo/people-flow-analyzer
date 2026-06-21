from __future__ import annotations

import cv2
import numpy as np


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    intersection = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def bbox_iou_tuple(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    return bbox_iou(np.array(box1, dtype=np.float32), np.array(box2, dtype=np.float32))


def bbox_mask_coverage(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, width, height = bbox
    roi = mask[y : y + height, x : x + width]
    if not roi.size:
        return 0.0
    return round(float(cv2.countNonZero(roi) / roi.size), 4)
