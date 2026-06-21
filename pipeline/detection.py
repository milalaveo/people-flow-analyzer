from __future__ import annotations

import cv2
import numpy as np

from .config import MIN_COMPONENT_AREA
from .geometry import bbox_iou, bbox_mask_coverage
from .models import Detection


def detect_candidates(mask: np.ndarray, view_mode_hint: str) -> list[Detection]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    detections: list[Detection] = []

    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA or width < 8 or height < 8:
            continue

        split_count = estimate_split_count(width, height, area, view_mode_hint)
        slice_width = max(int(round(width / split_count)), 10)
        for split_index in range(split_count):
            sx = x + split_index * slice_width
            sw = x + width - sx if split_index == split_count - 1 else slice_width
            if sw < 8:
                continue

            bbox = (sx, y, sw, height)
            coverage = bbox_mask_coverage(mask, bbox)
            head_score = head_likeness_score(mask, bbox)
            body_score = body_likeness_score(mask, bbox, coverage)
            source = choose_detection_source(head_score, body_score, view_mode_hint)
            if source == "reject":
                continue

            detections.append(
                Detection(
                    bbox=bbox,
                    centroid=(int(sx + sw / 2), int(y + height / 2)),
                    area=float(sw * height),
                    mask_coverage=coverage,
                    head_score=head_score,
                    body_score=body_score,
                    source=source,
                )
            )

    return merge_detections(detections)


def estimate_split_count(width: int, height: int, area: int, view_mode_hint: str) -> int:
    aspect = width / max(height, 1)
    if view_mode_hint == "overhead":
        if aspect > 1.4 and width > 26:
            return max(1, min(4, int(round(width / max(height * 0.9, 14.0)))))
        if area > 2500 and width > 50:
            return max(1, min(4, int(round(area / 1800.0))))
        return 1

    if aspect > 1.2 and width > 36:
        return max(1, min(3, int(round(width / max(height * 0.72, 22.0)))))
    if area > 11000 and width > 90:
        return max(1, min(3, int(round(area / 6500.0))))
    return 1


def choose_detection_source(head_score: float, body_score: float, view_mode_hint: str) -> str:
    if view_mode_hint == "overhead":
        return "head" if head_score >= 1.4 else "reject"
    if view_mode_hint == "side":
        return "body" if body_score >= 1.55 else ("head" if head_score >= 1.7 else "reject")
    if view_mode_hint == "front":
        return "body" if body_score >= 1.5 else ("head" if head_score >= 1.7 else "reject")
    return "body" if body_score >= 1.55 else ("head" if head_score >= 1.5 else "reject")


def body_likeness_score(mask: np.ndarray, bbox: tuple[int, int, int, int], mask_coverage: float) -> float:
    _, _, width, height = bbox
    aspect_ratio = height / max(width, 1)
    area = width * height
    score = 0.0

    if 14 <= width <= 160:
        score += 0.5
    if 28 <= height <= 260:
        score += 0.6
    if 1.15 <= aspect_ratio <= 4.6:
        score += 0.85
    elif 0.95 <= aspect_ratio <= 5.2:
        score += 0.35
    if 0.14 <= mask_coverage <= 0.86:
        score += 0.55
    elif 0.08 <= mask_coverage <= 0.94:
        score += 0.25
    if 450 <= area <= 18000:
        score += 0.45
    elif 250 <= area <= 24000:
        score += 0.2

    score += 0.4 * min(head_likeness_score(mask, bbox), 1.0)
    return score


def head_likeness_score(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, width, height = bbox
    roi = mask[y : y + height, x : x + width]
    if roi.size == 0 or width < 8 or height < 8:
        return 0.0

    binary = (roi > 0).astype(np.uint8)
    aspect_ratio = height / max(width, 1)
    area = width * height
    score = 0.0

    if 10 <= width <= 70 and 10 <= height <= 80:
        score += 0.35
    if 0.8 <= aspect_ratio <= 2.2:
        score += 0.5
    elif 0.65 <= aspect_ratio <= 2.8:
        score += 0.2
    if 120 <= area <= 3500:
        score += 0.25

    top_end = max(1, int(height * 0.28))
    mid_start = int(height * 0.28)
    mid_end = max(mid_start + 1, int(height * 0.62))
    bottom_start = int(height * 0.62)

    top_width = float(np.median(np.sum(binary[:top_end], axis=1))) if top_end > 0 else 0.0
    mid_width = float(np.median(np.sum(binary[mid_start:mid_end], axis=1))) if mid_end > mid_start else 0.0
    bottom_width = float(np.median(np.sum(binary[bottom_start:], axis=1))) if height > bottom_start else 0.0

    if mid_width > 1.0:
        top_ratio = top_width / mid_width
        bottom_ratio = bottom_width / mid_width if bottom_width > 0 else 0.0
        if top_ratio < 0.82:
            score += 0.45
        if 0.65 <= bottom_ratio <= 1.25:
            score += 0.15

    coverage = bbox_mask_coverage(mask, bbox)
    if 0.35 <= coverage <= 0.95:
        score += 0.2
    return score


def merge_detections(detections: list[Detection], iou_threshold: float = 0.35) -> list[Detection]:
    if not detections:
        return []

    boxes = np.array([detection.bbox for detection in detections], dtype=np.float32)
    scores = np.array([detection.person_score for detection in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    kept: list[Detection] = []

    while order.size:
        current_index = int(order[0])
        current = detections[current_index]
        merged = current
        remaining: list[int] = []
        for candidate in order[1:]:
            candidate_index = int(candidate)
            if bbox_iou(boxes[current_index], boxes[candidate_index]) >= iou_threshold:
                merged = Detection(
                    bbox=merged.bbox,
                    centroid=merged.centroid,
                    area=merged.area,
                    mask_coverage=max(merged.mask_coverage, detections[candidate_index].mask_coverage),
                    head_score=max(merged.head_score, detections[candidate_index].head_score),
                    body_score=max(merged.body_score, detections[candidate_index].body_score),
                    source=merged.source if merged.person_score >= detections[candidate_index].person_score else detections[candidate_index].source,
                )
            else:
                remaining.append(candidate_index)
        kept.append(merged)
        order = np.array(remaining, dtype=np.int32)

    return kept
