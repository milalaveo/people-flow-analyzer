from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile
from typing import Any, Iterator

import cv2
import numpy as np


PROC_WIDTH = 640
MIN_BLOB = 500
PREVIEW_EVERY = 12
MAX_VIDEO_DURATION_SEC = 15
TARGET_ANALYSIS_FPS = 12.0
MIN_TRACK_HITS = 4
MAX_TRACK_MISSING = 16
MAX_MATCH_DISTANCE = 95.0
MIN_COMPONENT_AREA = 220
STATIC_FLICKER_MAX_DISPLACEMENT = 18.0
STATIC_FLICKER_MIN_HITS = 5
STATIC_FLICKER_MIN_AREA_CHANGE = 0.35
IGNORED_ZONE_TTL = 90
IGNORED_ZONE_IOU = 0.35
VIDEO_SUFFIX = ".webm"
VIEW_INFERENCE_MIN_TRACKS = 2


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: float
    mask_coverage: float
    head_score: float
    body_score: float
    source: str

    @property
    def person_score(self) -> float:
        return max(self.head_score, self.body_score)


@dataclass
class Track:
    track_id: int
    centroids: list[tuple[int, int]] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    coverages: list[float] = field(default_factory=list)
    head_scores: list[float] = field(default_factory=list)
    body_scores: list[float] = field(default_factory=list)
    sources: Counter[str] = field(default_factory=Counter)
    hits: int = 0
    hit_streak: int = 0
    misses: int = 0
    counted: bool = False
    counted_label: str | None = None

    def update(self, detection: Detection) -> None:
        self.centroids.append(detection.centroid)
        self.bboxes.append(detection.bbox)
        self.areas.append(detection.area)
        self.coverages.append(detection.mask_coverage)
        self.head_scores.append(detection.head_score)
        self.body_scores.append(detection.body_score)
        self.sources[detection.source] += 1
        self.hits += 1
        self.hit_streak += 1
        self.misses = 0

    @property
    def latest_bbox(self) -> tuple[int, int, int, int]:
        return self.bboxes[-1]

    @property
    def latest_centroid(self) -> tuple[int, int]:
        return self.centroids[-1]

    @property
    def is_confirmed(self) -> bool:
        return (self.hit_streak >= MIN_TRACK_HITS or self.hits >= MIN_TRACK_HITS + 2) and not self.looks_like_static_flicker

    @property
    def preferred_source(self) -> str:
        if not self.sources:
            return "unknown"
        return self.sources.most_common(1)[0][0]

    def predicted_centroid(self) -> tuple[float, float]:
        if len(self.centroids) < 2:
            cx, cy = self.latest_centroid
            return float(cx), float(cy)
        (x1, y1), (x2, y2) = self.centroids[-2], self.centroids[-1]
        return float(x2 + (x2 - x1)), float(y2 + (y2 - y1))

    def recent_area(self) -> float:
        return float(np.median(self.areas[-3:])) if self.areas else 0.0

    def median_aspect(self) -> float:
        if not self.bboxes:
            return 2.0
        values = [height / max(width, 1) for _, _, width, height in self.bboxes[-5:]]
        return float(np.median(values))

    def median_coverage(self) -> float:
        return float(np.median(self.coverages[-5:])) if self.coverages else 0.0

    def mean_area_change(self) -> float:
        if len(self.areas) < 2:
            return 0.0
        values: list[float] = []
        for first, second in zip(self.areas[:-1], self.areas[1:], strict=False):
            if first > 0 and second > 0:
                values.append(abs(math.log((second + 1e-6) / (first + 1e-6))))
        return float(np.mean(values)) if values else 0.0

    def motion_vector(self) -> tuple[float, float]:
        if len(self.centroids) < 2:
            return 0.0, 0.0
        start = self.centroids[max(0, len(self.centroids) - 6)]
        end = self.centroids[-1]
        return float(end[0] - start[0]), float(end[1] - start[1])

    @property
    def looks_like_static_flicker(self) -> bool:
        if self.hits < STATIC_FLICKER_MIN_HITS or len(self.centroids) < 2:
            return False
        start_x, start_y = self.centroids[0]
        end_x, end_y = self.centroids[-1]
        displacement = math.hypot(end_x - start_x, end_y - start_y)
        if displacement > STATIC_FLICKER_MAX_DISPLACEMENT:
            return False
        min_area = max(min(self.areas), 1.0)
        relative_area_change = (max(self.areas) - min_area) / min_area
        return (
            self.preferred_source == "head"
            and relative_area_change >= STATIC_FLICKER_MIN_AREA_CHANGE
            and self.median_coverage() >= 0.55
        )

    def ignored_zone_bbox(self) -> tuple[int, int, int, int]:
        widths = [width for _, _, width, _ in self.bboxes[-5:]]
        heights = [height for _, _, _, height in self.bboxes[-5:]]
        xs = [x for x, _, _, _ in self.bboxes[-5:]]
        ys = [y for _, y, _, _ in self.bboxes[-5:]]
        return (
            int(np.median(xs)),
            int(np.median(ys)),
            int(np.median(widths)),
            int(np.median(heights)),
        )


class BackgroundSegmenter:
    def __init__(self, history: int = 500, var_threshold: int = 50) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask


class CentroidTracker:
    def __init__(self) -> None:
        self.next_id = 0
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        if not detections:
            for track in list(self.tracks.values()):
                track.misses += 1
                track.hit_streak = 0
            self._drop_stale_tracks()
            return self.tracks

        if not self.tracks:
            for detection in detections:
                self._register(detection)
            return self.tracks

        track_ids = list(self.tracks)
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id in track_ids:
                cost = self._match_cost(self.tracks[track_id], detection)
                if cost < float("inf"):
                    candidates.append((cost, track_id, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates, key=lambda item: item[0]):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            self.tracks[track_id].update(detections[detection_index])
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in used_detections:
                self._register(detection)

        for track_id, track in self.tracks.items():
            if track_id not in used_tracks:
                track.misses += 1
                track.hit_streak = 0

        self._drop_stale_tracks()
        return self.tracks

    def _register(self, detection: Detection) -> None:
        track = Track(self.next_id)
        track.update(detection)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _drop_stale_tracks(self) -> None:
        for track_id in list(self.tracks):
            if self.tracks[track_id].misses > MAX_TRACK_MISSING:
                self.tracks.pop(track_id, None)

    def _match_cost(self, track: Track, detection: Detection) -> float:
        predicted_x, predicted_y = track.predicted_centroid()
        cx, cy = detection.centroid
        distance = math.hypot(cx - predicted_x, cy - predicted_y)
        if distance > MAX_MATCH_DISTANCE * (1.35 if track.is_confirmed else 1.0):
            return float("inf")

        area_penalty = 0.0
        reference_area = track.recent_area()
        if reference_area > 0 and detection.area > 0:
            area_penalty = abs(math.log((detection.area + 1e-6) / (reference_area + 1e-6))) * 18.0
            if area_penalty > 28.0:
                return float("inf")

        source_penalty = 0.0
        if track.preferred_source != "unknown" and detection.source != track.preferred_source:
            source_penalty = 6.0

        return distance + area_penalty + source_penalty + track.misses * 4.0


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


def infer_view_mode(tracks: dict[int, Track]) -> tuple[str, dict[str, float | int | str]]:
    stable_tracks = [track for track in tracks.values() if len(track.centroids) >= 2 and not track.looks_like_static_flicker]
    if len(stable_tracks) < VIEW_INFERENCE_MIN_TRACKS:
        return "mixed", {
            "stable_track_count": len(stable_tracks),
            "median_aspect_ratio": 0.0,
            "median_mask_coverage": 0.0,
            "mean_log_area_change": 0.0,
            "horizontal_motion_score": 0.0,
            "vertical_motion_score": 0.0,
            "decision_basis": "insufficient_tracks",
        }

    aspect_values = [track.median_aspect() for track in stable_tracks]
    coverage_values = [track.median_coverage() for track in stable_tracks]
    area_changes = [track.mean_area_change() for track in stable_tracks]
    horizontal_motion = sum(abs(track.motion_vector()[0]) for track in stable_tracks)
    vertical_motion = sum(abs(track.motion_vector()[1]) for track in stable_tracks)
    head_fraction = sum(track.preferred_source == "head" for track in stable_tracks) / len(stable_tracks)
    depth_alignment_values: list[float] = []
    for track in stable_tracks:
        if len(track.areas) < 2 or len(track.centroids) < 2:
            continue
        area_delta = track.areas[-1] - track.areas[0]
        _, dy = track.motion_vector()
        if abs(area_delta) > max(track.areas[0] * 0.08, 80.0) and abs(dy) > 4:
            depth_alignment_values.append(math.copysign(1.0, area_delta) * math.copysign(1.0, dy))
    depth_alignment = float(np.mean(depth_alignment_values)) if depth_alignment_values else 0.0

    median_aspect = float(np.median(aspect_values))
    median_coverage = float(np.median(coverage_values))
    mean_area_change = float(np.mean(area_changes))
    debug = {
        "stable_track_count": len(stable_tracks),
        "median_aspect_ratio": round(median_aspect, 4),
        "median_mask_coverage": round(median_coverage, 4),
        "mean_log_area_change": round(mean_area_change, 4),
        "horizontal_motion_score": round(horizontal_motion, 4),
        "vertical_motion_score": round(vertical_motion, 4),
        "head_track_fraction": round(head_fraction, 4),
        "depth_alignment": round(depth_alignment, 4),
    }

    if head_fraction > 0.55 and median_aspect < 1.8:
        debug["decision_basis"] = "head_dominant_compact"
        return "overhead", debug
    if median_aspect < 1.45 and median_coverage > 0.6 and vertical_motion > horizontal_motion * 0.5:
        debug["decision_basis"] = "compact_dense_top_view"
        return "overhead", debug
    if median_aspect < 1.35 and median_coverage > 0.54 and mean_area_change > 0.38:
        debug["decision_basis"] = "compact_dense_front_view"
        return "front", debug
    if depth_alignment > 0.35 and mean_area_change > 0.3 and median_coverage > 0.5:
        debug["decision_basis"] = "depth_alignment_front"
        return "front", debug
    if mean_area_change > 0.55 and median_coverage > 0.45:
        debug["decision_basis"] = "depth_change_dense"
        return "front", debug
    if horizontal_motion > vertical_motion * 1.25:
        debug["decision_basis"] = "horizontal_motion_dominant"
        return "side", debug
    if vertical_motion > horizontal_motion * 1.6 and head_fraction > 0.4:
        debug["decision_basis"] = "vertical_motion_head_dominant"
        return "overhead", debug
    if mean_area_change > 0.35:
        debug["decision_basis"] = "depth_change_secondary"
        return "front", debug
    debug["decision_basis"] = "ambiguous_mixed"
    return "mixed", debug


def determine_counting_line(
    frame_shape: tuple[int, int],
    view_mode: str,
    tracks: dict[int, Track],
) -> dict[str, Any]:
    height, width = frame_shape
    confirmed = [track for track in tracks.values() if track.is_confirmed]
    horizontal_motion = sum(abs(track.motion_vector()[0]) for track in confirmed)
    vertical_motion = sum(abs(track.motion_vector()[1]) for track in confirmed)

    if view_mode == "side":
        return {"orientation": "vertical", "value": int(width * 0.5)}
    if view_mode == "front":
        return {"orientation": "horizontal", "value": int(height * 0.55)}
    if horizontal_motion >= vertical_motion:
        return {"orientation": "vertical", "value": int(width * 0.5)}
    return {"orientation": "horizontal", "value": int(height * 0.5)}


def classify_track_flow(track: Track, view_mode: str) -> str:
    if len(track.centroids) < 2:
        return "unknown"
    dx, dy = track.motion_vector()
    if abs(dx) < 5 and abs(dy) < 5:
        return "stationary"

    if view_mode == "side":
        return "left_to_right" if dx > 0 else "right_to_left"

    if view_mode == "front":
        if len(track.areas) >= 2:
            area_delta = track.areas[-1] - track.areas[0]
            if abs(area_delta) > max(track.areas[0] * 0.1, 120.0):
                return "toward_camera" if area_delta > 0 else "away_from_camera"
        return "toward_camera" if dy > 0 else "away_from_camera"

    if view_mode == "overhead":
        if abs(dx) >= abs(dy):
            return "rightward" if dx > 0 else "leftward"
        return "downward" if dy > 0 else "upward"

    if abs(dx) >= abs(dy):
        return "rightward" if dx > 0 else "leftward"
    return "downward" if dy > 0 else "upward"


def decide(tracks: dict[int, Track], frame_shape: tuple[int, int], counts: dict[str, int]) -> dict[str, Any]:
    confirmed_tracks = {track_id: track for track_id, track in tracks.items() if track.is_confirmed}
    view_mode, view_debug = infer_view_mode(confirmed_tracks)
    counting_line = determine_counting_line(frame_shape, view_mode, confirmed_tracks)

    filtered_static_flicker = sum(1 for track in tracks.values() if track.looks_like_static_flicker)
    flow_labels: list[str] = []
    per_track: dict[int, str] = {}
    for track_id, track in confirmed_tracks.items():
        flow_label = classify_track_flow(track, view_mode)
        per_track[track_id] = flow_label
        if flow_label not in {"unknown", "stationary"}:
            flow_labels.append(flow_label)

    dominant_flow = Counter(flow_labels).most_common(1)
    return {
        "count": len(confirmed_tracks),
        "flow": dominant_flow[0][0] if dominant_flow else "no clear flow",
        "view_mode": view_mode,
        "view_debug": view_debug,
        "counting_line": counting_line,
        "crossings": counts.copy(),
        "filtered_static_flicker_tracks": filtered_static_flicker,
        "per_track": per_track,
    }


def update_crossings(
    tracks: dict[int, Track],
    view_mode: str,
    counting_line: dict[str, Any],
    counts: dict[str, int],
) -> None:
    orientation = counting_line["orientation"]
    value = counting_line["value"]
    for track in tracks.values():
        if not track.is_confirmed or track.counted or len(track.centroids) < 2:
            continue

        prev = track.centroids[-2]
        curr = track.centroids[-1]
        prev_side = prev[0] - value if orientation == "vertical" else prev[1] - value
        curr_side = curr[0] - value if orientation == "vertical" else curr[1] - value
        if prev_side == 0 or curr_side == 0 or prev_side * curr_side > 0:
            continue

        flow_label = classify_track_flow(track, view_mode)
        primary, secondary = crossing_labels(view_mode, orientation)
        if flow_label in primary:
            counts["primary"] += 1
            track.counted_label = primary[0]
        else:
            counts["secondary"] += 1
            track.counted_label = secondary[0]
        track.counted = True


def crossing_labels(view_mode: str, orientation: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if view_mode == "front":
        return ("toward_camera",), ("away_from_camera",)
    if orientation == "vertical":
        return ("left_to_right", "rightward"), ("right_to_left", "leftward")
    return ("downward",), ("upward",)


def draw_overlay(
    frame: np.ndarray,
    tracks: dict[int, Track],
    decision: dict[str, Any],
) -> np.ndarray:
    output = frame.copy()

    line = decision["counting_line"]
    if line["orientation"] == "vertical":
        cv2.line(output, (line["value"], 0), (line["value"], output.shape[0] - 1), (0, 180, 255), 2)
    else:
        cv2.line(output, (0, line["value"]), (output.shape[1] - 1, line["value"]), (0, 180, 255), 2)

    for track_id, track in tracks.items():
        if not track.is_confirmed:
            continue
        x, y, width, height = track.latest_bbox
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 200, 0), 2)
        cx, cy = track.latest_centroid
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        label = decision["per_track"].get(track_id, "unknown")
        cv2.putText(output, f"ID {track_id} {label}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    banner = (
        f"active: {decision['count']} | flow: {decision['flow']} | "
        f"in: {decision['crossings']['primary']} | out: {decision['crossings']['secondary']} | "
        f"view: {decision['view_mode']}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(output, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def filter_ignored_zones(detections: list[Detection], ignored_zones: list[dict[str, Any]]) -> list[Detection]:
    if not ignored_zones:
        return detections
    filtered: list[Detection] = []
    for detection in detections:
        if any(bbox_iou_tuple(detection.bbox, zone["bbox"]) >= IGNORED_ZONE_IOU for zone in ignored_zones):
            continue
        filtered.append(detection)
    return filtered


def update_ignored_zones(ignored_zones: list[dict[str, Any]], tracks: dict[int, Track]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for zone in ignored_zones:
        if zone["ttl"] > 1:
            updated.append({"bbox": zone["bbox"], "ttl": zone["ttl"] - 1})

    for track in tracks.values():
        if not track.looks_like_static_flicker:
            continue
        bbox = track.ignored_zone_bbox()
        merged = False
        for zone in updated:
            if bbox_iou_tuple(zone["bbox"], bbox) >= 0.4:
                zone["bbox"] = bbox
                zone["ttl"] = max(zone["ttl"], IGNORED_ZONE_TTL)
                merged = True
                break
        if not merged:
            updated.append({"bbox": bbox, "ttl": IGNORED_ZONE_TTL})
    return updated


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


def analyze_video(video_path: str | Path) -> dict[str, Any]:
    final_update = None
    for update in analyze_video_stream(video_path):
        final_update = update
    if final_update is None:
        raise RuntimeError("Video analysis produced no output.")
    return final_update


def analyze_video_stream(video_path: str | Path) -> Iterator[dict[str, Any]]:
    from utils import to_rgb

    if not video_path:
        raise ValueError("Upload a video before running analysis.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if fps else 0.0
    if duration_sec > MAX_VIDEO_DURATION_SEC:
        cap.release()
        raise ValueError(
            f"Video is too long ({duration_sec:.1f}s). Please upload a clip up to {MAX_VIDEO_DURATION_SEC}s."
        )

    segmenter = BackgroundSegmenter()
    tracker = CentroidTracker()
    votes: Counter[str] = Counter()
    counts = {"primary": 0, "secondary": 0}
    ignored_zones: list[dict[str, Any]] = []
    peak_people_count = 0
    current_index = 0
    analyzed_index = 0
    frame_step = max(1, int(round(fps / TARGET_ANALYSIS_FPS))) if fps else 1
    effective_fps = fps / frame_step if fps else TARGET_ANALYSIS_FPS
    sample_target = max((frame_count // frame_step) // 2, 0) if frame_count else 0
    sample_payload: dict[str, Any] | None = None
    view_mode_hint = "mixed"
    last_decision: dict[str, Any] | None = None

    output_path = Path(tempfile.gettempdir()) / f"people-flow-{next(tempfile._get_candidate_names())}{VIDEO_SUFFIX}"
    writer: cv2.VideoWriter | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if current_index % frame_step != 0:
                current_index += 1
                continue

            processed = resize_for_processing(frame)
            enhanced = enhance(processed)
            raw_mask = segmenter.apply(enhanced)
            cleaned = clean_mask(raw_mask)
            detections = detect_candidates(cleaned, view_mode_hint)
            detections = filter_ignored_zones(detections, ignored_zones)
            tracks = tracker.update(detections)
            ignored_zones = update_ignored_zones(ignored_zones, tracks)

            provisional = decide(tracks, processed.shape[:2], counts)
            view_mode_hint = provisional["view_mode"] if provisional["view_mode"] != "mixed" else view_mode_hint
            update_crossings(tracks, view_mode_hint, provisional["counting_line"], counts)
            decision = decide(tracks, processed.shape[:2], counts)
            last_decision = decision
            overlay = draw_overlay(enhanced, tracks, decision)

            if writer is None:
                height, width = overlay.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"VP80"),
                    effective_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create output video writer.")

            writer.write(overlay)
            peak_people_count = max(peak_people_count, decision["count"])
            if decision["flow"] not in {"unknown", "stationary", "no clear flow"}:
                votes[decision["flow"]] += 1

            detection_details = [
                {
                    "id": index,
                    "bbox": detection.bbox,
                    "centroid": detection.centroid,
                    "mask_coverage": detection.mask_coverage,
                    "head_score": round(detection.head_score, 3),
                    "body_score": round(detection.body_score, 3),
                    "source": detection.source,
                }
                for index, detection in enumerate(detections)
            ]

            if sample_payload is None and analyzed_index >= sample_target:
                sample_payload = {
                    "stages": {
                        "original": to_rgb(processed),
                        "enhanced": to_rgb(enhanced),
                        "segmentation_mask": raw_mask,
                        "cleaned_mask": cleaned,
                        "detection_overlay": to_rgb(overlay),
                        "final_output": to_rgb(overlay),
                    },
                    "detections": detection_details,
                    "decision": decision,
                    "sample_frame_index": analyzed_index,
                }

            if analyzed_index % PREVIEW_EVERY == 0:
                yield {
                    "stages": {
                        "original": to_rgb(processed),
                        "enhanced": to_rgb(enhanced),
                        "segmentation_mask": raw_mask,
                        "cleaned_mask": cleaned,
                        "detection_overlay": to_rgb(overlay),
                        "final_output": to_rgb(overlay),
                    },
                    "metrics": {
                        "status": "processing",
                        "current_frame": analyzed_index,
                        "frames_total": max(frame_count // frame_step, 1),
                        "source_fps": round(float(fps), 2),
                        "analysis_fps": round(float(effective_fps), 2),
                        "frame_step": frame_step,
                        "active_tracked_people": decision["count"],
                        "peak_tracked_people": peak_people_count,
                        "inferred_view_mode": decision["view_mode"],
                        "view_debug": decision["view_debug"],
                        "crossings": decision["crossings"],
                        "ignored_static_zones": len(ignored_zones),
                        "filtered_static_flicker_tracks": decision["filtered_static_flicker_tracks"],
                        "dominant_flow_so_far": decision["flow"],
                    },
                    "detections": detection_details,
                    "decision": decision,
                    "annotated_video": None,
                    "is_final": False,
                }

            analyzed_index += 1
            current_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError("Video contains no readable frames.")
    if sample_payload is None:
        raise RuntimeError("Could not extract a sample frame from the video.")
    if last_decision is None:
        raise RuntimeError("Video analysis did not produce a final decision.")

    metrics = {
        "peak_tracked_people": peak_people_count,
        "dominant_flow": last_decision["flow"],
        "tracked_people_total": sum(track.is_confirmed for track in tracker.tracks.values()),
        "frames_processed": analyzed_index,
        "sample_frame_index": sample_payload["sample_frame_index"],
        "inferred_view_mode": last_decision["view_mode"],
        "view_debug": last_decision["view_debug"],
        "crossings": counts.copy(),
        "ignored_static_zones": len(ignored_zones),
        "filtered_static_flicker_tracks": last_decision["filtered_static_flicker_tracks"],
        "source_fps": round(float(fps), 2),
        "analysis_fps": round(float(effective_fps), 2),
        "frame_step": frame_step,
        "video_duration_sec": round(float(duration_sec), 2),
        "max_video_duration_sec": MAX_VIDEO_DURATION_SEC,
        "cpu_only": True,
        "detector": "Motion-driven head/body candidates",
        "segmentation": "MOG2 background subtraction",
        "annotated_video": str(output_path),
    }

    yield {
        "stages": sample_payload["stages"],
        "metrics": metrics,
        "detections": sample_payload["detections"],
        "decision": sample_payload["decision"],
        "annotated_video": str(output_path),
        "is_final": True,
    }
