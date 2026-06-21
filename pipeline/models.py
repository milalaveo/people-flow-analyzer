from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math

import numpy as np

from .config import (
    MIN_TRACK_HITS,
    STATIC_FLICKER_MAX_DISPLACEMENT,
    STATIC_FLICKER_MIN_AREA_CHANGE,
    STATIC_FLICKER_MIN_HITS,
)


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
