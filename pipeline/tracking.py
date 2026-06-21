from __future__ import annotations

import math

from .config import MAX_MATCH_DISTANCE, MAX_TRACK_MISSING
from .models import Detection, Track


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
