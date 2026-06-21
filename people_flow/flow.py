from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np

from .config import VIEW_INFERENCE_MIN_TRACKS
from .models import Track


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
