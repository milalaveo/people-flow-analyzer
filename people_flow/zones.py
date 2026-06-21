from __future__ import annotations

from typing import Any

from .config import IGNORED_ZONE_IOU, IGNORED_ZONE_TTL
from .geometry import bbox_iou_tuple
from .models import Detection, Track


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
