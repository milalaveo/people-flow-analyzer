from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .models import Track


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
