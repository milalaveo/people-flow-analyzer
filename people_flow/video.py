from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
from typing import Any, Iterator

import cv2

from .config import MAX_VIDEO_DURATION_SEC, PREVIEW_EVERY, TARGET_ANALYSIS_FPS, VIDEO_SUFFIX
from .detection import detect_candidates
from .flow import decide, update_crossings
from .overlay import draw_overlay
from .preprocessing import clean_mask, enhance, resize_for_processing
from .segmentation import BackgroundSegmenter
from .tracking import CentroidTracker
from .zones import filter_ignored_zones, update_ignored_zones


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
