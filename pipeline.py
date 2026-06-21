from __future__ import annotations

from people_flow.config import (
    IGNORED_ZONE_IOU,
    IGNORED_ZONE_TTL,
    MAX_MATCH_DISTANCE,
    MAX_TRACK_MISSING,
    MAX_VIDEO_DURATION_SEC,
    MIN_BLOB,
    MIN_COMPONENT_AREA,
    MIN_TRACK_HITS,
    PREVIEW_EVERY,
    PROC_WIDTH,
    STATIC_FLICKER_MAX_DISPLACEMENT,
    STATIC_FLICKER_MIN_AREA_CHANGE,
    STATIC_FLICKER_MIN_HITS,
    TARGET_ANALYSIS_FPS,
    VIDEO_SUFFIX,
    VIEW_INFERENCE_MIN_TRACKS,
)
from people_flow.detection import (
    body_likeness_score,
    choose_detection_source,
    detect_candidates,
    estimate_split_count,
    head_likeness_score,
    merge_detections,
)
from people_flow.flow import (
    classify_track_flow,
    crossing_labels,
    decide,
    determine_counting_line,
    infer_view_mode,
    update_crossings,
)
from people_flow.geometry import bbox_iou, bbox_iou_tuple, bbox_mask_coverage
from people_flow.models import Detection, Track
from people_flow.overlay import draw_overlay
from people_flow.preprocessing import clean_mask, enhance, resize_for_processing
from people_flow.segmentation import BackgroundSegmenter
from people_flow.tracking import CentroidTracker
from people_flow.video import analyze_video, analyze_video_stream
from people_flow.zones import filter_ignored_zones, update_ignored_zones

__all__ = [
    "BackgroundSegmenter",
    "CentroidTracker",
    "Detection",
    "IGNORED_ZONE_IOU",
    "IGNORED_ZONE_TTL",
    "MAX_MATCH_DISTANCE",
    "MAX_TRACK_MISSING",
    "MAX_VIDEO_DURATION_SEC",
    "MIN_BLOB",
    "MIN_COMPONENT_AREA",
    "MIN_TRACK_HITS",
    "PREVIEW_EVERY",
    "PROC_WIDTH",
    "STATIC_FLICKER_MAX_DISPLACEMENT",
    "STATIC_FLICKER_MIN_AREA_CHANGE",
    "STATIC_FLICKER_MIN_HITS",
    "TARGET_ANALYSIS_FPS",
    "Track",
    "VIDEO_SUFFIX",
    "VIEW_INFERENCE_MIN_TRACKS",
    "analyze_video",
    "analyze_video_stream",
    "bbox_iou",
    "bbox_iou_tuple",
    "bbox_mask_coverage",
    "body_likeness_score",
    "choose_detection_source",
    "classify_track_flow",
    "clean_mask",
    "crossing_labels",
    "decide",
    "detect_candidates",
    "determine_counting_line",
    "draw_overlay",
    "enhance",
    "estimate_split_count",
    "filter_ignored_zones",
    "head_likeness_score",
    "infer_view_mode",
    "merge_detections",
    "resize_for_processing",
    "update_crossings",
    "update_ignored_zones",
]
