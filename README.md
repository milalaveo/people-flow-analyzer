---
title: People Flow CV
emoji: "🚶"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.18.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# People Flow CV

People Flow CV is a Gradio app for analyzing pedestrian movement in short
surveillance-style videos. It uses a classical CPU-only OpenCV pipeline to
segment motion, track moving people-like regions, infer the dominant flow
direction, and produce an annotated output video.

The app is designed for quick demos, experiments, and Hugging Face CPU Spaces.
It does not require a GPU or a deep learning model.

## What the app produces

- Annotated video with tracked objects, IDs, movement labels, and a counting line.
- Live processing preview while the final video is being generated.
- Frame-stage previews: original frame, enhanced frame, segmentation mask,
  cleaned mask, and tracking overlay.
- JSON metrics with tracked counts, flow direction, inferred camera view,
  crossing counts, FPS information, and processing settings.

## Pipeline overview

1. **Frame sampling**
   The video is read with OpenCV and sampled to a target analysis FPS. This keeps
   processing lightweight on CPU.

2. **Resize**
   Frames are resized to a fixed processing width for predictable runtime.

3. **Enhancement**
   Contrast is improved with CLAHE on the LAB lightness channel, followed by a
   light Gaussian blur.

4. **Motion segmentation**
   Moving regions are extracted with MOG2 background subtraction.

5. **Mask cleanup**
   Morphological opening, closing, dilation, and connected-component filtering
   remove small noise and stabilize motion blobs.

6. **Candidate detection**
   The app builds pedestrian candidates from cleaned motion components using
   shape, area, mask coverage, and head/body likeness heuristics.

7. **Centroid tracking**
   Detections are linked across frames with a centroid tracker that accounts for
   predicted position, area changes, source consistency, and temporary misses.

8. **Flow decision**
   Confirmed tracks are used to infer the view mode, classify movement direction,
   count line crossings, and select the dominant flow.

9. **Overlay rendering**
   The annotated output video is rendered with bounding boxes, track IDs,
   direction labels, a counting line, and summary metrics.

## Configuration

Main pipeline settings live in `people_flow/config.py`.

```python
PROC_WIDTH = 640
MAX_VIDEO_DURATION_SEC = 15
TARGET_ANALYSIS_FPS = 12.0
MIN_TRACK_HITS = 4
MAX_TRACK_MISSING = 16
MAX_MATCH_DISTANCE = 95.0
```

The defaults are tuned for short pedestrian clips and CPU execution:

- `PROC_WIDTH` keeps frame processing fast and consistent.
- `MAX_VIDEO_DURATION_SEC` prevents long uploads from blocking the app.
- `TARGET_ANALYSIS_FPS` reduces workload while preserving motion continuity.
- Tracking thresholds control when a moving candidate becomes a confirmed track
  and how long it can disappear before being dropped.

## Project structure

```text
.
├── app.py                  # Gradio UI and streaming updates
├── app.css                 # Retro dark UI styling
├── pipeline.py             # Backward-compatible public API wrapper
├── people_flow/
│   ├── config.py           # Pipeline constants
│   ├── models.py           # Detection and Track dataclasses
│   ├── preprocessing.py    # Resize, enhancement, mask cleanup
│   ├── segmentation.py     # MOG2 background subtraction
│   ├── detection.py        # Motion-driven candidate detection
│   ├── tracking.py         # Centroid tracker
│   ├── flow.py             # View mode, flow labels, crossing counts
│   ├── overlay.py          # Annotated frame rendering
│   ├── zones.py            # Static flicker filtering
│   └── video.py            # Video IO and streaming analysis
├── utils.py                # Asset discovery and image helpers
├── assets/                 # Example images/videos
├── requirements.txt
└── README.md
```

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL and upload a short video, or use one of the
example videos from the UI.

## Deploy to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/spaces>.
2. Select **Gradio** as the SDK and CPU hardware.
3. Upload or commit the project files, including `people_flow/`, `assets/`,
   `app.py`, `app.css`, `pipeline.py`, `utils.py`, `requirements.txt`, and
   `README.md`.
4. Wait for the Space build to finish.
5. Open the Space and run the analyzer from the browser.

## Notes and limits

- Best suited for short videos with a mostly static camera.
- Works on motion cues, so heavy camera shake, fast lighting changes, or dense
  occlusion can reduce tracking quality.
- This is not a YOLO-style person detector. It is a classical OpenCV pipeline
  based on motion segmentation and tracking heuristics.

## Advantages

- Runs on CPU with a small dependency set.
- No model weights or GPU setup required.
- Fast enough for short demo clips.
- Transparent pipeline: every major processing stage is visible in the UI.
- Modular code structure for easier tuning and extension.
