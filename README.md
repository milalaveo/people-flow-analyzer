---
title: People Flow CV
emoji: 🚶
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.18.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# People Flow Tracking

This Hugging Face Spaces app is a production-ready Gradio conversion of the
`people_flow_cv.ipynb` Colab notebook. It uses a classical CPU-only OpenCV
pipeline, tuned for short pedestrian videos on Hugging Face CPU Spaces.

## Pipeline

1. **Enhance**: CLAHE on the LAB lightness channel plus a light Gaussian blur.
2. **Segment**: MOG2 background subtraction on video frames.
3. **Clean**: morphological opening, closing, dilation, and connected-component filtering.
4. **Detect**: OpenCV HOG + mask-driven pedestrian proposals.
5. **Track**: centroid tracking with per-track direction smoothing.
6. **Decide**: produce annotated video, counts, and dominant pedestrian flow.

## Project Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── pipeline.py
├── utils.py
└── assets/
```

## CPU/GPU Compatibility

GPU is not required. The notebook pipeline uses OpenCV image processing and
OpenCV's built-in HOG/SVM detector, both of which run on CPU. A CPU-only Hugging
Face Space is sufficient.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/spaces>.
2. Choose **Gradio** as the SDK and **CPU basic** as the hardware.
3. Upload or commit these files to the Space repository:
   `app.py`, `pipeline.py`, `utils.py`, `requirements.txt`, `README.md`, and `assets/`.
4. Wait for the Space build to finish.
5. Open the Space URL, upload a short video and press **Analyze video**.

Example videos can be added to `assets/videos/` as `.mp4`, `.webm`, `.ogv`,
`.mov`, `.avi`, or `.mkv` files. They will automatically appear in the Gradio
examples panel.
