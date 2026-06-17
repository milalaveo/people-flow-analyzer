from __future__ import annotations

import json
import gradio as gr

from pipeline import MAX_VIDEO_DURATION_SEC, analyze_video_stream
from utils import discover_video_examples


def run_video_analysis(video):
    last_output = None
    for update in analyze_video_stream(video):
        stages = update["stages"]
        metrics = update["metrics"]
        last_output = (
            update["annotated_video"] if update["is_final"] else gr.update(),
            stages["original"],
            stages["enhanced"],
            stages["segmentation_mask"],
            stages["cleaned_mask"],
            stages["detection_overlay"],
            json.dumps(metrics, indent=2, ensure_ascii=False),
            json.dumps(update["detections"], indent=2, ensure_ascii=False),
        )
        yield last_output

    if last_output is None:
        raise RuntimeError("No video preview was generated.")


video_examples = discover_video_examples()

with gr.Blocks(title="People Flow CV") as demo:
    gr.Markdown("# Automated People Flow Tracking")
    gr.Markdown(
        f"Upload a short surveillance-style video up to {MAX_VIDEO_DURATION_SEC}s. "
        "The system enhances each frame, segments moving regions, cleans the motion mask, "
        "tracks people-like motion over time, and produces an annotated output video with "
        "flow direction and crossing statistics."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload video")
            analyze_button = gr.Button("Analyze video", variant="primary")
            if video_examples:
                gr.Examples(
                    examples=video_examples,
                    inputs=input_video,
                    label="Example videos",
                )
        with gr.Column(scale=1):
            output_video = gr.Video(label="Annotated output video")
            metrics = gr.Code(label="Video metrics", language="json")

    with gr.Tabs():
        with gr.Tab("Live Preview"):
            with gr.Row():
                original = gr.Image(label="1. Current frame")
                enhanced = gr.Image(label="2. Enhanced frame")
            with gr.Row():
                segmentation_mask = gr.Image(label="3. Segmentation mask")
                cleaned_mask = gr.Image(label="4. Cleaned mask")
            detection_overlay = gr.Image(label="5. Tracking overlay")
        with gr.Tab("Detection Details"):
            detections = gr.Code(label="Current detections and centroids", language="json")

    analyze_button.click(
        fn=run_video_analysis,
        inputs=input_video,
        outputs=[
            output_video,
            original,
            enhanced,
            segmentation_mask,
            cleaned_mask,
            detection_overlay,
            metrics,
            detections,
        ],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch()
