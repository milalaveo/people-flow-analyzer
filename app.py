from __future__ import annotations

import asyncio
import json
import sys
import gradio as gr

from pipeline import MAX_VIDEO_DURATION_SEC, analyze_video_stream
from utils import discover_video_examples


DEFAULT_UNRAISABLE_HOOK = sys.unraisablehook


def suppress_asyncio_cleanup_noise(unraisable) -> None:
    exc = unraisable.exc_value
    obj = unraisable.object
    is_asyncio_del = obj is asyncio.BaseEventLoop.__del__
    is_invalid_fd = isinstance(exc, ValueError) and "Invalid file descriptor" in str(exc)
    if is_asyncio_del and is_invalid_fd:
        return
    DEFAULT_UNRAISABLE_HOOK(unraisable)


sys.unraisablehook = suppress_asyncio_cleanup_noise


def run_video_analysis(video):
    last_output = None
    for update in analyze_video_stream(video):
        stages = update["stages"]
        metrics = update["metrics"]
        if update["is_final"]:
            status = render_progress(100, "Analysis complete. Annotated video is ready.")
            video_output = gr.update(value=update["annotated_video"], visible=True)
            preview_output = gr.update(value=None, visible=False)
        else:
            frames_total = max(int(metrics["frames_total"]), 1)
            current_frame = min(int(metrics["current_frame"]), frames_total)
            progress = int(round((current_frame / frames_total) * 100))
            label = (
                f"Processing frame {metrics['current_frame']} of {metrics['frames_total']} "
                f"at {metrics['analysis_fps']} FPS..."
            )
            status = render_progress(progress, label)
            video_output = gr.update(visible=False)
            preview_output = gr.update(value=stages["detection_overlay"], visible=True)
        last_output = (
            video_output,
            status,
            preview_output,
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


def render_progress(percent: int, label: str) -> str:
    percent = max(0, min(percent, 100))
    return (
        '<div class="pf-progress" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        '<div class="pf-progress-meta">'
        f'<span>{label}</span><strong>{percent}%</strong>'
        '</div>'
        '<div class="pf-progress-track">'
        f'<div class="pf-progress-fill" style="width: {percent}%"></div>'
        '</div>'
        '</div>'
    )


video_examples = discover_video_examples()

theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#080a12",
    body_text_color="#f1e6b8",
    block_background_fill="#10131f",
    block_border_color="#2ad6b7",
    block_label_background_fill="#05070d",
    block_label_text_color="#f1e6b8",
    button_primary_background_fill="#f2b84b",
    button_primary_text_color="#080a12",
)

with gr.Blocks(title="People Flow CV", analytics_enabled=False) as demo:
    gr.Markdown(
        (
            "# Automated People Flow Tracking\n"
            f"Upload a video up to {MAX_VIDEO_DURATION_SEC}s and generate tracked flow overlays."
        ),
        elem_classes="pf-title",
    )

    with gr.Row(elem_classes="pf-workspace"):
        with gr.Column(scale=1, elem_classes="pf-panel"):
            input_video = gr.Video(label="Upload video")
            analyze_button = gr.Button("Analyze video", variant="primary")
            if video_examples:
                gr.Examples(
                    examples=video_examples,
                    inputs=input_video,
                    label="Example videos",
                )
        with gr.Column(scale=1, elem_classes="pf-output-panel"):
            output_video = gr.Video(label="Annotated output video", visible=False)
            processing_status = gr.HTML(render_progress(0, "Idle. Upload a video and start analysis."), elem_classes="pf-status")
            annotated_preview = gr.Image(label="Annotated preview while processing", visible=False)
            metrics = gr.Code(label="Video metrics", language="json")

    with gr.Tabs():
        with gr.Tab("Live Preview"):
            with gr.Row():
                original = gr.Image(label="1. Current frame", elem_classes="pf-stage")
                enhanced = gr.Image(label="2. Enhanced frame", elem_classes="pf-stage")
            with gr.Row():
                segmentation_mask = gr.Image(label="3. Segmentation mask", elem_classes="pf-stage")
                cleaned_mask = gr.Image(label="4. Cleaned mask", elem_classes="pf-stage")
            detection_overlay = gr.Image(label="5. Tracking overlay", elem_classes="pf-stage")
        with gr.Tab("Detection Details"):
            detections = gr.Code(label="Current detections and centroids", language="json")

    analyze_button.click(
        fn=run_video_analysis,
        inputs=input_video,
        outputs=[
            output_video,
            processing_status,
            annotated_preview,
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
    demo.launch(theme=theme, css_paths="app.css", ssr_mode=False)
