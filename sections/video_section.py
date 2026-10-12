"""
sections/video_section.py
=========================

Video section: preview a video source, run YOLO detection on video
frames, or run YOLO detection on a still image.

Every run is bounded by a "Maximum frames" setting so the browser is
never locked. The YOLO model is loaded once and cached.
"""

from pathlib import Path
import tempfile

import cv2
import streamlit as st

from ui import menu, number_input, show_image
from utils.file_io import decode_image, encode_image


@st.cache_resource
def load_yolo_model(weights_path):
    """
    Load and cache an Ultralytics YOLO model.

    Caching with ``st.cache_resource`` means the model is only loaded
    the first time; subsequent page reruns reuse it.
    """
    from ultralytics import YOLO
    if not Path(weights_path).is_file():
        raise ValueError(f"No YOLO weights found at: {weights_path}")
    return YOLO(weights_path)


def draw_detection_controls():
    """
    Draw the shared controls for detection tasks.

    Returns
    -------
    dict
        {"weights", "confidence", "class_ids", "show_labels", "show_scores"}
    """
    weights = st.text_input(
        "YOLO weights path",
        "models/yolov8n.pt",
        help="Full path to a local .pt file. Model files are not bundled.",
    )
    confidence = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5,
        help="Detections with score below this value are ignored.",
    )
    class_ids_text = st.text_input(
        "Class IDs (optional)", "",
        help="Comma-separated IDs, e.g. 0,2 for person and car. Empty means all.",
    )
    class_ids = None
    if class_ids_text.strip():
        try:
            class_ids = [int(x.strip()) for x in class_ids_text.split(",")]
            if any(i < 0 for i in class_ids):
                raise ValueError
        except ValueError:
            st.error("Class IDs must be non-negative integers separated by commas.")
            st.stop()

    show_labels = st.checkbox("Show class labels", value=True,
                              help="Draw class names next to each box.")
    show_scores = st.checkbox("Show confidence scores", value=True,
                              help="Draw confidence next to each label.")

    return {
        "weights": weights,
        "confidence": confidence,
        "class_ids": class_ids,
        "show_labels": show_labels,
        "show_scores": show_scores,
    }


def render():
    """Draw the Video section."""
    task = menu(
        "Video tasks",
        {
            "Preview":          "Read frames from a file, camera or stream.",
            "Detect in video":  "Run YOLO detection on video frames.",
            "Detect in image":  "Run YOLO detection on a single image.",
        },
        "video_task",
    )
    if task is None:
        return

    # --- Detect in image ------------------------------------------------
    if task == "Detect in image":
        uploaded = st.file_uploader("Image for detection",
                                    type=["png", "jpg", "jpeg", "bmp"], help='Upload the still image on which to run the chosen YOLO detector.')
        controls = draw_detection_controls()

        if uploaded and st.button("Detect objects", help='Run the configured detector on the uploaded image and show its boxes.'):
            image = decode_image(uploaded.getvalue())[:, :, :3]
            model = load_yolo_model(controls["weights"])
            results = model.predict(
                image,
                conf=controls["confidence"],
                classes=controls["class_ids"],
                verbose=False,
            )
            st.session_state.image_detection_result = results[0]

        if "image_detection_result" in st.session_state:
            result = st.session_state.image_detection_result
            annotated = result.plot(
                labels=controls["show_labels"],
                conf=controls["show_scores"],
            )
            show_image(annotated, "Annotated image")
            st.download_button("Save annotated image",
                               encode_image(annotated, "png"),
                               "detections.png", help='Download the current detection image with labels and boxes.')

            rows = result.boxes.data.tolist()
            st.write(rows)
            csv_text = "x1,y1,x2,y2,confidence,class_id\n" + "\n".join(
                ",".join(str(v) for v in row) for row in rows
            )
            st.download_button("Save detection table", csv_text, "detections.csv", help='Download box coordinates, confidence scores and class IDs as CSV.')

            for index, row in enumerate(rows):
                x1, y1, x2, y2 = [int(v) for v in row[:4]]
                crop = result.orig_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    st.download_button(
                        f"Save crop {index + 1}",
                        encode_image(crop, "png"),
                        f"crop_{index + 1}.png",
                        key=f"crop_download_{index}",
                     help='Download this result or sample file to your browser download folder.')
        return

    # --- Preview and Detect in video share the source selector ----------
    source_type = menu(
        "Video source",
        {
            "Upload file": "Upload a video file from your computer.",
            "Camera":      "Read from a camera attached to the computer running Python.",
            "Stream URL":  "Read from an RTSP or HTTP stream reachable from this computer.",
        },
        "video_source",
    )
    if source_type is None:
        return

    source = None
    uploaded_video = None

    if source_type == "Upload file":
        uploaded_video = st.file_uploader(
            "Video file",
            type=["mp4", "avi", "mov", "mkv"],
         help='Upload a video to preview or process frame by frame.')
        if uploaded_video is not None:
            st.video(uploaded_video.getvalue())
    elif source_type == "Camera":
        source = number_input("Camera index", 0, 10, 0, "camera_index")
    else:
        source = st.text_input("Stream URL", type="password",
                               help="May contain credentials; it is used only locally.")

    max_frames = number_input(
        "Maximum frames this run", 1, 3000, 100, "max_frames",
        help_text="The loop stops at this many frames even if the video is longer.",
    )

    controls = draw_detection_controls() if task == "Detect in video" else None

    if not st.button("Start", help='Process frames until the chosen frame limit or the end of the source.'):
        return

    if source_type == "Upload file" and uploaded_video is None:
        st.error("Upload a video file first.")
        return
    if source_type == "Stream URL" and not source:
        st.error("Enter a stream URL.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        if uploaded_video is not None:
            path = Path(temp_dir) / ("input" + Path(uploaded_video.name).suffix)
            path.write_bytes(uploaded_video.getvalue())
            source = str(path)

        capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                st.error("Cannot open the video source. Check the file, camera or URL.")
                return

            model = load_yolo_model(controls["weights"]) if controls else None
            frame_slot = st.empty()
            progress = st.progress(0.0)

            frame_count = 0
            for frame_index in range(max_frames):
                ok, frame = capture.read()
                if not ok:
                    break

                if model is not None:
                    result = model.predict(
                        frame,
                        conf=controls["confidence"],
                        classes=controls["class_ids"],
                        verbose=False,
                    )[0]
                    frame = result.plot(labels=controls["show_labels"],
                                        conf=controls["show_scores"])

                frame_slot.image(frame, channels="BGR",
                                 caption=f"Frame {frame_index + 1}")
                st.session_state.last_video_frame = frame
                frame_count += 1
                progress.progress((frame_index + 1) / max_frames)

            if frame_count == 0:
                st.warning("The source opened but no frames were read.")
            else:
                st.success(f"Processed {frame_count} frames.")
        finally:
            capture.release()

    if "last_video_frame" in st.session_state:
        st.download_button("Save last frame",
                           encode_image(st.session_state.last_video_frame, "png"),
                           "frame.png", help='Download the last processed video frame as a PNG.')
