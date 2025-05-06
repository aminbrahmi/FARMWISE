
import streamlit as st
from components import image_detection, video_detection, webcam_detection

def show_detection_hub(model_yolo):
    st.title("🔍 Smart Detection Hub")

    detection_type = st.radio(
        "Select Detection Mode:",
        ["📷 Image Detection", "🎥 Video Detection", "📹 Real-Time Webcam"],
        horizontal=True
    )

    st.markdown("---")

    if detection_type == "📷 Image Detection":
        image_detection.show(model_yolo)
    elif detection_type == "🎥 Video Detection":
        video_detection.show(model_yolo)
    elif detection_type == "📹 Real-Time Webcam":
        webcam_detection.show(model_yolo)
