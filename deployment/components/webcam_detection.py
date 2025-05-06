import streamlit as st
import cv2
import pandas as pd
import numpy as np

@st.cache_data
def load_pest_info():
    try:
        return pd.read_csv(r"utils\detailed_pests_solutions.csv")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def show(model):
    st.markdown("<div class='title'>📹 Real-Time Detection</div>", unsafe_allow_html=True)
    st.warning("⚠️ Click 'Start Webcam' to begin real-time detection")

    # Initialize session state
    if 'detected_pests' not in st.session_state:
        st.session_state.detected_pests = {}

    run = st.checkbox("▶️ Start Webcam")

    if run:
        pest_info_df = load_pest_info()
        if pest_info_df.empty:
            st.warning("⚠️ Pest info file missing or empty.")
            return

        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.success("✅ Streaming from webcam...")

        frame_count = 0
        frame_skip = 3  # Process every 3 frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Webcam not detected.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 640))  # Resize for faster detection

            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_container_width=True)

            names = results[0].names
            probs = results[0].probs

            if probs is not None and len(probs) > 0:
                probs_data = probs.data.cpu().numpy()
                for i in range(len(probs_data)):
                    score = float(probs_data[i])
                    name = names[i].lower()
                    if score > 0.70:  # Confidence threshold
                        if name not in st.session_state.detected_pests or score > st.session_state.detected_pests[name]:
                            st.session_state.detected_pests[name] = score

        cap.release()
        stframe.empty()
        st.success("✅ Webcam stopped.")

    # After unchecking "Start Webcam", display pest info
    if not run and st.session_state.detected_pests:
        st.markdown("## 🐛 Detected Pests Information")
        pest_info_df = load_pest_info()

        for pest, score in st.session_state.detected_pests.items():
            row = pest_info_df[pest_info_df["Pest Name"].str.lower() == pest]
            st.markdown(f"### 🐛 {pest.title()} ({score * 100:.1f}% confidence)")
            if not row.empty:
                st.markdown(f"**🔬 Scientific Name:** *{row.iloc[0]['Scientific Name']}*")
                st.markdown(f"**📖 Description:** {row.iloc[0]['Description']}")
                st.markdown(f"**🛠️ Management Strategies:** {row.iloc[0]['Management Strategies']}")
            else:
                st.warning(f"No additional info found in CSV for {pest}.")
            st.markdown("---")

        # Optional: Clear the detected pests after showing
        # st.session_state.detected_pests = {}
