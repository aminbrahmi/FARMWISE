import streamlit as st
import cv2
import tempfile
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
    st.markdown("<h1 style='text-align:center;'>🎞️ Video Detection</h1>", unsafe_allow_html=True)
    st.markdown("Upload a video to detect pests. Pest info will appear after playback.")

    uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        pest_info_df = load_pest_info()
        if pest_info_df.empty:
            st.warning("⚠️ Pest info file missing or empty.")
            return

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.info("🔄 Playing video with live detection...")

        detected_pests = {}
        frame_count = 0
        frame_skip = 5  # 📈 Skip every 5 frames (process 1 frame out of 5)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip this frame

            # 📏 Resize frame smaller for faster detection
            frame = cv2.resize(frame, (640, 640))

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
                    if score > 0.70:
                        if name not in detected_pests or score > detected_pests[name]:
                            detected_pests[name] = score

        cap.release()
        st.success("✅ Video finished. Displaying pest info...")

        if detected_pests:
            for pest, score in detected_pests.items():
                row = pest_info_df[pest_info_df["Pest Name"].str.lower() == pest]
                st.markdown(f"### 🐛 {pest.title()} ({score * 100:.1f}% confidence)")
                if not row.empty:
                    st.markdown(f"**🔬 Scientific Name:** *{row.iloc[0]['Scientific Name']}*")
                    st.markdown(f"**📖 Description:** {row.iloc[0]['Description']}")
                    st.markdown(f"**🛠️ Management Strategies:** {row.iloc[0]['Management Strategies']}")
                else:
                    st.warning("No additional info found in the CSV.")
                st.markdown("---")
        else:
            st.info("✅ No pests detected with confidence above 70%.")
