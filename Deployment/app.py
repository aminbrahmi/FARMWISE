import streamlit as st
import cv2
from ultralytics import YOLO

from auth.login import login_screen
from components import home, image_detection, video_detection, webcam_detection , PestDetection , fertilizer_page
from model.yolo_model import load_model_yolo


# Login check
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
    st.stop()

# Load YOLO model

model_yolo = load_model_yolo()

# Sidebar
st.set_page_config(page_title="🌾 FarmWise", layout="wide")
# --- Custom Sidebar Styling ---
st.markdown("""
    <style>
    /* General sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7f7f7;
        padding: 40px 20px;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.1);
        border-right: 1px solid #e0e0e0;
    }

    /* Sidebar logo */
    [data-testid="stSidebar"] img {
        display: block;
        margin: 0 auto 20px;
        border-radius: 50%;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 28px;
        font-weight: 600;
        color: #388e3c;
        text-align: center;
        margin-bottom: 15px;
        letter-spacing: 1px;
    }

    /* Sidebar subtitle */
    .sidebar-subtitle {
        font-size: 14px;
        text-align: center;
        color: #757575;
        margin-bottom: 30px;
    }

    /* Radio buttons */
    div[role="radiogroup"] {
        margin-top: 20px;
    }

    div[role="radiogroup"] > label {
        font-size: 16px;
        padding: 12px;
        margin: 6px 0;
        border-radius: 8px;
        color: #444;
        background-color: #e8f5e9;
        transition: all 0.3s ease;
        cursor: pointer;
        display: block;
        text-align: center;
        border: 1px solid transparent;
    }

    div[role="radiogroup"] > label:hover {
        background-color: #c8e6c9;
        transform: translateY(-2px);
    }

    div[role="radiogroup"] > label[data-selected="true"] {
        background-color: #81c784;
        color: white;
        font-weight: 700;
        border: 1px solid #388e3c;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Button styling */
    .stButton > button {
        background-color: #388e3c;
        color: white;
        font-size: 16px;
        padding: 12px 25px;
        border: none;
        border-radius: 30px;
        width: 100%;
        transition: background-color 0.3s, transform 0.2s ease;
        margin-top: 15px;
    }

    .stButton > button:hover {
        background-color: #2c6e1f;
        transform: scale(1.05);
    }

    /* Logout button styling */
    .stButton > button.logout {
        background-color: #f44336;
    }

    .stButton > button.logout:hover {
        background-color: #d32f2f;
        transform: scale(1.05);
    }

    </style>
""", unsafe_allow_html=True)



st.sidebar.image("https://cdn-icons-png.freepik.com/512/1886/1886966.png", width=100)
st.sidebar.title("🌾 FarmWise")
st.sidebar.markdown("Smart Detection for Farmers")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()



page = st.sidebar.radio(
    "📌 Navigate to:",
    ["🏡 Home", "🔍 Detection", "🌾 Fertilizer Recommendation"]
)

if page == "🏡 Home":
    home.show()

elif page == "🔍 Detection":
    PestDetection.show_detection_hub(model_yolo)
elif page == "🌾 Fertilizer Recommendation":
    fertilizer_page.show_fertilizer_page()
