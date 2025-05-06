import streamlit as st

def show():
    st.markdown("""
    <style>
    .title {
        font-size: 42px;
        text-align: center;
        color: #4CAF50;
        padding: 20px 0;
    }
    .content-box {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    h2 {
        color: #2E7D32;
    }
    ul {
        font-size: 18px;
        line-height: 1.8;
    }
    hr {
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🌿 Welcome to FarmWise: Smart Detection for Agriculture 🌿</div>", unsafe_allow_html=True)

    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXHsvQ31b86uvV1PYVF22DWMK8g9jY9wyzw_jbmbKuO5FpKZcME7bSBXtF0rB9HMIvRYo&usqp=CAU",
        use_container_width=True
    )

    st.markdown("""
    <div class='content-box'>
        <h2>🚀 What You Can Do</h2>
        <ul>
            <li>📸 <b>Upload images</b> to detect pests instantly.</li>
            <li>🎬 <b>Upload videos</b> for automatic frame-by-frame pest detection.</li>
            <li>📹 <b>Use your webcam</b> for real-time smart pest detection!</li>
        </ul>
        <hr>
        <h2>✅ Powered By</h2>
        <ul>
            <li>⚡ <b>YOLO Model</b> for fast and accurate detections</li>
            <li>🖥️ <b>Streamlit</b> for a smooth web experience</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
