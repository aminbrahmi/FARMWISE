import streamlit as st
from auth.auth_utils import init_db, verify_user, add_user, user_exists

def login_screen():
    init_db()  # Ensure DB is initialized

    # --- CSS Animations and Styling ---
    st.markdown("""
    <style>
    @keyframes fadeIn {
      0% { opacity: 0; transform: translateY(-20px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes shake {
      0% { transform: translateX(0); }
      25% { transform: translateX(-5px); }
      50% { transform: translateX(5px); }
      75% { transform: translateX(-5px); }
      100% { transform: translateX(0); }
    }

    body {
      background-color: #f0f8ff;
    }

    .login-title {
        text-align: center;
        font-size: 45px;
        color: #2E8B57;
        margin-bottom: 40px;
        animation: fadeIn 1s ease-in;
    }

    .stTextInput > div > div > input {
        font-size: 20px;
        padding: 15px;
        margin-bottom: 15px;
        width: 100%;
    }

    .stButton button {
        font-size: 20px;
        padding: 12px 24px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 10px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }

    .error {
        animation: shake 0.5s;
    }

    .agri-images {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
        animation: fadeIn 3s ease;
    }

    .agri-images img {
        width: 150px;
        height: 150px;
        object-fit: cover;
        border-radius: 50%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown("<div class='login-title'>🌾 Welcome to FarmWise</div>", unsafe_allow_html=True)

    # --- Agriculture Images ---
    st.markdown("""
    <div class='agri-images'>
        <img src="https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg?auto=compress&cs=tinysrgb&w=600">
        <img src="https://images.pexels.com/photos/2255920/pexels-photo-2255920.jpeg?auto=compress&cs=tinysrgb&w=600">
        <img src="https://images.pexels.com/photos/247616/pexels-photo-247616.jpeg?auto=compress&cs=tinysrgb&w=600">
    </div>
    """, unsafe_allow_html=True)

    # --- Toggle between login and register ---
    page = st.radio("Choose an option:", ["🔐 Login", "📝 Create Account"], horizontal=True)

    # --- Input Fields ---
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if page == "🔐 Login":
        if st.button("🔓 Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.success("✅ Login successful! Redirecting...")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
                st.markdown("<div class='error'></div>", unsafe_allow_html=True)
    else:
        confirm_password = st.text_input("🔁 Confirm Password", type="password")
        if st.button("🆕 Create Account"):
            if user_exists(username):
                st.warning("⚠️ Username already exists.")
            elif password != confirm_password:
                st.warning("⚠️ Passwords do not match.")
            elif not username or not password:
                st.warning("⚠️ All fields are required.")
            else:
                add_user(username, password)
                st.success("✅ Account created successfully! Please login.")
