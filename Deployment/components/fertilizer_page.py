import streamlit as st
import pandas as pd
import pickle
import os


# Load the model and data once
@st.cache_resource
def load_fertilizer_model():
    model_path = os.path.join("model", "fertilizer.pkl")
    with open(model_path, "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_fertilizer_info():
    csv_path = os.path.join("utils", "fertilizer_instructions.csv")
    return pd.read_csv(csv_path)

def show_fertilizer_page():
    st.title("🌾 Fertilizer Recommendation")
    st.markdown("Get the most suitable fertilizer based on your soil and crop data.")

    # User inputs
    N = st.slider("Nitrogen (N)", 0, 140, 34)
    P = st.slider("Phosphorous (P)", 5, 145, 65)
    K = st.slider("Potassium (K)", 5, 205, 62)
    temp = st.slider("Temperature (°C)", 0, 50, 30)
    humidity = st.slider("Humidity (%)", 0, 100, 65)
    moisture = st.slider("Moisture (%)", 0, 100, 7)
    soil_type = st.selectbox("Soil Type", ["Black", "Clayey", "Loamy", "Red", "Sandy"])
    crop_type = st.selectbox("Crop Type", ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"])

    # Encode categorical values
    soil_dict = {"Black": 0, "Clayey": 1, "Loamy": 2, "Red": 3, "Sandy": 4}
    crop_dict = {
        "Barley": 0, "Cotton": 1, "Ground Nuts": 2, "Maize": 3, "Millets": 4,
        "Oil seeds": 5, "Paddy": 6, "Pulses": 7, "Sugarcane": 8, "Tobacco": 9, "Wheat": 10
    }

    if st.button("Predict Fertilizer"):
        model = load_fertilizer_model()
        info = load_fertilizer_info()
        
        # Predict
        input_data = [[N, P, K, soil_dict[soil_type], crop_dict[crop_type], temp, humidity, moisture]]
        ans = model.predict(input_data)

        fertilizer_mapping = {
            0: "10-26-26",
            1: "14-35-14",
            2: "17-17-17",
            3: "20-20",
            4: "28-28",
            5: "DAP",
            6: "Urea"
        }

        predicted_fertilizer = fertilizer_mapping.get(ans[0], "Unknown")

        # Get more info
        row = info[info['Fertilizer Name'].str.replace('/', '-').str.strip() == predicted_fertilizer]

        st.subheader("🌿 Recommended Fertilizer:")
        st.success(predicted_fertilizer)

        if not row.empty:
            st.markdown(f"**🔍 Description:** {row.iloc[0]['Description']}")
            st.markdown(f"**🌱 Best Used For:** {row.iloc[0]['Best Used For']}")
            st.markdown(f"**💡 Application:** {row.iloc[0]['Application']}")
        else:
            st.warning("⚠️ Fertilizer information not found in the CSV.")
