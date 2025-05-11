# fertilizer.py
import pickle
import pandas as pd
import os

def get_fertilizer_recommendation(n_input, p_input, k_input, temp_input, humidity_input, soil_moisture_input, crop_type_input, soil_type_input):
    try:
        # 1. Load the model (ensure the path is correct for your environment)
        model_path = os.path.join("content", "fertilizer.pkl")
        model = pickle.load(open(model_path, 'rb'))

        # 2. Load the fertilizer information (ensure the path is correct)
        csv_path = os.path.join("utils", "fertilizer_instructions.csv")
        fertilizer_info = pd.read_csv(csv_path)

        # 3. Make the prediction
        ans = model.predict([[n_input, p_input, k_input, temp_input, humidity_input, soil_moisture_input, crop_type_input, soil_type_input]])

        # 4. Map the predicted result to the fertilizer name
        fertilizer_mapping = {
            0: "10-26-26",
            1: "14-35-14",
            2: "17-17-17",
            3: "20-20",
            4: "28-28",
            5: "DAP",
            6: "Urea"
        }
        predicted_fertilizer = fertilizer_mapping[ans[0]]

        # 5. Find the full information for the predicted fertilizer
        row = fertilizer_info[fertilizer_info['Fertilizer Name'].str.replace('/', '-').str.strip() == predicted_fertilizer]

        if not row.empty:
            # Extract details from the CSV
            description = row.iloc[0]['Description']
            best_used_for = row.iloc[0]['Best Used For']
            application = row.iloc[0]['Application']

            return {
                'predicted_fertilizer': predicted_fertilizer,
                'description': description,
                'best_used_for': best_used_for,
                'application': application
            }
        else:
            return {'error_message': "⚠️ Fertilizer information not found in the CSV."}

    except Exception as e:
        return {'error_message': f"Error processing request: {e}"}