# land_price_prediction.py
import os
import joblib
import pandas as pd
from flask import render_template
import sklearn
import pickle


print(f"Scikit-learn version in land_price_prediction: {sklearn.__version__}")

model_path = r'content\xgboost_model2.pkl'
print(f"Model path in land_price_prediction: {model_path}")

df_landprice = pd.read_csv("utils/agriculture_lands_tn.csv")


def predict_land_price(form_data):
    land_price_pipeline = None
    error_message = None
    prediction_result = None

    try:
        print(f"Scikit-learn version during prediction: {sklearn.__version__}")
        land_price_pipeline = joblib.load(model_path)
        print(f"Loaded model in land_price_prediction: {land_price_pipeline}")
        print("Land price model loaded successfully within predict_land_price!")

        # Create a DataFrame from the form data
        new_terrain = pd.DataFrame([{
            'Gouvernorat': form_data.get('Gouvernorat'),
            'Délégation': form_data.get('Delegation'),
            'Proximité': form_data.get('Proximity'),
            'Infrastructure': form_data.get('Infrastructure'),
            'Type_Agriculture': form_data.get('Type_Agriculture'),
            'Additional_Features': form_data.get('Additional_Features'),
            'Taille_m2': float(form_data.get('Taille_m2')) if form_data.get('Taille_m2') else None  # Handle potential missing value
        }])

        # Handle potential missing Taille_m2
        if new_terrain['Taille_m2'].isnull().any():
            return None, "Please provide the land size."

        prix_m2_prevu = land_price_pipeline.predict(new_terrain)[0]
        prix_total_prevu = prix_m2_prevu * new_terrain["Taille_m2"].values[0]

        prediction_result = {
            'prix_m2_prevu': f'{prix_m2_prevu:.2f}',
            'prix_total_prevu': f'{prix_total_prevu:.2f}'
        }
        return prediction_result, None  # Return None for error

    except Exception as e:
        return None, f'Erreur lors du chargement ou de la prédiction du modèle : {e}'