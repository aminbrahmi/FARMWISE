# crop_predictor.py
import numpy as np
import joblib  # Pour charger le modèle sauvegardé

# Assurez-vous que le chemin vers votre modèle est correct
MODEL_PATH = 'content\Gaussien_naive_bayes_Model (1).pkl'

try:
    DecisionTree = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Erreur: Le modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}")
    DecisionTree = None  # Gérez le cas où le modèle n'est pas chargé

def predict_crop(N, P, K, Temp, Humidity, pH, Rainfall):
    if DecisionTree is None:
        return "Erreur: Le modèle n'est pas disponible."
    input_data = np.array([[N, P, K, Temp, Humidity, pH, Rainfall]])
    predicted_crop = DecisionTree.predict(input_data)
    return f"Culture recommandée : {predicted_crop[0]}"
