#leaf_prediction.py
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Load the trained VGG16 model for leaf disease
try:
    leaf_disease_model = load_model('content\plantVillage_VGG16.h5') 
    leaf_disease_class_names = [
        'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]
    print("Leaf disease model loaded successfully from leaf_prediction.py!")
except Exception as e:
    print(f"Error loading leaf disease model in leaf_prediction.py: {e}")
    leaf_disease_model = None
    leaf_disease_class_names = []

def preprocess_leaf_image(image_bytes):
    """Preprocesses the image for the leaf disease VGG16 model."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array) # VGG16 preprocessing
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing in leaf_prediction.py: {e}")
        return None

def predict_leaf_disease(image_bytes):
    """Predicts the disease of a leaf from the image bytes."""
    if leaf_disease_model is None:
        return {"error": "Leaf disease model not loaded."}

    processed_image = preprocess_leaf_image(image_bytes)
    if processed_image is None:
        return {"error": "Error during image preprocessing."}

    try:
        predictions = leaf_disease_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = leaf_disease_class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        return {"prediction": predicted_class_name, "confidence": f"{confidence:.2f}%"}
    except Exception as e:
        print(f"Error during prediction in leaf_prediction.py: {e}")
        return {"error": f"Error during prediction: {e}"}