# components\imagePestDetection.py
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import csv

# Define your list of harmful pests
harmful_pests = [
    "Africanized Honey Bees (Killer Bees)",
    "Aphids",
    "Armyworms",
    "Brown Marmorated Stink Bugs",
    "Cabbage Loopers",
    "Citrus Canker",
    "Colorado Potato Beetles",
    "Corn Borers",
    "Corn Earworms",
    "Fall Armyworms",
    "Fruit Flies",
    "Spider Mites",
    "Thrips",
    "Tomato Hornworms",
    "Western Corn Rootworms"
]

# Load the trained YOLO model
try:
    model = YOLO('content/pestDetectionbestV11.pt')
    print("YOLO model loaded successfully from components/image.py.")
except Exception as e:
    print(f"Error loading YOLO model in components/image.py: {e}")
    model = None

def load_pest_data(csv_filepath):
    pest_info = {}
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pest_name = row['Pest Name'].strip()
                pest_info[pest_name] = {
                    'scientific_name': row['Scientific Name'],
                    'description': row['Description'],
                    'management_strategies': row['Management Strategies']
                }
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return pest_info

# Load pest data globally within image.py
PEST_DATA = load_pest_data('utils\detailed_pests_solutions.csv')

def process_image_for_prediction(image_path):
    """Processes the image and returns prediction results."""
    if model is None:
        return {'error': 'YOLO model not loaded'}

    try:
        results = model(image_path)
        pest_description = None
        management_strategies = None

        if results and hasattr(results[0], 'xyxy') and len(results[0].xyxy) > 0 and len(results[0].xyxy[0]) > 0:
            pred = results[0].probs
            if pred is not None:
                class_index = pred.top1
                confidence = pred.data[class_index].item() * 100
                class_name = results[0].names[class_index]

                is_harmful = class_name in harmful_pests and confidence <= 75

                if class_name in PEST_DATA:
                    pest_description = PEST_DATA[class_name]['description']
                    management_strategies = PEST_DATA[class_name]['management_strategies']

                prediction_result = {
                    'pest_detected': class_name,
                    'confidence': f"{confidence:.2f}%",
                    'is_harmful': is_harmful,
                    'description': pest_description,
                    'management': management_strategies
                }
                return prediction_result
            else:
                return {'error': 'No prediction probabilities found'}
        elif results and hasattr(results[0], 'probs') and results[0].probs is not None:
            # Classification result
            pred = results[0].probs
            class_index = pred.top1
            confidence = pred.data[class_index].item() * 100
            class_name = results[0].names[class_index]

            is_harmful = class_name in harmful_pests and confidence >= 75

            if class_name in PEST_DATA:
                pest_description = PEST_DATA[class_name]['description']
                management_strategies = PEST_DATA[class_name]['management_strategies']

            prediction_result = {
                'pest_detected': class_name,
                'confidence': f"{confidence:.2f}%",
                'is_harmful': is_harmful,
                'description': pest_description,
                'management': management_strategies
            }
            return prediction_result
        else:
            return {'message': 'No pests detected in the image'}

    except Exception as e:
        return {'error': f'Error during prediction: {e}'}