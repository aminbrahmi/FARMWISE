# components\VideoPestDetection.py
from ultralytics import YOLO
import cv2
import os
import csv
import numpy as np

# Define harmful pests
harmful_pests = [
    "Africanized Honey Bees (Killer Bees)", "Aphids", "Armyworms",
    "Brown Marmorated Stink Bugs", "Cabbage Loopers", "Citrus Canker",
    "Colorado Potato Beetles", "Corn Borers", "Corn Earworms",
    "Fall Armyworms", "Fruit Flies", "Spider Mites",
    "Thrips", "Tomato Hornworms", "Western Corn Rootworms"
]

# Load the trained YOLO model
try:
    video_model = YOLO('content/pestDetectionbestV11.pt')
    print("YOLO video model loaded successfully from components/video.py.")
except Exception as e:
    print(f"Error loading YOLO video model in components/video.py: {e}")
    video_model = None

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

# Load pest data globally within video.py
PEST_DATA_VIDEO = load_pest_data('utils/detailed_pests_solutions.csv')

def process_video(video_path, output_path):
    if video_model is None:
        return {'error': 'YOLO video model not loaded', 'predictions': []}

    predictions = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Error opening video file', 'predictions': []}

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices_to_process = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
        processed_frame_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i in frame_indices_to_process:
                results = video_model(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                processed_frame_count += 1

                if results and results[0].probs is not None:
                    pred = results[0].probs
                    class_index = pred.top1
                    confidence = pred.data[class_index].item()
                    class_name = results[0].names[class_index]

                    pest_details = PEST_DATA_VIDEO.get(class_name)
                    is_harmful = class_name in harmful_pests
                    prediction_data = {'frame': i, 'pest': class_name, 'confidence': float(confidence), 'is_harmful': is_harmful}
                    if pest_details:
                        prediction_data.update(pest_details)
                    predictions.append(prediction_data)

        cap.release()
        out.release()
        return {'output_path': output_path, 'predictions': predictions}

    except Exception as e:
        return {'error': f'Error processing video: {e}', 'predictions': []}