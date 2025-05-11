# components/webcamPestDetection.py
from ultralytics import YOLO
import cv2
import csv
import time

# Load the trained YOLO model
try:
    webcam_model = YOLO('content/pestDetectionbestV11.pt')
    print("YOLO webcam model loaded successfully from components/webcam.py.")
except Exception as e:
    print(f"Error loading YOLO webcam model in components/webcam.py: {e}")
    webcam_model = None

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

# Load pest data globally within webcam.py
PEST_DATA_WEBCAM = load_pest_data('utils\detailed_pests_solutions.csv')

# Control variables for webcam feed
is_webcam_active = False
cap = None
latest_webcam_results = None

def start_webcam_feed():
    global is_webcam_active
    global cap
    is_webcam_active = True
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

def stop_webcam_feed():
    global is_webcam_active
    global cap
    is_webcam_active = False
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
        print("Webcam released.")

def process_webcam_frame(frame):
    global webcam_model
    if webcam_model is None:
        return None

    results = webcam_model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame, results

def process_latest_webcam_frame(model, pest_data):
    global latest_webcam_results  # Declare it as global at the beginning
    cap_local = cv2.VideoCapture(0)
    if cap_local.isOpened():
        ret, frame = cap_local.read()
        cap_local.release()
        if ret:
            results_list = model(frame)  # Get the list of Results objects
            if results_list and len(results_list) > 0:
                results = results_list[0]  # Access the Results object for the first image
                if results.boxes and len(results.boxes.xyxy) > 0:
                    # Assuming only one detection for simplicity
                    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
                    confidence = results.boxes.conf[0].cpu().numpy()
                    class_id = int(results.boxes.cls[0].cpu().numpy())
                    pest_name = model.names[class_id]
                    is_harmful = "harmful" if pest_name in ["aphid", "caterpillar"] else "not-harmful" # Example logic
                    pest_info = pest_data.get(pest_name)
                    description = pest_info.get('description', 'No description available') if pest_info else 'No description available'
                    management = pest_info.get('management_strategies', 'No management strategies available') if pest_info else 'No management strategies available'

                    latest_webcam_results = {
                        'pest_detected': pest_name,
                        'confidence': float(confidence) * 100,
                        'is_harmful': is_harmful,
                        'description': description,
                        'management': management
                    }
                else:
                    latest_webcam_results = None
            else:
                latest_webcam_results = None
        else:
            latest_webcam_results = None
    else:
        latest_webcam_results = None

def generate_webcam_frames():
    global is_webcam_active
    global cap
    try:
        while True:
            if is_webcam_active:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise IOError("Cannot open webcam")
                        break
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, results = process_webcam_frame(frame)
                if annotated_frame is not None:
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

if __name__ == '__main__':
    start_webcam_feed()
    time.sleep(10)
    stop_webcam_feed()