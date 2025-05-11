# app.py
from flask import Flask, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
from components.imagePestDetection import process_image_for_prediction
from components.VideoPestDetection import process_video
from components.webcamPestDetection import generate_webcam_frames, stop_webcam_feed, start_webcam_feed, process_latest_webcam_frame, latest_webcam_results, webcam_model, PEST_DATA_WEBCAM
from components.fertilizer import get_fertilizer_recommendation

app = Flask(__name__)

# Configure allowed file types
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
UPLOAD_FOLDER = 'static/uploads'  # Define upload folder

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/image_detection', methods=['GET'])
def image_detection_page():
    return render_template('ImagePestDetection.html')

@app.route('/video_detection', methods=['GET'])
def video_detection_page():
    return render_template('VideoPestDetection.html')

@app.route('/webcam_detection', methods=['GET'])
def webcam_detection_page():
    return render_template('WebcamPestDetection.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return render_template('error.html', message='No image file provided')
    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', message='No image selected')
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image_' + filename)
        file.save(input_path)
        try:
            prediction_result = process_image_for_prediction(input_path)
            prediction_result['uploaded_image_url'] = '/' + input_path  # Corrected path for HTML
            return jsonify(prediction_result) # Keep JSON for image prediction
        except Exception as e:
            return render_template('error.html', message=f'Error processing image: {e}')
    else:
        return render_template('error.html', message='Invalid image file type')



@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return render_template('error.html', message='No video file provided')
    file = request.files['video']
    if file.filename == '':
        return render_template('error.html', message='No video selected')
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video_' + filename)
        output_filename = 'processed_video_' + filename  # Corrected:  filename, not path
        output_path_server = os.path.join('static', output_filename) #save in static
        output_video_url = '/static/' + output_filename
        file.save(input_path)
        try:
            processing_result = process_video(input_path, output_path_server)
            if 'error' in processing_result: #error from process_video
                return render_template('error.html', message=processing_result['error'])
            # Pass necessary data to the template
            return render_template('video_result.html',
                                   output_video_url=output_video_url,
                                   predictions=processing_result.get('predictions', []),
                                   harmful_detections=processing_result.get('harmful_detections', []))
        except Exception as e:
            return render_template('error.html', message=f'Error processing video: {e}')
    else:
        return render_template('error.html', message='Invalid video file type')



@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
def start_webcam():
    start_webcam_feed()
    return jsonify({'status': 'webcam started'})

@app.route('/stop_webcam')
def stop_webcam():
    stop_webcam_feed()
    return jsonify({'status': 'webcam stopped'})

@app.route('/get_webcam_prediction')
def get_webcam_prediction():
    global webcam_model
    global PEST_DATA_WEBCAM
    process_latest_webcam_frame(webcam_model, PEST_DATA_WEBCAM)
    global latest_webcam_results
    if latest_webcam_results:
        return jsonify(latest_webcam_results)
    else:
        return jsonify({})


# Fertilizer Recommendation Route
@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        n_input = int(request.form['nitrogen'])
        p_input = int(request.form['phosphorus'])
        k_input = int(request.form['potassium'])
        temp_input = int(request.form['temperature'])
        humidity_input = int(request.form['humidity'])
        soil_moisture_input = int(request.form['soil_moisture'])
        crop_type_input = int(request.form['crop_type'])
        soil_type_input = int(request.form['soil_type'])

        # Call the function from fertilizer.py
        result = get_fertilizer_recommendation(
            n_input, p_input, k_input, temp_input, humidity_input,
            soil_moisture_input, crop_type_input, soil_type_input
        )

        if 'error_message' in result:
            return render_template('fertilizer_result.html', error_message=result['error_message'])
        else:
            return render_template(
                'fertilizer_result.html',
                predicted_fertilizer=result['predicted_fertilizer'],
                description=result['description'],
                best_used_for=result['best_used_for'],
                application=result['application']
            )

    # Render the form initially
    return render_template('fertilizer_form.html')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Threaded=True for webcam
