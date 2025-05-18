from flask import Flask, request, render_template, Response, jsonify, url_for, redirect, session, g, flash
from werkzeug.utils import secure_filename
import os
import pickle
import joblib
import pandas as pd
import sys
from importlib import reload
from components.imagePestDetection import process_image_for_prediction
from components.VideoPestDetection import process_video
from components.webcamPestDetection import generate_webcam_frames, stop_webcam_feed, start_webcam_feed, process_latest_webcam_frame, latest_webcam_results, webcam_model, PEST_DATA_WEBCAM
from components.fertilizer import get_fertilizer_recommendation
from components.PriceEstimation import detect_objects
from components.supplier_logic import find_nearby_suppliers, create_supplier_map
from components.crop_predictor import predict_crop
from components.land_price_prediction import predict_land_price
from components.leaf_prediction import predict_leaf_disease 
from components.toxic_plant_logic import predict_toxic_plant
from components.landslide_logic import detect_landslide_logic
from components.login import get_db, register_user, authenticate_user
from functools import wraps



app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session
DATABASE = 'users.db'


# Configure allowed file types
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
UPLOAD_FOLDER = 'static/uploads'  # Define upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

@app.cli.command('initdb')
def initdb_command():
    """Initializes the database."""
    init_db()
    print('Initialized the database.')


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = register_user(username, password)
        if error is None:
            return redirect(url_for('login'))
        flash(error)
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user, error = authenticate_user(username, password)
        if user:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('home'))
        flash(error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if session.get('user_id') is None:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

@app.route('/', methods=['GET'])
@login_required
def home():
    return render_template('home.html')

@app.route('/image_detection', methods=['GET'])
@login_required
def image_detection_page():
    return render_template('ImagePestDetection.html')

@app.route('/video_detection', methods=['GET'])
@login_required
def video_detection_page():
    return render_template('VideoPestDetection.html')

@app.route('/webcam_detection', methods=['GET'])
@login_required
def webcam_detection_page():
    return render_template('WebcamPestDetection.html')

@app.route('/predict_image', methods=['POST'])
@login_required
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
@login_required
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
@login_required
def webcam_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
@login_required
def start_webcam():
    start_webcam_feed()
    return jsonify({'status': 'webcam started'})

@app.route('/stop_webcam')
@login_required
def stop_webcam():
    stop_webcam_feed()
    return jsonify({'status': 'webcam stopped'})

@app.route('/get_webcam_prediction')
@login_required
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
@login_required
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


#Mehdi yoloV5 donne le prix qu'on peut ganier
@app.route('/PriceEstimation', methods=['GET', 'POST'])
@login_required
def PriceEstimation():
    if request.method == 'POST':
        img = request.files['image']
        prix_kilo = float(request.form['prix_kilo'])
        poids_fruit = float(request.form['poids_fruit'])

        img_path = "static/uploads/" + img.filename
        img.save(img_path)

        results, labels = detect_objects(img_path)
        nb_fruits = len(labels)
        poids_total_kg = (nb_fruits * poids_fruit) / 1000
        gain = poids_total_kg * prix_kilo

        return render_template('PriceEstimation.html', result=round(gain, 2), count=nb_fruits, uploaded_image_path=img_path)

    return render_template('PriceEstimation.html')


#Chercher le plus proche fournisseur
@app.route('/supplier_search', methods=['GET', 'POST'])
@login_required
def supplier_search():
    nearby_suppliers = None
    folium_map_html = None
    error = None

    if request.method == 'POST':
        try:
            user_lat = float(request.form['latitude'])
            user_lon = float(request.form['longitude'])
            rayon = float(request.form['rayon'])

            nearby, all_suppliers = find_nearby_suppliers(user_lat, user_lon, rayon)
            nearby_suppliers = nearby

            if not nearby_suppliers.empty:
                m = create_supplier_map(user_lat, user_lon, nearby_suppliers, all_suppliers)
                map_filename = 'supplier_map.html'
                # Save to the static directory
                map_filepath = os.path.join('static', map_filename)
                m.save(map_filepath)
                # Generate the URL for the static file
                folium_map_html = url_for('static', filename=map_filename)
            else:
                folium_map_html = None # No map needed if no suppliers found

        except ValueError:
            error = "Invalid input. Please enter numeric values for latitude, longitude, and radius."
        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template('supplier_search.html', nearby_suppliers=nearby_suppliers, folium_map_html=folium_map_html, error=error)
# oumaima model 1 
@app.route('/crop_prediction', methods=['GET', 'POST'])
@login_required
def crop_prediction():
    prediction = None

    if request.method == 'POST':
        try:
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            prediction = predict_crop(n, p, k, temp, humidity, ph, rainfall)
        except ValueError:
            prediction = "Veuillez entrer des valeurs numériques valides."
        except Exception as e:
            prediction = f"Une erreur s'est produite lors de la prédiction : {e}"

    return render_template('crop_prediction.html', prediction=prediction)

# Amine's work :
@app.route('/land_price_test')
def land_price_test():
    prediction, error = predict_land_price({}) # Pass an empty dict
    return render_template('land_price_prediction.html', prediction_result=prediction, error=error)


##
@app.route('/land_price_prediction', methods=['GET', 'POST'])
@login_required
def land_price_prediction_route():
    gouvernorats = sorted(pd.read_csv("utils/agriculture_lands_tn.csv")["Gouvernorat"].dropna().unique())
    delegations = sorted(pd.read_csv("utils/agriculture_lands_tn.csv")["Délégation"].dropna().unique())
    agriculture_types = sorted(pd.read_csv("utils/agriculture_lands_tn.csv")["Type_Agriculture"].dropna().unique())
    prediction_text = session.pop("prediction_text", "")
    if request.method == 'POST':
        try:
            data = {
                'Gouvernorat': request.form['Gouvernorat'],
                'Délégation': request.form['Délégation'],
                'Proximité': request.form.get('Proximité') or 'aucun',
                'Infrastructure': ', '.join(request.form.getlist('Infrastructure')) or 'aucun',
                'Type_Agriculture': request.form.get('Type_Agriculture') or 'aucun',
                'Additional_Features': ', '.join(request.form.getlist('Additional_Features')) or 'aucun',
                'Taille_m2': request.form['Taille_m2']
            }
            prediction_result, error = predict_land_price(data)

            if error:
                session["prediction_text"] = f"Error: {error}"
                return redirect(url_for("land_price_prediction_route"))
            else:
                prix_total = float(prediction_result['prix_total_prevu'])
                prix_m2 = float(prediction_result['prix_m2_prevu'])
                session["prediction_text"] = f"Estimated Price: {prix_total:,.2f} TND ({prix_m2:.2f} TND/m²)"
                return redirect(url_for("land_price_result", prediction_text=session["prediction_text"], **data))

        except Exception as e:
            session["prediction_text"] = f"Error processing prediction: {str(e)}"
            return redirect(url_for("land_price_prediction_route"))

    return render_template("land_price_prediction.html", gouvernorats=gouvernorats, delegations=delegations, agriculture_types=agriculture_types, prediction_text=prediction_text)


@app.route("/land_price_result")
@login_required
def land_price_result():
    prediction_text = request.args.get("prediction_text")
    governorate = request.args.get("Gouvernorat")
    delegation = request.args.get("Délégation")
    proximity = request.args.get("Proximité")
    infrastructure = request.args.get("Infrastructure")
    agriculture_type = request.args.get("Type_Agriculture")
    additional_features = request.args.get("Additional_Features")
    size = request.args.get("Taille_m2")

    return render_template("land_price_result.html", prediction_text=prediction_text,
                           governorate=governorate, delegation=delegation, proximity=proximity,
                           infrastructure=infrastructure, agriculture_type=agriculture_type,
                           additional_features=additional_features, size=size)

@app.route("/get_land_delegations", methods=["POST"])
@login_required
def get_land_delegations():
    selected_gov = request.json.get("gouvernorat")
    df = pd.read_csv("utils/agriculture_lands_tn.csv")
    filtered_delegations = df[df["Gouvernorat"] == selected_gov]["Délégation"].dropna().unique().tolist()
    return jsonify({"delegations": filtered_delegations})

@app.route("/get_land_proximites", methods=["POST"])
@login_required
def get_land_proximites():
    data = request.json
    gov = data.get("gouvernorat")
    delg = data.get("delegation")
    df = pd.read_csv("utils/agriculture_lands_tn.csv")
    filtered = df[(df["Gouvernorat"] == gov) & (df["Délégation"] == delg)]
    proximites = filtered["Proximité"].dropna().unique().tolist()
    return jsonify({"proximites": proximites})

@app.route("/get_land_agriculture_types", methods=["POST"])
@login_required
def get_land_agriculture_types():
    data = request.json
    gov = data.get("gouvernorat")
    delg = data.get("delegation")
    df = pd.read_csv("utils/agriculture_lands_tn.csv")
    filtered_data = df[(df["Gouvernorat"] == gov) & (df["Délégation"] == delg)]
    agriculture_types = filtered_data["Type_Agriculture"].dropna().unique().tolist()
    return jsonify({"agriculture_types": agriculture_types})


#oumaima model 2 :
@app.route('/predict_leaf_disease1', methods=['GET', 'POST'])
@login_required
def handle_leaf_disease_prediction():  # ✅ nom différent ici
    if 'leaf_image' not in request.files:
        return render_template('LeafDiseaseDetection.html', error='No leaf image file provided')

    file = request.files['leaf_image']

    if file.filename == '':
        return render_template('LeafDiseaseDetection.html', error='No leaf image selected')

    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            img_bytes = file.read()
            prediction_result = predict_leaf_disease(img_bytes)  # ✅ appelle la fonction importée
            if "error" in prediction_result:
                return render_template('LeafDiseaseDetection.html', error=prediction_result["error"])
            else:
                return render_template(
                    'LeafDiseaseDetection.html',
                    prediction=f'Predicted disease: {prediction_result["prediction"]}',
                    confidence=prediction_result["confidence"]
                )
        except Exception as e:
            return render_template('LeafDiseaseDetection.html', error=f'Error processing leaf image: {e}')
    else:
        return render_template('LeafDiseaseDetection.html', error='Invalid leaf image file type')

#ines
@app.route('/predict_toxic_plant', methods=['POST'])
@login_required
def predict_toxic_plant_endpoint():
    """Endpoint for toxic plant image classification."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if image_file and allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        image_bytes = image_file.read()
        prediction = predict_toxic_plant(image_bytes)
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Invalid image file type'}), 400

@app.route('/toxic_plant_detection', methods=['GET'])
@login_required
def toxic_plant_detection_page():
    """Renders a page for toxic plant image upload."""
    return render_template('ToxicPlantDetection.html') 


#Amine model 2
@app.route('/landslide_detection_page', methods=['GET'])
@login_required
def landslide_detection_page():
    return render_template('LandslideDetection.html')

@app.route('/detect_landslide', methods=['POST'])
@login_required
def detect_landslide_route():
    result = detect_landslide_logic(request)
    if 'error' in result:
        return render_template('error.html', message=result['error'])
    else:
        return render_template(
            'landslide_result.html',
            result=result['result'],
            uploaded_image_url=result['uploaded_image_url'],
            prediction_url=result['prediction_url'],
            show_landslide=result['show_landslide'],
        )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Threaded=True for webcam