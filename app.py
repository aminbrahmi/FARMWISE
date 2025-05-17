from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import pandas as pd
import os
import numpy as np
from keras.models import load_model
from utils.utils_landslide import preprocess_h5_image
import keras
import keras.backend as K
from flask import send_from_directory
from utils.utils_landslide import preprocess_h5_image, save_prediction_preview_from_raw
import h5py

import matplotlib.pyplot as plt



app = Flask(__name__)
app.secret_key = 'secret_key_for_session'  # Required for session

#-----------------------------------------Land Price-------------------------------------------------

# Update file paths to the new folder structure
df_landprice = pd.read_csv("utils/agriculture_lands_tn.csv")
pipeline_landprice = joblib.load("content/xgboost_model2.pkl")

@app.route("/", methods=["GET"])
def landprice():
    gouvernorats = sorted(df_landprice["Gouvernorat"].dropna().unique())
    delegations = sorted(df_landprice["Délégation"].dropna().unique())
    agriculture_types = sorted(df_landprice["Type_Agriculture"].dropna().unique())  # Ensure this comes from the cleaned data
    prediction_text = session.pop("prediction_text", "")
    return render_template("landprice.html", gouvernorats=gouvernorats, delegations=delegations, agriculture_types=agriculture_types, prediction_text=prediction_text)


@app.route("/get_delegations", methods=["POST"])
def get_delegations():
    selected_gov = request.json.get("gouvernorat")
    filtered_delegations = df_landprice[df_landprice["Gouvernorat"] == selected_gov]["Délégation"].dropna().unique().tolist()
    return {"delegations": filtered_delegations}

@app.route("/get_proximites", methods=["POST"])
def get_proximites():
    data = request.json
    gov = data.get("gouvernorat")
    delg = data.get("delegation")
    filtered = df_landprice[(df_landprice["Gouvernorat"] == gov) & (df_landprice["Délégation"] == delg)]
    proximites = filtered["Proximité"].dropna().unique().tolist()
    return {"proximites": proximites}

@app.route("/get_agriculture_types", methods=["POST"])
def get_agriculture_types():
    data = request.json
    gov = data.get("gouvernorat")
    delg = data.get("delegation")
    
    # Filter the dataset based on Governorate and Delegation
    filtered_data = df_landprice[(df_landprice["Gouvernorat"] == gov) & (df_landprice["Délégation"] == delg)]
    agriculture_types = filtered_data["Type_Agriculture"].dropna().unique().tolist()
    
    return {"agriculture_types": agriculture_types}


@app.route("/predictPrice", methods=["POST"])
def predict():
    try:
        # Handling optional fields: Proximity, Infrastructure, Type_Agriculture, and Additional_Features
        infrastructure_selected = request.form.getlist('Infrastructure')  # List of selected Infrastructure options
        additional_features_selected = request.form.getlist('Additional_Features')  # List of selected Additional Features

        # Convert the lists into a string (comma-separated values), or use "aucun" if empty
        infrastructure_str = ', '.join(infrastructure_selected) if infrastructure_selected else 'aucun'
        additional_features_str = ', '.join(additional_features_selected) if additional_features_selected else 'aucun'

        # Collecting data from the form
        data = {
            'Gouvernorat': request.form['Gouvernorat'],
            'Délégation': request.form['Délégation'],
            'Proximité': request.form.get('Proximité') or 'aucun',  # Handle empty proximity
            'Infrastructure': infrastructure_str,
            'Type_Agriculture': request.form.get('Type_Agriculture') or 'aucun',  # Handle empty agriculture type
            'Additional_Features': additional_features_str,
            'Taille_m2': float(request.form['Taille_m2'])
        }

        # Convert data to DataFrame for prediction
        input_df = pd.DataFrame([data])

        # Predict price per m² using the pipeline model
        prix_m2 = pipeline_landprice.predict(input_df)[0]
        prix_total = prix_m2 * input_df["Taille_m2"].values[0]

        # Save the selected features and the predicted result
        session["prediction_text"] = f"Estimated Price: {prix_total:,.2f} TND ({prix_m2:.2f} TND/m²)"
        return redirect(url_for("show_result", prediction_text=session["prediction_text"], **data))

    except Exception as e:
        session["prediction_text"] = f"Error: {str(e)}"
        return redirect(url_for("landprice"))


@app.route("/result")
def show_result():
    prediction_text = request.args.get("prediction_text")
    governorate = request.args.get("Gouvernorat")
    delegation = request.args.get("Délégation")
    proximity = request.args.get("Proximité")
    infrastructure = request.args.get("Infrastructure")
    agriculture_type = request.args.get("Type_Agriculture")
    additional_features = request.args.get("Additional_Features")
    size = request.args.get("Taille_m2")
    
    return render_template("landprice_result.html", prediction_text=prediction_text, 
                           governorate=governorate, delegation=delegation, proximity=proximity,
                           infrastructure=infrastructure, agriculture_type=agriculture_type, 
                           additional_features=additional_features, size=size)

#-----------------------------------------Landslide-------------------------------------------------



@keras.saving.register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@keras.saving.register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

@keras.saving.register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

landslide_model = load_model("content/landslide_model.keras")

@app.route("/detect-landslide", methods=["GET", "POST"])
def detect_landslide():
    result = None
    uploaded_image_url = None
    prediction_url = None
    show_landslide = False  # New flag to control display

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".h5"):
            filename = file.filename
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            with h5py.File(filepath, "r") as f:
                raw_data = f["img"][:]

            # Preprocess image
            img_array = preprocess_h5_image(filepath)
            prediction = landslide_model.predict(np.expand_dims(img_array, axis=0))
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

            pixel_count = np.count_nonzero(mask)
            if pixel_count > 10:
                result = "🌋 Landslide Detected"
                show_landslide = True
            else:
                result = "✅ No Landslide Detected"

            # Save uploaded image (RGB)
            uploaded_filename = filename.replace(".h5", "_uploaded.png")
            rgb = raw_data[:, :, [3, 2, 1]]
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-5)
            uploaded_path = os.path.join("static/preview", uploaded_filename)
            plt.imsave(uploaded_path, rgb)
            uploaded_image_url = "/" + uploaded_path.replace("\\", "/")

            # Only save and show mask if landslide detected
            if show_landslide:
                mask_filename = filename.replace(".h5", "_mask.png")
                prediction_url = save_prediction_preview_from_raw(raw_data, mask, filename=mask_filename)

    return render_template(
        "landslide.html",
        result=result,
        uploaded_image_url=uploaded_image_url,
        prediction_url=prediction_url,
        show_landslide=show_landslide  # Pass the flag to template
    )


def save_prediction_preview(original_array, prediction_mask, filename, folder="static/preview"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    rgb = original_array[:, :, :3]
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-5)

    ax1.imshow(rgb)
    ax1.set_title("Uploaded Image")
    ax1.axis("off")

    ax2.imshow(rgb)
    ax2.imshow(prediction_mask, cmap='Reds', alpha=0.5)
    ax2.set_title("Predicted Mask")
    ax2.axis("off")

    output_path = os.path.join(folder, filename)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return "/" + output_path.replace("\\", "/")




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
    app.run(debug=True)
