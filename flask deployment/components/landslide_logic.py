import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour éviter tkinter en arrière-plan
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.saving
import os
from flask import request  # You'll need this if you process request data here
from werkzeug.utils import secure_filename  # Import secure_filename

# Define your upload and preview folders here as well
LANDSLIDE_UPLOAD_FOLDER = 'uploads'
LANDSLIDE_PREVIEW_FOLDER = 'static/preview'

# Ensure upload folders exist (you might want to do this in app.py)
if not os.path.exists(LANDSLIDE_UPLOAD_FOLDER):
    os.makedirs(LANDSLIDE_UPLOAD_FOLDER)
if not os.path.exists(LANDSLIDE_PREVIEW_FOLDER):
    os.makedirs(LANDSLIDE_PREVIEW_FOLDER)

def allowed_file(filename, allowed_extensions):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    )

def preprocess_h5_image(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["img"][:]  # assuming dataset is called 'img'

        if data.shape[-1] != 14:
            raise ValueError("Expected 14 channels in the .h5 image")

        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        red = data[:, :, 3]
        green = data[:, :, 2]
        blue = data[:, :, 1]
        nir = data[:, :, 7]

        # NDVI
        ndvi = np.divide(nir - red, nir + red + 1e-5)  # avoid division by zero

        # Prepare final input array
        processed = np.zeros((128, 128, 6), dtype=np.float32)
        processed[:, :, 0] = 1 - red / mid_rgb
        processed[:, :, 1] = 1 - green / mid_rgb
        processed[:, :, 2] = 1 - blue / mid_rgb
        processed[:, :, 3] = ndvi
        processed[:, :, 4] = 1 - data[:, :, 12] / mid_slope    # slope
        processed[:, :, 5] = 1 - data[:, :, 13] / mid_elevation  # elevation

        print("✅ Normalized shape:", processed.shape)
        return processed

def save_prediction_preview_from_raw(raw_data, prediction_mask, filename, folder=LANDSLIDE_PREVIEW_FOLDER):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 🎨 Display real RGB from raw_data (channels 1=Blue, 2=Green, 3=Red)
    red = raw_data[:, :, 3]
    green = raw_data[:, :, 2]
    blue = raw_data[:, :, 1]

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-5)

    ax1.imshow(rgb)
    ax1.set_title("Uploaded Image")
    ax1.axis("off")

    ax2.imshow(rgb)
    ax2.imshow(prediction_mask, cmap="Reds", alpha=0.5)
    ax2.set_title("Predicted Mask")
    ax2.axis("off")

    output_path = os.path.join(folder, filename.replace(".h5", ".png"))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return "/" + output_path.replace("\\", "/")

@tensorflow.keras.saving.register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

@tensorflow.keras.saving.register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

@tensorflow.keras.saving.register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Load the landslide detection model (keep this in the logic file)
try:
    landslide_model = load_model(
        "content/landslide_model.keras",
        custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m},
    )
except Exception as e:
    print(f"Error loading landslide model in landslide_logic: {e}")
    landslide_model = None

def detect_landslide_logic(request):
    if landslide_model is None:
        return render_template(
            'error.html', message='Landslide detection model not loaded.'
        )

    result = None
    uploaded_image_url = None
    prediction_url = None
    show_landslide = False  # New flag to control display

    if 'file' not in request.files:
        return render_template('error.html', message='No .h5 image file provided')
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message='No .h5 image selected')
    if file and allowed_file(file.filename, {'h5'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join(LANDSLIDE_UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
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
            uploaded_path = os.path.join(LANDSLIDE_PREVIEW_FOLDER, uploaded_filename)
            plt.imsave(uploaded_path, rgb)
            uploaded_image_url = "/" + uploaded_path.replace("\\", "/")

            # Only save and show mask if landslide detected
            if show_landslide:
                mask_filename = filename.replace(".h5", "_mask.png")
                prediction_url = save_prediction_preview_from_raw(
                    raw_data, mask, filename=mask_filename
                )

            return {
                'result': result,
                'uploaded_image_url': uploaded_image_url,
                'prediction_url': prediction_url,
                'show_landslide': show_landslide,
            }

        except Exception as e:
            return {'error': f'Error processing .h5 file: {e}'}
    else:
        return {'error': 'Invalid file type. Please upload a .h5 file.'}