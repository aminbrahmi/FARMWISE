import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Pour éviter tkinter en arrière-plan


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
        processed[:, :, 4] = 1 - data[:, :, 12] / mid_slope     # slope
        processed[:, :, 5] = 1 - data[:, :, 13] / mid_elevation # elevation

        print("✅ Normalized shape:", processed.shape)
        return processed

def save_prediction_preview_from_raw(raw_data, prediction_mask, filename, folder="static/preview"):
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

