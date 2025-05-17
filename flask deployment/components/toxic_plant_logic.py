import torch
from torchvision import transforms
from PIL import Image
import io
import os
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import torch.nn as nn

# Mapping from folder name to (Plant name, Toxicity)
plant_labels = {
    '000': ("Western Poison Oak", True),
    '001': ("Eastern Poison Oak", True),
    '002': ("Eastern Poison Ivy", True),
    '003': ("Western Poison Ivy", True),
    '004': ("Poison Sumac", True),
    '100': ("Virginia creeper", False),
    '101': ("Boxelder", False),
    '102': ("Jack-in-the-pulpit", False),
    '103': ("Bear Oak", False),
    '104': ("Fragrant Sumac", False)
}

# Construct the path relative to the current file's directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, '..', 'content', 'efficientnet_b4_toxicplant_multiclass_final1.pth')

# Recreate the model architecture
weights = EfficientNet_B4_Weights.DEFAULT
model = efficientnet_b4(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(plant_labels)) # Adjust output size

# Load the trained weights (state_dict)
print(f"Attempting to load model weights from: {MODEL_PATH}")
try:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print('Toxic plant model loaded successfully (from logic file)!')
except Exception as e:
    print(f'Error loading toxic plant model (from logic file): {e}')
    model = None

# Define transformations for the toxic plant model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define class labels for the toxic plant model (must match the order of classes during training)
class_labels = list(plant_labels.keys())

def predict_toxic_plant(image_bytes):
    """Performs inference for the toxic plant classification model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transformed_image = transform(image).unsqueeze(0).to(torch.device('cpu')) # Ensure tensor is on CPU for inference

        if model is not None:
            with torch.no_grad():
                outputs = model(transformed_image)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_folder_code = class_labels[predicted_idx.item()]
                plant_name, is_toxic = plant_labels[predicted_folder_code]
                probability = torch.softmax(outputs, dim=1)[0, predicted_idx].item()
                return {'predicted_plant': plant_name, 'is_toxic': is_toxic, 'probability': probability}
        else:
            return {'error': 'Toxic plant model not loaded (in logic)'}
    except Exception as e:
        return {'error': f'Toxic plant prediction error (in logic): {e}'}