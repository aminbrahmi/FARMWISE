import torch

# Charger le modèle une seule fois
model = torch.hub.load('ultralytics/yolov5', 'custom', path='content\ModelYoloMehdi.pt', source='github')  # 'yolov5' = nom du dossier cloné

def detect_objects(img_path):
    results = model(img_path)
    labels = results.pandas().xyxy[0]['name']
    return results, labels
