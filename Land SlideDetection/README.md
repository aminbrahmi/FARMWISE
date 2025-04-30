🏔️ Landslide Detection Using Deep Learning


This project focuses on building a semantic segmentation model to automatically detect landslides from satellite images using deep learning techniques in Keras. The goal is to support rapid response and risk assessment in areas prone to landslides.

🧠 Objective
To develop and evaluate a semantic segmentation model capable of detecting landslides in satellite images, using a U-Net-like architecture with custom metrics such as F1-score, precision, and recall.

📁 Dataset
Input: RGB satellite images with potential landslide areas.

Labels: Binary mask images where landslide regions are marked.

The dataset is divided into:

Training set

Validation set

Test set

🛠️ Model Architecture
The model is built using:

Convolutional layers with ReLU activation

MaxPooling and UpSampling

Sigmoid activation at the output

Custom metrics: f1_m, precision_m, and recall_m

✅ Results
Evaluation is done by comparing predicted masks with ground truth labels.

The results are visualized using side-by-side comparison of:

Original image

True label

Predicted mask

The trained model achieved satisfactory performance, highlighting landslide regions with good precision and recall.

📊 Example Output
For one test image:

Prediction Mask: Highlights the detected landslide areas.

Ground Truth: Actual labeled regions from the dataset.

Input Image: RGB satellite image.