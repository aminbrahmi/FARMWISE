🌾 Agricultural Land Price Prediction in Tunisia

This project aims to predict the price per square meter (TND/m²) of agricultural lands in Tunisia based on a wide range of features such as location, land size, proximity to the sea or city, infrastructure availability, type of agriculture, and other property characteristics.

🎯 Objective
To build an intelligent and accurate machine learning model capable of:

Estimating the price of a given agricultural land parcel.

Helping stakeholders (buyers, investors, and agencies) make informed decisions about land value.

Detecting price anomalies and supporting fair land valuation practices across Tunisia.

🧠 Methodology
Data Collection & Cleaning
A real estate dataset of agricultural lands in Tunisia was used. The data was cleaned, standardized (e.g., land size in m²), and enriched with additional features extracted from text (proximity, infrastructure, etc.).

Unsupervised Segmentation (Clustering)
Lands were segmented into meaningful clusters using PCA and KMeans, capturing patterns like:

Coastal premium lands

Rural low-cost fields

City-adjacent mixed-value lands
➕ The cluster label was then used as a feature to improve model learning.

Modeling
Several regression models were tested including:

Linear Regression

Random Forest

XGBoost (final model)
The best performance was achieved with XGBoost, reaching an R² score of ~0.71 and a MAE of ~35 TND/m².

Evaluation & Deployment
The trained model was tested on new unseen examples and generalized well across different regions. A pipeline was built using scikit-learn and XGBoost, allowing fast inference for any new land input.

🚀 Example Output
For a land of 2100 m² in Kelibia (Nabeul) near the sea with electricity and vineyard potential, the model predicted:

Estimated price/m²: 🌾 Agricultural Land Price Prediction in Tunisia
This project aims to predict the price per square meter (TND/m²) of agricultural lands in Tunisia based on a wide range of features such as location, land size, proximity to the sea or city, infrastructure availability, type of agriculture, and other property characteristics.

🎯 Objective
To build an intelligent and accurate machine learning model capable of:

Estimating the price of a given agricultural land parcel.

Helping stakeholders (buyers, investors, and agencies) make informed decisions about land value.

Detecting price anomalies and supporting fair land valuation practices across Tunisia.

🧠 Methodology
Data Collection & Cleaning
A real estate dataset of agricultural lands in Tunisia was used. The data was cleaned, standardized (e.g., land size in m²), and enriched with additional features extracted from text (proximity, infrastructure, etc.).

Unsupervised Segmentation (Clustering)
Lands were segmented into meaningful clusters using PCA and KMeans, capturing patterns like:

Coastal premium lands

Rural low-cost fields

City-adjacent mixed-value lands
➕ The cluster label was then used as a feature to improve model learning.

Modeling
Several regression models were tested including:

Linear Regression

Random Forest

XGBoost (final model)
The best performance was achieved with XGBoost, reaching an R² score of ~0.71 and a MAE of ~35 TND/m².

Evaluation & Deployment
The trained model was tested on new unseen examples and generalized well across different regions. A pipeline was built using scikit-learn and XGBoost, allowing fast inference for any new land input.

🚀 Example Output
For a land of 2100 m² in Kelibia (Nabeul) near the sea with electricity and vineyard potential, the model predicted:

Estimated price/m²: 593.81 TND

Total estimated price: 1247003.30 TND

📦 Technologies
Python, Pandas, scikit-learn, XGBoost, Matplotlib

Clustering (PCA + KMeans)

Model evaluation (MAE, RMSE, R²)

Optional deployment-ready pipeline using joblib TND

📦 Technologies
Python, Pandas, scikit-learn, XGBoost, Matplotlib

Clustering (PCA + KMeans)

Model evaluation (MAE, RMSE, R²)

Optional deployment-ready pipeline using joblib