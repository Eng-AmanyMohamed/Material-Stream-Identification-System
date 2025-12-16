from train_knn import predict_with_unknown
from feature_extraction import extract_feature_vector_single
import joblib
import numpy as np

# Load the full pipeline (model + preprocessing)
data = joblib.load("models/knn_model.pkl")
knn_model = data['model']
scaler = data['scaler']
pca = data['pca']
classes = data['classes']
threshold = data.get('rejection_threshold', 0.85)

image_path = "images/bricks.jpg"

# Step 1: Extract RAW features (538-D)
raw_features = extract_feature_vector_single(image_path)  # Shape: (1, 538)

# Step 2: Apply SAME preprocessing as during training
scaled_features = scaler.transform(raw_features)   # Standardize
pca_features = pca.transform(scaled_features)      # Reduce to 100-D

# Step 3: Predict with rejection
pred, confidence = predict_with_unknown(pca_features,threshold=threshold)

if pred == "Unknown":
    print("\nPredicted: UNKNOWN MATERIAL")
else:
    print(f"\nPredicted: {classes[pred]}")
# print(f"Confidence: {confidence:.4f}")
